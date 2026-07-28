"""Measure a vehicle classifier per VEHICLE on the deployment camera.

CLAUDE.md §4 carries a table comparing the size heuristic, two trained
classifiers and the hybrid — and until this script existed that table could not
be reproduced. It was measured once, by hand, and every later decision leaned on
numbers nobody could re-derive. A measurement that cannot be re-run is an
opinion with a decimal point.

Three things this gets right that a naive `model.val()` does not:

**Scored per vehicle, not per crop.** The pipeline classifies a vehicle from
every view it gets and votes once. Scoring loose crops measures something the
system never does, and flatters classes that happen to have more crops.

**The same voting the pipeline uses.** Predictions are accumulated through the
real ``VehicleClassifier``, so the hybrid C/D deferral and the crop-size floor
are exercised here exactly as they are in a run. A number measured through a
different code path is a number about different code.

**Compared against the thing it replaces.** A classifier is only worth shipping
if it beats the heuristic already in the tree, so both are scored on the same
vehicles in the same pass, plus the hybrid of the two.

    python -m tools.eval_vehicle_cls --model models/vehicle_cls.pt
    python -m tools.eval_vehicle_cls --model models/vehicle_cls.pt --json-out r.json

Ground truth is the `label` column of data/dataset/manifest.csv, one label per
(clip, track, carriageway) — never per crop, since the harvester's two passes
number tracks independently and (clip, track) alone merges two different
vehicles. That bug has appeared in three separate files; see CLAUDE.md §5.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.anpr import imread_unicode                        # noqa: E402
from pipeline.classify import (                                 # noqa: E402
    HEAVY_MIN_FRONTAL_AREA_M2,
    MENTOR_CLASSES,
    CropClassifier,
    EVIDENCE_REF_PX,
)

MANIFEST = ROOT / "data" / "dataset" / "manifest.csv"
CROPS = ROOT / "data" / "dataset" / "crops"
CODES = list(MENTOR_CLASSES)


def _heuristic(coco: int, h_m: float, w_m: float) -> str:
    """The shipping size heuristic, as classify.VehicleClassifier._map applies it."""
    if coco in (1, 3):
        return "G"
    if coco == 5:
        return "E"
    if coco == 2:
        return "A"
    if coco == 7:
        return "D" if h_m * w_m >= HEAVY_MIN_FRONTAL_AREA_M2 else "C"
    return "F"


def load_vehicles() -> dict[tuple, dict]:
    """Ground truth and crops, grouped one entry per real vehicle."""
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found — run tools.harvest_dataset first")
    with open(MANIFEST, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    by: dict[tuple, dict] = defaultdict(
        lambda: {"crops": [], "labels": Counter(), "coco": Counter(),
                 "h": [], "w": []})
    for r in rows:
        label = (r.get("label") or "").strip().upper()
        if label not in CODES:
            continue
        # Carriageway is part of identity — see the module docstring.
        key = (r["clip"], r["track"], r.get("carriageway", ""))
        v = by[key]
        v["crops"].append(CROPS / r["file"])
        v["labels"][label] += 1
        try:
            v["coco"][int(r["coco"])] += 1
            v["h"].append(float(r["height_m"]))
            v["w"].append(float(r["width_m"]))
        except (ValueError, KeyError, TypeError):
            pass
    return dict(by)


def evaluate(model_path: str, min_px: int, device: str) -> dict:
    vehicles = load_vehicles()
    clf = CropClassifier(model_path, device=device, min_px=min_px)
    if clf.model is None:
        raise SystemExit(f"classifier unusable: {clf.error}")

    got = {"heuristic": {}, "classifier": {}, "hybrid": {}}
    truth: dict[tuple, str] = {}
    declined = 0
    for key, v in vehicles.items():
        truth[key] = v["labels"].most_common(1)[0][0]
        h_m = float(np.median(v["h"])) if v["h"] else 0.0
        w_m = float(np.median(v["w"])) if v["w"] else 0.0
        coco = v["coco"].most_common(1)[0][0] if v["coco"] else 2
        heur = _heuristic(coco, h_m, w_m)
        got["heuristic"][key] = heur

        # Same size weighting the live classifier applies, so a near view counts
        # for more than a distant one here too.
        votes: dict[str, float] = defaultdict(float)
        for p in v["crops"]:
            img = imread_unicode(p)
            if img is None:
                continue
            res = clf.predict(img)
            if res is None:
                continue                      # below the crop-size floor
            code, conf = res
            votes[code] += conf * max(img.shape[0] / EVIDENCE_REF_PX, 1e-3)
        if not votes:
            declined += 1
            got["classifier"][key] = heur     # nothing to say; keep the heuristic
            got["hybrid"][key] = heur
            continue
        best = max(votes, key=votes.get)
        got["classifier"][key] = best
        # The shipped hybrid: the model names the vehicle, and where that lands
        # on C-or-D the frontal-area estimate decides which.
        got["hybrid"][key] = (
            _heuristic(7, h_m, w_m) if best in ("C", "D") and h_m and w_m else best)

    out = {"vehicles": len(truth), "declined_all_crops": declined, "methods": {}}
    for name, pred in got.items():
        correct = sum(1 for k in truth if pred.get(k) == truth[k])
        per_class = {}
        for c in CODES:
            keys = [k for k in truth if truth[k] == c]
            if keys:
                per_class[c] = {
                    "n": len(keys),
                    "acc": round(sum(1 for k in keys if pred.get(k) == c) / len(keys), 3),
                }
        confusion = Counter((truth[k], pred.get(k, "-")) for k in truth)
        out["methods"][name] = {
            "overall": round(correct / len(truth), 3) if truth else 0.0,
            "per_class": per_class,
            "confusion": {f"{a}->{b}": n for (a, b), n in
                          sorted(confusion.items()) if a != b},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/vehicle_cls.pt")
    ap.add_argument("--min-px", type=int, default=64,
                    help="crop-size floor below which the classifier declines "
                         "(classify.MIN_CLS_CROP_PX)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    r = evaluate(args.model, args.min_px, args.device)
    print(f"model    : {args.model}")
    print(f"vehicles : {r['vehicles']} from the DEPLOYMENT camera "
          f"(hand-labelled ground truth)")
    if r["declined_all_crops"]:
        print(f"           {r['declined_all_crops']} had no crop above the "
              f"{args.min_px}px floor — heuristic used for those")
    classes = sorted({c for m in r["methods"].values() for c in m["per_class"]})
    print(f"\n{'method':<12} {'overall':>8}  " +
          "  ".join(f"{c:>5}" for c in classes))
    for name, m in r["methods"].items():
        cells = "  ".join(
            f"{m['per_class'][c]['acc']:>5.3f}" if c in m["per_class"] else "    -"
            for c in classes)
        print(f"{name:<12} {m['overall']:>8.3f}  {cells}")
    print("\nn per class: " + "  ".join(
        f"{c}={r['methods']['hybrid']['per_class'][c]['n']}"
        for c in classes if c in r["methods"]["hybrid"]["per_class"]))
    worst = r["methods"]["hybrid"]["confusion"]
    if worst:
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:6]
        print(f"hybrid's main confusions: {dict(top)}")
    thin = [c for c in classes
            if r["methods"]["hybrid"]["per_class"][c]["n"] < 10]
    if thin:
        print(f"\n[CHECK] classes with fewer than 10 vehicles {thin} — each one "
              f"moves that column by ~10 points.\n        Treat those figures as "
              f"directional, not as measurements.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
