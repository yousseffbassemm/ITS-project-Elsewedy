"""Measure the ANPR cascade end to end, on held-out EALPR vehicles.

Every accuracy number this project quotes has to come from somewhere a reader
can re-run, and for ANPR there are three different things that can be measured
and only one of them is the answer:

    stage 2 alone   did we find the plate on the vehicle?      (recall / IoU)
    stage 3 alone   given a perfect plate crop, what do we read? (char accuracy)
    END TO END      vehicle photo in, plate string out           <- the answer

Reporting stage 3 alone is the trap. It is measured on ground-truth plate crops,
so it silently assumes stage 2 was perfect, and it is always the flattering
number. This script reports all three and leads with the end-to-end one.

The ground truth is EALPR's own: each vehicle photo `0001.jpg` has a plate box
in `Vehicles Labeling/0001.txt`, and the characters of that same plate are boxed
in `Characters Labeling/0001_license_plate_1.txt`. The two were annotated
separately and are joined here by name, which is what makes an end-to-end score
possible at all.

    python -m tools.eval_anpr                          # existing pretrained weights
    python -m tools.eval_anpr --stage2 models/plate_on_vehicle.pt \\
                              --stage3 models/plate_chars.pt

**This is an in-domain number.** EALPR is close-up photography; the deployment
camera is 720p CCTV at 30 m. CLAUDE.md §5 records a plate detector that scored
mAP50 0.985 here and regressed on real footage. Treat this as "does the cascade
work at all", never as "this is what it will do on the street".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.anpr import (                                    # noqa: E402
    CharReader,
    PlateLocator,
    assemble,
    imread_unicode,
)

EALPR = ROOT / "data" / "plates" / "EALPR"
VEH_IMG = EALPR / "EALPR Vechicles dataset" / "Vehicles"
VEH_LBL = EALPR / "EALPR Vechicles dataset" / "Vehicles Labeling"
PLATE_IMG = EALPR / "EALPR- Plates dataset"
CHAR_LBL = EALPR / "EALPR- LP characters dataset" / "Characters Labeling"
CHARMAP = ROOT / "data" / "plates" / "EALPR_charmap.json"
STAGE2_DS = ROOT / "data" / "plates" / "stage2_plate_on_vehicle"


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _rows(path: Path) -> list[list[float]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
        p = line.split()
        if len(p) == 5:
            try:
                out.append([float(v) for v in p])
            except ValueError:
                pass
    return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _truth_string(stem: str, names: dict[int, str]) -> tuple[str, str]:
    """(arabic, visual) ground-truth plate string for vehicle `stem`."""
    lbl = CHAR_LBL / f"{stem}_license_plate_1.txt"
    if not lbl.exists():
        return "", ""
    chars = [{"glyph": names[int(r[0])], "x": r[1], "conf": 1.0}
             for r in _rows(lbl) if int(r[0]) in names]
    if not chars:
        return "", ""
    arabic, _latin, visual, _l, _d = assemble(chars)
    return arabic, visual


def _char_accuracy(pred: str, truth: str) -> float:
    """Fraction of ground-truth glyphs recovered, order-insensitive.

    A multiset comparison, not a positional one: a read that finds every
    character but drops one in the middle would score near zero positionally
    while being obviously a near-miss, and the distinction between "read the
    plate badly" and "read a different plate" is what matters here.
    """
    if not truth:
        return 0.0
    have, want = Counter(pred), Counter(truth)
    return sum((have & want).values()) / sum(want.values())


def evaluate(stage2: str, stage3: str, limit: int | None, device: str) -> dict:
    if not CHARMAP.exists():
        raise SystemExit("run `python -m tools.ealpr_charmap` first")
    names = {int(k): v for k, v in
             json.loads(CHARMAP.read_text(encoding="utf-8"))["map"].items()}

    # Evaluate on the HELD-OUT split only. Scoring on images the detector
    # trained on is the single easiest way to publish a number that means
    # nothing — and both of these weights may well have seen the train split.
    val_dir = STAGE2_DS / "val" / "images"
    if val_dir.exists():
        stems = sorted(p.stem for p in val_dir.glob("*.*"))
        split = "held-out val split"
    else:
        stems = sorted(p.stem for p in VEH_LBL.glob("*.txt"))
        split = "ALL images (no split built — run tools.build_anpr_datasets)"
    if limit:
        stems = stems[:limit]

    loc = PlateLocator(stage2, device=device)
    rdr = CharReader(stage3, device=device)
    if loc.model is None:
        raise SystemExit(f"stage 2 weights unusable: {loc.error}")
    if rdr.model is None:
        raise SystemExit(f"stage 3 weights unusable: {rdr.error}")

    found = ious = 0
    n_box = 0
    e2e_exact = e2e_char = e2e_n = 0
    s3_exact = s3_char = s3_n = 0
    ious_list: list[float] = []

    for stem in stems:
        img_p = next((p for ext in (".jpg", ".png", ".jpeg")
                      if (p := VEH_IMG / f"{stem}{ext}").exists()), None)
        gt_rows = _rows(VEH_LBL / f"{stem}.txt")
        if img_p is None or not gt_rows:
            continue
        img = imread_unicode(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy, bw, bh = gt_rows[0][1:]
        gt = [(cx - bw / 2) * w, (cy - bh / 2) * h,
              (cx + bw / 2) * w, (cy + bh / 2) * h]
        n_box += 1

        truth_ar, _truth_vis = _truth_string(stem, names)

        # --- stage 2: locate the plate on the whole vehicle image -----------
        got = loc.locate(img)
        if got is not None:
            (x1, y1, x2, y2), _c = got
            iou = _iou((x1, y1, x2, y2), gt)
            ious_list.append(iou)
            if iou >= 0.5:
                found += 1
                ious += iou

        # --- end to end: read from the plate stage 2 actually produced ------
        if truth_ar:
            e2e_n += 1
            pred_ar = ""
            if got is not None:
                (x1, y1, x2, y2), _c = got
                crop = img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                chars = rdr.read(crop) if crop.size else []
                if chars:
                    pred_ar = assemble(chars)[0]
            e2e_exact += int(pred_ar == truth_ar)
            e2e_char += _char_accuracy(pred_ar.replace(" ", ""),
                                       truth_ar.replace(" ", ""))

            # --- stage 3 alone: read from the GROUND-TRUTH plate crop -------
            plate_p = next((p for ext in (".png", ".jpg")
                            if (p := PLATE_IMG / f"{stem}_license_plate_1{ext}").exists()),
                           None)
            if plate_p is not None:
                pimg = imread_unicode(plate_p)
                if pimg is not None:
                    s3_n += 1
                    chars = rdr.read(pimg)
                    pred = assemble(chars)[0] if chars else ""
                    s3_exact += int(pred == truth_ar)
                    s3_char += _char_accuracy(pred.replace(" ", ""),
                                              truth_ar.replace(" ", ""))

    return {
        "split": split,
        "vehicles": n_box,
        "stage2": {
            "recall_iou50": round(found / n_box, 3) if n_box else 0.0,
            "mean_iou_when_found": round(ious / found, 3) if found else 0.0,
            "median_iou": round(float(np.median(ious_list)), 3) if ious_list else 0.0,
        },
        "stage3_on_truth_crops": {
            "n": s3_n,
            "exact_plate": round(s3_exact / s3_n, 3) if s3_n else 0.0,
            "char_accuracy": round(s3_char / s3_n, 3) if s3_n else 0.0,
        },
        "end_to_end": {
            "n": e2e_n,
            "exact_plate": round(e2e_exact / e2e_n, 3) if e2e_n else 0.0,
            "char_accuracy": round(e2e_char / e2e_n, 3) if e2e_n else 0.0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", default="models/plate_detect.pt")
    ap.add_argument("--stage3", default="models/eg_alpr.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    _utf8_stdout()

    r = evaluate(args.stage2, args.stage3, args.limit, args.device)
    print(f"stage 2: {args.stage2}")
    print(f"stage 3: {args.stage3}")
    print(f"split  : {r['split']}   vehicles {r['vehicles']}\n")
    print(f"stage 2  plate recall @IoU0.5 : {r['stage2']['recall_iou50']:.1%}"
          f"   median IoU {r['stage2']['median_iou']:.3f}")
    s3 = r["stage3_on_truth_crops"]
    print(f"stage 3  on TRUTH crops (n={s3['n']}): "
          f"exact {s3['exact_plate']:.1%}  chars {s3['char_accuracy']:.1%}")
    e = r["end_to_end"]
    print(f"END TO END (n={e['n']}): "
          f"exact {e['exact_plate']:.1%}  chars {e['char_accuracy']:.1%}")
    print("\nend-to-end is the honest number; stage 3 alone assumes a perfect "
          "plate crop.\nBoth are IN-DOMAIN on close-up photography — see the "
          "module docstring.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
