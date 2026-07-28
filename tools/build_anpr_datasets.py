"""Build the two YOLO datasets the ANPR cascade trains on, from EALPR.

The mentor's architecture is a cascade, and each arrow is a separate model with
a separate training set:

    frame --[COCO YOLO]--> vehicle --[stage 2]--> plate --[stage 3]--> characters
                                                        \\-> colour (no model)

Stage 1 needs no training: base COCO already finds vehicles, and that is not the
weak part of this pipeline. Stages 2 and 3 are what this script prepares.

**Stage 2 — plate localisation ON A VEHICLE.** Built from EALPR's vehicle
photographs with the plate box annotated. Trained on the whole vehicle rather
than on the frame, because that is what the cascade actually hands it: a crop
the tracker produced. This is also what makes the mentor's "the plate might be
on the sides, not always the middle" requirement testable rather than assumed —
the script reports where the plate actually sits across the training set, and on
this data it is off-centre far more often than not.

**Stage 3 — character recognition.** Built from EALPR's plate crops with each
character boxed. The class ids are EALPR's own integers, which mean nothing
without the legend recovered by tools/ealpr_charmap.py — so this script refuses
to run until that legend exists, and writes the glyph names into data.yaml so a
trained model reports Arabic rather than integers.

**The split is by PLATE.** Both datasets have several rows per underlying
object; letting two crops of one plate land either side of the split measures
memorisation. Same rule as tools/train_vehicle_classes.py, same reason.

    python -m tools.build_anpr_datasets              # both stages
    python -m tools.build_anpr_datasets --stage 2    # just plate localisation
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EALPR = ROOT / "data" / "plates" / "EALPR"
VEH_IMG = EALPR / "EALPR Vechicles dataset" / "Vehicles"
VEH_LBL = EALPR / "EALPR Vechicles dataset" / "Vehicles Labeling"
PLATE_IMG = EALPR / "EALPR- Plates dataset"
CHAR_LBL = EALPR / "EALPR- LP characters dataset" / "Characters Labeling"
CHARMAP = ROOT / "data" / "plates" / "EALPR_charmap.json"

OUT_PLATE = ROOT / "data" / "plates" / "stage2_plate_on_vehicle"
OUT_CHARS = ROOT / "data" / "plates" / "stage3_chars"


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _rows(path: Path) -> list[list[float]]:
    """YOLO label rows, tolerant of the stray bad bytes in this dataset."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            vals = [float(v) for v in p]
        except ValueError:
            continue
        # A box outside the image, or with no area, is a corrupt row. Ultralytics
        # accepts these silently and trains on them.
        if not all(0.0 <= v <= 1.0 for v in vals[1:]) or vals[3] <= 0 or vals[4] <= 0:
            continue
        out.append(vals)
    return out


def _write(out: Path, pairs: list[tuple[Path, list[list[float]]]],
           names: dict[int, str], val_frac: float, seed: int) -> dict:
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    pairs = sorted(pairs, key=lambda t: t[0].name)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    counts = Counter()
    per_class = Counter()
    for i, (img, rows) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        # Copy rather than symlink: this dataset gets zipped and uploaded to a
        # Colab runtime, where a symlink into a path that does not exist there
        # is an empty image and a silent training-set hole.
        shutil.copy(img, out / split / "images" / img.name)
        (out / split / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}"
                      for r in rows), encoding="utf-8")
        counts[split] += 1
        for r in rows:
            per_class[int(r[0])] += 1

    # Ultralytics resolves `path` relative to its own settings dir unless it is
    # absolute, which silently yields an empty dataset. Written absolute here and
    # rewritten by the Colab notebook for the runtime's own layout.
    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.resolve().as_posix()}",
            "train: train/images",
            "val: val/images",
            f"nc: {len(names)}",
            "names:",
            *[f"  {i}: {n}" for i, n in sorted(names.items())],
            "",
        ]), encoding="utf-8")
    return {"train": counts["train"], "val": counts["val"],
            "instances": dict(sorted(per_class.items()))}


def stage2(val_frac: float, seed: int) -> dict:
    """Plate localisation on a vehicle photograph."""
    if not VEH_LBL.exists():
        raise SystemExit(f"{VEH_LBL} not found — clone EALPR first.")
    pairs, positions = [], []
    for lbl in sorted(VEH_LBL.glob("*.txt")):
        img = next((p for ext in (".jpg", ".png", ".jpeg")
                    if (p := VEH_IMG / f"{lbl.stem}{ext}").exists()), None)
        rows = _rows(lbl)
        if img is None or not rows:
            continue
        # EALPR labels the plate as class 0 already; force it so a stray id
        # cannot create a phantom second class.
        rows = [[0.0, *r[1:]] for r in rows]
        pairs.append((img, rows))
        positions += [r[1] for r in rows]

    res = _write(OUT_PLATE, pairs, {0: "plate"}, val_frac, seed)
    # The mentor's specific concern, measured rather than assumed.
    off = sum(1 for x in positions if abs(x - 0.5) > 0.15)
    res["plate_x_off_centre_pct"] = round(100.0 * off / max(len(positions), 1), 1)
    res["plate_x_range"] = (round(min(positions), 3), round(max(positions), 3))
    return res


def stage3(val_frac: float, seed: int) -> dict:
    """Character recognition on a plate crop."""
    if not CHARMAP.exists():
        raise SystemExit(
            f"{CHARMAP} not found. EALPR's character class ids have no legend in "
            "the dataset; recover it first with\n"
            "    python -m tools.ealpr_charmap\n"
            "Training without it produces a model that predicts integers and "
            "cannot say which character each one is.")
    cm = json.loads(CHARMAP.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cm["map"].items()}

    pairs, lengths = [], Counter()
    for lbl in sorted(CHAR_LBL.glob("*.txt")):
        img = next((p for ext in (".png", ".jpg", ".jpeg")
                    if (p := PLATE_IMG / f"{lbl.stem}{ext}").exists()), None)
        rows = _rows(lbl)
        if img is None or not rows:
            continue
        # Drop any row whose class the legend does not cover — an unmapped id
        # would train a class the pipeline cannot name.
        rows = [r for r in rows if int(r[0]) in names]
        if not rows:
            continue
        pairs.append((img, rows))
        lengths[len(rows)] += 1

    # Contiguous ids: EALPR's set has a hole at 17 (the digit zero, which modern
    # Egyptian plates do not use), and Ultralytics requires 0..nc-1.
    remap = {old: new for new, old in enumerate(sorted(names))}
    pairs = [(img, [[float(remap[int(r[0])]), *r[1:]] for r in rows])
             for img, rows in pairs]
    compact = {remap[o]: names[o] for o in sorted(names)}

    res = _write(OUT_CHARS, pairs, compact, val_frac, seed)
    res["chars_per_plate"] = dict(sorted(lengths.items()))
    res["alphabet"] = "".join(compact[i] for i in sorted(compact))
    res["id_remap"] = remap
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(2, 3), default=None,
                    help="build only one stage (default: both)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _utf8_stdout()

    if args.stage in (None, 2):
        r = stage2(args.val_frac, args.seed)
        print(f"stage 2  plate-on-vehicle -> {OUT_PLATE}")
        print(f"  images   train {r['train']}  val {r['val']}")
        print(f"  plate centre-x spans {r['plate_x_range']}; "
              f"{r['plate_x_off_centre_pct']}% sit off-centre (>0.15 from the "
              f"middle) — the cascade must search the whole vehicle, not the middle")
    if args.stage in (None, 3):
        r = stage3(args.val_frac, args.seed)
        print(f"\nstage 3  characters -> {OUT_CHARS}")
        print(f"  images   train {r['train']}  val {r['val']}")
        print(f"  alphabet {len(r['alphabet'])} classes: {r['alphabet']}")
        print(f"  chars per plate: {r['chars_per_plate']}")
        rare = {i: n for i, n in r["instances"].items() if n < 50}
        if rare:
            print(f"  [CHECK] classes with <50 instances: {rare} — these will "
                  f"score badly and their errors will be invisible in the mean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
