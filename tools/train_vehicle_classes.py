"""Train the 7-class vehicle classifier (A/C/D/E/G/V/F) on a GPU.

This is the fix for the class problem: base COCO has four vehicle classes and the
mentor taxonomy has seven, so **C** (light truck) and **V** (van) cannot be
expressed at all and are currently guessed from a monocular size heuristic. A
lorry gets called a pickup and a van gets called a car because nothing in the
model knows those categories exist.

**Why a classifier and not a detector.** docs/finetuning-plan.md assumes a
detection fine-tune. Detection is not the broken part — YOLO finds the vehicles
reliably, and the tracker holds them across occlusions. What is wrong is the
LABEL. A second-stage classifier over the tracked crop is a much better fit here:

  * it trains on the crops tools/harvest_dataset.py already produces, so the
    annotation job is "pick one of seven letters per vehicle" rather than
    redrawing boxes on full frames
  * it inherits the pipeline's per-track voting, so a vehicle is classified from
    every view of it rather than from one frame
  * detection quality is untouched, so a bad classifier run cannot cost counts

The pipeline consumes it through the same path a fine-tuned detector would —
class codes — so nothing downstream changes.

    # 1. harvest crops, then correct the `label` column in manifest.csv
    python -m tools.harvest_dataset
    # 2. build a split dataset from the corrected labels
    python -m tools.train_vehicle_classes --prepare
    # 3. on a Colab/Kaggle GPU
    python -m tools.train_vehicle_classes --train --epochs 60 --device 0

**The split is by VEHICLE, never by crop.** Three crops of one car are three
views of the same object taken seconds apart; letting one land in train and
another in val measures memorisation and reports it as accuracy. This is the
single easiest way to produce a model that scores 0.97 and helps nothing.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.classify import MENTOR_CLASSES          # noqa: E402

DATASET = ROOT / "data" / "dataset"
CLS_DIR = DATASET / "cls"
CODES = list(MENTOR_CLASSES)                           # A C D E G V F

# Classes too rare to learn. Below this many VEHICLES a class contributes noise
# and an inflated per-class metric on two or three examples; the plan's own
# target is >=400 each for C and V.
MIN_VEHICLES_PER_CLASS = 12


def _load(manifest: Path) -> list[dict]:
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} not found — run `python -m tools.harvest_dataset` first")
    with open(manifest, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def prepare(manifest: Path, val_frac: float, seed: int,
            use_draft: bool, include_opposite: bool) -> int:
    rows = _load(manifest)
    if not include_opposite:
        rows = [r for r in rows if r.get("carriageway") == "analysed"]

    # One label per VEHICLE. The human corrects a vehicle, not a crop, so a
    # disagreement between crops of one vehicle is a labelling slip — take the
    # majority and report how often it happened.
    # The carriageway is part of a vehicle's identity: the harvester's two
    # passes number their tracks independently, so track 2 exists in both and
    # means two different vehicles. Grouping on (clip, track) alone silently
    # merged those pairs, which showed up as bogus "conflicting" labels — two
    # genuinely different vehicles disagreeing about what they are — and lost
    # 7 of 71 vehicles from the training set.
    by_vehicle: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_vehicle[(r["clip"], r["track"], r["carriageway"])].append(r)

    labelled: dict[tuple, str] = {}
    conflicts = 0
    unlabelled = 0
    for key, items in by_vehicle.items():
        votes = Counter(r["label"].strip().upper() for r in items
                        if r["label"].strip())
        if not votes:
            if not use_draft:
                unlabelled += 1
                continue
            votes = Counter(r["draft_label"].strip().upper() for r in items
                            if r["draft_label"].strip())
            if not votes:
                unlabelled += 1
                continue
        if len(votes) > 1:
            conflicts += 1
        code = votes.most_common(1)[0][0]
        if code in CODES:
            labelled[key] = code

    per_class = Counter(labelled.values())
    usable = {c for c, n in per_class.items() if n >= MIN_VEHICLES_PER_CLASS}
    dropped = {c: n for c, n in per_class.items() if c not in usable}

    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)
    for split in ("train", "val"):
        for c in sorted(usable):
            (CLS_DIR / split / c).mkdir(parents=True, exist_ok=True)

    # Split by vehicle, stratified per class so a rare class is present in both.
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}
    by_class: dict[str, list[tuple]] = defaultdict(list)
    for key, code in labelled.items():
        if code in usable:
            by_class[code].append(key)
    for code, keys in by_class.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_frac))
        val = set(keys[:n_val])
        for key in keys:
            split = "val" if key in val else "train"
            for r in by_vehicle[key]:
                src = DATASET / "crops" / r["file"]
                if src.exists():
                    shutil.copy(src, CLS_DIR / split / code / r["file"])
                    counts[split] += 1

    print(f"vehicles labelled : {len(labelled)}  "
          f"(unlabelled {unlabelled}, conflicting {conflicts})")
    print(f"per class         : {dict(sorted(per_class.items()))}")
    if dropped:
        print(f"DROPPED (< {MIN_VEHICLES_PER_CLASS} vehicles): {dropped}")
        print("  -> supplement these from public data before training; a class "
              "with a handful of examples\n     produces a confident model and a "
              "meaningless per-class score. See docs/finetuning-plan.md §2.")
    print(f"crops             : train {counts['train']}  val {counts['val']}")
    print(f"dataset           : {CLS_DIR}")
    if len(usable) < 2:
        raise SystemExit(
            "fewer than two usable classes — nothing to train. Label more "
            "vehicles (or pass --use-draft to bootstrap from the pipeline's own "
            "guesses, which trains the model to copy the heuristic it is meant "
            "to replace).")
    return 0


def train(epochs: int, imgsz: int, batch: int, device: str, base: str,
          out: Path, data: Path | None = None, project: str | None = None,
          workers: int = 8, name: str = "its7class") -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed:  pip install ultralytics")
    data = Path(data) if data else CLS_DIR
    if not (data / "train").exists():
        raise SystemExit(
            f"{data} not built. Either:\n"
            f"  python -m tools.prep_vehicle_dataset      (MIO-TCD, ~30k crops)\n"
            f"  python -m tools.train_vehicle_classes --prepare   (the 194 "
            f"hand-labelled deployment crops)")

    model = YOLO(base)
    kwargs = dict(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, name=name, patience=15, cos_lr=True, workers=workers,
        # Vehicles are photographed from behind by a fixed camera, so a mirrored
        # car is still that car — horizontal flip is safe here and doubles the
        # data. (Unlike PLATES, where mirroring reverses reading order; see
        # tools/train_plates.py.) Vertical flip is not a thing that happens.
        fliplr=0.5, flipud=0.0, degrees=8.0,
        # The deployment crops are small, blurry and JPEG-damaged. Training on
        # clean upscaled crops alone produces a model that meets nothing like
        # them, so lean on photometric augmentation.
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, erasing=0.3,
    )
    # Colab reclaims runtimes without warning and takes /content with it. A
    # Drive-backed project dir means best.pt survives a disconnect; this cost
    # three training runs before it was done. See CLAUDE.md §4.
    if project:
        kwargs["project"] = project
    model.train(**kwargs)

    best = Path(model.trainer.best)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.pt"
    shutil.copy(best, dest)
    print(f"\nclassifier -> {dest}")
    print(f"Deploy with:  ITS_VEHICLE_CLS={dest.as_posix()}")
    print("Then MEASURE it on the deployment camera, which is the only test "
          "that has ever predicted real behaviour here:\n"
          f"  python -m tools.eval_vehicle_cls --model {dest.as_posix()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true",
                    help="build the split dataset from manifest.csv")
    ap.add_argument("--train", action="store_true", help="fine-tune on a GPU")
    ap.add_argument("--manifest", default=str(DATASET / "manifest.csv"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-draft", action="store_true",
                    help="fall back to the pipeline's draft labels where a "
                         "human has not corrected one (bootstrap only)")
    ap.add_argument("--analysed-only", action="store_true",
                    help="ignore opposite-carriageway vehicles")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--base", default="yolov8s-cls.pt")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--data", default=None,
                    help="dataset root with train/ and val/ (default: the "
                         "harvested set; use data/vehicle_ext/cls for MIO-TCD)")
    ap.add_argument("--project", default=None,
                    help="Ultralytics project dir; point at Google Drive on "
                         "Colab so a reclaimed runtime does not cost the run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="its7class",
                    help="run name, and the output weights filename")
    args = ap.parse_args()

    if not (args.prepare or args.train):
        ap.error("nothing to do: pass --prepare and/or --train")
    if args.prepare:
        prepare(Path(args.manifest), args.val_frac, args.seed,
                args.use_draft, not args.analysed_only)
    if args.train:
        return train(args.epochs, args.imgsz, args.batch, args.device,
                     args.base, Path(args.out),
                     data=Path(args.data) if args.data else None,
                     project=args.project, workers=args.workers,
                     name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
