"""Build the 7-class vehicle-classification dataset from MIO-TCD.

The project's own harvested set is 194 crops of 71 vehicles from one clip. That
is enough to MEASURE a classifier and nowhere near enough to train one — the v1
model trained on it scores 0.647 on the deployment camera and reports 2 of ~6
microbuses in the live pipeline. The fix is not more tuning, it is more data of
the right kind.

**Why MIO-TCD specifically.** It is 519,164 crops of vehicles taken by traffic
cameras — the same *kind* of image this pipeline classifies: small, oblique,
compressed, variable weather and lighting. Datasets of clean 3/4-view press
photography (Stanford Cars and friends) are far larger and far worse for this,
because the domain gap is exactly the thing that breaks classifiers here. Its
class list also contains the two categories that COCO cannot express and that
the mentor's taxonomy needs:

    work_van      -> V   (base COCO calls every one of these a car)
    pickup_truck  -> C   (base COCO calls every one of these a truck)

    MIO-TCD                 mentor    note
    car                     A         private car
    pickup_truck            C         light truck
    single_unit_truck       D         rigid lorry
    articulated_truck       D         semi-trailer
    bus                     E         bus and microbus
    motorcycle, bicycle     G         two-wheelers, as the pipeline groups them
    work_van                V         panel van
    non-motorized_vehicle   F         carts etc — genuinely "other"
    pedestrian, background  -         dropped; the classifier only ever sees
                                      crops the vehicle detector produced

**Classes are capped, and that is a decision, not tidiness.** Raw MIO-TCD is
260,518 cars against 9,679 vans. Trained on that, the cheapest way to a high
score is to answer "car" — which is the behaviour being replaced. Capping every
class to the same ceiling costs data on the classes that have plenty and buys
per-class accuracy on the classes the report is actually about. CLAUDE.md §4
records the opposite mistake: tripling class C without re-checking the others
lifted C and cost 12 points overall.

**The test set is the deployment camera, and it is never trained on.** A
held-out MIO-TCD split measures how well the model learned MIO-TCD. This project
has been burned by exactly that twice — a plate detector at mAP50 0.985 that
regressed on real footage, and a classifier at 0.952 in-domain and 0.676 on the
real camera. So the project's own hand-labelled crops are copied out as a
separate `test/` split, and tools/eval_vehicle_cls.py leads with that number.

    python -m tools.prep_vehicle_dataset            # extract, map, split
    python -m tools.prep_vehicle_dataset --cap 8000
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXT = ROOT / "data" / "vehicle_ext"
TAR = EXT / "MIO-TCD-Classification.tar"
OUT = EXT / "cls"
DEPLOY_MANIFEST = ROOT / "data" / "dataset" / "manifest.csv"
DEPLOY_CROPS = ROOT / "data" / "dataset" / "crops"

# MIO-TCD folder -> mentor code. None means "not a vehicle this pipeline
# classifies" and the folder is skipped entirely.
MIOTCD_TO_CODE: dict[str, str | None] = {
    "car": "A",
    "pickup_truck": "C",
    "single_unit_truck": "D",
    "articulated_truck": "D",
    "bus": "E",
    "motorcycle": "G",
    "bicycle": "G",
    "work_van": "V",
    "non-motorized_vehicle": "F",
    "pedestrian": None,
    "background": None,
}

# Why single_unit_truck lands on D and not C: "light truck" (C) in the Egyptian
# scheme is the pickup/small-lorry category, and the pipeline resolves the C/D
# boundary a second way anyway — classify.VehicleClassifier.resolve defers a
# C-or-D answer to the monocular frontal-area estimate, because a pickup and a
# lorry look alike from behind and differ mainly in size. So this mapping sets
# the coarse answer and the size test refines it; it does not have to be perfect.
# It IS the assumption most worth re-testing if C/D accuracy disappoints.


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def extract(tar_path: Path, cap: int) -> dict[str, list[Path]]:
    """Stream the tar and write capped, mapped crops to a staging directory.

    Streamed rather than unpacked: the archive is 3.1 GB and only the ~40k
    images that survive the cap are wanted, so unpacking it whole would cost
    another 3 GB of disk for files that are deleted immediately.
    """
    stage = EXT / "_staged"
    if stage.exists():
        shutil.rmtree(stage)
    kept: dict[str, list[Path]] = defaultdict(list)
    counts, seen = Counter(), Counter()

    with tarfile.open(tar_path, "r|") as t:
        for m in t:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            # train/<class>/<id>.jpg — the `test/` tree is unlabelled.
            if len(parts) < 3 or parts[-3] != "train":
                continue
            # Count EVERY folder before deciding whether to keep it. Counting
            # only the mapped ones made the "folders actually seen" listing
            # below omit the folders that were skipped — which is exactly the
            # listing CLAUDE.md §5 requires, printed in a form that could not
            # reveal a dataset whose real contents differ from its docs.
            seen[parts[-2]] += 1
            code = MIOTCD_TO_CODE.get(parts[-2])
            if code is None:
                continue
            if counts[code] >= cap:
                continue
            dest = stage / code / f"{parts[-2]}_{parts[-1]}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = t.extractfile(m)
            if data is None:
                continue
            dest.write_bytes(data.read())
            kept[code].append(dest)
            counts[code] += 1

    print("MIO-TCD folders actually seen in the archive:")
    for k, v in sorted(seen.items()):
        print(f"  {v:>7}  {k:<24} -> {MIOTCD_TO_CODE.get(k) or '(skipped)'}")
    return kept


def split_and_write(kept: dict[str, list[Path]], val_frac: float,
                    seed: int) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(seed)
    counts = {"train": Counter(), "val": Counter()}
    for code, files in kept.items():
        files = sorted(files)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        for i, src in enumerate(files):
            split = "val" if i < n_val else "train"
            dest = OUT / split / code / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest)
            counts[split][code] += 1
    stage = EXT / "_staged"
    if stage.exists():
        shutil.rmtree(stage)
    return {k: dict(sorted(v.items())) for k, v in counts.items()}


def write_deployment_test() -> dict:
    """Copy the project's hand-labelled deployment crops out as `test/`.

    These never enter training. They are the only images in this whole dataset
    that come from the camera the system is actually deployed on, which makes
    them the only honest measure of whether any of this helped.
    """
    if not DEPLOY_MANIFEST.exists():
        return {}
    with open(DEPLOY_MANIFEST, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    counts = Counter()
    for r in rows:
        code = (r.get("label") or "").strip().upper()
        src = DEPLOY_CROPS / r["file"]
        if code not in MIOTCD_TO_CODE.values() or not src.exists():
            continue
        dest = OUT / "test" / code / r["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        counts[code] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default=str(TAR))
    ap.add_argument("--cap", type=int, default=6000,
                    help="max images per mentor class (default 6000)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _utf8_stdout()

    tar_path = Path(args.tar)
    if not tar_path.exists():
        raise SystemExit(
            f"{tar_path} not found. Download it first (3.1 GB, CC BY-NC-SA 4.0):\n"
            "  curl -L -C - -o data/vehicle_ext/MIO-TCD-Classification.tar \\\n"
            "    https://tcd.miovision.com/static/dataset/MIO-TCD-Classification.tar")

    kept = extract(tar_path, args.cap)
    if not kept:
        raise SystemExit(
            "no images extracted — the archive layout is not train/<class>/<id>.jpg "
            "as expected. Print the real listing before trusting any dataset's "
            "documentation (CLAUDE.md §5).")
    split = split_and_write(kept, args.val_frac, args.seed)
    test = write_deployment_test()

    print(f"\ndataset -> {OUT}")
    print(f"  train : {split['train']}  (n={sum(split['train'].values())})")
    print(f"  val   : {split['val']}  (n={sum(split['val'].values())})")
    if test:
        print(f"  test  : {test}  (n={sum(test.values())})  "
              f"<- DEPLOYMENT CAMERA, never trained on")
    else:
        print("  test  : none — data/dataset/manifest.csv not found, so there is "
              "no deployment-camera\n          test set and any score below will "
              "be in-domain only. See CLAUDE.md §5.")
    missing = sorted({c for c in MIOTCD_TO_CODE.values() if c} - set(split["train"]))
    if missing:
        print(f"  [CHECK] no training images for {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
