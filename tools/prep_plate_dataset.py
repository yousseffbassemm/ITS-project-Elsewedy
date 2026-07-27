"""Build a plate-detection training set MATCHED to this project's camera.

The naive version of this script — point YOLO at EALPR and train — produces a
model that is worse than useless here, and the reason is a domain gap that is
invisible unless you measure it.

    EALPR plates      median 136 px wide, 0.173 of image width
    street_egypt.mp4  20-35 px wide,      0.027 of image width

EALPR is a set of close-up vehicle photographs; this project's input is a
traffic-overview camera. Training on 136 px plates optimises the model for
objects 4.5x bigger than any it will ever be asked to find, and small-object
recall is precisely what the deployment needs. Fine-tuning on the raw set would
report excellent val mAP and change nothing on the real footage — the most
expensive kind of wrong answer, because the metrics look like success.

So each source image is rescaled until its plate lands in the DEPLOYMENT range
and composited onto a frame the size of the real camera's. The model then trains
on plates the size it will actually meet.

    python -m tools.prep_plate_dataset
    python -m tools.prep_plate_dataset --target-px 18 46 --native-frac 0.25

A fraction is kept at native scale (`--native-frac`) so the model retains its
ability on close-up plates and does not overfit to one narrow size.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "plates" / "EALPR" / "EALPR Vechicles dataset"
OUT = ROOT / "data" / "plates" / "detect_ds"

# The deployment frame. Matching it matters as much as matching plate size: the
# model learns an object's scale RELATIVE to the input it is given.
FRAME_W, FRAME_H = 1280, 720


def read_label(path: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append(tuple(float(v) for v in p[1:5]))   # cx cy w h (normalised)
    return out


def compose(im: np.ndarray, boxes, target_px: float, rng: random.Random):
    """Rescale so the first plate is ~target_px wide, then paste onto a frame.

    Returns (frame, new_boxes) or None if the result would be degenerate.
    """
    h, w = im.shape[:2]
    plate_px = boxes[0][2] * w
    if plate_px < 4:
        return None
    scale = target_px / plate_px
    nw, nh = max(int(w * scale), 8), max(int(h * scale), 8)
    if nw > FRAME_W or nh > FRAME_H:
        # Source is bigger than the frame even after scaling — crop around the
        # plate rather than shrinking further, which would undershoot target_px.
        return None
    small = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

    # Mid-grey rather than black: a black surround creates a hard synthetic edge
    # the model can key on, and it would learn "plates appear near a black
    # border" instead of what a plate looks like.
    frame = np.full((FRAME_H, FRAME_W, 3), 114, np.uint8)
    ox = rng.randint(0, FRAME_W - nw)
    oy = rng.randint(0, FRAME_H - nh)
    frame[oy:oy + nh, ox:ox + nw] = small

    new = []
    for cx, cy, bw, bh in boxes:
        ncx = (ox + cx * nw) / FRAME_W
        ncy = (oy + cy * nh) / FRAME_H
        nbw = (bw * nw) / FRAME_W
        nbh = (bh * nh) / FRAME_H
        if nbw <= 0 or nbh <= 0:
            continue
        new.append((ncx, ncy, nbw, nbh))
    return (frame, new) if new else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--target-px", type=float, nargs=2, default=[18.0, 46.0],
                    help="plate width range to synthesise (deployment range)")
    ap.add_argument("--native-frac", type=float, default=0.25,
                    help="fraction kept at original scale")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    imgs = sorted((src / "Vehicles").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images under {src / 'Vehicles'}")

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    # Split by SOURCE IMAGE before any augmentation, so a rescaled copy of a
    # training vehicle can never appear in val. Getting this backwards inflates
    # val mAP with what is effectively the training set.
    order = imgs[:]
    rng.shuffle(order)
    n_val = int(len(order) * args.val_frac)
    val_set = set(order[:n_val])

    stats = {"train": 0, "val": 0, "skipped": 0, "native": 0, "scaled": 0}
    widths = []
    for p in imgs:
        lp = src / "Vehicles Labeling" / (p.stem + ".txt")
        if not lp.exists():
            stats["skipped"] += 1
            continue
        boxes = read_label(lp)
        im = cv2.imread(str(p))
        if im is None or not boxes:
            stats["skipped"] += 1
            continue
        split = "val" if p in val_set else "train"

        if rng.random() < args.native_frac:
            frame, new = im, boxes
            stats["native"] += 1
            widths.append(boxes[0][2] * im.shape[1])
        else:
            target = rng.uniform(*args.target_px)
            res = compose(im, boxes, target, rng)
            if res is None:
                stats["skipped"] += 1
                continue
            frame, new = res
            stats["scaled"] += 1
            widths.append(target)

        cv2.imwrite(str(out / split / "images" / f"{p.stem}.jpg"), frame)
        (out / split / "labels" / f"{p.stem}.txt").write_text(
            "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in new))
        stats[split] += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\n"
        "names:\n  0: plate\n")

    if not widths:
        # Nothing was written, so data.yaml above points at empty splits and a
        # train run would fail much later with a confusing message.
        raise SystemExit(
            f"no usable images: {stats['skipped']} of {len(imgs)} were skipped. "
            f"Check --src points at '{Path(args.src).name}' with its 'Vehicles' "
            "and 'Vehicles Labeling' subfolders.")
    widths = np.array(widths)
    print(f"train {stats['train']}  val {stats['val']}  skipped {stats['skipped']}")
    print(f"  scaled-to-deployment {stats['scaled']}   kept-native {stats['native']}")
    print(f"plate width px: p10 {np.percentile(widths,10):.0f}  "
          f"median {np.median(widths):.0f}  p90 {np.percentile(widths,90):.0f}")
    print(f"target domain (street_egypt.mp4): 20-35 px")
    print(f"\ndata.yaml -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
