"""Harvest a labelling-ready vehicle dataset from the project's footage.

Step 2-3 of docs/finetuning-plan.md: every distinct vehicle in every clip is
detected, tracked, and saved as a handful of crops, pre-labelled with the current
pipeline's best guess. A human then only has to CORRECT labels rather than draw
boxes, which is where nearly all the annotation cost lives.

Deduplication is by TRACK, not by frame: one vehicle passing the camera yields a
few crops of itself rather than 60 near-identical ones, so the set stays balanced
and a single lorry cannot dominate training.

    python -m tools.harvest_dataset                    # all clips in samples/
    python -m tools.harvest_dataset --videos a.mp4 b.mp4 --per-track 4

Outputs under data/dataset/:
    crops/<clip>_t<track>_<n>.jpg   the images
    manifest.csv                    one row per crop, with the draft label
    sheets/sheet_XX.jpg             contact sheets for eyeball labelling
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classify import VehicleClassifier          # noqa: E402
from pipeline.config import PipelineConfig, VEHICLE_CLASSES  # noqa: E402
from pipeline.detect_track import VehicleDetector        # noqa: E402
from pipeline.speed import ViewTransformer               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dataset"
CROP_PX = 128


def harvest(video: Path, cfg: PipelineConfig, per_track: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  !! cannot open {video}")
        return []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    det_or = VehicleDetector(cfg, fps, frame_size=(w, h))
    clf = VehicleClassifier(ViewTransformer(cfg.source_px(w, h), cfg.target_px()))
    # keep the N largest views of each track — the closest, best-resolved ones
    best: dict[int, list[tuple[float, np.ndarray, int, int]]] = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = det_or.track(frame, fi)
        if det.tracker_id is not None and len(det):
            confs = det.confidence if det.confidence is not None else np.ones(len(det))
            for t, b, c, cf in zip(det.tracker_id, det.xyxy, det.class_id, confs):
                if int(c) not in VEHICLE_CLASSES:
                    continue
                t = int(t)
                clf.observe(t, int(c), b, float(cf))
                x1, y1, x2, y2 = (max(int(v), 0) for v in b)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 - x1 < 24 or y2 - y1 < 24:
                    continue                      # too small to label by eye
                area = (x2 - x1) * (y2 - y1)
                keep = best.setdefault(t, [])
                keep.append((area, frame[y1:y2, x1:x2].copy(), fi, int(c)))
                keep.sort(key=lambda r: -r[0])
                del keep[per_track:]
        fi += 1
        if total and fi % 200 == 0:
            print(f"  {video.name}: {fi}/{total}", flush=True)
    cap.release()

    rows = []
    for t, items in best.items():
        h_m, w_m = clf.size_estimate(t)
        draft = clf.resolve(t)
        for n, (area, crop, frame_no, coco) in enumerate(items):
            name = f"{video.stem}_t{t:04d}_{n}.jpg"
            cv2.imwrite(str(OUT / "crops" / name),
                        cv2.resize(crop, (CROP_PX, CROP_PX)))
            rows.append({
                "file": name, "clip": video.stem, "track": t, "frame": frame_no,
                "coco": coco, "draft_label": draft, "label": "",
                "height_m": round(h_m, 2), "width_m": round(w_m, 2),
                "px_area": int(area),
            })
    print(f"  {video.name}: {len(best)} distinct vehicles -> {len(rows)} crops")
    return rows


def contact_sheets(rows: list[dict], cols: int = 8, per_sheet: int = 48) -> int:
    """One tile per TRACK so a human labels each vehicle once."""
    seen, tiles = set(), []
    for r in rows:
        key = (r["clip"], r["track"])
        if key in seen:
            continue
        seen.add(key)
        img = cv2.imread(str(OUT / "crops" / r["file"]))
        if img is None:
            continue
        img = cv2.resize(img, (CROP_PX, CROP_PX))
        cv2.rectangle(img, (0, 0), (CROP_PX - 1, 16), (0, 0, 0), -1)
        cv2.putText(img, f'{r["clip"][:6]} t{r["track"]} {r["draft_label"]}',
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        tiles.append(img)
    n = 0
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        while len(chunk) % cols:
            chunk.append(np.zeros((CROP_PX, CROP_PX, 3), np.uint8))
        grid = np.vstack([np.hstack(chunk[i:i + cols])
                          for i in range(0, len(chunk), cols)])
        cv2.imwrite(str(OUT / "sheets" / f"sheet_{n:02d}.jpg"), grid)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--per-track", type=int, default=3)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=736)
    args = ap.parse_args()

    for sub in ("crops", "sheets"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    videos = ([Path(v) for v in args.videos] if args.videos
              else sorted((ROOT / "samples").glob("*.mp4")))
    cfg = PipelineConfig()
    cfg.model, cfg.imgsz = args.model, args.imgsz
    # Harvest EVERY vehicle in view, including the opposite carriageway: more
    # training examples is the point here, and the ROI gate is scene-specific.
    cfg.roi_gated_tracking = False

    rows: list[dict] = []
    for v in videos:
        print(f"harvesting {v.name} ...", flush=True)
        rows.extend(harvest(v, cfg, args.per_track))

    if not rows:
        print("no vehicles harvested")
        return 1
    man = OUT / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    sheets = contact_sheets(rows)

    tracks = len({(r["clip"], r["track"]) for r in rows})
    print()
    print(f"{tracks} distinct vehicles, {len(rows)} crops -> {man}")
    print(f"{sheets} contact sheet(s) in {OUT / 'sheets'}")
    from collections import Counter
    draft = Counter(r["draft_label"] for r in rows
                    if (r["clip"], r["track"]) and True)
    per_track_label = Counter()
    seen = set()
    for r in rows:
        k = (r["clip"], r["track"])
        if k not in seen:
            seen.add(k)
            per_track_label[r["draft_label"]] += 1
    print("draft label mix (per vehicle):", dict(per_track_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
