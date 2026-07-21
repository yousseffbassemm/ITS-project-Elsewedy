"""Train the plate detector and the plate-character (OCR) model on a GPU.

Runs on Colab/Kaggle, but it is a plain script, not a notebook, on purpose: it
lives in the repo, so it can be edited in VS Code with real linting and
autocomplete, reviewed in a diff, and re-run reproducibly. The notebook in
notebooks/ is only a thin driver that pulls this file and calls it.

Two separate models, because they answer different questions:

    stage 1  plate DETECTION   where is the plate on the vehicle    1 class
    stage 2  plate OCR         which characters are on the plate    ~27 classes

Stage 2 is character DETECTION rather than a text recogniser (CRNN etc.). On
Arabic plates this is the approach the published Egyptian ALPR work uses, and it
degrades more gracefully: a partly-read plate returns the characters it is sure
of instead of one wrong string.

    python -m tools.train_plates --stage detect --epochs 80
    python -m tools.train_plates --stage ocr --epochs 120
    python -m tools.train_plates --stage both --device 0

Datasets are NOT vendored — they are large and separately licensed. See
docs/anpr-plan.md for the sources and how to fetch each one.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Character classes on an Egyptian plate: Arabic-Indic digits plus the subset of
# Arabic letters the traffic authority actually issues. Kept explicit (rather
# than read from whatever data.yaml a download happens to ship) so a dataset with
# a different class ORDER cannot silently permute every label.
EG_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
EG_LETTERS = ["alef", "beh", "geem", "dal", "reh", "seen", "sad", "tah",
              "ain", "feh", "qaf", "lam", "meem", "noon", "heh", "waw", "yeh"]
EG_CHARS = EG_DIGITS + EG_LETTERS          # 27 classes, matching EALPR


def _require_ultralytics():
    try:
        from ultralytics import YOLO       # noqa: F401
    except ImportError:
        sys.exit("ultralytics is not installed:  pip install ultralytics")


def train_detector(data_yaml: str, epochs: int, imgsz: int, batch: int,
                   device: str, base: str, name: str) -> Path:
    """Stage 1 — a one-class 'plate' detector.

    imgsz defaults higher than the plate itself needs because the model sees the
    WHOLE frame: a plate that is 100 px wide in a 1920 px frame is a small
    object, and small-object recall is what this stage lives or dies on.
    """
    _require_ultralytics()
    from ultralytics import YOLO

    model = YOLO(base)
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        name=name, patience=20, cos_lr=True,
        # Plates are rigid, near-planar and always upright on the vehicle, so
        # the augmentations that help general detection actively hurt here.
        # Rotation stays small, and vertical flip is off entirely — an upside
        # down plate is not a thing this camera will ever see.
        degrees=5.0, fliplr=0.0, flipud=0.0, mosaic=0.5, scale=0.5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    )
    return Path(model.trainer.best)


def train_ocr(data_yaml: str, epochs: int, imgsz: int, batch: int,
              device: str, base: str, name: str) -> Path:
    """Stage 2 — character detection within an already-cropped plate."""
    _require_ultralytics()
    from ultralytics import YOLO

    model = YOLO(base)
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        name=name, patience=25, cos_lr=True,
        # Horizontal flip is DISABLED and this matters more here than anywhere
        # else: mirroring a plate reverses reading order and turns some Arabic
        # glyphs into each other. It is the single augmentation most likely to
        # quietly wreck this model.
        fliplr=0.0, flipud=0.0, degrees=3.0,
        # Mosaic stitches four images together, which for character detection
        # invents plates that do not exist and splits real ones. Off.
        mosaic=0.0, scale=0.3, hsv_v=0.5,
    )
    return Path(model.trainer.best)


def validate_against_footage(weights: Path, clip: Path, plate_px_floor: float) -> dict:
    """Sanity-check a trained detector on THIS project's own footage.

    A model can score well on its training domain and still find nothing on the
    target camera. This runs it on a real clip and reports what it actually
    detects, so that gap shows up here rather than in a demo.
    """
    _require_ultralytics()
    import cv2
    from ultralytics import YOLO

    if not clip.exists():
        return {"error": f"clip not found: {clip}"}
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(clip))
    widths, frames, hits = [], 0, 0
    while frames < 400:
        ok, frame = cap.read()
        if not ok:
            break
        if frames % 10 == 0:
            r = model.predict(frame, verbose=False, conf=0.25)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                widths.append(float(b[2] - b[0]))
                hits += 1
        frames += 1
    cap.release()
    if not widths:
        return {"clip": clip.name, "plates_found": 0,
                "note": "no plates detected — check domain gap before blaming OCR"}
    import numpy as np
    p90 = float(np.percentile(widths, 90))
    return {
        "clip": clip.name,
        "plates_found": hits,
        "plate_px_p90": round(p90, 1),
        "ocr_viable": bool(p90 >= plate_px_floor),
        "note": ("plates are large enough to read" if p90 >= plate_px_floor else
                 f"plates average {p90:.0f}px, below the {plate_px_floor:.0f}px "
                 "OCR floor — detection and colour only on this footage"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("detect", "ocr", "both"), default="both")
    ap.add_argument("--detect-data", default="data/plates/detect/data.yaml")
    ap.add_argument("--ocr-data", default="data/plates/ocr/data.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="default: 960 for detect, 320 for ocr")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--base", default="yolov8s.pt")
    ap.add_argument("--out", default="models", help="where to copy the weights")
    ap.add_argument("--validate-clip", default="samples/street_egypt.mp4")
    ap.add_argument("--plate-px-floor", type=float, default=100.0)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    if args.stage in ("detect", "both"):
        data = args.detect_data
        if not Path(data).exists():
            sys.exit(f"detector dataset not found: {data}\n"
                     "See docs/anpr-plan.md for how to fetch it.")
        best = train_detector(data, args.epochs, args.imgsz or 960, args.batch,
                              args.device, args.base, "plate_detect")
        dest = out / "plate_detect.pt"
        shutil.copy(best, dest)
        print(f"detector -> {dest}")
        report["detector"] = str(dest)
        report["footage_check"] = validate_against_footage(
            dest, ROOT / args.validate_clip, args.plate_px_floor)

    if args.stage in ("ocr", "both"):
        data = args.ocr_data
        if not Path(data).exists():
            sys.exit(f"OCR dataset not found: {data}\n"
                     "See docs/anpr-plan.md for how to fetch it.")
        # Characters are small and the crop is already tight, so a modest imgsz
        # is enough and keeps training fast.
        best = train_ocr(data, args.epochs, args.imgsz or 320, args.batch,
                         args.device, args.base, "plate_ocr")
        dest = out / "plate_ocr.pt"
        shutil.copy(best, dest)
        print(f"ocr -> {dest}")
        report["ocr"] = str(dest)

    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
