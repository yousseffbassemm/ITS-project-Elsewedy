"""Train the two ANPR cascade stages on a GPU (Colab T4 or similar).

    # once, locally — recover EALPR's class legend and build both datasets
    python -m tools.ealpr_charmap
    python -m tools.build_anpr_datasets

    # on the GPU
    python -m tools.train_anpr --stage 2 --epochs 80
    python -m tools.train_anpr --stage 3 --epochs 120

Stage 2 (plate on a vehicle) and stage 3 (characters on a plate) are trained
separately because they are different problems with different failure modes, and
training them together is what produces a model that is good at the easy one and
quietly bad at the hard one.

**Augmentation differs between the two stages, and getting it wrong is silent.**

*Stage 2* sees whole vehicles from a fixed camera. A mirrored car is still a
car, so horizontal flip is free data — but it MOVES THE PLATE to the other side
of the vehicle, which is exactly the variation this stage needs to be robust to
(EALPR's plates span 0.03-0.96 of vehicle width), so flip is not merely safe
here, it is the point.

*Stage 3* reads characters. Horizontal flip reverses reading order AND mirrors
every glyph into something that is not an Arabic letter, so it must be OFF.
Rotation is kept small for the same reason: a plate photographed at an angle is
common, a plate rotated 15 degrees is not, and heavy rotation teaches the model
that orientation carries no information when it carries all of it.

The deployment images are CCTV-grade — small, motion-blurred, JPEG-damaged —
while EALPR is close-up photography. That domain gap is the single biggest risk
in this whole approach, so photometric augmentation is turned up hard on both
stages and the eval below is run on real footage, never only on the val split.
See CLAUDE.md §5: a plate detector that scored mAP50 0.985 in-domain was a
straight regression on the deployment camera.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE2_DS = ROOT / "data" / "plates" / "stage2_plate_on_vehicle"
STAGE3_DS = ROOT / "data" / "plates" / "stage3_chars"
MODELS = ROOT / "models"

# name -> (dataset dir, output weights, base model, imgsz, augmentation)
STAGES = {
    2: {
        "data": STAGE2_DS,
        "out": "plate_on_vehicle.pt",
        "base": "yolov8n.pt",
        "imgsz": 480,
        # One small object in a large image: mosaic helps a lot here, and the
        # flip is deliberate (see the module docstring).
        "aug": dict(fliplr=0.5, flipud=0.0, degrees=5.0, scale=0.5,
                    mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.5),
    },
    3: {
        "data": STAGE3_DS,
        "out": "plate_chars.pt",
        "base": "yolov8n.pt",
        # Characters are small and densely packed across a wide, short crop, so
        # this stage needs more input resolution than stage 2 despite the image
        # being far smaller.
        "imgsz": 640,
        # NO horizontal flip — it mirrors Arabic glyphs into non-characters and
        # reverses reading order. Mosaic off: it splices four plates together and
        # teaches the model that characters from different plates belong to one
        # string, which is precisely the error the cascade must not make.
        "aug": dict(fliplr=0.0, flipud=0.0, degrees=3.0, scale=0.3,
                    mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.6),
    },
}


def train(stage: int, epochs: int, batch: int, device: str,
          base: str | None, out_dir: Path, project: str | None) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed:  pip install ultralytics")

    spec = STAGES[stage]
    data_yaml = spec["data"] / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"{data_yaml} not found — build it first with\n"
            f"    python -m tools.build_anpr_datasets --stage {stage}")

    model = YOLO(base or spec["base"])
    kwargs = dict(
        data=str(data_yaml), epochs=epochs, imgsz=spec["imgsz"], batch=batch,
        device=device, name=f"anpr_stage{stage}", patience=20, cos_lr=True,
        **spec["aug"],
    )
    # Colab reclaims runtimes without warning and takes /content with them. A
    # Drive-backed project dir means last.pt and best.pt survive a disconnect —
    # this cost three training runs before it was done. See CLAUDE.md §5.
    if project:
        kwargs["project"] = project
    model.train(**kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / spec["out"]
    shutil.copy(Path(model.trainer.best), dest)
    print(f"\nstage {stage} weights -> {dest}")
    print(f"Deploy with:  ITS_ANPR_STAGE{stage}={dest.as_posix()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(2, 3), required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--base", default=None, help="base weights (default per stage)")
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--project", default=None,
                    help="Ultralytics project dir; point this at Google Drive on "
                         "Colab so a reclaimed runtime does not cost the run")
    args = ap.parse_args()
    return train(args.stage, args.epochs, args.batch, args.device, args.base,
                 Path(args.out), args.project)


if __name__ == "__main__":
    raise SystemExit(main())
