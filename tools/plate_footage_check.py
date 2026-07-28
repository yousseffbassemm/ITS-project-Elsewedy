"""Decide whether a clip's plates are BIG ENOUGH to read, before training anything.

This tool exists because of a failure mode that wastes weeks: a plate OCR model
is trained on a large online dataset, run on the target footage, and reads
nothing — which looks like a model problem and is actually a camera problem.
Training data cannot add pixels that the sensor never captured, so this is the
first thing to check, not the last.

Two independent estimates are produced and cross-checked:

  * GEOMETRIC — the pipeline's calibrated road-plane homography already knows how
    many pixels one metre spans at any image row. Multiply by the plate's real
    width and you get its pixel width anywhere in frame, without detecting a
    single plate. This is the honest upper bound: it assumes a perfectly
    face-on plate.
  * EMPIRICAL — detect vehicles, take the widest boxes actually observed, and
    scale by the plate-to-vehicle width ratio. This catches scenes where the
    calibration is off or traffic never comes near the camera.

Agreement between the two is the signal that both are trustworthy. On
samples/street_egypt.mp4 they agree at ~34 px, and a hand-located plate in that
clip measured 34x16 px.

    python -m tools.plate_footage_check                       # all clips in samples/
    python -m tools.plate_footage_check --videos clip.mp4 --plate-width-m 0.52

The ONE assumption is the plate's real-world width (--plate-width-m), treated the
same way as config.dash_pitch_m: documented, overridable, and the only number to
change if the vehicle population differs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classify import VehicleClassifier          # noqa: E402
from pipeline.config import PipelineConfig, VEHICLE_CLASSES  # noqa: E402
from pipeline.plates import (                            # noqa: E402
    MARGINAL_PX_FOR_FUSED_OCR, MIN_PX_FOR_FUSED_OCR, MIN_PX_FOR_OCR,
)
from pipeline.speed import ViewTransformer               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Real-world plate widths (metres). Egypt's single-row private plate is the
# default because it is this project's target domain; the EU/long format is
# offered for the stock 4K clips, which are European.
PLATE_WIDTH_M = {"eg": 0.32, "eu": 0.52}

# Readability thresholds, in PLATE PIXEL WIDTH. Imported from the pipeline so
# this tool and the runtime gate cannot drift apart — see pipeline/plates.py for
# the measured accuracy curve these come from (docs/anpr-plan.md §5c).
#
#   >= READABLE_PX  a single frame reads (95.8% character accuracy)
#   >= FUSION_PX    multi-frame fusion reaches the same (93.1%); one frame gives 78%
#   >= MARGINAL_PX  fusion is partial (66.7%); expect errors
#   below           not a training problem, and no model will fix it
READABLE_PX = MIN_PX_FOR_OCR
FUSION_PX = MIN_PX_FOR_FUSED_OCR
MARGINAL_PX = MARGINAL_PX_FOR_FUSED_OCR

# Fraction of a vehicle's box width taken up by its plate. A car is ~1.8 m wide
# and an Egyptian plate ~0.32 m, so ~0.18. Used only for the empirical estimate.
PLATE_TO_VEHICLE_W = 0.18


def geometric_estimate(cfg: PipelineConfig, w: int, h: int,
                       plate_w_m: float) -> list[tuple[float, float, float]]:
    """(y_fraction, px_per_metre, plate_px) sampled down the analysed roadway."""
    clf = VehicleClassifier(ViewTransformer(cfg.source_px(w, h), cfg.target_px()))
    rows = []
    for yf in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        ppm = clf._px_per_metre(w * 0.45, yf * h)
        rows.append((yf, ppm, ppm * plate_w_m))
    return rows


def empirical_estimate(video: Path, cfg: PipelineConfig, max_frames: int,
                       plate_w_m: float) -> tuple[float, int]:  # noqa: ARG001
    """Widest vehicle boxes actually seen -> implied plate width in px.

    `plate_w_m` is accepted and deliberately NOT used: this estimate scales the
    observed vehicle width by a plate/vehicle RATIO instead, so it shares no
    input with the geometric estimate above. That independence is the point —
    docs/anpr-plan.md §1 rests on three methods agreeing (34 / 38 / 34 px), and
    an agreement between two calculations fed the same constant would prove
    nothing. The parameter stays so both estimators present one interface to
    the caller.
    """
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.0, 0
    model = YOLO(cfg.model)
    widths: list[float] = []
    fi = 0
    while fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % 25 == 0:                       # ~1 fps is plenty for a maximum
            r = model.predict(frame, imgsz=cfg.imgsz, conf=0.35,
                              classes=sorted(VEHICLE_CLASSES), verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                widths.append(float(b[2] - b[0]))
        fi += 1
    cap.release()
    if not widths:
        return 0.0, 0
    # The 95th percentile, not the single max: one clipped box straddling the
    # frame edge would otherwise set the answer.
    p95 = float(np.percentile(widths, 95))
    return p95 * PLATE_TO_VEHICLE_W, len(widths)


def verdict(plate_px: float) -> tuple[str, str]:
    if plate_px >= READABLE_PX:
        return "OCR FEASIBLE", "single-frame OCR reads this footage (~96% chars)"
    if plate_px >= FUSION_PX:
        return "OCR FEASIBLE (fusion)", (
            "run with --enhance: multi-frame fusion reads this (~93% chars), "
            "a single frame only manages ~78%")
    if plate_px >= MARGINAL_PX:
        return "MARGINAL", (
            "partial reads even with --enhance (~67% chars); expect errors and "
            "do not report plate numbers unattended")
    return "OCR NOT FEASIBLE", (
        f"needs ~{FUSION_PX / max(plate_px, 1e-6):.1f}x more plate resolution "
        f"to reach the {FUSION_PX:.0f}px fusion floor — a camera change, not a "
        "training problem"
    )


def report(video: Path, cfg: PipelineConfig, plate_w_m: float, region: str,
           max_frames: int, skip_empirical: bool) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"!! cannot open {video}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n{'=' * 72}\n{video.name}  —  {w}x{h}  (plate assumed {plate_w_m} m, '{region}')\n{'=' * 72}")

    geo = geometric_estimate(cfg, w, h, plate_w_m)
    print("  geometric (calibrated homography):")
    for yf, ppm, ppx in geo:
        print(f"    y={yf:.2f}  {ppm:7.1f} px/m  ->  plate {ppx:6.1f} px")
    geo_best = max(p for _, _, p in geo)

    emp_best = 0.0
    if not skip_empirical:
        emp_best, n = empirical_estimate(video, cfg, max_frames, plate_w_m)
        print(f"  empirical ({n} vehicle boxes, p95 width x {PLATE_TO_VEHICLE_W}):"
              f"  plate {emp_best:.1f} px")

    best = max(geo_best, emp_best)
    tag, advice = verdict(best)
    print(f"\n  best achievable plate width: {best:.0f} px "
          f"(single-frame >= {READABLE_PX:.0f}, with fusion >= {FUSION_PX:.0f}, "
          f"marginal >= {MARGINAL_PX:.0f})")
    print(f"  VERDICT: {tag} — {advice}")
    # Detection and colour survive far below the OCR floor: a 30 px plate is
    # still a findable rectangle with a recoverable dominant hue.
    print(f"  plate DETECTION + COLOUR: {'feasible' if best >= 20 else 'unreliable'} "
          f"(these need ~20 px, far less than OCR)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--region", choices=sorted(PLATE_WIDTH_M), default="eg",
                    help="plate size convention (default: eg)")
    ap.add_argument("--plate-width-m", type=float, default=None,
                    help="override the plate's real width in metres")
    ap.add_argument("--max-frames", type=int, default=750)
    ap.add_argument("--geometric-only", action="store_true",
                    help="skip YOLO (fast; no model load)")
    args = ap.parse_args()

    plate_w_m = args.plate_width_m or PLATE_WIDTH_M[args.region]
    videos = ([Path(v) for v in args.videos] if args.videos
              else sorted((ROOT / "samples").glob("*.mp4")))
    if not videos:
        print("no videos found")
        return 1

    cfg = PipelineConfig()
    for v in videos:
        report(v, cfg, plate_w_m, args.region, args.max_frames, args.geometric_only)

    print(f"\n{'=' * 72}")
    print("NOTE: the geometric figure assumes the calibrated homography in "
          "pipeline/config.py\napplies to this clip. It is exact for the clip it "
          "was calibrated on (street_egypt)\nand indicative elsewhere — trust the "
          "empirical number for other cameras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
