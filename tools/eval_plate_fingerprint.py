"""Measure whether the plate fingerprint can actually tell plates apart.

A perceptual hash always produces a number. Whether that number IDENTIFIES
anything is an empirical question, and at 34 px it is not obvious — the whole
point of the resolution finding is that very little information survives. So this
tool answers it with real Egyptian plates rather than by assertion, exactly the
way tools/plate_footage_check.py answers the OCR question.

Method: take real EALPR plate crops, degrade each to the deployment plate width
with independent per-frame sub-pixel shifts, noise and JPEG damage — simulating
several frames of one vehicle passing this camera — then ask two questions:

    SAME  plate, different simulated frames  -> how far apart are the hashes?
    OTHER plates                             -> how far apart are those?

If the two distributions overlap, the fingerprint cannot support corridor
matching and must not be deployed for it. The gap between them, if any, is what
sets the operational threshold.

    python -m tools.eval_plate_fingerprint
    python -m tools.eval_plate_fingerprint --width 34 --plates 300 --frames 8

Reported: the equal-error rate and the false-match rate at a threshold chosen to
keep most true matches, plus what those mean for a corridor deployment.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.plate_id import (                        # noqa: E402
    FINGERPRINT_BITS, distance, fuse,
)

EALPR = ROOT / "data" / "plates" / "EALPR" / "EALPR- Plates dataset"
# Aspect band of a usable plate crop, mirroring pipeline.plates — the scraped set
# contains screenshots and whole-vehicle photos that are not plates.
MIN_ASPECT, MAX_ASPECT = 1.4, 6.0


def sighting_conditions(rng: random.Random) -> dict:
    """Capture conditions for one SIGHTING (one camera, one pass).

    Held constant across the frames of a sighting and re-drawn between
    sightings, because that is the structure of the real problem: the frames
    from camera A share its exposure, angle and framing, and camera B's differ.

    Without this the evaluation is circular. Degrading one photograph twice and
    finding the hashes agree proves only that the hash is deterministic — it
    could be keying on that photograph's lighting, its background, or where the
    crop happened to be cut, none of which transfer to a second camera. The
    project made exactly this mistake once already, calibrating speed by
    plausibility; a flattering measurement is worse than none.
    """
    return {
        "gain": rng.uniform(0.70, 1.35),          # exposure differs per camera
        "bias": rng.uniform(-25, 25),
        "gamma": rng.uniform(0.75, 1.35),
        "angle": rng.uniform(-3.5, 3.5),          # mounting / vehicle yaw
        "pad": rng.uniform(-0.06, 0.06),          # detector cuts the box differently
        "jpeg": rng.choice([30, 40, 50, 60]),
    }


def degrade(img: np.ndarray, target_w: int, rng: random.Random,
            cond: dict) -> np.ndarray:
    """One simulated frame of this plate, under a given sighting's conditions.

    Sub-pixel shift, downscale to the deployment width, sensor noise and JPEG
    damage — the same degradation model docs/anpr-plan.md §5c used to measure the
    OCR curve — plus the per-sighting exposure, angle and framing above.
    """
    h, w = img.shape[:2]
    # Crop jitter: the plate detector does not cut the same box twice.
    pad = int(abs(cond["pad"]) * w)
    if cond["pad"] > 0 and w - 2 * pad > 16 and h - 2 * pad > 8:
        img = img[pad:h - pad, pad:w - pad]
    elif pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]

    dx, dy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), cond["angle"], 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    moved = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)

    # Exposure and gamma, i.e. a different camera with different settings.
    f = np.clip(moved.astype(np.float32) * cond["gain"] + cond["bias"], 0, 255)
    f = np.power(f / 255.0, cond["gamma"]) * 255.0

    scale = target_w / float(w)
    small = cv2.resize(f.astype(np.uint8),
                       (target_w, max(int(round(h * scale)), 4)),
                       interpolation=cv2.INTER_AREA)
    noisy = np.clip(small.astype(np.float32)
                    + np.random.normal(0, 2.0, small.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, cond["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else noisy


def load_plates(src: Path, limit: int, rng: random.Random) -> list[np.ndarray]:
    files = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
    if not files:
        raise SystemExit(f"no plate images in {src}")
    rng.shuffle(files)
    out = []
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        if w < 90 or h < 30 or not (MIN_ASPECT <= w / h <= MAX_ASPECT):
            continue                        # too small to degrade FROM, or not a plate
        out.append(im)
        if len(out) >= limit:
            break
    return out


def evaluate(plates: list[np.ndarray], width: int, frames: int,
             rng: random.Random) -> dict:
    # Two independent "sightings" of each plate, each fused from `frames` views —
    # this is a camera-A / camera-B corridor match, not a re-hash of one image.
    a, b = [], []
    for im in plates:
        ca, cb = sighting_conditions(rng), sighting_conditions(rng)
        fa = fuse([degrade(im, width, rng, ca) for _ in range(frames)])
        fb = fuse([degrade(im, width, rng, cb) for _ in range(frames)])
        if fa is not None and fb is not None:
            a.append(fa)
            b.append(fb)
    n = len(a)
    if n < 2:
        raise SystemExit("not enough usable plates to evaluate")

    same = np.array([distance(a[i], b[i]) for i in range(n)])
    # All cross-plate pairs, which is the population a corridor deployment
    # actually faces: every passing vehicle is compared against every candidate.
    other = np.array([distance(a[i], b[j])
                      for i in range(n) for j in range(n) if i != j])

    rows = []
    for thr in range(0, FINGERPRINT_BITS + 1):
        tpr = float((same <= thr).mean())            # true matches accepted
        fpr = float((other <= thr).mean())           # different plates accepted
        rows.append((thr, tpr, fpr))
    eer = min(rows, key=lambda r: abs((1 - r[1]) - r[2]))
    # An operating point that keeps most genuine matches, which is what a
    # journey-time system needs — it can tolerate misses, not false links.
    at95 = next((r for r in rows if r[1] >= 0.95), rows[-1])
    return {"n": n, "same": same, "other": other, "rows": rows,
            "eer": eer, "at95": at95}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(EALPR))
    ap.add_argument("--width", type=int, default=34,
                    help="deployment plate width in px (this camera: 34)")
    ap.add_argument("--plates", type=int, default=250)
    ap.add_argument("--frames", type=int, default=8,
                    help="frames fused per sighting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", type=int, nargs="*", default=None,
                    help="also evaluate at these widths, e.g. --compare 50 65 100")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    plates = load_plates(Path(args.src), args.plates, rng)
    print(f"{len(plates)} real Egyptian plates loaded from {Path(args.src).name}")
    print(f"fingerprint: {FINGERPRINT_BITS} bits, {args.frames} frames fused "
          f"per sighting\n")

    widths = [args.width] + list(args.compare or [])
    for w in widths:
        r = evaluate(plates, w, args.frames, rng)
        same, other = r["same"], r["other"]
        thr, tpr, fpr = r["at95"]
        ethr, etpr, efpr = r["eer"]
        print(f"=== plate width {w} px  ({r['n']} plates, "
              f"{len(other)} impostor pairs) ===")
        print(f"  same plate, two sightings : mean {same.mean():5.1f} bits differ "
              f"(p95 {np.percentile(same, 95):.0f})")
        print(f"  different plates          : mean {other.mean():5.1f} bits differ "
              f"(p5  {np.percentile(other, 5):.0f})")
        print(f"  equal-error rate          : {efpr*100:5.1f}%  at <= {ethr} bits")
        print(f"  keeping 95% of true matches: threshold <= {thr} bits, "
              f"FALSE-MATCH RATE {fpr*100:.1f}%")
        if fpr > 0.01:
            print(f"  -> at {fpr*100:.1f}% false matches, 1 in "
                  f"{max(int(1/max(fpr,1e-9)),1)} unrelated vehicles is wrongly "
                  "linked.\n     NOT usable as a standalone corridor identifier "
                  "at this width.")
        else:
            print("  -> usable as corroborating evidence alongside time, "
                  "direction and vehicle class.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
