"""Turn the EALPR plate crops into a labelled plate-COLOUR training set.

Egypt colour-codes the plate's top band by vehicle category, so the band is a
free, independent signal for vehicle class — and unlike the characters, it
survives the heavy downscaling of a traffic-overview camera. This builds the
data needed to fit that rule properly instead of on a handful of hand-measured
plates.

    python -m tools.build_plate_colour_set --stage features
    python -m tools.build_plate_colour_set --stage cluster --k 8

Two stages, because labelling 2,000 plates one by one is not sensible:

  features  measure each plate's band and body in HSV -> features.csv
  cluster   group by band colour and emit one contact sheet per cluster, so a
            human labels a whole cluster at once by looking at it

Every measurement is band-versus-body. The plate's white body is a built-in
neutral reference, and judging the band against it cancels white balance,
exposure and codec together — which is what makes low-saturation CCTV plates
classifiable at all. See pipeline.plates.classify_band.

Screenshots and other junk are dropped: the set is scraped, and a UI screenshot
or a near-black crop contributes nothing but noise to a colour fit.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.plates import classify_band            # noqa: E402

DEFAULT_SRC = ROOT / "data" / "plates" / "EALPR" / "EALPR- Plates dataset"
OUT = ROOT / "data" / "plates" / "colour"

# A plate is a wide, shallow rectangle. Anything far off that is a screenshot, a
# whole-vehicle photo, or a crop that failed — none of which belong in a colour
# fit. Real Egyptian plates run about 2:1 to 5:1.
MIN_ASPECT, MAX_ASPECT = 1.4, 6.0
MIN_W, MIN_H = 40, 18


def reject_reason(im) -> str | None:
    """Why this image is not a usable plate crop, or None if it is."""
    if im is None:
        return "unreadable"
    h, w = im.shape[:2]
    if w < MIN_W or h < MIN_H:
        return "too small"
    ar = w / h
    if not (MIN_ASPECT <= ar <= MAX_ASPECT):
        # Screenshots and full-vehicle frames land here.
        return f"aspect {ar:.1f}"
    if im.ndim != 3:
        return "not colour"
    # A crop with almost no variation is a blank or a solid fill, not a plate.
    if float(np.std(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))) < 12.0:
        return "flat"
    return None


def features(im) -> dict:
    """Band and body HSV medians, plus the margin between them."""
    h, w = im.shape[:2]
    split = max(h // 3, 1)
    band, body = im[:split, :], im[split:, :]
    hb = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    bo = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    H, S, V = (float(np.median(hb[..., k])) for k in range(3))
    bS, bV = (float(np.median(bo[..., k])) for k in (1, 2))
    return {"H": round(H, 1), "S": round(S, 1), "V": round(V, 1),
            "body_S": round(bS, 1), "body_V": round(bV, 1),
            "S_margin": round(S - bS, 1),
            "width": w, "height": h,
            "draft": classify_band(H, S, V, bS, bV)}


def stage_features(src: Path, out: Path) -> int:
    files = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
    if not files:
        raise SystemExit(f"no images in {src}")
    rows, rejected = [], {}
    for f in files:
        im = cv2.imread(str(f))
        why = reject_reason(im)
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        rows.append({"file": f.name, **features(im), "label": ""})
    if not rows:
        # Every candidate was rejected — almost always a wrong --src pointing at
        # whole-vehicle photos rather than plate crops. Say that, instead of an
        # IndexError from rows[0] three lines down.
        raise SystemExit(
            f"none of the {len(files)} images in {src} are usable plate crops "
            f"({rejected}). Check --src points at the PLATES dataset, not the "
            "vehicles one.")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "features.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} usable plates -> {out / 'features.csv'}")
    print(f"rejected {sum(rejected.values())}: {rejected}")
    print("draft colour mix:", dict(Counter(r["draft"] for r in rows).most_common()))
    return 0


def stage_cluster(src: Path, out: Path, k: int, per_sheet: int) -> int:
    """Group plates by band colour so a human can label a cluster at a time."""
    import csv as _csv
    with open(out / "features.csv", encoding="utf-8-sig") as fh:
        rows = list(_csv.DictReader(fh))
    # Cluster in a space where hue is CIRCULAR. Treating H as a line puts red at
    # 0 and red at 179 in different clusters, which is exactly the colour that
    # matters most here (red = truck).
    ang = np.array([float(r["H"]) for r in rows]) * (2 * np.pi / 180.0)
    S = np.array([float(r["S"]) for r in rows])
    V = np.array([float(r["V"]) for r in rows])
    M = np.array([float(r["S_margin"]) for r in rows])
    X = np.column_stack([np.cos(ang) * S / 128.0, np.sin(ang) * S / 128.0,
                         V / 255.0, M / 128.0]).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    _, labels, _ = cv2.kmeans(X, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    sheets = out / "clusters"
    sheets.mkdir(parents=True, exist_ok=True)
    for c in range(k):
        idx = np.where(labels == c)[0]
        tiles = []
        for i in idx[:per_sheet]:
            im = cv2.imread(str(src / rows[i]["file"]))
            if im is not None:
                tiles.append(cv2.resize(im, (150, 75)))
        if not tiles:
            continue
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.zeros((75, 150, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols])
                          for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(sheets / f"cluster_{c:02d}.png"), grid)
        med_h = np.median([float(rows[i]["H"]) for i in idx])
        med_s = np.median([float(rows[i]["S"]) for i in idx])
        med_m = np.median([float(rows[i]["S_margin"]) for i in idx])
        print(f"  cluster {c:02d}: {len(idx):5d} plates  H~{med_h:5.0f} "
              f"S~{med_s:5.0f} S_margin~{med_m:5.0f}")

    for i, r in enumerate(rows):
        r["cluster"] = int(labels[i])
    with open(out / "features.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncontact sheets -> {sheets}")
    print("Label each cluster by looking at its sheet, then put the colour in the "
          "`label` column for every row of that cluster.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("features", "cluster"), default="features")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-sheet", type=int, default=36)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if args.stage == "features":
        return stage_features(src, out)
    return stage_cluster(src, out, args.k, args.per_sheet)


if __name__ == "__main__":
    raise SystemExit(main())
