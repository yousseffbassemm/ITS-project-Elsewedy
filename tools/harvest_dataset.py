"""Harvest a labelling-ready vehicle dataset from the project's footage.

Step 2-3 of docs/finetuning-plan.md: every distinct vehicle in every clip is
detected, tracked, and saved as a handful of crops, pre-labelled with the current
pipeline's best guess. A human then only has to CORRECT labels rather than draw
boxes, which is where nearly all the annotation cost lives.

Three properties this has to get right, each of which it previously got wrong and
each of which quietly poisons the fine-tune:

**One vehicle, one identity.** Crops are bucketed by track id while the video
plays, but the re-identification layer can decide at frame 900 that ids 5 and 3
were always the same car. That merge does not retroactively re-key the crops
already filed under 5, so the same vehicle appeared twice under two names — and a
model trained on that learns to tell one car from itself. Every bucket is now
resolved through ``VehicleDetector.resolve_id`` at the END, after all merges are
known, and buckets that collapse together are re-scored and re-trimmed as one.

**Say which carriageway a vehicle came from.** This camera also watches the
opposite carriageway. Those vehicles are NOT noise — for training, more examples
is the point, and they are the same vehicle population from a different angle.
But they are not part of the analysed traffic, so mixing them into a report is
wrong. Every crop is therefore labelled ``analysed`` or ``opposite`` and the
report-facing set is the analysed, counted subset. Filter on the column; don't
throw the data away.

**Number vehicles the way the video does.** A crop called ``t0137`` is an
internal allocation counter that matches nothing a human can see. Analysed
vehicles that cross the counting line are numbered 1..N in order of appearance,
which is exactly the ``#N`` drawn on annotated.mp4 — so a labeller can pause the
video and check.

    python -m tools.harvest_dataset                    # all clips in samples/
    python -m tools.harvest_dataset --videos a.mp4 --per-track 4

Outputs under data/dataset/:
    crops/<clip>_v0001_0.jpg        analysed vehicles, numbered as on the video
    crops/<clip>_x0042_0.jpg        opposite carriageway ('x' = excluded from the report)
    manifest.csv                    one row per crop, with the draft label
    sheets/sheet_XX.jpg             contact sheets for eyeball labelling
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classify import VehicleClassifier          # noqa: E402
from pipeline.config import PipelineConfig               # noqa: E402
from pipeline.counting import LineCounter                # noqa: E402
from pipeline.detect_track import VehicleDetector        # noqa: E402
from pipeline.lanes import LaneModel                     # noqa: E402
from pipeline.speed import ViewTransformer               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dataset"
CROP_PX = 128
# Minimum box size worth saving. Below this a human cannot label it either, so it
# is annotation cost with no training value.
MIN_BOX_PX = 24


def _scan(video: Path, cfg: PipelineConfig, per_track: int, gate: bool):
    """One pass over the video. Returns (crops, roi_votes, first_frame, counted).

    ``gate`` arms the ROI filter inside VehicleDetector, and it changes more than
    which boxes survive: the gate runs BEFORE the re-identification layer, so
    with it off the stabiliser also sees the opposite carriageway — 126 canonical
    ids instead of 28 on this clip. That extra churn re-keys analysed vehicles
    mid-track, and a vehicle whose id changes at the counting line has no
    previous side recorded, so it is never counted. Measured: the analysed count
    falls from 18 to 5. Identifying the analysed traffic therefore REQUIRES the
    gate, which is why this file makes two passes instead of one.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  !! cannot open {video}")
        return {}, {}, {}, set()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    cfg = replace(cfg, roi_gated_tracking=gate)
    detector = VehicleDetector(cfg, fps, frame_size=(w, h))
    clf = VehicleClassifier(ViewTransformer(cfg.source_px(w, h), cfg.target_px()),
                            detector.native_codes)
    # The counter decides which vehicles are part of the ANALYSED traffic, and in
    # what order they appear. It is fed only on-carriageway detections, which
    # reproduces exactly what the reporting pipeline (ROI gate on) counts.
    counter = LineCounter(cfg, w, h, LaneModel(cfg, w, h), scheme=detector.scheme)

    best: dict[int, list[dict]] = defaultdict(list)
    roi_votes: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # [off, on]
    first_frame: dict[int, int] = {}
    seq = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detector.track(frame, fi)
        if det.tracker_id is not None and len(det):
            confs = det.confidence if det.confidence is not None else np.ones(len(det))
            on_road = np.array([detector.on_roadway(b) for b in det.xyxy], dtype=bool)
            # Count only the analysed carriageway, so 'counted' here means the
            # same thing it means in analytics.json.
            if on_road.any():
                counter.update(det[on_road])
            for i, (t, b, c, cf) in enumerate(
                    zip(det.tracker_id, det.xyxy, det.class_id, confs)):
                if not detector.scheme.is_vehicle(c):
                    continue
                t = int(t)
                clf.observe(t, int(c), b, float(cf))
                roi_votes[t][1 if on_road[i] else 0] += 1
                first_frame.setdefault(t, fi)
                x1, y1, x2, y2 = (max(int(v), 0) for v in b)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                    continue
                best[t].append({
                    "area": (x2 - x1) * (y2 - y1), "img": frame[y1:y2, x1:x2].copy(),
                    "frame": fi, "coco": int(c), "seq": seq,
                })
                seq += 1
                # Trim opportunistically so memory stays bounded; the final trim
                # after merging is the one that decides what is written.
                if len(best[t]) > per_track * 4:
                    best[t].sort(key=lambda r: -r["area"])
                    del best[t][per_track * 2:]
        fi += 1
        if total and fi % 400 == 0:
            print(f"  {video.name} [{'analysed' if gate else 'all'}]: "
                  f"{fi}/{total}", flush=True)
    cap.release()

    # --- collapse ids the re-id layer merged AFTER those crops were filed -------
    merged: dict[int, list[dict]] = defaultdict(list)
    merged_roi: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    merged_first: dict[int, int] = {}
    alias_hits = 0
    for t, items in best.items():
        root = detector.resolve_id(t)
        if root != t:
            alias_hits += 1
        merged[root].extend(items)
    for t, v in roi_votes.items():
        root = detector.resolve_id(t)
        merged_roi[root][0] += v[0]
        merged_roi[root][1] += v[1]
    for t, f in first_frame.items():
        root = detector.resolve_id(t)
        merged_first[root] = min(merged_first.get(root, f), f)
    # --- fold together tracks that are one vehicle split in time ---------------
    # Re-id merges ids that COEXIST or reappear within its gap window. A track
    # that fragments and restarts outside that window survives as two, and no
    # amount of id resolution finds it.
    #
    # Appearance alone cannot settle this, and the reason is in this very clip:
    # two white microbuses one frame apart (correlation 0.970) are one vehicle
    # whose track broke, while two white box trucks 962 frames — 38 seconds —
    # apart at correlation 0.977 are two different lorries from the same fleet.
    # Identical pixels, opposite answers. TIME is what separates them: one
    # vehicle cannot be in two places seconds apart on a carriageway, and cannot
    # pass the same camera twice in 90 seconds.
    order = sorted(merged, key=lambda t: merged_first.get(t, 1 << 30))
    # Signature from each track's LARGEST crop, which is the one that gets
    # written and therefore the one the post-hoc audit compares. Using
    # whichever crop happened to be appended first made this step and the audit
    # disagree about the same pair of vehicles, so the merge silently declined
    # a duplicate the audit then reported.
    sigs = {t: _sig(max(items, key=lambda r: r["area"])["img"])
            for t, items in merged.items() if items}
    for i, a in enumerate(order):
        if a not in merged or sigs.get(a) is None:
            continue
        for b in order[i + 1:]:
            if b not in merged or sigs.get(b) is None:
                continue
            gap = abs(merged_first.get(a, 0) - merged_first.get(b, 0))
            if gap > TEMPORAL_SPLIT_FRAMES:
                break                       # ordered by time — no closer pair left
            if cv2.compareHist(sigs[a], sigs[b],
                               cv2.HISTCMP_CORREL) >= DUP_SAME_VEHICLE_NEARBY:
                merged[a].extend(merged.pop(b))
                merged_roi[a][0] += merged_roi[b][0]
                merged_roi[a][1] += merged_roi[b][1]
                alias_hits += 1

    for items in merged.values():
        items.sort(key=lambda r: -r["area"])
        del items[per_track:]

    counted = {detector.resolve_id(t) for t in counter.vehicle_events}
    print(f"  {video.name} [{'analysed' if gate else 'all'}]: {len(merged)} "
          f"vehicles, {counter.total_vehicles()} crossed the line, "
          f"{alias_hits} id(s) folded back by re-id", flush=True)
    return merged, merged_roi, merged_first, counted, clf


def _write(video: Path, merged, first, counted, clf, tag_fn, carriageway,
           rows: list[dict]) -> None:
    """Write one pass's crops and manifest rows."""
    for t, items in merged.items():
        no, tag = tag_fn(t)
        h_m, w_m = clf.size_estimate(t)
        draft = clf.resolve(t)
        for n, it in enumerate(items):
            name = f"{video.stem}_{tag}_{n}.jpg"
            cv2.imwrite(str(OUT / "crops" / name),
                        cv2.resize(it["img"], (CROP_PX, CROP_PX)))
            rows.append({
                "file": name, "clip": video.stem,
                "vehicle_no": no or "",
                "track": t,
                "carriageway": carriageway,
                "counted": "yes" if t in counted else "no",
                "frame": it["frame"], "coco": it["coco"],
                "draft_label": draft, "label": "",
                "height_m": round(h_m, 2), "width_m": round(w_m, 2),
                "px_area": int(it["area"]),
            })


def harvest(video: Path, cfg: PipelineConfig, per_track: int,
            include_opposite: bool = True) -> list[dict]:
    """Two passes: the analysed carriageway, then everything else.

    Pass 1 runs with the ROI gate ARMED, which is the only configuration in
    which the analysed vehicles get stable ids and the counting line agrees with
    analytics.json — so these crops carry the same #N the annotated video draws.

    Pass 2 runs with the gate off and keeps only the vehicles pass 1 could not
    see: the opposite carriageway. They are real vehicles and good training data,
    they are simply not part of the analysed traffic, so they are labelled rather
    than discarded (or silently mixed in, which is what used to happen).
    """
    rows: list[dict] = []

    merged, _roi, first, counted, clf = _scan(video, cfg, per_track, gate=True)
    order = sorted((t for t in merged if t in counted),
                   key=lambda t: first.get(t, 1 << 30))
    vehicle_no = {t: i + 1 for i, t in enumerate(order)}

    def _analysed_tag(t):
        no = vehicle_no.get(t)
        return no, (f"v{no:04d}" if no else f"a{t:04d}")

    _write(video, merged, first, counted, clf, _analysed_tag, "analysed", rows)
    n_analysed = len(vehicle_no)

    n_opp = 0
    if include_opposite:
        m2, roi2, first2, counted2, clf2 = _scan(video, cfg, per_track, gate=False)
        # Keep only vehicles that were NEVER on the analysed carriageway.
        #
        # A majority vote is not enough and gets this badly wrong: the ROI
        # starts at y=0.30, so an analysed vehicle spends most of its tracked
        # life in the far field ABOVE it and reads as majority-off-road even
        # though it crossed the counting line. Measured: 12 of the 18 analysed
        # vehicles were duplicated into the opposite set at histogram
        # correlation 1.000 — the same crop filed under two names, which is the
        # exact defect this rewrite exists to remove.
        #
        # A genuine opposite-carriageway vehicle never touches the analysed one,
        # so the tolerance only has to absorb mask-boundary jitter.
        off = {t: v for t, v in m2.items()
               if roi2[t][1] <= max(2, 0.02 * (roi2[t][0] + roi2[t][1]))}
        # Geometry alone does not finish the job. With the gate off, an analysed
        # vehicle's track can FRAGMENT, and a fragment covering only its far-field
        # portion has no ROI overlap at all — so it passes the test above while
        # its crops show a vehicle pass 1 already filed. Measured: the geometric
        # filter cut cross-pass duplicates from 12 to 3, and the survivors were
        # exactly such fragments.
        #
        # Appearance settles it, because these are not similar vehicles — they
        # are the same pixels, at correlation 1.000.
        ref = [s for s in (_sig(items[0]["img"]) for items in merged.values())
               if s is not None]
        drop = set()
        for t, items in off.items():
            s = _sig(items[0]["img"]) if items else None
            if s is None:
                continue
            if any(cv2.compareHist(s, r, cv2.HISTCMP_CORREL) >= DUP_SAME_VEHICLE
                   for r in ref):
                drop.add(t)
        if drop:
            print(f"  {video.name}: dropped {len(drop)} opposite-pass "
                  f"track(s) that duplicate an analysed vehicle", flush=True)
        off = {t: v for t, v in off.items() if t not in drop}
        _write(video, off, first2, set(), clf2,
               lambda t: (None, f"x{t:04d}"), "opposite", rows)
        n_opp = len(off)

    print(f"  {video.name}: {n_analysed} analysed vehicles (#1..#{n_analysed}), "
          f"{n_opp} opposite-carriageway, {len(rows)} crops")
    return rows


def _sig(img) -> "np.ndarray | None":
    """HS colour histogram of one crop, for duplicate detection."""
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(img, (CROP_PX, CROP_PX)), cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1], None, [32, 32],
                        [0, 180, 0, 256]).flatten().astype(np.float32)


# A duplicate is the SAME pixels filed twice, so it lands at ~1.000. Two
# genuinely similar vehicles in one clip — this footage has several near-identical
# white microbuses — measured 0.970 and 0.977. 0.99 sits in the gap with room on
# both sides, so removing above it cannot delete a real second vehicle.
DUP_SAME_VEHICLE = 0.99

# Two tracks that look alike AND start within this many raw frames of each other
# are one vehicle whose track fragmented. Two seconds at 25 fps: a vehicle cannot
# be in two places two seconds apart on this carriageway, and no vehicle passes
# the same camera twice within a 90 s clip — so beyond it, lookalikes are
# different vehicles and must be kept.
TEMPORAL_SPLIT_FRAMES = 50
# Looser than DUP_SAME_VEHICLE because the time constraint is doing the work: a
# fragmented track resumes at a slightly different scale and lighting, so its
# two halves measured 0.970 rather than 1.000.
DUP_SAME_VEHICLE_NEARBY = 0.96


def audit_duplicates(rows: list[dict], corr: float = 0.97) -> list[tuple]:
    """Report vehicles that look like the same vehicle filed twice.

    "One vehicle, one identity" is the property this tool exists to guarantee,
    and it has now been broken twice by two different mechanisms — ids merged
    after their crops were filed, and a vehicle appearing in both passes. Both
    were found by eye on a contact sheet, which does not scale and will not
    catch the third one.

    An HS colour histogram over each vehicle's best crop is a crude descriptor,
    but duplicates here are the SAME pixels rather than two similar vehicles, so
    they land at correlation ~1.0 and stand well clear of genuine lookalikes.
    Reported, never auto-merged: two identical white microbuses in the same clip
    are a real possibility and deleting one would be worse than flagging both.
    """
    first: dict[tuple, dict] = {}
    frames: dict[tuple, int] = {}
    for r in rows:
        key = (r["clip"], r["track"], r["carriageway"])
        first.setdefault(key, r)
        frames[key] = min(frames.get(key, 1 << 30), int(r["frame"]))
    sigs = {}
    for key, r in first.items():
        im = cv2.imread(str(OUT / "crops" / r["file"]))
        if im is None:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        sigs[key] = hist.flatten().astype(np.float32)
    keys = sorted(sigs)
    dupes = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue                      # different clip
            c = float(cv2.compareHist(sigs[a], sigs[b], cv2.HISTCMP_CORREL))
            if c > corr:
                # The frame gap is what makes the report actionable: seconds
                # apart means one fragmented track, half a minute apart means
                # two vehicles from the same fleet.
                dupes.append((a, b, c, abs(frames[a] - frames[b])))
    return dupes


def contact_sheets(rows: list[dict], cols: int = 8, per_sheet: int = 48) -> int:
    """One tile per VEHICLE so a human labels each vehicle once."""
    seen, tiles = set(), []
    for r in rows:
        # The carriageway MUST be part of the key. The two passes number their
        # tracks independently, so track 2 exists in both and means two
        # different vehicles — keying on the number alone silently dropped one
        # of them from the sheet and captioned the survivor with the other's
        # label. A labeller cannot correct a vehicle that is not shown.
        key = (r["clip"], r["track"], r["carriageway"])
        if key in seen:
            continue
        seen.add(key)
        img = cv2.imread(str(OUT / "crops" / r["file"]))
        if img is None:
            continue
        img = cv2.resize(img, (CROP_PX, CROP_PX))
        cv2.rectangle(img, (0, 0), (CROP_PX - 1, 16), (0, 0, 0), -1)
        # Three distinct states, three distinct captions. Collapsing the last
        # two under one 'x' prefix made an analysed vehicle look like
        # oncoming traffic:
        #   #N   analysed AND counted — this number is drawn on the video
        #   aN   analysed but never crossed the line (no number on the video)
        #   xN   opposite carriageway
        if r["vehicle_no"]:
            who, colour = f'#{r["vehicle_no"]}', (0, 255, 255)      # yellow
        elif r["carriageway"] == "analysed":
            who, colour = f'a{r["track"]}', (120, 220, 120)         # green
        else:
            who, colour = f'x{r["track"]}', (150, 150, 150)         # grey
        cv2.putText(img, f'{r["clip"][:6]} {who} {r["draft_label"]}',
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1)
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--per-track", type=int, default=3)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=736)
    ap.add_argument("--analysed-only", action="store_true",
                    help="skip the second pass entirely: only the analysed "
                         "carriageway. Halves the runtime, but gives far fewer "
                         "training examples")
    args = ap.parse_args()

    for sub in ("crops", "sheets"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    videos = ([Path(v) for v in args.videos] if args.videos
              else sorted((ROOT / "samples").glob("*.mp4")))
    cfg = PipelineConfig()
    cfg.model, cfg.imgsz = args.model, args.imgsz

    rows: list[dict] = []
    for v in videos:
        print(f"harvesting {v.name} ...", flush=True)
        rows.extend(harvest(v, cfg, args.per_track,
                            include_opposite=not args.analysed_only))

    if not rows:
        print("no vehicles harvested")
        return 1
    man = OUT / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    sheets = contact_sheets(rows)

    # The two passes number their tracks independently, so the carriageway is
    # part of a vehicle's identity. Leaving it out of these keys under-counted
    # the set and, in contact_sheets, hid vehicles outright.
    vehicles = {(r["clip"], r["track"], r["carriageway"]) for r in rows}
    analysed = {(r["clip"], r["track"]) for r in rows
                if r["carriageway"] == "analysed" and r["counted"] == "yes"}
    per_vehicle_label = Counter()
    seen: set = set()
    for r in rows:
        k = (r["clip"], r["track"], r["carriageway"])
        if k not in seen:
            seen.add(k)
            per_vehicle_label[r["draft_label"]] += 1
    print()
    print(f"{len(vehicles)} distinct vehicles, {len(rows)} crops -> {man}")
    print(f"  of those, {len(analysed)} are analysed-carriageway vehicles that "
          f"crossed the counting line (these carry a #N matching the video)")
    print(f"{sheets} contact sheet(s) in {OUT / 'sheets'}")
    print("draft label mix (per vehicle):", dict(per_vehicle_label))

    dupes = audit_duplicates(rows)
    if dupes:
        print(f"\n[WARNING] {len(dupes)} pair(s) of vehicles look identical — the "
              "same vehicle may be filed twice,\n          which teaches the model "
              "to tell one vehicle from itself. Check these on the sheets:")
        for a, b, c, gap in dupes[:12]:
            verdict = ("LIKELY ONE VEHICLE" if gap <= TEMPORAL_SPLIT_FRAMES
                       else "probably two similar vehicles")
            print(f"          {a[2]}/{a[1]} == {b[2]}/{b[1]}  corr {c:.3f}  "
                  f"{gap} frames apart — {verdict}")
    else:
        print("\nduplicate audit: no vehicle appears twice")
    print("\nNext: correct the `label` column against the contact sheets, then\n"
          "      python -m tools.train_vehicle_classes --prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
