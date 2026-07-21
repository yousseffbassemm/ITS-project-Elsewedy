"""Regression tests for the ITS pipeline.

Every test here encodes a bug that was actually found and fixed, or an invariant
the analytics must satisfy. Dependency-free on purpose — no pytest needed:

    .venv\\Scripts\\python -m tests.test_pipeline
    .venv\\Scripts\\python -m tests.test_pipeline data/jobs/street_v2/analytics.json

Passing an analytics.json additionally checks that report's internal consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.classify import VehicleClassifier, native_code_map   # noqa: E402
from pipeline.config import PipelineConfig               # noqa: E402
from pipeline.congestion import CongestionMonitor        # noqa: E402
from pipeline.counting import LineCounter                # noqa: E402
from pipeline.lanes import LaneModel                     # noqa: E402
from pipeline.detect_track import visible_mask           # noqa: E402
from pipeline.reid import IdStabilizer, _group, _iou     # noqa: E402
from pipeline.plates import (                            # noqa: E402
    COLOR_TO_CODES, PlateReader, classify_band,
)
from pipeline.speed import SpeedEstimator                # noqa: E402

W, H, FPS = 1280, 720, 25.0
CFG = PipelineConfig()

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        _failures.append(name)


def _det(x1, y1, x2, y2, cls=2, tid=1):
    return sv.Detections(xyxy=np.array([[x1, y1, x2, y2]], dtype=float),
                         class_id=np.array([cls]), tracker_id=np.array([tid]))


# --------------------------------------------------------------------------
def test_counting_is_stride_independent():
    """A crossing must be counted no matter how coarsely the video is sampled.

    Regression: the old test compared the post-crossing POSITION against a
    +/-40 px band, so at stride 2-4 (74-111 px per step) the vehicle leapt the
    band and the crossing was silently discarded.
    """
    (_, ly), _ = CFG.line_px(W, H)[0], CFG.line_px(W, H)[1]
    for step in (37, 74, 111, 150, 200):
        c = LineCounter(CFG, W, H, None)
        for k in range(8):
            yy = ly + 3 * step - k * step
            c.update(_det(615, yy - 45, 665, yy, tid=7))
        check(f"crossing counted at ~{step}px/frame", c.total_vehicles() == 1,
              f"got {c.total_vehicles()}")


def test_counting_line_spans_the_whole_carriageway():
    """Both endpoints must sit on the carriageway edges at the line's own row.

    Regression: the line spanned x=0.15..0.87 while lane 1 sits at x~0.00..0.23
    at that row, so lane-1 vehicles crossed OUTSIDE it and were never counted.
    """
    (sx, sy), (ex, _) = CFG.line_px(W, H)
    y = float(sy)
    left = -1.0 * y + 426.5          # measured carriageway edge lines
    right = 1.0 * y + 726.0
    check("line starts at the left carriageway edge", abs(sx - left) <= 12,
          f"line {sx:.0f} vs edge {left:.0f}")
    check("line ends at the right carriageway edge", abs(ex - right) <= 12,
          f"line {ex:.0f} vs edge {right:.0f}")


def test_counting_rejects_off_segment():
    """A vehicle crossing the line's infinite extension, but past the segment
    ends, must NOT be counted (that is the opposite carriageway / off-road)."""
    (_, ly), _ = CFG.line_px(W, H)[0], CFG.line_px(W, H)[1]
    for x, expect, label in [(615, 1, "mid-carriageway"),
                             (1500, 0, "past the right end"),
                             (-500, 0, "past the left end")]:
        c = LineCounter(CFG, W, H, None)
        for k in range(8):
            yy = ly + 111 - k * 37
            c.update(_det(x - 25, yy - 45, x + 25, yy, tid=7))
        check(f"off-segment rejection: {label}", c.total_vehicles() == expect,
              f"got {c.total_vehicles()} expected {expect}")


def test_counting_counts_each_vehicle_once():
    """Jitter across the line must not produce a second count."""
    (_, ly), _ = CFG.line_px(W, H)[0], CFG.line_px(W, H)[1]
    c = LineCounter(CFG, W, H, None)
    for yy in [ly + 60, ly + 20, ly - 20, ly + 20, ly - 20, ly - 60]:
        c.update(_det(615, yy - 45, 665, yy, tid=7))
    check("vehicle wobbling across the line counts once",
          c.total_vehicles() == 1, f"got {c.total_vehicles()}")


# --------------------------------------------------------------------------
def test_lane_straddle_majority_rule():
    """A straddling vehicle belongs to the lane it covers most."""
    lm = LaneModel(CFG, W, H)
    y = 650.0

    def bx(i):
        (tx, ty), (nx, ny) = CFG.lane_dividers[i]
        t = (y / H - ty) / (ny - ty)
        return (tx + (nx - tx) * t) * W

    for lane in range(4):
        c = (bx(lane) + bx(lane + 1)) / 2
        check(f"box fully inside L{lane+1}",
              lm.assign([c - 20, y - 40, c + 20, y]) == lane)
    b = bx(2)
    check("straddle 80% in L2", lm.assign([b - 80, y - 40, b + 20, y]) == 1)
    check("straddle 80% in L3", lm.assign([b - 20, y - 40, b + 80, y]) == 2)
    check("straddle 60% in L3", lm.assign([b - 40, y - 40, b + 60, y]) == 2)
    check("off-carriageway box has no lane",
          lm.assign([-700, y - 40, -600, y]) is None)


def test_rightmost_lane_is_widest():
    """L4 absorbs the shoulder and must be the widest zone (measured geometry)."""
    lm = LaneModel(CFG, W, H)
    y = 600.0

    def bx(i):
        (tx, ty), (nx, ny) = CFG.lane_dividers[i]
        t = (y / H - ty) / (ny - ty)
        return (tx + (nx - tx) * t) * W

    widths = [bx(i + 1) - bx(i) for i in range(lm.n)]
    check("4 lanes configured", lm.n == 4, f"got {lm.n}")
    check("rightmost lane is the widest",
          int(np.argmax(widths)) == 3, f"widths {np.round(widths).tolist()}")


# --------------------------------------------------------------------------
def _drive(stride, true_kmh=85.0, drops=frozenset()):
    s = SpeedEstimator(CFG, W, H, FPS, stride)
    inv = np.linalg.inv(s.transformer.m)
    mps = true_kmh / 3.6
    out = None
    for f in range(0, 400, stride):
        if f in drops:
            continue
        y_m = 40.0 - mps * (f / FPS)
        img = cv2.perspectiveTransform(
            np.array([[[8.0, y_m]]], dtype=np.float32), inv)[0][0]
        if not (0.32 <= img[1] / H <= 0.90):
            continue
        r = s.update(_det(img[0]-20, img[1]-40, img[0]+20, img[1], tid=7), f)
        if 7 in r:
            out = r[7]
    return out


def test_speed_is_stride_independent():
    """Regression: the position deque held fps/stride entries, so at stride 8 it
    could never reach the old 6-sample gate and NO speed was ever emitted."""
    for st in (1, 2, 3, 4, 8, 12):
        got = _drive(st)
        ok = got is not None and abs(got - 85.0) < 2.0
        check(f"speed correct at stride {st}", ok,
              "no speed emitted" if got is None else f"got {got:.1f}")


def test_speed_survives_dropped_detections():
    """Regression: elapsed time assumed consecutive frames, so missed detections
    shortened the baseline and inflated speed by ~17%."""
    got = _drive(1, drops=frozenset({5, 12, 19, 26, 33, 40, 47, 60, 61, 62}))
    ok = got is not None and abs(got - 85.0) < 2.0
    check("speed correct with dropped detections", ok,
          "no speed" if got is None else f"got {got:.1f}")


def test_speed_histogram_is_complete():
    """Regression: bins stopped at 130 km/h while the ceiling was 160, so fast
    vehicles were counted but silently dropped from the chart."""
    s = SpeedEstimator(CFG, W, H, FPS, 1)
    s._veh_samples = {i: [v] * 4 for i, v in enumerate([5, 45, 88, 129, 141, 155])}
    summ = s.summary(VehicleClassifier())
    total = sum(summ["histogram"]["counts"])
    check("histogram counts sum to vehicles timed",
          total == summ["n_vehicles_timed"],
          f"{total} vs {summ['n_vehicles_timed']}")


# --------------------------------------------------------------------------
def test_reid_primitives():
    check("iou identical boxes", abs(_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1) < 1e-9)
    check("iou disjoint boxes", _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)
    check("car/truck/bus share a class group",
          _group(2) == _group(7) == _group(5))
    check("motorcycle is a different group", _group(3) != _group(2))
    s = IdStabilizer(FPS)
    s._alias = {3: 2, 2: 1}
    check("alias chain resolves with path compression",
          s._resolve(3) == 1 and s._alias[3] == 1)


def test_reid_tunables_scale_with_stride():
    """Anything measured in RAW frames must scale, or it breaks at high stride."""
    for st in (1, 2, 3, 4, 8, 12):
        s = IdStabilizer(FPS, st)
        sp = SpeedEstimator(CFG, W, H, FPS, st)
        check(f"stride {st}: gap tolerances exceed the step size",
              s.dup_gap >= 2 * st and sp.max_gap >= 3 * st and sp.window >= 6,
              f"dup_gap={s.dup_gap} speed_gap={sp.max_gap} window={sp.window}")


def test_reid_never_merges_distinct_vehicles():
    """Two vehicles visible at the same time must keep separate ids."""
    s = IdStabilizer(FPS)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :640] = (20, 20, 200)
    frame[:, 640:] = (200, 200, 20)
    boxes = np.array([[300., 400., 380., 480.], [800., 400., 880., 480.]])
    ids = s.assign(frame, boxes, np.array([1, 2]), np.array([2, 2]), 0)
    check("two simultaneous vehicles get distinct ids", ids[0] != ids[1],
          f"got {ids.tolist()}")
    for f in range(1, 12):
        ids = s.assign(frame, boxes, np.array([1, 2]), np.array([2, 2]), f)
    check("they stay distinct over time", ids[0] != ids[1], f"got {ids.tolist()}")


# --------------------------------------------------------------------------
def test_duplicate_boxes_collapse_quickly():
    """Two boxes on one vehicle must collapse fast.

    Regression: needing 4 qualifying frames left half the real duplicates
    unmerged, so a vehicle visibly carried two boxes with two ids — the id
    'flicking and coming back'.
    """
    s = IdStabilizer(FPS)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    boxes = np.array([[300., 400., 380., 500.], [302., 398., 378., 500.]])
    ids = None
    for f in range(4):
        ids = s.assign(frame, boxes, np.array([1, 2]), np.array([2, 2]), f)
    check("duplicate boxes share one id within 4 frames",
          ids[0] == ids[1], f"got {ids.tolist()}")
    check("exactly one merge was recorded", s.merges == 1, f"got {s.merges}")


def test_display_numbers_are_sequential():
    """On-screen numbers must run 1, 2, 3... in order of appearance.

    Regression: internal ids are allocation counters, so a car that briefly wore
    two boxes consumed two ids before they merged — the FIRST car on screen was
    labelled '#3' and the truck behind it '#4'.
    """
    s = IdStabilizer(FPS, display_delay=3)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    dup = np.array([[300., 400., 380., 500.], [302., 398., 378., 500.]])
    ids = None
    for f in range(6):                      # first car, detected twice
        ids = s.assign(frame, dup, np.array([1, 2]), np.array([2, 2]), f)
    first = s.display_id(int(ids[0]))
    for f in range(6, 14):                  # a truck joins behind it
        both = np.vstack([dup, [[900., 600., 1010., 700.]]])
        ids = s.assign(frame, both, np.array([1, 2, 5]), np.array([2, 2, 7]), f)
    second = s.display_id(int(ids[-1]))
    check("first vehicle on screen is #1", first == 1, f"got {first}")
    check("next vehicle is #2", second == 2, f"got {second}")
    check("duplicate boxes did not burn a number", s.merges == 1)


def test_numbering_has_no_gaps_when_a_duplicate_lingers():
    """A duplicate that survives a while before merging must not leave a gap.

    Numbers are retired when a duplicate merges away, so if one were numbered
    first the sequence would read #1, #2, #4. The delay is set from measured
    duplicate lifetimes (<=10 frames) versus real track lengths (>=39), and the
    display number is requested every frame exactly as the renderer does.
    """
    s = IdStabilizer(FPS)                    # default (measured) delay
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    apart = np.array([[300., 400., 380., 500.], [700., 300., 760., 380.]])
    close = np.array([[300., 400., 380., 500.], [302., 398., 378., 500.]])
    for f in range(10):                      # two separate boxes for 10 frames
        ids = s.assign(frame, apart, np.array([1, 2]), np.array([2, 2]), f)
        for t in ids:
            s.display_id(int(t))
    for f in range(10, 60):                  # they converge -> one vehicle
        ids = s.assign(frame, close, np.array([1, 2]), np.array([2, 2]), f)
        for t in ids:
            s.display_id(int(t))
    r = s.numbering_report()
    check("numbering stays contiguous from 1", r["contiguous_from_1"], str(r))
    check("no number is used twice", r["no_duplicate_numbers"], str(r))
    check("highest number equals vehicles numbered",
          r["highest_number"] == r["vehicles_numbered"], str(r))


def test_vehicle_past_the_line_is_not_counted():
    """A vehicle already beyond the line when the clip starts must NOT be counted.

    It never crosses, so it is not traffic passing this section. This is
    deliberate: it is the difference between the reported 17 and a naive manual
    tally of 18 on street_egypt.mp4.
    """
    (_, ly), _ = CFG.line_px(W, H)[0], CFG.line_px(W, H)[1]
    c = LineCounter(CFG, W, H, None)
    for k in range(10):                      # starts above the line, drives away
        yy = ly - 20 - k * 25
        c.update(_det(615, yy - 45, 665, yy, tid=7))
    check("vehicle starting past the line is not counted",
          c.total_vehicles() == 0, f"got {c.total_vehicles()}")


def test_draw_time_duplicate_suppression():
    """One vehicle must never be drawn with two boxes, but a genuinely occluded
    vehicle behind another must still be drawn."""
    def d(boxes, conf):
        return sv.Detections(xyxy=np.array(boxes, dtype=float),
                             class_id=np.array([2] * len(boxes)),
                             tracker_id=np.arange(1, len(boxes) + 1),
                             confidence=np.array(conf))
    m = visible_mask(d([[300, 400, 380, 500], [302, 398, 378, 500]], [0.9, 0.7]))
    check("second box on the same vehicle is hidden",
          m.tolist() == [True, False], f"got {m.tolist()}")
    m = visible_mask(d([[300, 400, 380, 500], [800, 400, 880, 500]], [0.9, 0.8]))
    check("two separate vehicles both drawn", m.all(), f"got {m.tolist()}")
    m = visible_mask(d([[300, 300, 380, 460], [300, 380, 380, 500]], [0.9, 0.8]))
    check("a vehicle behind another is still drawn", m.all(), f"got {m.tolist()}")
    check("empty frame is handled",
          visible_mask(sv.Detections.empty()).tolist() == [])


def test_class_mapping():
    """COCO truck spans pickups and lorries; size decides which.

    Regression: every COCO truck was mapped to D (heavy), which was wrong on 6 of
    the 7 trucks in the clip — they are pickups and flatbeds.
    """
    m = VehicleClassifier._map
    check("car -> A", m(2, 1.5, 1.8) == "A")
    check("bus -> E", m(5, 3.0, 2.5) == "E")
    check("motorcycle -> G", m(3, 1.5, 0.8) == "G")
    check("pickup-sized truck -> C (light)", m(7, 2.0, 1.9) == "C")
    check("lorry-sized truck -> D (heavy)", m(7, 3.5, 3.0) == "D")
    check("truck with no size estimate defaults to C, not D",
          m(7, 0.0, 0.0) == "C")
    check("unknown class -> F", m(99, 0.0, 0.0) == "F")


def test_class_vote_prefers_near_field_evidence():
    """A clear close-up must outweigh many distant, low-confidence blobs."""
    clf = VehicleClassifier()                  # no transformer: confidence only
    for _ in range(10):
        clf.observe(1, 7, conf=0.31)           # distant, weakly called a truck
    for _ in range(6):
        clf.observe(1, 2, conf=0.92)           # near, confidently a car
    check("weighted vote follows the confident evidence",
          clf.resolve(1) == "A", f"got {clf.resolve(1)}")


def test_finetuned_model_is_detected_and_takes_over():
    """A model trained on the 7 classes must bypass the COCO heuristics.

    The test is not "do these names look like vehicles" — COCO's car/truck/bus
    would pass that — but "does the model know a class COCO cannot express",
    i.e. C or V. That is precisely what the fine-tune adds.
    """
    check("plain COCO model uses the heuristic",
          native_code_map({0: "person", 2: "car", 5: "bus", 7: "truck"}) is None)
    check("code-named model detected",
          native_code_map(["A", "C", "D", "E", "G", "V", "F"])[1] == "C")
    check("descriptively named model detected",
          native_code_map({0: "private car", 1: "light truck", 2: "heavy truck",
                           3: "bus", 4: "motorcycle", 5: "van", 6: "unknown"})[5] == "V")
    check("snake_case names detected",
          native_code_map({0: "private_car", 1: "light_truck", 2: "heavy_truck",
                           3: "microbus", 4: "motorbike", 5: "panel_van",
                           6: "other"})[3] == "E")
    check("a model with an alien class is rejected",
          native_code_map({0: "car", 1: "van", 2: "aeroplane"}) is None)
    # and the classifier must then trust the model instead of measuring size
    clf = VehicleClassifier(None, {0: "A", 1: "C", 2: "D", 5: "V"})
    clf.observe(1, 5, conf=0.9)
    check("native model's class head wins over the size heuristic",
          clf.resolve(1) == "V", f"got {clf.resolve(1)}")


def test_vans_are_not_guessed():
    """Vans are reported as A rather than guessed, by design.

    Measured heights of vans and cars overlap completely, and the best threshold
    tuned on the test data still misassigned 4 of 11. Emitting V would mean
    labelling saloons as vans.
    """
    m = VehicleClassifier._map
    check("a van-shaped COCO car is still A", m(2, 2.1, 1.8) == "A")


def test_congestion_needs_actual_traffic():
    """Regression: occupancy is an image-space area fraction, so one lorry close
    to the camera covered a third of the ROI and headlined the report as
    'Heavy' on a road that was Free-flow 96.8% of the time."""
    cm = CongestionMonitor(CFG, W, H)
    check("0 vehicles cannot be congested",
          cm.classify(0.60, 60.0, 0) == "Free-flow")
    check("1 vehicle cannot be congested",
          cm.classify(0.60, 60.0, 1) == "Free-flow")
    check("2 vehicles cap at Moderate", cm.classify(0.60, 60.0, 2) == "Moderate")
    check("real congestion still reports Jam",
          cm.classify(0.60, 5.0, 6) == "Jam")


def test_congestion_ignores_blips():
    """A single frame at a level is not a traffic condition."""
    cm = CongestionMonitor(CFG, W, H)
    for i in range(200):
        cm.samples.append({"t": i * 0.04, "occ": 0.5 if i == 100 else 0.01,
                           "n": 6, "avg_speed": 60.0,
                           "level": "Heavy" if i == 100 else "Free-flow"})
    s = cm.summary()
    check("headline level ignores a one-frame blip",
          s["worst_level"] == "Free-flow", f"got {s['worst_level']}")
    check("the blip is still reported as peak_instant_level",
          s["peak_instant_level"] == "Heavy")
    check("zero-length peak periods are dropped", s["peak_periods"] == [])


def test_empty_clip_does_not_crash():
    """A clip with no vehicles must still produce a valid report."""
    empty = sv.Detections.empty()
    ctr = LineCounter(CFG, W, H, LaneModel(CFG, W, H))
    ctr.update(empty)
    sp = SpeedEstimator(CFG, W, H, FPS, 1)
    sp.update(empty, 0)
    cg = CongestionMonitor(CFG, W, H)
    cg.update(empty, 0.0, 0.0)
    clf = VehicleClassifier()
    check("empty counting summary", ctr.summary(clf)["total_vehicles"] == 0)
    check("empty speed summary", sp.summary(clf)["n_vehicles_timed"] == 0)
    check("empty congestion summary", cg.summary()["worst_level"] == "Free-flow")


def test_analytics_invariants(path: str) -> None:
    """Cross-check a produced report for internal consistency."""
    a = json.loads(Path(path).read_text())
    c, s = a["counting"], a["speed"]
    total = c["total_vehicles"]
    check("by_direction sums to total",
          c["by_direction"]["in"] + c["by_direction"]["out"] == total)
    check("by_class_code sums to total", sum(c["by_class_code"].values()) == total)
    check("lane counts do not exceed total",
          sum(l["count"] for l in a["lanes"]) <= total)
    check("histogram sums to vehicles timed",
          sum(s["histogram"]["counts"]) == s["n_vehicles_timed"])
    check("histogram bins match counts",
          len(s["histogram"]["bins"]) == len(s["histogram"]["counts"]))
    ts = a["timeseries"]
    check("timeseries arrays are equal length",
          len({len(ts["t_sec"]), len(ts["vehicle_count"]),
               len(ts["avg_speed_kmh"]), len(ts["congestion_level"])}) == 1)
    check("max speed >= median speed", s["max_kmh"] >= s["median_kmh"])
    check("lane percentages are sane",
          all(0 <= l["pct"] <= 100 for l in a["lanes"]))
    if total:
        check("busiest lane is a real lane",
              a["throughput"]["busiest_lane"] in {l["lane"] for l in a["lanes"]})


# --- plates -----------------------------------------------------------------------
def test_washed_out_band_is_still_read() -> None:
    """A real red plate band measured S=57 and an absolute S>=70 rule rejected it.

    At a 32 px plate width, H.264 chroma subsampling leaves the band ~2 px of
    chroma, so saturation is genuinely low for EVERY plate at this scale — the
    threshold was tuned on clean plate photos that this camera never produces.
    Judging the band against the plate's own white body fixes it without
    loosening the rule into calling grey things coloured.

    Numbers are the ones actually measured off the box truck in
    samples/street_egypt.mp4 (band H=169 S=57 V=120, body S=16).
    """
    check("washed-out red band reads as red",
          classify_band(169, 57, 120, s_body=16, v_body=150) == "red",
          f"got {classify_band(169, 57, 120, s_body=16, v_body=150)}")
    # ...and red must map to the truck codes, which is the whole point of
    # reading colour: it is evidence for the C/D classes.
    check("red band supports the truck classes",
          set(COLOR_TO_CODES["red"]) == {"C", "D"})


def test_neutral_surfaces_are_not_given_a_colour() -> None:
    """The relative rule must not turn every bright patch into a category.

    These are measured off the same frame: the white truck body, its roof, and
    road asphalt. A colour reading on any of them would be a confident wrong
    answer that then votes on the vehicle's class.
    """
    for name, (h, s, v, sb) in {
        "white truck body": (3, 15, 192, 19),
        "truck roof": (8, 25, 216, 22),
        "road asphalt": (120, 32, 63, 44),
    }.items():
        got = classify_band(h, s, v, s_body=sb)
        check(f"{name} is not given a plate colour",
              got in ("white", "unknown"), f"got {got}")


def test_plate_colour_needs_repeated_evidence() -> None:
    """One frame is not enough — a brake light bleeding onto the band is red too.

    A single sample must resolve to 'unknown' rather than committing, because at
    this resolution any one frame's reading is noisy.
    """
    r = PlateReader()
    r._colors[7]["red"] = 5.0
    r._widths[7] = [30.0]                       # only one observation
    check("a single plate sample does not resolve a colour",
          r.color_of(7) == "unknown", f"got {r.color_of(7)}")
    r._widths[7] = [30.0, 31.0, 29.0]           # now enough
    r._cache.pop(7, None)
    check("repeated samples do resolve", r.color_of(7) == "red",
          f"got {r.color_of(7)}")


def test_missing_plate_model_does_not_crash() -> None:
    """A missing plate model must degrade, not take the whole run down.

    The rest of the analytics — counting, speed, lanes, congestion — do not
    depend on plates, so a missing weights file must never cost the user their
    entire job.
    """
    r = PlateReader(weights="definitely_not_a_real_model.pt")
    check("missing weights are reported, not raised", r.load_error is not None)
    check("reader still usable without a model", r.color_of(1) == "unknown")
    s = r.summary([1])
    check("summary works with no model", s["plates_detected"] == 0)


def test_ocr_absence_is_explained() -> None:
    """An empty OCR column must say WHY, or it reads as a broken model.

    This is the failure this whole stage is designed around: on 34 px footage
    OCR cannot work, and if the report is silent about that, the natural
    conclusion is that the model needs more training data — which would be
    weeks spent on the wrong problem.
    """
    r = PlateReader()
    r._widths[3] = [34.0, 33.0, 35.0]           # the real street_egypt range
    s = r.summary([3])
    check("no-OCR case is attributed to resolution",
          "camera-resolution limit" in s["ocr_note"], s["ocr_note"])
    check("the offending plate size is quoted",
          f"{s['plate_px_p90']:.0f}px" in s["ocr_note"], s["ocr_note"])
    check("the shortfall factor is quoted", "x short" in s["ocr_note"],
          s["ocr_note"])

    # Adequate footage with no engine must NOT be blamed on resolution — that
    # would send someone off to buy a camera they already have.
    r2 = PlateReader()
    r2._widths[4] = [150.0, 155.0, 160.0]
    note = r2.summary([4])["ocr_note"]
    check("adequate footage is not blamed on resolution",
          "COULD be read" in note, note)


def main() -> int:
    print("counting")
    test_counting_is_stride_independent()
    test_counting_line_spans_the_whole_carriageway()
    test_counting_rejects_off_segment()
    test_counting_counts_each_vehicle_once()
    print("lanes")
    test_lane_straddle_majority_rule()
    test_rightmost_lane_is_widest()
    print("speed")
    test_speed_is_stride_independent()
    test_speed_survives_dropped_detections()
    test_speed_histogram_is_complete()
    print("re-identification")
    test_reid_primitives()
    test_reid_tunables_scale_with_stride()
    test_reid_never_merges_distinct_vehicles()
    test_duplicate_boxes_collapse_quickly()
    test_display_numbers_are_sequential()
    test_numbering_has_no_gaps_when_a_duplicate_lingers()
    test_vehicle_past_the_line_is_not_counted()
    test_draw_time_duplicate_suppression()
    print("classification")
    test_class_mapping()
    test_class_vote_prefers_near_field_evidence()
    test_vans_are_not_guessed()
    test_finetuned_model_is_detected_and_takes_over()
    print("plates")
    test_washed_out_band_is_still_read()
    test_neutral_surfaces_are_not_given_a_colour()
    test_plate_colour_needs_repeated_evidence()
    test_missing_plate_model_does_not_crash()
    test_ocr_absence_is_explained()
    print("congestion")
    test_congestion_needs_actual_traffic()
    test_congestion_ignores_blips()
    print("robustness")
    test_empty_clip_does_not_crash()
    if len(sys.argv) > 1:
        print(f"analytics invariants ({sys.argv[1]})")
        test_analytics_invariants(sys.argv[1])
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
