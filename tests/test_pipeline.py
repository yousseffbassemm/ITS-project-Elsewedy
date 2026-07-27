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
from pipeline.config import (                            # noqa: E402
    COCO_SCHEME, UNGROUPED, PipelineConfig, scheme_for_codes,
)
from pipeline.congestion import CongestionMonitor        # noqa: E402
from pipeline.counting import LineCounter                # noqa: E402
from pipeline.lanes import LaneModel                     # noqa: E402
from pipeline.detect_track import VehicleDetector, visible_mask  # noqa: E402
from pipeline.reid import IdStabilizer, _group, _iou     # noqa: E402
from pipeline.plates import (                            # noqa: E402
    COLOR_TO_CODES, PlateReader, classify_band,
)
from pipeline.plate_ocr import (                          # noqa: E402
    PlateEnhancer, crop_score, sharpness, vote,
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


def test_finetuned_model_works_downstream_not_just_in_classify():
    """A 7-class model must be understood by COUNTING, SPEED and OCCUPANCY too.

    Regression, and the most damaging one found: classify.py honoured a
    fine-tuned model's class head, but counting, speed, congestion and the re-id
    class gate all hard-coded the COCO ids. On the model docs/finetuning-plan.md
    §7 describes as a drop-in:

        class 0 is 'A' (private car), and PERSON_CLASS is 0  -> every private
            car crossing the line was recorded as a PEDESTRIAN, so
            total_vehicles omitted the commonest class entirely
        classes 4 (G) and 6 (F) are in no COCO set -> dropped from counts,
            from the speed sample and from the occupancy measure
        re-id groups are COCO-keyed -> 'A' landed in the pedestrian group and
            'G' in no group at all, so the class gate was nonsense

    Everything now takes its class meanings from config.ClassScheme.
    """
    names = {0: "A", 1: "C", 2: "D", 3: "E", 4: "G", 5: "V", 6: "F"}
    sch = scheme_for_codes(native_code_map(names))
    (_, ly), _ = CFG.line_px(W, H)[0], CFG.line_px(W, H)[1]
    cm = CongestionMonitor(CFG, W, H, scheme=sch)
    for cid, code in names.items():
        c = LineCounter(CFG, W, H, None, scheme=sch)
        for k in range(8):
            yy = ly + 3 * 74 - k * 74
            c.update(_det(615, yy - 45, 665, yy, cls=cid, tid=7))
        occ, _n = cm._occupancy(_det(400, 500, 500, 600, cls=cid, tid=1))
        ok = (c.total_vehicles() == 1
              and c.pedestrians_in + c.pedestrians_out == 0
              and occ > 0)
        check(f"native class {cid} ({code}) counts as a vehicle", ok,
              f"counted={c.total_vehicles()} "
              f"peds={c.pedestrians_in + c.pedestrians_out} occ={occ:.4f}")
    check("native motorcycle keeps its own re-id group",
          sch.group(4) != sch.group(0), f"G={sch.group(4)} A={sch.group(0)}")
    check("native unknown (F) is ungrouped, so it never stitches",
          sch.group(6) == UNGROUPED, f"got {sch.group(6)}")
    check("a fine-tuned model has no pedestrian class at all",
          not sch.person_ids, f"got {sch.person_ids}")
    # ...and a plain COCO model must be completely unaffected.
    check("COCO scheme still calls id 0 a person", COCO_SCHEME.is_person(0))
    check("COCO scheme still calls id 2 a vehicle", COCO_SCHEME.is_vehicle(2))
    check("COCO scheme is what an un-fine-tuned model gets",
          scheme_for_codes(None) is COCO_SCHEME)


def test_ungrouped_classes_never_merge_with_each_other():
    """Two 'unknown' vehicles agreeing on being unknown is not evidence.

    The re-id gate compares class GROUPS, so without this an id whose class the
    scheme does not recognise would match any other unrecognised id — the gate
    would pass on -1 == -1 and weld two different vehicles together.
    """
    sch = scheme_for_codes({0: "F", 1: "F"})
    s = IdStabilizer(FPS, scheme=sch)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    a = np.array([[300., 400., 380., 480.]])
    s.assign(frame, a, np.array([1]), np.array([0]), 0)
    # same place a few frames later, a brand-new raw id: a stitch candidate
    ids = s.assign(frame, a, np.array([2]), np.array([1]), 4)
    check("an ungrouped class is not stitched to another ungrouped one",
          s.stitches == 0, f"stitches={s.stitches} ids={ids.tolist()}")


def test_near_field_views_outweigh_distant_ones():
    """The class vote must actually weight by apparent size, as documented.

    Regression: the weight divided pixel height by the image scale at the
    vehicle's ground point, which converts it to a PHYSICAL height and cancels
    distance exactly — a 15 px blob counted as much as a 200 px close-up, while
    two separate docstrings claimed the opposite. Distant frames vastly
    outnumber near ones, so unweighted they decide the class.
    """
    sp = SpeedEstimator(CFG, W, H, FPS, 1)
    clf = VehicleClassifier(sp.transformer)
    far = [630.0, 0.35 * H - 20, 650.0, 0.35 * H]        # 20 px tall
    near = [600.0, 0.85 * H - 200, 700.0, 0.85 * H]      # 200 px tall
    clf.observe(1, 7, far, conf=0.9)
    w_far = clf._votes[1][7]
    clf2 = VehicleClassifier(sp.transformer)
    clf2.observe(1, 7, near, conf=0.9)
    w_near = clf2._votes[1][7]
    check("a near view carries more vote weight than a distant one",
          w_near > 5 * w_far, f"near={w_near:.3f} far={w_far:.3f}")

    # And the consequence: many distant weak calls must not beat a few close
    # confident ones.
    c = VehicleClassifier(sp.transformer)
    for _ in range(20):
        c.observe(1, 7, far, conf=0.35)          # distant, weakly "truck"
    for _ in range(3):
        c.observe(1, 2, near, conf=0.90)         # near, confidently "car"
    check("20 distant blobs do not outvote 3 clear close-ups",
          c.resolve(1) == "A", f"got {c.resolve(1)} votes={dict(c._votes[1])}")

    # Metric size must still be recorded — it is what the C/D split runs on.
    h_m, w_m = c.size_estimate(1)
    check("metric size is still measured for the C/D test", h_m > 0 and w_m > 0,
          f"h={h_m:.2f} w={w_m:.2f}")


def test_plate_overlay_box_does_not_go_stale():
    """The drawn plate box must belong to the CURRENT frame.

    Regression: _boxes was never cleared, so once a plate was located the
    annotator kept drawing a rectangle at that position for the rest of the
    vehicle's life — a box detached from its vehicle, sliding backwards down the
    road. Plate detection is intermittent at 34 px, so this was the normal case.
    """
    r = PlateReader()
    frame = np.full((720, 1280, 3), 120, np.uint8)
    det = sv.Detections(xyxy=np.array([[560.0, 360.0, 700.0, 460.0]]),
                        class_id=np.array([2]), tracker_id=np.array([7]))
    r._observe_one(7, frame, (600, 400, 634, 417))
    check("a box found this frame is drawn", r.box_of(7) is not None)
    # A frame in which the detector finds no plate must clear it. (No model is
    # loaded, so _find_plates_frame returns nothing — exactly that case.)
    r.observe_frame(frame, det)
    check("a box not re-found this frame is not drawn", r.box_of(7) is None,
          f"got {r.box_of(7)}")


def test_speed_report_carries_per_vehicle_figures():
    """analytics.json must expose per-vehicle speeds.

    Regression: tools/export_plates_csv reads speed.per_vehicle to fill its
    speed column, and the pipeline never emitted that key — so the column was
    silently blank in every exported table.
    """
    s = SpeedEstimator(CFG, W, H, FPS, 1)
    s._veh_samples = {4: [80.0] * 4, 9: [40.0] * 4}
    summ = s.summary(VehicleClassifier())
    pv = summ.get("per_vehicle")
    check("per_vehicle is published", isinstance(pv, dict) and len(pv) == 2,
          f"got {pv}")
    check("keys survive a JSON round-trip as track ids",
          {int(k) for k in json.loads(json.dumps(pv))} == {4, 9}, f"got {pv}")
    check("empty runs still publish the key",
          SpeedEstimator(CFG, W, H, FPS, 1).summary(
              VehicleClassifier())["per_vehicle"] == {})


def test_crop_classifier_overrides_the_size_heuristic():
    """A trained crop classifier must win, and must never be a hard dependency.

    The heuristic below it is a monocular size estimate over COCO classes that
    cannot express C or V at all and has no microbus concept — measured on
    street_egypt.mp4 it calls bus/microbus right 1 time in 24, on a road where
    microbuses are a third of the traffic. Mixing the two votes would let the
    weaker signal dilute the stronger one, so code votes take priority outright.
    """
    class _Fake:
        model = True

        def __init__(self, code, conf=0.9):
            self.code, self.conf = code, conf
            self.calls = 0

        def predict(self, crop):
            self.calls += 1
            return (self.code, self.conf)

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    box = [500.0, 300.0, 600.0, 420.0]

    fake = _Fake("E")
    clf = VehicleClassifier(None, None, crop_classifier=fake, crop_every=1)
    clf.observe(7, 2, box, conf=0.95, frame=frame)      # COCO says car -> A
    check("the crop classifier's class wins over the COCO heuristic",
          clf.resolve(7) == "E", f"got {clf.resolve(7)}")

    # ...and with no frame, or no classifier, the heuristic still runs.
    plain = VehicleClassifier()
    plain.observe(7, 2, box, conf=0.95)
    check("without a crop classifier the heuristic still resolves",
          plain.resolve(7) == "A", f"got {plain.resolve(7)}")

    # Sampling: a second inference per vehicle per frame is expensive on CPU,
    # and a vehicle is visible for tens of frames.
    sampled = _Fake("E")
    c2 = VehicleClassifier(None, None, crop_classifier=sampled, crop_every=5)
    for _ in range(10):
        c2.observe(9, 2, box, conf=0.9, frame=frame)
    check("classifier is sampled, not run every frame", sampled.calls == 2,
          f"{sampled.calls} calls in 10 observations")

    # A classifier that failed to load must be ignored entirely, not raise:
    # a class label is not worth losing the counts over.
    class _Dead:
        model = None

    c3 = VehicleClassifier(None, None, crop_classifier=_Dead())
    c3.observe(7, 2, box, conf=0.95, frame=frame)
    check("unloadable weights fall back to the heuristic",
          c3.resolve(7) == "A", f"got {c3.resolve(7)}")


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


def test_confirm_rate_detects_a_broken_stride() -> None:
    """A stride too coarse to track must be REPORTED, not silently under-counted.

    ByteTrack has no notion of frame_stride: it treats consecutive calls as
    consecutive frames, so at a coarse stride its motion model is wrong by the
    stride factor and association fails. Counting needs an unbroken id either
    side of the line, so a modest drop in confirmation becomes a large drop in
    counts. Measured on street_egypt.mp4:

        stride 1  100% confirmed -> 18 counted
        stride 3   99% confirmed -> 17 counted
        stride 6   67% confirmed ->  5 counted    <- silent 72% under-count

    The first version of this guard measured displacement only for tracks that
    had ALREADY associated successfully, so it reported "OK" on the stride-6 run
    that counted 5 of 18 — it could only observe the successes. The rate is
    therefore measured on the RAW detection stream, before any tracker filtering.
    """
    d = VehicleDetector.__new__(VehicleDetector)
    d.raw_detections, d.confirmed_detections = 0, 0
    check("no detections reports healthy rather than dividing by zero",
          d.confirm_rate == 1.0)

    d.raw_detections, d.confirmed_detections = 100, 100
    check("stride 1 (all confirmed) is healthy", d.confirm_rate >= 0.99)

    # The real stride-6 measurement.
    d.raw_detections, d.confirmed_detections = 100, 67
    check("a broken stride is caught", d.confirm_rate < 0.95,
          f"{d.confirm_rate:.2f}")


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
    # The blue population measured across 2,017 EALPR plates is one cloud at
    # H 90-120, not two. Both ends must land on the same category.
    for h in (95, 104, 118):
        check(f"blue band H={h} reads as blue",
              classify_band(h, 150, 130, s_body=20) == "blue",
              f"got {classify_band(h, 150, 130, s_body=20)}")
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


def test_oversized_plate_box_is_rejected() -> None:
    """A 'plate' most of a vehicle wide is a false positive, not a plate.

    Running the plate detector on tight vehicle CROPS put it far outside its
    training domain — it was trained on full scenes where plates are small
    objects. On a 100x126 crop it returned a box 83% of the vehicle width at 0.73
    confidence: the whole rear panel. Those detections reported ~150 px plates in
    a clip whose real maximum is ~35 px, which made unreadable footage look
    adequate for OCR — precisely the wrong conclusion, arrived at confidently.

    Detection now runs per-frame, and this guard is the backstop.
    """
    det = sv.Detections(
        xyxy=np.array([[100.0, 100.0, 200.0, 226.0]]),   # a 100x126 vehicle
        class_id=np.array([2]), tracker_id=np.array([1]),
    )
    # 83 px wide inside a 100 px vehicle — the real false positive observed.
    check("an 83%-of-vehicle box is not accepted as a plate",
          PlateReader._owner((105.0, 180.0, 188.0, 200.0), det) is None)
    # ~18 px is the plausible real plate, and it must still be attached.
    check("a plausibly sized plate is attached to its vehicle",
          PlateReader._owner((140.0, 200.0, 158.0, 210.0), det) == 1)
    # A plate on no vehicle at all belongs to nobody.
    check("a plate outside every vehicle is dropped",
          PlateReader._owner((600.0, 600.0, 618.0, 610.0), det) is None)


def test_plate_goes_to_the_tightest_containing_vehicle() -> None:
    """Overlapping boxes: the smallest container owns the plate.

    On this camera a distant vehicle is often framed inside a nearer one's box,
    and attaching its plate to the wrong vehicle would corrupt that vehicle's
    colour vote and therefore its class evidence.
    """
    det = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 400.0, 400.0],        # big near vehicle
                       [100.0, 100.0, 180.0, 180.0]]),  # small far vehicle
        class_id=np.array([2, 2]), tracker_id=np.array([7, 9]),
    )
    check("the tighter vehicle owns the plate",
          PlateReader._owner((130.0, 150.0, 142.0, 158.0), det) == 9)


def test_plate_colour_needs_repeated_evidence() -> None:
    """One frame is not enough — a brake light bleeding onto the band is red too.

    A single sample must resolve to 'unknown' rather than committing, because at
    this resolution any one frame's reading is noisy.

    The gate counts COLOUR readings, not plate detections. It used to count
    len(_widths), which is appended for EVERY plate box including the sub-20px
    ones no band is ever read from — so three tiny detections plus one colour
    sample named a category off that single frame, which is exactly what the
    gate exists to prevent.
    """
    r = PlateReader()
    r._colors[7]["red"] = 5.0
    r._color_n[7] = 1                           # only one colour reading
    r._widths[7] = [30.0]
    check("a single plate sample does not resolve a colour",
          r.color_of(7) == "unknown", f"got {r.color_of(7)}")

    # The closed loophole: plenty of DETECTIONS, still only one colour read.
    r._widths[7] = [12.0, 14.0, 13.0, 30.0]
    r._cache.pop(7, None)
    check("detections without colour readings do not satisfy the gate",
          r.color_of(7) == "unknown", f"got {r.color_of(7)}")

    r._color_n[7] = 3                           # now enough real colour reads
    r._cache.pop(7, None)
    check("repeated colour samples do resolve", r.color_of(7) == "red",
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


def test_ocr_gate_blocks_hallucinated_reads() -> None:
    """Below the floor the OCR engine must not be CALLED, not merely ignored.

    Measured, not hypothetical: a real Egyptian ALPR model (30-class Arabic
    character detector) run on street_egypt.mp4 emitted 43 character detections.
    Every one was a false positive — none fell inside a detected plate, and their
    median box was 13 px wide where a real character on a 30 px plate is ~4-6 px.
    Mean confidence 0.28, with one class accounting for 31 of the 43.

    So OCR on inadequate footage does not fail silently; it produces
    confident-looking garbage. That is more dangerous than an empty column,
    because a plate number that looks plausible gets believed. The resolution
    gate is what stops it, and it has to gate the CALL.
    """
    calls: list = []

    def spy(crop):
        calls.append(crop.shape)
        return ("HALLUCINATED", 0.9)

    frame = np.full((720, 1280, 3), 120, np.uint8)
    r = PlateReader(ocr=spy)
    for w in (24, 30, 34, 35):                  # the real street_egypt range
        r._observe_one(1, frame, (100, 400, 100 + w, 400 + w // 2))
    check("OCR is never invoked below the floor", not calls, f"{len(calls)} calls")
    check("no hallucinated text reaches the report", r.text_of(1)[0] is None)

    r2 = PlateReader(ocr=spy)
    for _ in range(3):
        r2._observe_one(2, frame, (100, 400, 260, 455))    # 160 px — adequate
    check("OCR does run on adequate footage", len(calls) == 3, f"{len(calls)}")
    check("its text is captured", r2.text_of(2)[0] == "HALLUCINATED")


def test_sharper_crop_scores_higher() -> None:
    """Best-crop selection must prefer a sharp small plate over a blurred big one.

    This ordering is the whole point of scoring: a blurred plate is unreadable at
    any size, so sharpness is weighted above size. A selector that just took the
    largest crop would spend the super-resolution pass on motion blur.
    """
    sharp_small = crop_score(conf=0.5, sharp=4000, width=25)
    blurry_big = crop_score(conf=0.9, sharp=40, width=60)
    check("a sharp small crop beats a blurred large one",
          sharp_small > blurry_big, f"{sharp_small:.3f} vs {blurry_big:.3f}")

    # A flat image has essentially no Laplacian energy; an edge image has a lot.
    flat = np.full((40, 80, 3), 128, np.uint8)
    edges = flat.copy()
    edges[:, ::4] = 255
    check("sharpness separates flat from detailed",
          sharpness(edges) > sharpness(flat) + 100,
          f"{sharpness(edges):.1f} vs {sharpness(flat):.1f}")


def test_vote_prefers_agreement_over_count() -> None:
    """Voting is confidence-weighted, so blurry agreement cannot outvote clarity.

    Three low-confidence reads agreeing on the wrong answer is exactly what
    correlated failure looks like at this resolution, and it must not beat one
    confident read. Weighting by count instead of confidence would get this
    backwards.
    """
    text, conf, detail = vote([("ABC", 0.9), ("ABD", 0.2), ("ABD", 0.2)])
    check("the confident read wins the disputed position", text == "ABC", text)
    check("agreement is reported per position",
          len(detail["per_position_agreement"]) == 3, str(detail))

    # Reads of different lengths are not comparable position-by-position; the
    # best-supported length must win before any character voting happens.
    text2, _, d2 = vote([("AB", 0.9), ("ABCD", 0.3), ("ABCE", 0.3)])
    check("the best-supported length is chosen first", text2 == "AB", text2)
    check("only same-length reads are voted", d2["reads_at_best_length"] == 1,
          str(d2))

    check("no reads gives no answer", vote([])[0] == "")


def test_tailgate_badges_are_not_accepted_as_plates() -> None:
    """The widest 'plate' in street_egypt.mp4 was a CHEVROLET badge.

    Measured, and it is the most expensive false positive in the stage: at
    94x17 px it was bright, sharp and horizontal, so the crop scorer ranked it
    the BEST crop for that vehicle (0.872), spent super-resolution on it, fused
    it with the two genuine plate views — degrading them — and inflated the
    reported plate size, making the footage look more readable than it is.

    Shape is what separates them. Real Egyptian plates run ~2:1 to 3:1; the
    badge is 5.5:1. The upper bound stays generous because a plate viewed from
    a steep angle foreshortens vertically and its aspect ratio rises.
    """
    frame = np.full((720, 1280, 3), 120, np.uint8)
    r = PlateReader()
    r._observe_one(1, frame, (100, 400, 194, 417))       # 94x17 badge
    check("a 5.5:1 strip is rejected as not plate-shaped",
          r.plate_px(1) == 0.0 and r.rejected_shape == 1,
          f"px={r.plate_px(1)} rejected={r.rejected_shape}")

    # ...and the three genuine plates measured off the same clip must survive.
    for w, h, aspect in [(54, 21, "2.6"), (46, 16, "2.9"), (43, 14, "3.1")]:
        r2 = PlateReader()
        r2._observe_one(1, frame, (100, 400, 100 + w, 400 + h))
        check(f"a real {w}x{h} plate ({aspect}:1) is kept",
              r2.plate_px(1) == float(w), f"got {r2.plate_px(1)}")


def test_character_height_is_reported_not_just_width() -> None:
    """Plate WIDTH is not the binding number; character height is.

    Every ANPR specification is written in character height, and the two come
    apart at steep viewing angles — a 94px-wide detection 17px tall carries the
    same ~9px characters as a 34px plate seen square-on. Reporting width alone
    is what made a badge look like the readable plate in the clip.
    """
    frame = np.full((720, 1280, 3), 120, np.uint8)
    r = PlateReader()
    r._observe_one(1, frame, (100, 400, 154, 421))       # 54x21, a real plate
    check("best-view width is reported alongside the p90",
          r.plate_px_best(1) == 54.0, f"got {r.plate_px_best(1)}")
    check("character height is derived from HEIGHT, not width",
          abs(r.char_px_best(1) - 21 * 0.55) < 0.1, f"got {r.char_px_best(1)}")
    check("and it lands below the 20px ANPR minimum for this footage",
          r.char_px_best(1) < 20.0, f"got {r.char_px_best(1)}")


def test_fusion_ignores_crops_of_a_different_scale() -> None:
    """Fusing a 94px crop with 33px ones degrades the best view.

    Multi-frame fusion combines samples of the same signal. A crop a third the
    width carries a ninth of the information, and upscaling it to match
    contributes interpolated mush that the median mixes into the result.
    """
    crops = [{"img": np.zeros((17, 94, 3), np.uint8), "w": 94},
             {"img": np.zeros((13, 36, 3), np.uint8), "w": 36},
             {"img": np.zeros((14, 33, 3), np.uint8), "w": 33}]
    check("mismatched-scale crops are not fused", PlateEnhancer._fuse(crops) is None)
    same = [{"img": np.zeros((21, 54, 3), np.uint8), "w": 54},
            {"img": np.zeros((20, 50, 3), np.uint8), "w": 50}]
    check("comparable-scale crops still fuse",
          PlateEnhancer._fuse(same) is not None)


def test_a_lone_read_is_never_certified() -> None:
    """One read cannot corroborate itself.

    Regression, measured on street_egypt.mp4: for a 38px plate all three crop
    reads came back EMPTY and only the fused image produced anything. Being the
    only read, every character position "agreed" with itself, mean agreement was
    1.0 by construction, and the pipeline published a two-character plate at
    **confidence 1.0** — the exact confident-garbage failure the whole plate
    stage is designed around.
    """
    _t, lone, d1 = vote([("٣F", 0.56)])
    _t, pair, d2 = vote([("ABC123", 0.9), ("ABC123", 0.8)])
    check("a single read cannot reach full confidence", lone <= 0.5,
          f"got {lone:.2f}")
    check("corroborated reads still can", pair > 0.9, f"got {pair:.2f}")
    check("corroboration is reported", d1["corroboration"] < d2["corroboration"])


def test_implausible_and_undersized_reads_are_not_published() -> None:
    """Evidence is kept; only fit-to-publish answers reach plate_text.

    Two independent gates, because they catch different lies:
      * a real Egyptian plate carries at least 5 glyphs, so a 2-character
        string is the character detector firing on noise, at ANY resolution
      * below the measured fusion floor, 12-frame fusion reads 0 of 50 plates
        (docs/anpr-plan.md §5c), so whatever comes out is not a read
    """
    class _FakeOCR:
        model = True

        def __init__(self, text):
            self.text = text

        def read(self, img):
            return (self.text, 0.9, [1] * len(self.text))

    def run(text, width):
        e = PlateEnhancer(sr=None, ocr=_FakeOCR(text), keep=3, min_px=12.0)
        for _ in range(3):
            e.offer(1, np.zeros((max(width // 2, 4), width, 3), np.uint8), 0.8)
        return e.resolve(1)

    r = run("٣F", 38)
    check("a 2-character 'plate' is not published", r["plate_text"] == "",
          f"got {r['plate_text']!r}")
    check("but the discarded read is kept for inspection",
          r.get("untrusted_read", {}).get("text") == "٣F", str(r.get("untrusted_read")))

    r = run("٣٤٥AB", 34)          # plausible length, unreadable size
    check("a plausible read below the fusion floor is not published",
          r["plate_text"] == "", f"got {r['plate_text']!r}")
    check("and the reason names the resolution floor",
          "fusion" in r.get("untrusted_read", {}).get("rejected_because", ""),
          str(r.get("untrusted_read")))

    r = run("٣٤٥AB", 120)         # plausible length, adequate size
    check("a plausible read on adequate footage IS published",
          r["plate_text"] == "٣٤٥AB", f"got {r['plate_text']!r}")


def test_carriageway_is_labelled_even_with_the_gate_off() -> None:
    """Turning the ROI gate off must not make every vehicle 'analysed'.

    The harvester deliberately tracks the whole frame — the opposite
    carriageway is extra training data, not noise — and then labels which
    carriageway each vehicle came from. The mask used to be built only when the
    gate was ARMED, so with the gate off on_roadway() answered True for
    everything and the label was meaningless.
    """
    cfg = PipelineConfig()
    cfg.roi_gated_tracking = False
    d = VehicleDetector.__new__(VehicleDetector)
    d.cfg = cfg
    d._roi_gate = False
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [cfg.roi_px(W, H)], 1)
    d._roi_mask = m.astype(bool)
    on = [640.0, 600.0, 740.0, 690.0]            # mid-carriageway, near field
    off = [10.0, 40.0, 60.0, 80.0]               # top-left corner, off the road
    check("a vehicle on the analysed carriageway is labelled so",
          d.on_roadway(on) is True)
    check("a vehicle off it is not", d.on_roadway(off) is False)


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
    test_speed_report_carries_per_vehicle_figures()
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
    test_near_field_views_outweigh_distant_ones()
    test_crop_classifier_overrides_the_size_heuristic()
    test_vans_are_not_guessed()
    test_finetuned_model_is_detected_and_takes_over()
    test_finetuned_model_works_downstream_not_just_in_classify()
    test_ungrouped_classes_never_merge_with_each_other()
    test_confirm_rate_detects_a_broken_stride()
    print("plates")
    test_washed_out_band_is_still_read()
    test_neutral_surfaces_are_not_given_a_colour()
    test_oversized_plate_box_is_rejected()
    test_plate_goes_to_the_tightest_containing_vehicle()
    test_plate_colour_needs_repeated_evidence()
    test_plate_overlay_box_does_not_go_stale()
    test_missing_plate_model_does_not_crash()
    test_sharper_crop_scores_higher()
    test_vote_prefers_agreement_over_count()
    test_tailgate_badges_are_not_accepted_as_plates()
    test_character_height_is_reported_not_just_width()
    test_fusion_ignores_crops_of_a_different_scale()
    test_a_lone_read_is_never_certified()
    test_implausible_and_undersized_reads_are_not_published()
    test_ocr_gate_blocks_hallucinated_reads()
    test_ocr_absence_is_explained()
    test_carriageway_is_labelled_even_with_the_gate_off()
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
