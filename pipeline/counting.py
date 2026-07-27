"""Line-crossing vehicle counting with per-class (mentor taxonomy) breakdown.

Implemented directly (not via supervision.LineZone, whose 2-D ``np.cross`` was
removed in NumPy 2.5). We track which side of the counting line each vehicle's
bottom-centre sits on and record ONE crossing event per tracker id when that side
flips (jitter can't double-count). Mentor classes (A/C/D/E/G/V/F) are resolved at
summary time from each track's accumulated size evidence.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from .classify import VehicleClassifier, display_name
from .config import COCO_SCHEME, PipelineConfig


class LineCounter:
    def __init__(self, cfg: PipelineConfig, w: int, h: int, lane_model=None,
                 scheme=COCO_SCHEME):
        # Which class ids are vehicles and which are people depends on the
        # MODEL, not on COCO — see config.ClassScheme.
        self.scheme = scheme
        (sx, sy), (ex, ey) = cfg.line_px(w, h)
        self.a = np.array([sx, sy], dtype=float)
        self.b = np.array([ex, ey], dtype=float)
        # Allow a small overhang past each end of the segment (~30 px), so a
        # vehicle straddling the very edge of the line still counts.
        seg_len = float(np.linalg.norm(self.b - self.a)) or 1.0
        self._span_tol = 30.0 / seg_len
        self.lane_model = lane_model
        self._last_side: dict[int, float] = {}
        self._last_pos: dict[int, np.ndarray] = {}
        self._counted: set[int] = set()
        # crossing events: tracker_id -> direction ("in"/"out")
        self.vehicle_events: dict[int, str] = {}
        self.lane_of: dict[int, int] = {}          # tracker_id -> lane index
        self.pedestrians_in = 0
        self.pedestrians_out = 0
        self._live_in = 0
        self._live_out = 0
        self._live_lane: dict[int, int] = defaultdict(int)

    def _side(self, p) -> float:
        return ((self.b[0] - self.a[0]) * (p[1] - self.a[1])
                - (self.b[1] - self.a[1]) * (p[0] - self.a[0]))

    def _crossed_segment(self, p0, p1, s0: float, s1: float) -> bool:
        """Did the p0->p1 step cross the line *within the segment's extent*?

        Interpolates the exact crossing point and projects it onto the segment.
        The previous test compared the post-crossing POSITION against a +/-40 px
        box around the line, which fails as soon as a vehicle moves more than
        40 px between processed frames — i.e. at frame_stride 2-4, where a
        vehicle covers 74-111 px per step and leaps the band entirely. The
        crossing was then detected and silently discarded. This test depends on
        the geometry, not on how fast the vehicle happens to be sampled.
        """
        denom = s0 - s1
        if denom == 0:
            return False
        t = s0 / denom                      # in [0,1] because the sign flipped
        cross = np.asarray(p0, dtype=float) + t * (np.asarray(p1, dtype=float)
                                                   - np.asarray(p0, dtype=float))
        ab = self.b - self.a
        len2 = float(ab.dot(ab))
        if len2 == 0:
            return False
        u = float((cross - self.a).dot(ab) / len2)   # 0 = start, 1 = end
        return -self._span_tol <= u <= 1.0 + self._span_tol

    def update(self, det: sv.Detections) -> None:
        if det.tracker_id is None or len(det) == 0:
            return
        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for i, tid in enumerate(det.tracker_id):
            tid = int(tid)
            cid = int(det.class_id[i])
            p = anchors[i]
            s = self._side(p)
            prev = self._last_side.get(tid)
            prev_p = self._last_pos.get(tid)
            # A point landing exactly ON the line has side 0 and no sign. Keep
            # the last non-zero side so the crossing is still seen on the next
            # frame instead of being lost — and hold the POSITION back with it.
            # Advancing the position while keeping the old side pairs a side
            # with a point it was not measured at, so the next frame
            # interpolates the crossing from a mismatched pair and can place it
            # past the segment's end, silently rejecting a real crossing.
            if s != 0:
                self._last_side[tid] = s
                self._last_pos[tid] = p
            if prev is None or prev_p is None or s == 0 or prev == 0:
                continue
            if (prev < 0) == (s < 0):
                continue                      # same side, no crossing
            if tid in self._counted or not self._crossed_segment(prev_p, p, prev, s):
                continue                      # already counted / crossed off the end
            self._counted.add(tid)
            direction = "in" if (prev < 0 and s > 0) else "out"
            if self.scheme.is_person(cid):
                if direction == "in":
                    self.pedestrians_in += 1
                else:
                    self.pedestrians_out += 1
            elif self.scheme.is_vehicle(cid):
                self.vehicle_events[tid] = direction
                if direction == "in":
                    self._live_in += 1
                else:
                    self._live_out += 1
                if self.lane_model is not None:
                    lane = self.lane_model.assign(det.xyxy[i])
                    if lane is not None:
                        self.lane_of[tid] = lane
                        self._live_lane[lane] += 1

    def line_px(self):
        return (int(self.a[0]), int(self.a[1])), (int(self.b[0]), int(self.b[1]))

    def total_vehicles(self) -> int:
        return len(self.vehicle_events)

    def live_counts(self) -> tuple[int, int]:
        return self._live_in, self._live_out

    def live_lane_counts(self) -> dict[int, int]:
        return dict(self._live_lane)

    def summary(self, classifier: VehicleClassifier) -> dict:
        by_code: dict[str, int] = defaultdict(int)
        dir_by_code: dict[str, dict[str, int]] = {"in": defaultdict(int), "out": defaultdict(int)}
        by_dir = {"in": 0, "out": 0}
        for tid, direction in self.vehicle_events.items():
            code = classifier.resolve(tid)
            by_code[code] += 1
            dir_by_code[direction][code] += 1
            by_dir[direction] += 1
        name = lambda code: f"{code} · {display_name(code)}"
        return {
            "total_vehicles": self.total_vehicles(),
            "by_class": {name(c): by_code[c] for c in sorted(by_code)},
            "by_class_code": {c: by_code[c] for c in sorted(by_code)},
            "by_direction": by_dir,
            "by_class_direction": {
                "in": {name(c): n for c, n in dir_by_code["in"].items()},
                "out": {name(c): n for c, n in dir_by_code["out"].items()},
            },
            "pedestrians": {"in": self.pedestrians_in, "out": self.pedestrians_out},
        }
