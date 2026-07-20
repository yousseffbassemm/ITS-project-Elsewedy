"""Congestion analysis from vehicle density, road occupancy and average speed.

Per processed frame we measure how much of the region-of-interest (the roadway)
is covered by vehicle boxes (occupancy) and how many vehicles are present, then
combine that with the frame's average speed to assign a congestion level.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from .config import CONGESTION_LEVELS, VEHICLE_CLASSES, PipelineConfig

# Highest congestion level permitted for a given vehicle count in the ROI,
# indexed by count: 0 or 1 vehicle can never be worse than Free-flow, 2 at most
# Moderate, 3-4 at most Heavy, 5+ unrestricted.
MAX_LEVEL_FOR_COUNT = [0, 0, 1, 2, 2, 3]
# A level touched for a fraction of a second is a blip, not a traffic condition.
MIN_PERIOD_SEC = 1.0
MIN_LEVEL_SHARE_PCT = 1.0


class CongestionMonitor:
    def __init__(self, cfg: PipelineConfig, w: int, h: int):
        self.cfg = cfg
        self.roi = cfg.roi_px(w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi], 1)
        self.roi_mask = mask.astype(bool)
        self.roi_area = float(self.roi_mask.sum()) or float(w * h)
        self.w, self.h = w, h
        self.vehicles_in_roi = 0
        # Occupancy only ever needs pixels inside the ROI, so all per-frame work
        # happens in its bounding box rather than over the whole (4K) frame.
        rx, ry, rw, rh = cv2.boundingRect(self.roi)
        self._bx0, self._by0 = max(rx, 0), max(ry, 0)
        self._bx1, self._by1 = min(rx + rw, w), min(ry + rh, h)
        self._roi_crop = self.roi_mask[self._by0:self._by1, self._bx0:self._bx1]
        self._cover = np.zeros(self._roi_crop.shape, dtype=np.uint8)
        self.samples: list[dict] = []  # per-frame record

    def _occupancy(self, det: sv.Detections) -> tuple[float, int]:
        """Fraction of ROI covered by vehicle boxes, and vehicle count in ROI."""
        if det.class_id is None:
            return 0.0, 0
        cover = self._cover
        cover.fill(0)
        n = 0
        for i in range(len(det)):
            if int(det.class_id[i]) not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = det.xyxy[i].astype(int)
            cx, cy = (x1 + x2) // 2, y2  # bottom-centre
            if 0 <= cy < self.h and 0 <= cx < self.w and self.roi_mask[cy, cx]:
                n += 1
            # draw in ROI-bounding-box coordinates; cv2 clips to the canvas
            cv2.rectangle(cover, (x1 - self._bx0, y1 - self._by0),
                          (x2 - self._bx0, y2 - self._by0), 1, -1)
        covered = float(np.count_nonzero(np.logical_and(cover.astype(bool), self._roi_crop)))
        return covered / self.roi_area, n

    def classify(self, occ: float, avg_speed: float, n: int = 99) -> str:
        c = self.cfg
        if occ >= c.occ_jam:
            level = "Jam"
        elif occ >= c.occ_heavy:
            level = "Heavy"
        elif occ >= c.occ_moderate:
            level = "Moderate"
        else:
            level = "Free-flow"
        # Low average speed (with at least some traffic) bumps severity by one.
        if avg_speed and avg_speed < c.jam_speed_kmh and occ >= c.occ_moderate:
            idx = min(CONGESTION_LEVELS.index(level) + 1, len(CONGESTION_LEVELS) - 1)
            level = CONGESTION_LEVELS[idx]
        # Occupancy is an IMAGE-space area fraction, so one lorry passing close to
        # the camera can cover a third of the visible ROI on its own. Congestion
        # is a property of traffic, not of one large nearby vehicle: cap the level
        # by how many vehicles are actually on the road.
        cap = MAX_LEVEL_FOR_COUNT[min(n, len(MAX_LEVEL_FOR_COUNT) - 1)]
        return CONGESTION_LEVELS[min(CONGESTION_LEVELS.index(level), cap)]

    def update(self, det: sv.Detections, t_sec: float, avg_speed: float) -> str:
        occ, n = self._occupancy(det)
        self.vehicles_in_roi = n   # vehicles on THIS carriageway, this frame
        level = self.classify(occ, avg_speed, n)
        self.samples.append(
            {"t": t_sec, "occ": occ, "n": n, "avg_speed": avg_speed, "level": level}
        )
        return level

    def summary(self) -> dict:
        if not self.samples:
            return {"levels_pct": {}, "worst_level": "Free-flow", "peak_periods": [],
                    "avg_occupancy": 0.0}
        levels = [s["level"] for s in self.samples]
        total = len(levels)
        levels_pct = {
            lv: round(100.0 * levels.count(lv) / total, 1) for lv in CONGESTION_LEVELS
        }
        # The headline level must be one the road actually SUSTAINED. Reporting
        # the worst level ever touched made a single frame (0.4% of a clip that
        # was Free-flow 96.8% of the time) headline the dashboard as "Heavy".
        sustained = [lv for lv in CONGESTION_LEVELS
                     if levels_pct.get(lv, 0.0) >= MIN_LEVEL_SHARE_PCT]
        worst = sustained[-1] if sustained else "Free-flow"
        peak_instant = max(CONGESTION_LEVELS,
                           key=lambda lv: (levels.count(lv) > 0,
                                           CONGESTION_LEVELS.index(lv)))
        # Peak periods = contiguous runs of Heavy/Jam, ignoring blips.
        peaks, start, start_level = [], None, None
        for s in self.samples:
            heavy = s["level"] in ("Heavy", "Jam")
            if heavy and start is None:
                start, start_level = s["t"], s["level"]
            elif not heavy and start is not None:
                peaks.append({"start_sec": round(start, 1), "end_sec": round(s["t"], 1),
                              "level": start_level})
                start = None
        if start is not None:
            peaks.append({"start_sec": round(start, 1),
                          "end_sec": round(self.samples[-1]["t"], 1), "level": start_level})
        peaks = [p for p in peaks
                 if p["end_sec"] - p["start_sec"] >= MIN_PERIOD_SEC]
        return {
            "levels_pct": levels_pct,
            "worst_level": worst,
            "peak_instant_level": peak_instant,
            "peak_periods": peaks[:20],
            "avg_occupancy": round(float(np.mean([s["occ"] for s in self.samples])), 3),
        }
