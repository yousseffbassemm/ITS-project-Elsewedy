"""Per-lane model with explicit lane boundaries placed on the painted stripes.

Lanes are NOT assumed equal width — each boundary line is given in the config
(median, painted stripes, barrier), and consecutive boundaries bound one lane, so
the rightmost lane can be a single wide lane. A vehicle is assigned to the lane
its ground footprint (box bottom edge) falls in most — the "straddling → count it
in the lane it covers most" rule.
"""
from __future__ import annotations

import cv2
import numpy as np

# distinct, high-contrast lane colours (BGR)
LANE_COLORS_BGR = [
    (80, 180, 255),   # amber
    (90, 210, 90),    # green
    (230, 160, 60),   # blue
    (150, 100, 240),  # pink/red
    (230, 200, 80),   # cyan
    (120, 120, 240),  # coral
]
LANE_COLORS_HEX = ["#ffb44f", "#2fd07a", "#1f9fe6", "#f0649b", "#ffbe3d", "#ff7a3d"]


class LaneModel:
    def __init__(self, cfg, w: int, h: int):
        self.w, self.h = w, h
        # boundaries as pixel line segments: [(top_pt, bottom_pt), ...]
        self.boundaries = [
            (np.array([tx * w, ty * h]), np.array([bx * w, by * h]))
            for (tx, ty), (bx, by) in cfg.lane_dividers
        ]
        self.n = len(self.boundaries) - 1
        self.polys: list[np.ndarray] = []
        for i in range(self.n):
            lt, lb = self.boundaries[i]
            rt, rb = self.boundaries[i + 1]
            # polygon corners: top-left, top-right, bottom-right, bottom-left
            self.polys.append(np.array([lt, rt, rb, lb], dtype=np.int32))

    def assign(self, xyxy) -> int | None:
        """Lane whose polygon contains most of the box's ground footprint."""
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        votes = np.zeros(self.n, dtype=int)
        for x in np.linspace(x1, x2, 15):
            for i, poly in enumerate(self.polys):
                if cv2.pointPolygonTest(poly, (float(x), float(y2)), False) >= 0:
                    votes[i] += 1
                    break
        return int(np.argmax(votes)) if votes.sum() else None

    def draw(self, frame, counts: dict[int, int]) -> None:
        overlay = frame.copy()
        for i, poly in enumerate(self.polys):
            cv2.fillPoly(overlay, [poly], LANE_COLORS_BGR[i % len(LANE_COLORS_BGR)])
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        for i, poly in enumerate(self.polys):
            col = LANE_COLORS_BGR[i % len(LANE_COLORS_BGR)]
            cv2.polylines(frame, [poly], True, col, 2, cv2.LINE_AA)
            near_mid = poly[2:].mean(axis=0).astype(int)  # midpoint of BR,BL edge
            cx, cy = int(near_mid[0]), int(near_mid[1]) - 14
            txt = f"L{i+1}: {counts.get(i, 0)}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Edge lanes run off-frame, so their natural label anchor can land
            # outside the image — keep every lane's count visible.
            cx = int(np.clip(cx, tw // 2 + 8, self.w - tw // 2 - 8))
            cy = int(np.clip(cy, th + 10, self.h - 8))
            cv2.rectangle(frame, (cx - tw // 2 - 6, cy - th - 6), (cx + tw // 2 + 6, cy + 6),
                          (25, 25, 25), -1)
            cv2.putText(frame, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        col, 2, cv2.LINE_AA)
