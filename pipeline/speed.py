"""Per-vehicle speed estimation via perspective (bird's-eye) transform.

We map the road plane in the image to a metric rectangle with a homography, track
each vehicle's bottom-centre point in metres, and derive speed from how far it
moves per unit time.

Robustness measures (monocular speed is noise-prone):
  * Speed is only measured in the RELIABLE zone (near/mid field). Near the far
    edge of the calibration quad a few pixels map to many metres, so those
    samples are dropped rather than trusted.
  * A vehicle needs enough tracked history before any speed is emitted.
  * Samples above a highway-realistic ceiling are rejected as tracking jitter.
  * Reported figures aggregate PER VEHICLE (median speed per tracker id), so a
    slow truck lingering many frames can't dominate, and outliers are damped.

Absolute accuracy still depends on the calibration in PipelineConfig
(source_points + target_size_m).
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

from .config import COCO_SCHEME, PipelineConfig

# Only trust speed while the vehicle's ground point is in this image-y band
# (fractions of frame height). The lower bound excludes the far field, where a
# few pixels map to many metres. The upper bound stops just short of the frame
# edge, where a vehicle's box gets clipped and its ground point jumps.
#
# The band used to stop at 0.58 because the old calibration quad ended at 0.62 —
# beyond it, speeds were extrapolated outside the homography. The measured quad
# now spans 0.25..0.97, so the well-resolved near field is finally usable, which
# is also where monocular speed is most trustworthy.
RELIABLE_Y_MIN = 0.32
RELIABLE_Y_MAX = 0.90
# Speed accuracy depends on the TIME baseline, not the number of samples, so the
# gate is a minimum span plus a couple of points. Requiring a fixed 6 positions
# was stride-dependent and silently broke at coarse strides: the position deque
# holds ~fps/stride entries, which is 3 at stride 8, so `len >= 6` could never be
# satisfied and NO speed was ever emitted. At stride 1 these two rules are
# equivalent (0.25 s ~ 7 frames at 25 fps).
MIN_POINTS = 3            # minimum positions before emitting a speed
MIN_SPAN_SEC = 0.25       # ...and they must span at least this much time
SPEED_CEILING_KMH = 160.0  # reject samples above this as jitter
# If a track is unseen for longer than this, the two positions either side of the
# gap are too far apart to treat as one continuous motion sample — start fresh.
MAX_SAMPLE_GAP_FRAMES = 15


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(reshaped, self.m)
        return out.reshape(-1, 2)


class SpeedEstimator:
    def __init__(self, cfg: PipelineConfig, w: int, h: int, fps: float,
                 frame_stride: int, scheme=COCO_SCHEME):
        self.cfg = cfg
        # Which class ids count as vehicles depends on the model — see
        # config.ClassScheme.
        self.scheme = scheme
        self.h = h
        self.transformer = ViewTransformer(cfg.source_px(w, h), cfg.target_px())
        self.fps = fps
        # ~1 s of positions, but never fewer than the gate needs (see MIN_POINTS).
        self.window = max(int(round(fps / frame_stride)), MIN_POINTS + 3)
        # Consecutive samples are `frame_stride` frames apart by design, so the
        # "track was lost" threshold has to scale with the stride or a high
        # stride would discard every sample as if it were a gap.
        self.max_gap = max(MAX_SAMPLE_GAP_FRAMES, 3 * frame_stride)
        # each entry is (frame_idx, metric_point) — see update()
        self._pos: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.current: dict[int, float] = {}          # latest speed per id (HUD)
        self._veh_samples: dict[int, list[float]] = defaultdict(list)  # per id

    def update(self, det: sv.Detections, frame_idx: int) -> dict[int, float]:
        self.current = {}
        if det.tracker_id is None or len(det) == 0:
            return self.current
        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        metric = self.transformer.transform_points(anchors)
        for i, tid in enumerate(det.tracker_id):
            cid = int(det.class_id[i])
            if not self.scheme.is_vehicle(cid):
                continue
            y_frac = anchors[i][1] / self.h
            if not (RELIABLE_Y_MIN <= y_frac <= RELIABLE_Y_MAX):
                continue  # outside the trustworthy band
            tid = int(tid)
            hist = self._pos[tid]
            # A track can vanish for a few frames (occlusion, a missed detection)
            # or leave and re-enter the reliable band. Positions are therefore NOT
            # guaranteed one stride apart, so each sample carries its frame index
            # and elapsed time is measured from the real index delta. Assuming
            # consecutive frames here undercounts elapsed time and inflates speed.
            if hist and (frame_idx - hist[-1][0]) > self.max_gap:
                hist.clear()  # gap too long to interpolate across — restart
            hist.append((frame_idx, metric[i]))
            if len(hist) >= MIN_POINTS:
                f0, p0 = hist[0]
                f1, p1 = hist[-1]
                elapsed = (f1 - f0) / max(self.fps, 1e-6)
                if elapsed >= MIN_SPAN_SEC:
                    speed_kmh = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) / elapsed * 3.6
                    if speed_kmh <= SPEED_CEILING_KMH:
                        self._veh_samples[tid].append(speed_kmh)
                        # Display a stable running-median speed, not the noisy
                        # instantaneous value (keeps on-screen labels realistic).
                        self.current[tid] = float(
                            np.median(self._veh_samples[tid][-self.window:])
                        )
        return self.current

    def frame_avg(self) -> float:
        return float(np.median(list(self.current.values()))) if self.current else 0.0

    def per_vehicle(self) -> dict[int, float]:
        """Public: robust speed per vehicle (km/h)."""
        return self._per_vehicle()

    def _per_vehicle(self) -> dict[int, float]:
        """One robust speed per vehicle = median of its in-zone samples."""
        return {
            tid: float(np.median(s))
            for tid, s in self._veh_samples.items()
            if len(s) >= 3  # ignore vehicles with too few samples to trust
        }

    def summary(self, classifier=None) -> dict:
        veh = self._per_vehicle()
        speeds = np.array(list(veh.values()), dtype=float)
        if speeds.size == 0:
            return {
                "calibrated": self.cfg.calibrated, "avg_kmh": 0.0, "median_kmh": 0.0,
                "max_kmh": 0.0, "p85_kmh": 0.0, "n_vehicles_timed": 0,
                "histogram": {"bins": [], "counts": []}, "by_class_avg_kmh": {},
                "per_vehicle": {},
            }
        # Bins must reach the fastest vehicle, or fast vehicles are counted in
        # n_vehicles_timed but silently dropped from the chart and the counts no
        # longer sum to the total. Stop AT the fastest vehicle rather than at the
        # 160 km/h ceiling, so the chart isn't padded with empty bins.
        # np.histogram includes the right edge in the final bin, so the maximum
        # speed itself always lands inside a bin.
        top = max(int(np.ceil(float(speeds.max()) / 10.0) * 10), 10)
        bin_edges = list(range(0, top + 1, 10))
        counts, _ = np.histogram(speeds, bins=bin_edges)
        from .classify import display_name
        by_class: dict[str, list[float]] = defaultdict(list)
        for tid, sp in veh.items():
            code = classifier.resolve(tid) if classifier else "F"
            by_class[f"{code} · {display_name(code)}"].append(sp)
        return {
            "calibrated": self.cfg.calibrated,
            "avg_kmh": round(float(speeds.mean()), 1),
            "median_kmh": round(float(np.median(speeds)), 1),
            "max_kmh": round(float(speeds.max()), 1),
            "p85_kmh": round(float(np.percentile(speeds, 85)), 1),
            "n_vehicles_timed": int(speeds.size),
            "histogram": {
                "bins": [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)],
                "counts": [int(c) for c in counts],
            },
            "by_class_avg_kmh": {
                k: round(float(np.mean(v)), 1) for k, v in by_class.items()
            },
            # Per-vehicle speeds, keyed by track id. Published because the
            # aggregates above cannot be joined back to a specific vehicle, and
            # the plate export needs exactly that — it reports one row per
            # vehicle and was silently emitting a blank speed column without it.
            "per_vehicle": {str(tid): round(sp, 1) for tid, sp in veh.items()},
        }
