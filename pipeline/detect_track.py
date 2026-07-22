"""Detection + tracking: YOLO with built-in ByteTrack, wrapped for supervision.

ByteTrack is used (not DeepSORT) because there is no GPU here — ByteTrack has no
deep re-ID network and stays fast on CPU.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .classify import native_code_map
from .config import PERSON_CLASS, VEHICLE_CLASSES, PipelineConfig
from .reid import IdStabilizer

# Scene-tuned ByteTrack settings shipped with the project (longer track_buffer
# than the Ultralytics default). See the file for the reasoning.
TRACKER_CFG = str(Path(__file__).resolve().parent / "bytetrack.yaml")


def _one_box_per_id(det: sv.Detections) -> np.ndarray:
    """Boolean mask keeping the best detection per tracker id.

    When the re-id layer collapses two duplicate boxes onto one vehicle, BOTH
    rows survive carrying the same id. Leaving them in would let one vehicle
    append two positions per frame (wrecking the speed window), vote twice for
    its class, and be tested against the counting line twice. Keep the most
    confident box for each id.
    """
    ids = det.tracker_id
    mask = np.zeros(len(ids), dtype=bool)
    if det.confidence is None:
        _, first = np.unique(ids, return_index=True)
        mask[first] = True
        return mask
    order = np.argsort(-det.confidence)          # highest confidence first
    seen: set[int] = set()
    for i in order:
        t = int(ids[i])
        if t not in seen:
            seen.add(t)
            mask[i] = True
    return mask


def visible_mask(det: sv.Detections, iou_thr: float = 0.60,
                 ground_frac: float = 0.35) -> np.ndarray:
    """Mask hiding a box that visually duplicates another in the SAME frame.

    The re-id layer needs a couple of frames of evidence before it will merge two
    ids, which is correct for identity but leaves a 1-2 frame window where one
    vehicle wears two boxes with two labels. That is invisible at playback speed
    but obvious when stepping through frames. This is a draw-time filter only —
    counting, speed and class votes still see every detection, so nothing is lost
    from the analysis, and the merge still decides identity on real evidence.
    """
    n = len(det)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    conf = det.confidence if det.confidence is not None else np.ones(n)
    order = np.argsort(-np.asarray(conf))       # prefer the confident box
    for ii in range(n):
        i = order[ii]
        if not keep[i]:
            continue
        for jj in range(ii + 1, n):
            j = order[jj]
            if not keep[j]:
                continue
            a, b = det.xyxy[i], det.xyxy[j]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
            if union <= 0 or inter / union < iou_thr:
                continue
            mean_h = ((a[3] - a[1]) + (b[3] - b[1])) / 2.0
            gd = np.hypot((a[0]+a[2])/2 - (b[0]+b[2])/2, a[3] - b[3])
            if gd <= ground_frac * max(mean_h, 1.0):
                keep[j] = False                 # same vehicle, drawn twice
    return keep


class VehicleDetector:
    def __init__(self, cfg: PipelineConfig, fps: float = 25.0,
                 frame_size: tuple[int, int] | None = None):
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        # A model fine-tuned on the mentor taxonomy has its OWN class ids, so the
        # COCO id filter below would silently drop every detection. Detect that
        # case and take the model's classes as-is.
        self.native_codes = native_code_map(getattr(self.model, "names", None))
        if self.native_codes is not None:
            self._classes = None                  # keep everything the model emits
            self.vehicle_ids = {i for i, c in self.native_codes.items() if c != "F"}
        else:
            # Classes we ask YOLO to return: vehicles + pedestrians.
            self._classes = sorted(set(VEHICLE_CLASSES) | {PERSON_CLASS})
            self.vehicle_ids = set(VEHICLE_CLASSES)
        # ByteTrack does the frame-to-frame association; this re-attaches a
        # vehicle that was lost for longer than track_buffer to its original id.
        self.stabilizer = (IdStabilizer(fps, max(cfg.frame_stride, 1))
                           if cfg.stable_ids else None)
        # Unbiased tracking-health counters. Measured on RAW detections, before
        # any tracker filtering, so a stride at which association is failing
        # cannot hide the failure by producing fewer tracks to inspect.
        self.raw_detections = 0
        self.confirmed_detections = 0
        # Detections surviving the ROI gate. The ROI is a MEASURED polygon for one
        # specific camera, so on any other footage it can sit off the road and
        # silently discard most of the traffic.
        self.roi_kept = 0
        self._roi_mask = None
        if cfg.roi_gated_tracking and frame_size is not None:
            w, h = frame_size
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [cfg.roi_px(w, h)], 1)
            k = max(int(cfg.roi_margin_px), 1)
            self._roi_mask = cv2.dilate(
                m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            ).astype(bool)

    def _on_roadway(self, det: sv.Detections) -> np.ndarray:
        """Mask of detections whose ground point is on the analysed carriageway."""
        if self._roi_mask is None:
            return np.ones(len(det), dtype=bool)
        h, w = self._roi_mask.shape
        gx = np.clip(((det.xyxy[:, 0] + det.xyxy[:, 2]) / 2).astype(int), 0, w - 1)
        gy = np.clip(det.xyxy[:, 3].astype(int), 0, h - 1)
        return self._roi_mask[gy, gx]

    def track(self, frame: np.ndarray, frame_idx: int = 0) -> sv.Detections:
        """Run detection + tracking on one frame, return supervision Detections.

        Detections carry .tracker_id, .class_id, .confidence and .xyxy. When
        ``cfg.stable_ids`` is on, .tracker_id holds CANONICAL ids that survive
        occlusion gaps rather than raw ByteTrack ids.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            **({} if self._classes is None else {"classes": self._classes}),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            verbose=False,
        )[0]
        det = sv.Detections.from_ultralytics(results)
        self.raw_detections += len(det)
        # Keep only tracker-confirmed detections (needed for counting & speed).
        if det.tracker_id is None:
            return det[np.zeros(len(det), dtype=bool)]
        self.confirmed_detections += len(det)
        # ByteTrack still sees the whole frame (it needs unbroken motion history),
        # but everything downstream — ids, counting, annotation — is restricted to
        # the analysed carriageway.
        if len(det):
            det = det[self._on_roadway(det)]
        self.roi_kept += len(det)
        if self.stabilizer is not None and len(det):
            det.tracker_id = self.stabilizer.assign(
                frame, det.xyxy, det.tracker_id, det.class_id, frame_idx
            )
            det = det[_one_box_per_id(det)]
        return det

    @property
    def stitches(self) -> int:
        """Number of times a re-appearing vehicle was re-attached to its id."""
        return self.stabilizer.stitches if self.stabilizer else 0

    @property
    def merges(self) -> int:
        """Number of duplicate detections collapsed onto one vehicle."""
        return self.stabilizer.merges if self.stabilizer else 0

    @property
    def confirm_rate(self) -> float:
        """Fraction of detections the tracker managed to give an id.

        This is the honest way to detect a frame_stride that is too coarse.
        ByteTrack has no notion of stride — it treats consecutive calls as
        consecutive frames, so its motion model is wrong by exactly the stride
        factor and association starts failing. Measured on this clip:

            stride 1  100.0% confirmed -> 18 vehicles counted
            stride 3   99.9%           -> 17
            stride 6   84.0%           ->  5

        Counting needs an UNBROKEN id either side of the line, so a modest drop
        in confirmation becomes a large drop in counts. Anything below ~95% means
        the run is under-counting silently.
        """
        return (self.confirmed_detections / self.raw_detections
                if self.raw_detections else 1.0)

    @property
    def roi_pass_rate(self) -> float:
        """Fraction of tracked detections that fell on the analysed carriageway.

        A low rate means the ROI polygon in the config does not match this
        video. Every scene-specific value (roi, counting line, lane dividers,
        speed homography) was measured for samples/street_egypt.mp4, and the web
        app applies them unchanged to whatever is uploaded — so this is the
        normal case for any other clip, not an exotic failure.
        """
        return (self.roi_kept / self.confirmed_detections
                if self.confirmed_detections else 1.0)

    def display_id(self, tracker_id: int) -> int | None:
        """Sequential on-screen number for a vehicle (see IdStabilizer)."""
        if self.stabilizer is None:
            return int(tracker_id)
        return self.stabilizer.display_id(tracker_id)
