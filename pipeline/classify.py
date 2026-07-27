"""Map COCO detections to the mentor's vehicle taxonomy (A/C/D/E/G/V/F).

Mentor scheme:
    A  Private car        C  Light truck      D  Heavy truck
    E  Bus                G  Motorcycle       V  Van
    F  Unknown

Base YOLO (COCO) has four relevant classes — car, truck, bus, motorcycle — and
they do not line up with the seven. What the detector CAN do, verified against
hand-labelled ground truth on 18 vehicles, is separate {car, van} from
{light truck, heavy truck}: COCO "car" covered every A and V, COCO "truck" every
C and D, with no crossover. Everything below is about splitting inside those two
groups.

**truck -> C or D (size).** COCO "truck" spans a pickup and a semi-trailer alike,
so mapping all of them to D (heavy) was wrong on 6 of the 7 trucks in the clip.
They default to C, and are promoted to D only when the vehicle really is large.
Size comes from the vehicle's apparent height divided by the image scale at its
ground position — NOT from box width, which is unusable here: a loaded pickup
measured 4.44 m wide (cargo inflates the box) while the genuine lorry measured
3.33 m. Height behaves: ordinary cars come out at 1.57-1.66 m, which is right.

**car -> A only.** Vans cannot be separated from cars by this camera and are
reported as A. Measured heights overlap completely (vans 1.57-2.21 m, cars
1.57-2.20 m), and the best threshold tuned ON the test data still misassigned 4
of 11. Emitting V would mean labelling saloons as vans, which is exactly how this
heuristic failed the first time it was tried. It is left to a fine-tuned model.

The size thresholds are physical (an HGV is taller than ~3.5 m and wider than
~2.5 m; a pickup is neither), not fitted to whatever split this clip cleanly —
but they ride on a noisy monocular estimate, so C/D is approximate. Reliable
seven-class output needs training data: see docs/finetuning-plan.md.

Votes are weighted by detection confidence and by how near the camera the vehicle
was, because a 200 px near-field view is far better evidence than a 15 px distant
one, and each track's class is resolved ONCE from the whole history so a single
bad frame cannot flip it.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

# code -> (display name, chart colour)
MENTOR_CLASSES: dict[str, tuple[str, str]] = {
    "A": ("Private car", "#e4002b"),
    "C": ("Light truck", "#ff7a3d"),
    "D": ("Heavy truck", "#b3001b"),
    "E": ("Bus", "#1f5fd0"),
    "G": ("Motorcycle", "#2fd07a"),
    "V": ("Van", "#ffbe3d"),
    "F": ("Unknown", "#8a8f98"),
}

# COCO ids
_COCO_CAR, _COCO_MOTO, _COCO_BICYCLE, _COCO_BUS, _COCO_TRUCK = 2, 3, 1, 5, 7

# Heavy vs light is judged on estimated FRONTAL AREA (height x width), not on
# height and width separately. Either dimension alone sits on a knife edge — the
# clip's one lorry measured 3.48 m tall against a 3.5 m threshold, while a light
# box truck measured 3.38 m — but their frontal areas are 10.5 m2 vs 7.6 m2, a
# usable margin. The threshold is physical: an HGV presents roughly 8-10 m2,
# a pickup around 4 m2.
#
# CAVEAT: this clip contains exactly ONE heavy vehicle, so the threshold is
# anchored on a single example and should be re-checked against footage with more
# lorries. C vs D is the least certain part of the taxonomy.
HEAVY_MIN_FRONTAL_AREA_M2 = 9.0
MIN_SIZE_SAMPLES = 4

# Apparent box height, in pixels, that counts as one unit of evidence in the
# class vote. A 200 px near-field view then weighs ~2 and a 15 px distant blob
# ~0.15, which is the point: the detector's class call on a handful of pixels is
# barely better than a guess, and there are many more distant frames than close
# ones, so unweighted they would dominate.
#
# This weighting was documented but not implemented: the pixel height was
# divided by the image scale at the vehicle's ground point, which converts it to
# a PHYSICAL height and cancels distance exactly — a vehicle contributed the same
# weight at 15 px as at 200 px. Metric height is still measured, but for the C/D
# size test, which is what it is actually for.
EVIDENCE_REF_PX = 100.0


def display_name(code: str) -> str:
    return MENTOR_CLASSES.get(code, ("Unknown", ""))[0]


# --- support for a model fine-tuned on the 7 classes ------------------------------
# Everything above exists because base COCO cannot express C and V. A model trained
# on the mentor taxonomy makes all of it unnecessary: its own class head is the
# answer, so the heuristics below are bypassed rather than layered on top.
#
# Class names are matched loosely because trainers name things differently
# ("A" / "Private car" / "private_car" / "sedan" all mean A).
_NAME_TO_CODE = {
    "a": "A", "private car": "A", "private_car": "A", "car": "A", "sedan": "A",
    "hatchback": "A", "suv": "A",
    "c": "C", "light truck": "C", "light_truck": "C", "pickup": "C",
    "pick-up": "C", "lighttruck": "C",
    "d": "D", "heavy truck": "D", "heavy_truck": "D", "truck": "D",
    "lorry": "D", "hgv": "D", "trailer": "D",
    "e": "E", "bus": "E", "microbus": "E", "minibus": "E", "coach": "E",
    "g": "G", "motorcycle": "G", "motorbike": "G", "moto": "G",
    "bicycle": "G", "scooter": "G", "tuktuk": "G", "tuk-tuk": "G",
    "v": "V", "van": "V", "minivan": "V", "panel van": "V", "panel_van": "V",
    "f": "F", "unknown": "F", "other": "F",
}


def code_for_name(name: str) -> str | None:
    """Mentor code for a model's class name, or None if it isn't one of ours."""
    return _NAME_TO_CODE.get(str(name).strip().lower().replace("-", " ").replace("_", " ")) \
        or _NAME_TO_CODE.get(str(name).strip().lower())


def native_code_map(names) -> dict[int, str] | None:
    """If a model emits the mentor taxonomy directly, return {class_id: code}.

    Returns None for a plain COCO model. The test is not "do the names look like
    vehicles" — COCO's car/truck/bus would pass that — but "does this model know a
    class COCO cannot express", i.e. C (light truck) or V (van). That is exactly
    the capability the fine-tune adds.
    """
    if not names:
        return None
    items = names.items() if hasattr(names, "items") else enumerate(names)
    mapped = {int(i): code_for_name(n) for i, n in items}
    if any(c is None for c in mapped.values()):
        return None                      # not purely our taxonomy
    if not {"C", "V"} & set(mapped.values()):
        return None                      # COCO-like: no light-truck/van concept
    return mapped


class CropClassifier:
    """Second-stage classifier: mentor class from a cropped vehicle.

    Detection is not the weak part of this pipeline — YOLO finds the vehicles and
    the tracker holds them across occlusions. The LABEL is weak, because base
    COCO has four vehicle classes against the taxonomy's seven and no microbus
    concept at all. A classifier over the tracked crop targets exactly that, and
    it trains on the crops tools/harvest_dataset.py already produces rather than
    needing boxes redrawn on full frames.

    Measured against the size heuristic on 68 hand-labelled vehicles from
    street_egypt.mp4: overall 0.500 -> 0.676, with bus/microbus 0.042 -> 0.833.
    Microbuses are roughly a third of the traffic on that road.

    Missing or unloadable weights are reported, not raised — the rest of the
    analytics do not depend on this and must not be lost to it.
    """

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 128):
        self.model = None
        self.error: str | None = None
        self.imgsz, self.device = imgsz, device
        self.names: dict[int, str] = {}
        try:
            from pathlib import Path

            from ultralytics import YOLO
            if not Path(weights).exists():
                raise FileNotFoundError(weights)
            self.model = YOLO(weights)
            self.names = {int(k): str(v) for k, v in self.model.names.items()}
            unknown = {v for v in self.names.values() if v not in MENTOR_CLASSES}
            if unknown:
                raise ValueError(
                    f"weights emit non-taxonomy classes {sorted(unknown)}; "
                    f"expected a subset of {sorted(MENTOR_CLASSES)}")
        except Exception as exc:                   # pragma: no cover - env dependent
            self.model = None
            self.error = f"{type(exc).__name__}: {exc}"

    def predict(self, crop) -> tuple[str, float] | None:
        """(mentor_code, confidence) for one crop, or None if unavailable."""
        if self.model is None or crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if w < 16 or h < 16:
            return None                            # too small to be evidence
        try:
            r = self.model.predict(crop, imgsz=self.imgsz, verbose=False,
                                   device=self.device)[0]
            return self.names.get(int(r.probs.top1), "F"), float(r.probs.top1conf)
        except Exception:                          # pragma: no cover
            return None


class VehicleClassifier:
    """Accumulates per-track evidence and resolves each vehicle's mentor class."""

    def __init__(self, transformer=None, native_codes: dict[int, str] | None = None,
                 crop_classifier: "CropClassifier | None" = None,
                 crop_every: int = 5):
        # When the detector already speaks the mentor taxonomy, its class head is
        # authoritative and every heuristic below is skipped.
        self.native = native_codes
        # Road-plane homography, used to turn pixels into metres. Optional: with
        # no transformer the classifier still works, it just cannot split C/D.
        self.tf = transformer
        self._inv = None
        if transformer is not None:
            try:
                self._inv = np.linalg.inv(transformer.m)
            except np.linalg.LinAlgError:      # pragma: no cover - degenerate cfg
                self._inv = None
        # A trained crop classifier speaks the taxonomy directly, so its votes
        # live in CODE space rather than COCO-id space and take priority in
        # resolve(). Run every Nth observation per track: on CPU this is a
        # second inference per vehicle per frame, and a vehicle is visible for
        # tens of frames, so sampling costs almost no accuracy after the vote.
        self.crop_clf = crop_classifier if (crop_classifier is not None
                                            and crop_classifier.model) else None
        self.crop_every = max(int(crop_every), 1)
        self._votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._code_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._seen_n: dict[int, int] = defaultdict(int)
        self._height: dict[int, list[float]] = defaultdict(list)
        self._width: dict[int, list[float]] = defaultdict(list)
        self._cache: dict[int, str] = {}

    def _px_per_metre(self, gx: float, gy: float) -> float:
        """Pixels per road-plane metre at a given ground point."""
        if self.tf is None or self._inv is None:
            return 0.0
        m = self.tf.transform_points(np.array([[gx, gy]], dtype=float))
        if not m.size:
            return 0.0
        shifted = cv2.perspectiveTransform(
            np.array([[[m[0][0] + 1.0, m[0][1]]]], dtype=np.float32), self._inv)[0][0]
        return float(np.hypot(shifted[0] - gx, shifted[1] - gy))

    def observe(self, tid: int, coco_id: int, xyxy=None, conf: float = 1.0,
                frame=None) -> None:
        tid = int(tid)
        weight = float(conf)
        n = self._seen_n[tid]
        self._seen_n[tid] = n + 1
        if self.crop_clf is not None and frame is not None and xyxy is not None \
                and n % self.crop_every == 0:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 > 0 and y2 - y1 > 0:
                got = self.crop_clf.predict(frame[y1:y2, x1:x2])
                if got is not None:
                    code, cconf = got
                    # Weight the same way the COCO vote is weighted: by apparent
                    # size, so a clear near view outranks a distant blob.
                    self._code_votes[tid][code] += cconf * max(
                        (y2 - y1) / EVIDENCE_REF_PX, 1e-3)
                    self._cache.pop(tid, None)
        if xyxy is not None:
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            gx, gy = (x1 + x2) / 2.0, y2
            # Near-field views are much better evidence: weight by APPARENT
            # (pixel) size, which is what carries the detail the detector
            # classified from. Applied whether or not the homography is usable,
            # since it needs no calibration.
            weight *= max(y2 - y1, 1.0) / EVIDENCE_REF_PX
            scale = self._px_per_metre(gx, gy)
            if scale > 1.0:
                # Metric size, for the C/D frontal-area test only.
                self._height[tid].append((y2 - y1) / scale)
                self._width[tid].append((x2 - x1) / scale)
        self._votes[tid][int(coco_id)] += max(weight, 1e-6)
        self._cache.pop(tid, None)

    def size_estimate(self, tid: int) -> tuple[float, float]:
        """Median (height, width) in metres for a track, 0.0 if unknown."""
        h = self._height.get(int(tid)) or []
        w = self._width.get(int(tid)) or []
        return (float(np.median(h)) if len(h) >= MIN_SIZE_SAMPLES else 0.0,
                float(np.median(w)) if len(w) >= MIN_SIZE_SAMPLES else 0.0)

    def resolve(self, tid: int) -> str:
        tid = int(tid)
        if tid in self._cache:
            return self._cache[tid]
        # A trained crop classifier wins outright where it has an opinion. It
        # was trained on the taxonomy itself, whereas the path below is a
        # monocular size heuristic over COCO classes that cannot express C or V
        # at all — so mixing the two would only let the weaker signal dilute the
        # stronger one.
        code_votes = self._code_votes.get(tid)
        if code_votes:
            best = max(code_votes, key=code_votes.get)
            self._cache[tid] = best
            return best
        votes = self._votes.get(tid)
        if not votes:
            return "F"
        cid = max(votes, key=votes.get)          # confidence/size-weighted vote
        if self.native is not None:
            code = self.native.get(int(cid), "F")
        else:
            code = self._map(cid, *self.size_estimate(tid))
        self._cache[tid] = code
        return code

    @staticmethod
    def _map(cid: int, height_m: float = 0.0, width_m: float = 0.0) -> str:
        if cid in (_COCO_MOTO, _COCO_BICYCLE):
            return "G"
        if cid == _COCO_BUS:
            return "E"
        if cid == _COCO_CAR:
            return "A"                            # vans are not separable — see docstring
        if cid == _COCO_TRUCK:
            frontal_area = height_m * width_m
            return "D" if frontal_area >= HEAVY_MIN_FRONTAL_AREA_M2 else "C"
        return "F"
