"""Configuration for the ITS traffic pipeline.

Everything that is scene-specific (counting-line position, speed-calibration
points, region of interest) lives here so it can be tuned per video without
touching the processing code. Defaults are derived from the frame size so the
pipeline runs on ANY clip out of the box; swap in real values once the actual
street video is available (especially the speed calibration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- COCO class ids we care about -------------------------------------------------
# Egyptian streets: microbus -> bus/truck, tuk-tuk -> car/motorcycle (v1 limitation).
PERSON_CLASS = 0
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Coarse groups for the re-id class gate. Base YOLO flip-flops between car, truck
# and bus on the SAME vehicle from frame to frame, so demanding an exact class
# match would block legitimate re-attachments; grouping keeps the gate meaningful
# (a motorcycle can never be stitched to a lorry) while tolerating that.
TWO_WHEELER, FOUR_WHEELER, PEDESTRIAN, UNGROUPED = 0, 1, 2, -1
_COCO_GROUPS = {1: TWO_WHEELER, 3: TWO_WHEELER,          # bicycle, motorcycle
                2: FOUR_WHEELER, 5: FOUR_WHEELER, 7: FOUR_WHEELER,  # car/bus/truck
                0: PEDESTRIAN}
# Same grouping expressed over the mentor taxonomy, for a fine-tuned model.
_MENTOR_GROUPS = {"G": TWO_WHEELER,
                  "A": FOUR_WHEELER, "C": FOUR_WHEELER, "D": FOUR_WHEELER,
                  "E": FOUR_WHEELER, "V": FOUR_WHEELER,
                  "F": UNGROUPED, "P": PEDESTRIAN}


@dataclass(frozen=True)
class ClassScheme:
    """Which detector class ids mean 'vehicle', 'person', and how they group.

    Base COCO and a model fine-tuned on the mentor taxonomy use DIFFERENT id
    spaces for the same concepts, and everything downstream of the detector —
    counting, speed, occupancy, the re-id class gate — has to ask which is which.
    Hard-coding the COCO ids there is what made the fine-tuned path silently
    wrong: with a 7-class model, class 0 is "A" (private car), not "person", so
    every private car crossing the line was recorded as a PEDESTRIAN, and the
    classes COCO has no id for (G at 4, F at 6) were dropped from the counts, the
    speed sample and the occupancy measure entirely.

    docs/finetuning-plan.md §7 promises the fine-tune is a drop-in. This is the
    object that makes that true: build it once from the loaded model and pass it
    to every stage, instead of each stage assuming COCO.
    """

    vehicle_ids: frozenset
    person_ids: frozenset
    groups: dict

    def is_vehicle(self, class_id) -> bool:
        return int(class_id) in self.vehicle_ids

    def is_person(self, class_id) -> bool:
        return int(class_id) in self.person_ids

    def group(self, class_id) -> int:
        """Coarse group for the re-id gate; UNGROUPED never matches anything."""
        return self.groups.get(int(class_id), UNGROUPED)


COCO_SCHEME = ClassScheme(
    vehicle_ids=frozenset(VEHICLE_CLASSES),
    person_ids=frozenset({PERSON_CLASS}),
    groups=dict(_COCO_GROUPS),
)


def scheme_for_codes(native_codes: dict | None) -> ClassScheme:
    """Class scheme for a model, given its {class_id: mentor_code} map.

    ``None`` means a plain COCO model, which gets the COCO scheme. A fine-tuned
    model has no person class at all — the mentor taxonomy does not include one —
    so pedestrian counts are legitimately zero rather than silently full of cars.
    ``F`` (unknown) is a vehicle for counting purposes but is left UNGROUPED, so
    the re-id gate never stitches two vehicles together on the strength of both
    being unidentifiable.
    """
    if not native_codes:
        return COCO_SCHEME
    return ClassScheme(
        vehicle_ids=frozenset(i for i, c in native_codes.items() if c != "P"),
        person_ids=frozenset(i for i, c in native_codes.items() if c == "P"),
        groups={int(i): _MENTOR_GROUPS.get(c, UNGROUPED)
                for i, c in native_codes.items()},
    )


@dataclass
class PipelineConfig:
    # Model / inference
    # Calibrated for the Elsewedy highway CCTV clip (street_egypt.mp4, 1280x720):
    # yolov8s for accuracy, imgsz 736 to match 720p, every frame for exact counts.
    model: str = "yolov8s.pt"
    imgsz: int = 736
    conf: float = 0.30
    frame_stride: int = 1  # process every Nth frame (CPU speed lever)
    # Re-attach a vehicle that ByteTrack lost during an occlusion to its original
    # id, instead of letting it return as a new vehicle. See pipeline/reid.py.
    stable_ids: bool = True
    # Only track/annotate vehicles whose ground point is on the analysed
    # carriageway. This camera also sees the OPPOSITE carriageway: on the 90 s
    # clip, 99 of 126 ids belonged to traffic we do not analyse. They could never
    # be counted (they never reach the counting line) but they consumed id
    # numbers, so on-screen ids jumped around and looked unstable. The margin
    # keeps a vehicle tracked slightly outside the ROI so it is not dropped and
    # re-acquired at the boundary.
    roi_gated_tracking: bool = True
    roi_margin_px: int = 45

    # --- Vehicle class ------------------------------------------------------------
    # Second-stage classifier over each tracked crop, emitting the mentor
    # taxonomy directly. Empty means "use the COCO size heuristic", which cannot
    # express C (light truck) or V (van) and has no microbus concept — measured
    # on street_egypt.mp4 it gets bus/microbus right 1 time in 24, and
    # microbuses are about a third of that road's traffic.
    #
    # Train one with notebooks/vehicle_classes_colab.ipynb; deploy with
    # ITS_VEHICLE_CLS=models/vehicle_cls.pt. See tools/train_vehicle_classes.py.
    vehicle_cls_model: str = ""
    # Run the classifier on every Nth observation of a track. It is a second
    # inference per vehicle per frame on CPU, and a vehicle is visible for tens
    # of frames, so sampling costs almost nothing once the per-track vote runs.
    vehicle_cls_every: int = 5

    # --- Licence plates -----------------------------------------------------------
    # Plate detection and plate COLOUR need only ~20 px of plate width and work on
    # this camera. Plate OCR needs ~100 px and does not: measured four ways, plates
    # here top out at ~35 px. See docs/anpr-plan.md and tools/plate_footage_check.
    #
    # The stage is off by default because it costs a second inference pass per
    # vehicle crop, which matters on CPU. Turn it on with ITS_PLATES=1.
    plates: bool = False
    plate_model: str = "models/plate_detect.pt"
    # Plates are small objects in a full frame, so the plate pass runs at a
    # higher resolution than the vehicle pass — recall collapses below ~1280.
    plate_imgsz: int = 1280
    # OCR weights. Empty means no OCR engine — the report then says so rather than
    # leaving an unexplained empty column.
    plate_ocr_model: str = ""
    # Plate width below which character recognition is not attempted. Emitting a
    # confident string from a 30 px plate is worse than emitting nothing.
    plate_ocr_min_px: float = 100.0
    # Draw the plate box and its colour on the annotated video.
    draw_plates: bool = True

    # --- multi-frame OCR (best-crop -> super-resolution -> vote) -------------------
    # Keeps the best N views of each vehicle's plate, super-resolves only those,
    # OCRs each and votes. See pipeline/plate_ocr.py.
    plate_enhance: bool = False
    # Crops kept per vehicle for fusion. 12, not 3: the measured gain that
    # justifies fusion at all (+15-20 character-accuracy points, docs/anpr-plan.md
    # §5c) was measured with TWELVE-frame fusion, so keeping 3 was leaving most
    # of the only technique that genuinely recovers detail on the table. Crops
    # far smaller than the best one are excluded at fusion time regardless
    # (plate_ocr.FUSE_SCALE_TOLERANCE), so a larger keep costs little.
    plate_keep_crops: int = 12
    plate_sr_model: str = "models/RealESRGAN_x4.pth"
    plate_alpr_model: str = "models/eg_alpr.pt"
    # Super-resolution multiplies effective width by 4, so a 25px plate becomes
    # 100px. Whether that is a genuine read or an upscaling artefact is exactly
    # what the cross-crop vote is there to test, so the floor applied to the
    # ENHANCED pipeline is on the raw crop and deliberately low.
    plate_enhance_min_px: float = 12.0  # raw-crop floor; fusion reads start ~50px
    # Write every intermediate image (raw crop, super-resolved, OCR overlay).
    plate_debug_dir: str = ""

    # --- Annotation overlay -------------------------------------------------------
    # The counting line and its IN/OUT tag are an internal reference: counting
    # still happens, and in/out totals are still reported in analytics.json and on
    # the dashboard, they are just not drawn over the video.
    draw_counting_line: bool = False
    draw_in_out_hud: bool = False

    # Counting line, as fractions of (width, height). Spans the FULL main
    # carriageway — endpoints are the measured carriageway edges at this row (see
    # lane_dividers). A narrower span silently misses vehicles in lane 1, which
    # sits far left here.
    #
    # The HEIGHT was chosen by measurement, not taste. Traffic travels bottom ->
    # top, and sweeping the line over the clip shows two distinct failure modes:
    #   * too low (y > 0.55): a vehicle that is already up the road when the clip
    #     starts never reaches the line and is never counted — this is what left
    #     the first car uncounted at the old y=0.60.
    #   * too high... rather, too close to the camera (y < 0.41): vehicles cross
    #     within a few frames of appearing, BEFORE the re-id layer has the
    #     evidence to merge a duplicate box, so one vehicle is counted twice.
    # Between those, y = 0.41..0.55 is a plateau counting all 18 real vehicles
    # with zero duplicates. 0.48 sits in the middle, 7 rows from either edge.
    line_start: tuple[float, float] = (0.0637, 0.48)
    line_end: tuple[float, float] = (0.8367, 0.48)

    # Region of interest for congestion/occupancy: the true carriageway polygon
    # bounded by the same measured edge lines, from y=0.30 down to y=0.99.
    roi: tuple[tuple[float, float], ...] = (
        (0.1645, 0.30), (0.7359, 0.30), (1.1241, 0.99), (-0.2237, 0.99),
    )

    # --- Speed calibration --------------------------------------------------------
    # source_points: 4 image points (fractions of w,h) forming a road-plane quad,
    # order TL, TR, BR, BL. The left and right edges follow the two MEASURED
    # carriageway boundary lines (same fit as lane_dividers), which are genuinely
    # parallel on the road. Mapping that quad to a rectangle therefore sends both
    # vanishing points to infinity and rectifies the road plane correctly.
    #
    # This is verifiable, and it is how the numbers below were set: the painted
    # dashes are evenly spaced in reality, so after a correct rectification their
    # measured pitch must be CONSTANT. It is — 27.2 / 27.1 / 27.9 / 28.8, a 5.9%
    # spread. The previous quad gave 25.1 / 18.4 / 14.5 / 11.7 m, a 114% spread
    # that shrank with distance: proof it was not a road-plane rectangle, and the
    # reason speeds came out near 100 km/h on a road like this.
    #
    #   width  = 15.86 m — carriageway width implied by the measured lane widths
    #                      (three ~3.65 m lanes plus the wide L4 + shoulder).
    #   length = 43.6 m  — the value that makes the dash pitch equal DASH_PITCH_M.
    #
    # The one remaining assumption is dash_pitch_m. If this road uses a different
    # marking standard, change ONLY that number and re-derive length linearly;
    # everything else is measured. A surveyed distance between two ground marks
    # would remove the assumption entirely.
    calibrated: bool = True
    dash_pitch_m: float = 12.0   # 3 m dash + 9 m gap (common highway standard)
    source_points: tuple[tuple[float, float], ...] = (
        (0.1926, 0.2500), (0.7078, 0.2500), (1.1141, 0.9722), (-0.2137, 0.9722),
    )
    target_size_m: tuple[float, float] = (15.86, 43.6)  # width x length (metres)

    # --- Lanes -------------------------------------------------------------------
    # Explicit lane boundaries placed ON the painted white stripes (not an equal
    # split). Each boundary is a line ((far_x, far_y), (near_x, near_y)) in
    # fractions; consecutive boundaries bound one lane.
    #
    # These are MEASURED, not eyeballed: a 60-frame median background isolates the
    # markings, a Hough fit recovers each line as x = a*y + b, and rectifying the
    # road plane (which sends both vanishing points to infinity) confirms the fits
    # are consistent to ~1% across rows. See docs/methodology.md.
    #
    #   edge | stripe1 | stripe2 | stripe3 | right barrier  ->  L1 L2 L3 and a wide L4
    #
    # L4 deliberately runs from stripe 3 out to the barrier, absorbing the paved
    # shoulder, which makes it the widest zone (31% of the carriageway vs 25% for
    # L1). Vehicle-position sampling shows traffic uses all four, with nothing
    # travelling beyond the solid right edge line — so the shoulder only ever adds
    # a vehicle to L4 rather than creating a phantom fifth lane.
    #
    # Fractions outside [0,1] are intentional: the carriageway leaves the frame on
    # both sides before the bottom row, and the polygons must follow it.
    lane_dividers: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
        ((0.1532, 0.32), (-0.2237, 0.99)),  # solid edge beside median — left of L1
        ((0.3063, 0.32), (0.1132, 0.99)),   # stripe 1 — L1 | L2
        ((0.4246, 0.32), (0.3926, 0.99)),   # stripe 2 — L2 | L3
        ((0.5494, 0.32), (0.7178, 0.99)),   # stripe 3 — L3 | L4
        ((0.7472, 0.32), (1.1241, 0.99)),   # right barrier — right edge of wide L4
    )

    # Congestion occupancy thresholds (fraction of ROI area covered by vehicles)
    occ_moderate: float = 0.15
    occ_heavy: float = 0.35
    occ_jam: float = 0.55
    jam_speed_kmh: float = 10.0  # low avg speed bumps the level up

    def line_px(self, w: int, h: int):
        return (
            (int(self.line_start[0] * w), int(self.line_start[1] * h)),
            (int(self.line_end[0] * w), int(self.line_end[1] * h)),
        )

    def roi_px(self, w: int, h: int) -> np.ndarray:
        return np.array([[int(x * w), int(y * h)] for x, y in self.roi], dtype=np.int32)

    def source_px(self, w: int, h: int) -> np.ndarray:
        return np.array(
            [[x * w, y * h] for x, y in self.source_points], dtype=np.float32
        )

    def target_px(self) -> np.ndarray:
        tw, th = self.target_size_m
        # Map to a metric rectangle (metres). Order matches source_points order:
        # top-left, top-right, bottom-right, bottom-left.
        return np.array(
            [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
        )


CONGESTION_LEVELS = ["Free-flow", "Moderate", "Heavy", "Jam"]
CONGESTION_COLORS = {  # BGR for OpenCV overlay
    "Free-flow": (80, 200, 120),
    "Moderate": (60, 200, 240),
    "Heavy": (60, 120, 240),
    "Jam": (60, 60, 220),
}
