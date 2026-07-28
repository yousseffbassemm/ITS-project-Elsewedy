"""Licence-plate stage: locate the plate on each tracked vehicle, read its
COLOUR, and — where the footage allows — its characters.

Three capabilities with very different resolution needs, deliberately kept
separate so the two cheap ones are not held hostage by the expensive one:

    plate DETECTION  ~20 px plate width   a findable bright rectangle
    plate COLOUR     ~20 px               dominant hue of the top band
    plate OCR       ~100 px               individual glyph strokes

On this project's target footage (samples/street_egypt.mp4) plates top out at
34 px, so detection and colour work and OCR cannot — see
tools/plate_footage_check.py, which measures this for any clip. OCR is therefore
optional here: it activates when a plate is large enough, and records WHY it
declined otherwise, rather than emitting confident nonsense.

**Why colour is worth having on its own.** Egypt colour-codes the plate's top
band by vehicle CATEGORY, which is independent evidence for exactly the classes
docs/finetuning-plan.md calls hardest (C light truck / V van / A private car):

    blue        private car "malaky"      -> A
    red         truck / tractor           -> C or D
    orange      taxi                      -> A (commercial use)
    green       diplomatic
    yellow      customs unpaid

Measured on 2,017 EALPR plates, 89% are blue — so colour is a strong signal for
"this is NOT a private car" and a weak one for anything finer. Police plates are
a darker blue in principle and are not separable from private at this
resolution, so they are not claimed.

The band is a large flat patch of colour, which is precisely the kind of signal
that survives heavy downscaling — a 34 px plate still carries a usable hue even
though its characters are gone.

Evidence is accumulated PER TRACK and resolved once, mirroring
classify.VehicleClassifier: a single frame's colour reading is noisy at this
scale, but a vehicle is visible for tens of frames and the vote is stable.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# --- colour taxonomy --------------------------------------------------------------
# Reference hues in OpenCV HSV (H is 0-179, not 0-359). Ranges rather than points
# because plate paint fades, and CCTV white balance shifts hue by a few degrees
# between day and dusk.
#
# Each entry: (h_lo, h_hi, s_min, v_min). Achromatic classes (white/black) are
# handled separately since hue is meaningless at low saturation.
# Hue bands were re-fitted against the 2,017 usable EALPR plates rather than
# guessed. See tools/build_plate_colour_set.py; the measured distribution is in
# docs/anpr-plan.md.
#
# The big correction: light_blue and dark_blue were separate categories, and the
# data says they are ONE population. 89% of all Egyptian plates sit in H 90-120
# with no bimodality — the old split was cutting a single blue cloud in half on
# brightness, which is why a private car in street_egypt.mp4 came out "police".
# Police plates do exist, but this camera cannot separate them from private ones
# by band colour, and inventing the distinction is worse than declining it.
PLATE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red":    (0, 12, 60, 40),      # truck / tractor   (wraps — see classify_band)
    "orange": (13, 25, 70, 70),     # taxi
    "yellow": (26, 36, 80, 90),     # customs unpaid
    "green":  (40, 85, 45, 40),     # diplomatic
    "blue":   (86, 125, 35, 25),    # private "malaky" (and police — not separable)
}

# Plate colour -> mentor vehicle-class codes it supports. Used as EVIDENCE that
# nudges classification, never as an override: a truck towing a private car still
# shows its own red plate, and a repainted plate is not a reclassification.
COLOR_TO_CODES: dict[str, tuple[str, ...]] = {
    "red": ("C", "D"),
    "blue": ("A",),
    "orange": ("A",),
    "green": ("A",),
    "yellow": (),
    "white": (),
    "unknown": (),
}

# Draw colour for each plate category (BGR). Chosen to READ as the plate colour
# it represents, so the overlay is self-explaining without a legend.
PLATE_BGR: dict[str, tuple[int, int, int]] = {
    "red": (60, 60, 220),
    "orange": (60, 150, 250),
    "yellow": (60, 220, 240),
    "green": (90, 200, 90),
    "blue": (230, 150, 60),
    "white": (230, 230, 230),
    "unknown": (150, 150, 150),
}

COLOR_DISPLAY = {
    "red": "Red — truck/tractor",
    "orange": "Orange — taxi",
    "yellow": "Yellow — customs unpaid",
    "green": "Green — diplomatic",
    "blue": "Blue — private (malaky)",
    "white": "White — no band read",
    "unknown": "Unknown",
}

# Resolution floors, in plate pixel width. Kept here rather than inline so the
# footage checker and this module cannot drift apart.
#
# There are TWO OCR floors because there are two pipelines, and conflating them
# is what made the docs contradict themselves. Measured on 50 real Egyptian
# plates degraded to a fixed width (docs/anpr-plan.md §5c):
#
#     width | single frame | 12-frame fusion
#      50px |    46.2%     |     66.7%
#      65px |    77.8%     |     93.1%
#     100px |    95.8%     |     99.5%
#
# So ~100px is the floor for reading ONE frame, which is what MIN_PX_FOR_OCR
# gates; multi-frame fusion (pipeline/plate_ocr.PlateEnhancer) reaches the same
# accuracy at ~65px, and is partial down to ~50px. Neither rescues this
# project's 34px footage — fusion scores 12% there, 0 plates read.
MIN_PX_FOR_COLOR = 20.0
MIN_PX_FOR_OCR = 100.0        # single-frame read
MIN_PX_FOR_FUSED_OCR = 65.0   # multi-frame fusion + voting
MARGINAL_PX_FOR_FUSED_OCR = 50.0
MIN_SAMPLES = 3          # per-track colour votes before a colour is reported

# A plate wider than this fraction of its vehicle's box is not a plate. Real
# plates are ~18% of vehicle width (0.32 m on a 1.8 m car); 0.40 leaves room for
# a motorcycle, whose plate is a much larger share of a narrow vehicle, while
# still rejecting a detector that has locked onto the whole rear panel.
MAX_PLATE_FRAC_OF_VEHICLE = 0.40

# A plate is a rectangle of a particular SHAPE, and shape is the cheapest way to
# reject the false positive that costs most here: tailgate brand lettering.
#
# Measured on street_egypt.mp4 — the single widest "plate" in the whole clip, at
# 94px, was the CHEVROLET badge on a pickup's tailgate. It is bright, sharp,
# high-contrast and horizontal, so the crop scorer ranked it BEST of all crops
# for that vehicle (0.872) and spent the super-resolution and fusion passes on
# it. It also inflated the reported plate size statistics, making the footage
# look better than it is.
#
# Egyptian civilian plates run about 2:1 to 3:1. The band below is deliberately
# wider than that: a plate seen from a steep downward angle foreshortens
# vertically and its aspect ratio RISES, so the upper bound has to tolerate
# genuine perspective. The badge sits at 5.5:1, outside it. Anything beyond this
# band is a strip of something, not a plate — and at that foreshortening the
# characters are unreadable regardless.
MIN_PLATE_ASPECT, MAX_PLATE_ASPECT = 1.5, 4.5


# How much more saturated the band must be than the plate BODY before it counts
# as coloured, plus an absolute floor so a noisy dark crop cannot qualify on
# margin alone.
SAT_MARGIN_OVER_BODY = 20.0
SAT_ABSOLUTE_FLOOR = 30.0


def classify_band(h: float, s: float, v: float,
                  s_body: float = 0.0, v_body: float = 255.0) -> str:  # noqa: ARG001
    # `v_body` is accepted and deliberately NOT used. Every caller measures it
    # anyway (it comes free from the same median as s_body), and it is the
    # obvious next signal if this rule ever needs hardening — a very dark plate
    # body means the crop is underexposed and the saturation comparison below is
    # on thin ice. It is not used today because that behaviour has not been
    # measured, and adding an untested branch to the one colour rule that does
    # work is a bad trade. Keeping the parameter keeps the call sites ready.
    """Plate colour from the band's HSV, judged AGAINST the plate's own body.

    Absolute saturation thresholds do not survive this footage. The measured red
    band of a truck in samples/street_egypt.mp4 reads S=57 — far below the S>=70
    a clean plate photo would give — because at a 32 px plate width H.264 chroma
    subsampling leaves the 4 px band only ~2 px of chroma to work with. Any fixed
    floor tuned on clean imagery rejects it; any floor low enough to accept it
    starts calling grey bumpers coloured.

    The way out is that every Egyptian plate carries its own reference: the white
    body below the band. Judging the band relative to that neutral patch cancels
    the codec, the white balance and the exposure together, because both regions
    suffered them equally. On that same truck the band reads S=57 against a body
    at S=16 — a 41-point margin that is unambiguous even though the absolute
    value is low.

    The margin has since been checked against 2,017 EALPR plates: 82% clear a
    40-point band-vs-body margin, and the 18% that do not are genuinely faded or
    washed-out bands that no threshold should claim. The 20-point floor used here
    is deliberately below that, because a CCTV plate is dimmer than a EALPR
    close-up and the absolute floor catches the rest.
    """
    if s < max(s_body + SAT_MARGIN_OVER_BODY, SAT_ABSOLUTE_FLOOR):
        # Not meaningfully more colourful than the plate's own white body, so
        # there is no band to read. Saying "white" beats inventing a category.
        return "white" if v > 90 else "unknown"
    for name, (lo, hi, _s_min, v_min) in PLATE_COLORS.items():
        if v < v_min:
            continue
        if name == "red":
            # Red wraps in OpenCV's 0-179 hue encoding. The wrap arm starts at
            # 160 rather than 170 because the measured plate sits at H=169 and
            # 160-179 is entirely red-magenta — no other plate colour lives there.
            if h <= hi or h >= 160:
                return name
        elif lo <= h <= hi:
            return name
    return "unknown"


class PlateReader:
    """Per-track plate colour (and optionally characters) from vehicle crops.

    The detector weights are optional. Without them the class still runs and
    reports why it produced nothing, so the pipeline never hard-fails on a
    missing model — the rest of the analytics are unaffected.
    """

    def __init__(self, weights: str | None = None, ocr=None,
                 min_px_for_ocr: float = MIN_PX_FOR_OCR, enhancer=None):
        self.model = None
        self.load_error: str | None = None
        self.ocr = ocr
        self.min_px_for_ocr = min_px_for_ocr
        # Optional multi-frame pipeline: best-crop selection -> super-resolution
        # -> OCR -> confidence-weighted vote. See pipeline/plate_ocr.py.
        self.enhancer = enhancer
        if weights:
            try:
                from ultralytics import YOLO
                if not Path(weights).exists():
                    raise FileNotFoundError(weights)
                self.model = YOLO(weights)
            except Exception as exc:                  # pragma: no cover - env dependent
                self.load_error = f"{type(exc).__name__}: {exc}"

        self._colors: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._widths: dict[int, list[float]] = defaultdict(list)
        # Number of frames a COLOUR was actually read for this track. Distinct
        # from len(_widths), which counts every plate detection including the
        # sub-20px ones no colour is read from — see color_of.
        self._color_n: dict[int, int] = defaultdict(int)
        self._texts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Plate boxes located in the MOST RECENT frame only. Keeping them across
        # frames made the overlay draw a plate rectangle at the position it was
        # last seen for the rest of the vehicle's life — a box detached from its
        # vehicle, drifting backwards down the road. Plate detection is
        # intermittent at this scale, so "no box this frame" is the normal case
        # and drawing nothing is the honest answer.
        self._boxes: dict[int, tuple[int, int, int, int]] = {}
        self._cache: dict[int, str] = {}
        # Plate HEIGHT per track. Height is what OCR actually needs — characters
        # are ~55% of it — and width alone is misleading: a 94px-wide detection
        #17px tall carries the same 9px characters as a 34px plate.
        self._heights: dict[int, list[float]] = defaultdict(list)
        self.detections = 0
        self.rejected_shape = 0

    # --- geometry ---------------------------------------------------------------
    def _find_plates_frame(self, frame: np.ndarray, imgsz: int
                           ) -> list[tuple[float, float, float, float, float]]:
        """Every plate in the WHOLE frame, in frame coordinates.

        Detection runs on the full frame, not on per-vehicle crops, and this is a
        correctness issue rather than an optimisation. The detector was trained on
        full scenes, where a plate is a small object. Handed a tight 100x126
        vehicle crop it is far outside that domain — Ultralytics upscales the crop
        to imgsz, the entire rear panel becomes plate-shaped, and it returned a
        box 83% of the vehicle width at 0.73 confidence. Those false positives
        then reported plates as ~150 px wide in a clip whose real maximum is
        ~35 px, which would have made the footage look adequate for OCR when it
        is not — the exact wrong conclusion.

        Running once per frame is also ~10x cheaper than once per vehicle.
        """
        if self.model is None:
            return []
        r = self.model.predict(frame, verbose=False, conf=0.25, imgsz=imgsz)[0]
        confs = r.boxes.conf.cpu().numpy()
        return [(*(float(v) for v in b), float(c))   # type: ignore[misc]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), confs)]


    # --- evidence ---------------------------------------------------------------
    def observe_frame(self, frame: np.ndarray, det, imgsz: int = 1280) -> None:
        """Record one frame of plate evidence for every tracked vehicle in it.

        Plates are detected once across the frame and then matched to vehicles by
        containment, rather than each vehicle being cropped and searched.
        """
        # Boxes are per-frame (see _boxes): clear before this frame's detections
        # so box_of() can never hand the annotator a position from a past frame.
        self._boxes.clear()
        if det.tracker_id is None or not len(det):
            return
        for pb in self._find_plates_frame(frame, imgsz):
            tid = self._owner(pb[:4], det)
            if tid is None:
                continue
            self._observe_one(tid, frame, pb[:4], conf=pb[4])

    @staticmethod
    def _owner(plate_box, det) -> int | None:
        """Which tracked vehicle a plate belongs to, or None.

        A plate must sit INSIDE a vehicle box — a detection floating on the road
        belongs to nobody and is dropped rather than attached to whichever
        vehicle happens to be nearest. Where boxes overlap, the smallest
        containing vehicle wins: on this camera a distant vehicle is often framed
        inside a nearer one's box, and the tighter fit is the true owner.
        """
        px1, py1, px2, py2 = plate_box
        cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        best, best_area = None, float("inf")
        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i]
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            # A real plate is a small fraction of the vehicle it is bolted to.
            # This guard is what catches a detector that has locked onto the
            # whole rear panel instead of the plate.
            if (px2 - px1) > MAX_PLATE_FRAC_OF_VEHICLE * (x2 - x1):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best, best_area = int(det.tracker_id[i]), area
        return best

    def _observe_one(self, tid: int, frame: np.ndarray, plate_box,
                     conf: float = 1.0) -> None:
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = (int(v) for v in plate_box)
        px1, py1 = max(px1, 0), max(py1, 0)
        px2, py2 = min(px2, w), min(py2, h)
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 4:
            return
        # Shape gate — see MIN_PLATE_ASPECT. Rejecting here rather than at draw
        # time matters: a badge accepted as a plate votes on the vehicle's plate
        # COLOUR, is picked as the best crop for OCR, and is fused with the real
        # plate views, corrupting all three.
        if not (MIN_PLATE_ASPECT <= pw / max(ph, 1) <= MAX_PLATE_ASPECT):
            self.rejected_shape += 1
            return
        self.detections += 1
        self._widths[tid].append(float(pw))
        self._heights[tid].append(float(ph))
        self._boxes[tid] = (px1, py1, px2, py2)
        # Offer this view to the best-frame selector. It keeps only the top few
        # per vehicle, so the expensive super-resolution pass is spent on the
        # crops most likely to read rather than on all of them.
        if self.enhancer is not None:
            self.enhancer.offer(tid, frame[py1:py2, px1:px2], conf)

        if pw < MIN_PX_FOR_COLOR:
            return
        plate = frame[py1:py2, px1:px2]
        # The colour band occupies the top third; the rest is the white body
        # carrying the characters.
        split = max(ph // 3, 1)
        band, body = plate[:split, :], plate[split:, :]
        if band.size == 0 or body.size == 0:
            return
        # Median, not mean: the band has black lettering ("EGYPT") across it, and
        # averaging drags the reading toward grey. The median ignores that text.
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        hh, ss, vv = (float(np.median(hsv[..., k])) for k in range(3))
        # The body is the plate's built-in neutral reference — see classify_band.
        hsv_b = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        s_body, v_body = (float(np.median(hsv_b[..., k])) for k in (1, 2))
        name = classify_band(hh, ss, vv, s_body, v_body)
        # A bigger, brighter sample is better evidence — same weighting principle
        # the class vote uses.
        self._colors[tid][name] += pw * max(vv, 1.0) / 255.0
        self._color_n[tid] += 1
        self._cache.pop(tid, None)

        if self.ocr is not None and pw >= self.min_px_for_ocr:
            try:
                text, text_conf = self.ocr(plate)
            except Exception:                         # pragma: no cover
                return
            if text:
                self._texts[tid][text] += float(text_conf)

    # --- resolution -------------------------------------------------------------
    def color_of(self, tid: int) -> str:
        tid = int(tid)
        if tid in self._cache:
            return self._cache[tid]
        votes = self._colors.get(tid)
        if not votes:
            return "unknown"
        real = {k: v for k, v in votes.items() if k not in ("unknown", "white")}
        # Require a few consistent samples before naming a category — one frame
        # of a red brake light bleeding onto the band is not a red plate.
        #
        # The count is of COLOUR readings, not of plate detections. Gating on
        # len(_widths) let a track with three sub-20px detections (from which no
        # band is ever read) and a single colour sample name a category on that
        # one frame — exactly the evidence this gate exists to demand more of.
        pool = real or votes
        if self._color_n.get(tid, 0) < MIN_SAMPLES:
            return "unknown"
        best = max(pool, key=pool.get)
        self._cache[tid] = best
        return best

    def text_of(self, tid: int) -> tuple[str | None, float]:
        votes = self._texts.get(int(tid))
        if not votes:
            return None, 0.0
        best = max(votes, key=votes.get)
        total = sum(votes.values()) or 1.0
        return best, votes[best] / total

    def plate_px(self, tid: int) -> float:
        w = self._widths.get(int(tid)) or []
        return float(np.percentile(w, 90)) if w else 0.0

    def plate_px_best(self, tid: int) -> float:
        """Widest view of this plate — what OCR actually gets to work with.

        The p90 above describes the typical detection, most of which are distant.
        Feasibility depends on the BEST view, and reporting only the p90
        understated it by ~40% on this clip.
        """
        w = self._widths.get(int(tid)) or []
        return float(max(w)) if w else 0.0

    def char_px_best(self, tid: int) -> float:
        """Estimated character height in the best view — the binding number.

        Every ANPR specification is written in character height, not plate
        width, and the two come apart badly at steep viewing angles: a 94px-wide
        detection 17px tall carries 9px characters, the same as a 34px plate
        seen square-on. Egyptian glyphs occupy roughly 55% of plate height.
        """
        h = self._heights.get(int(tid)) or []
        return round(float(max(h)) * 0.55, 1) if h else 0.0

    def box_of(self, tid: int):
        return self._boxes.get(int(tid))

    def _ocr_note(self, p90: float) -> str:
        """Explain the OCR outcome, leading with resolution — the binding limit.

        Quotes the floor and the engine ACTUALLY in force. With --enhance the
        readable floor is the fusion one (~65px, not ~100px) and the engine
        belongs to the enhancer, so the plain single-frame wording reported a
        harder limit than the run faced and claimed "no OCR engine is
        configured" on a run that had one.
        """
        if not p90:
            return "no plates measured, so OCR feasibility is undetermined"
        fused = self.enhancer is not None
        floor = MIN_PX_FOR_FUSED_OCR if fused else self.min_px_for_ocr
        how = " with multi-frame fusion" if fused else ""
        has_engine = self.ocr is not None or (
            fused and getattr(self.enhancer, "ocr", None) is not None)
        if p90 < floor:
            short = floor / max(p90, 1e-6)
            return (
                f"plate p90 {p90:.0f}px is below the {floor:.0f}px floor for "
                f"character recognition{how} (~{short:.1f}x short); this is a "
                "camera-resolution limit, not a model limit — more training data "
                "cannot add pixels the sensor never captured "
                "(see tools/plate_footage_check.py)"
            )
        if not has_engine:
            return (f"plate p90 {p90:.0f}px clears the {floor:.0f}px floor, but "
                    "no OCR engine is configured — this footage COULD be read")
        return f"plate p90 {p90:.0f}px >= {floor:.0f}px floor{how} — OCR ran"

    def summary(self, tracks, classifier=None, lane_of=None,
                display_id=None) -> dict:
        """Report for analytics.json, including why OCR did or did not run.

        The optional arguments attach each plate to the vehicle it came from —
        its class, lane and on-screen number. A plate row is only actionable in
        context: "red plate" matters because it is on the lorry in lane 4.
        """
        rows, widths = [], []
        for tid in tracks:
            c = self.color_of(tid)
            px = self.plate_px(tid)
            if px:
                widths.append(px)
            text, tconf = self.text_of(tid)
            # The multi-frame pipeline is authoritative when it is enabled: it
            # looked at the best crops rather than whichever frame happened to be
            # processed, and its confidence is cross-crop agreement rather than
            # one model's self-assessment.
            enh = self.enhancer.resolve(tid) if self.enhancer is not None else None
            if enh and enh.get("plate_text"):
                text, tconf = enh["plate_text"], enh["plate_confidence"]
            code = classifier.resolve(tid) if classifier is not None else None
            lane = None if lane_of is None else lane_of.get(int(tid))
            rows.append({
                "track": int(tid),
                "vehicle_no": (display_id(tid) if display_id is not None
                               else None),
                "vehicle_class": code,
                # Lanes are 0-indexed internally and 1-indexed everywhere a human
                # reads them; match the rest of the report.
                "lane": None if lane is None else lane + 1,
                "plate_color": c,
                "plate_color_display": COLOR_DISPLAY.get(c, c),
                "supports_classes": list(COLOR_TO_CODES.get(c, ())),
                # Whether the plate colour AGREES with the class the pipeline
                # reached independently. Disagreement is the interesting case —
                # it is either a misread band or a genuinely unusual vehicle.
                "agrees_with_class": (None if not code or not COLOR_TO_CODES.get(c)
                                      else code in COLOR_TO_CODES[c]),
                "plate_px": round(px, 1),
                "plate_px_best": round(self.plate_px_best(tid), 1),
                # The number every ANPR spec is actually written in.
                "char_px_best": self.char_px_best(tid),
                "plate_text": text,
                "plate_text_confidence": round(tconf, 2),
                **({"enhance": enh} if enh else {}),
            })
        mix: dict[str, int] = defaultdict(int)
        for r in rows:
            mix[r["plate_color"]] += 1
        p90 = round(float(np.percentile(widths, 90)), 1) if widths else 0.0
        ocr_ran = any(r["plate_text"] for r in rows)
        return {
            "model": "trained" if self.model is not None else "heuristic fallback",
            "model_error": self.load_error,
            "plates_detected": self.detections,
            # Detections thrown out for not being plate-shaped (tailgate badges,
            # light bars). A large number here is not a problem being hidden — it
            # is false positives that used to reach the colour vote and the OCR.
            "rejected_wrong_shape": self.rejected_shape,
            "plate_px_p90": p90,
            "plate_px_best": round(max((r["plate_px_best"] for r in rows),
                                       default=0.0), 1),
            "char_px_best": round(max((r["char_px_best"] for r in rows),
                                      default=0.0), 1),
            # 20px is the ANPR industry minimum character height; 15px the
            # absolute floor below which no vendor claims a read.
            "char_px_required": 20.0,
            "color_mix": dict(mix),
            "ocr_attempted": self.ocr is not None or self.enhancer is not None,
            "ocr_produced_text": ocr_ran,
            # The floor actually in force, so a reader (and tools/export_plates_csv)
            # never has to guess which one applied. The enhanced pipeline uses a
            # much lower raw-crop floor because fusion+voting is what tests the
            # read, so quoting the single-frame floor for an --enhance run
            # reported "not attempted" for plates that were in fact attempted.
            "ocr_min_px": (self.enhancer.min_px if self.enhancer is not None
                           else self.min_px_for_ocr),
            # The whole point of the resolution note: an empty OCR column must be
            # attributable, or it reads as a broken model and sends the next two
            # weeks into collecting more training data that cannot help.
            #
            # The resolution verdict is stated whether or not an engine is
            # configured. Reporting only "no engine" when the footage ALSO cannot
            # support OCR hides the fact that actually matters — someone would
            # plug in an engine, get nothing, and still not know why.
            "ocr_note": self._ocr_note(p90),
            "enhance": (None if self.enhancer is None else {
                "enabled": True,
                "min_raw_px": self.enhancer.min_px,
                "crops_kept_per_vehicle": self.enhancer.keep,
                "super_resolution": self.enhancer.sr.mode if self.enhancer.sr else "off",
                "debug_dir": str(self.enhancer.debug) if self.enhancer.debug else None,
            }),
            "vehicles": rows,
        }
