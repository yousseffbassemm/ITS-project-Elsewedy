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

    light blue  private car "malaky"      -> A
    red         truck / tractor           -> C or D
    orange      taxi                      -> A (commercial use)
    brown       commercial vehicle        -> C / V
    dark blue   police
    green       diplomatic
    yellow      customs unpaid

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
PLATE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red":        (0, 10, 70, 50),      # trucks / tractors  (wraps, see _hue_match)
    "orange":     (11, 22, 90, 80),     # taxi
    "yellow":     (23, 33, 90, 90),     # customs unpaid
    "green":      (40, 85, 50, 40),     # diplomatic
    "light_blue": (86, 105, 40, 90),    # private car "malaky"
    "dark_blue":  (106, 130, 60, 25),   # police
    "brown":      (5, 20, 60, 25),      # commercial (dark, low-value orange)
}

# Plate colour -> mentor vehicle-class codes it supports. Used as EVIDENCE that
# nudges classification, never as an override: a truck towing a private car still
# shows its own red plate, and a repainted plate is not a reclassification.
COLOR_TO_CODES: dict[str, tuple[str, ...]] = {
    "red": ("C", "D"),
    "brown": ("C", "V"),
    "light_blue": ("A",),
    "orange": ("A",),
    "dark_blue": ("A",),
    "green": ("A",),
    "yellow": (),
    "white": (),
    "unknown": (),
}

COLOR_DISPLAY = {
    "red": "Red — truck/tractor",
    "orange": "Orange — taxi",
    "yellow": "Yellow — customs unpaid",
    "green": "Green — diplomatic",
    "light_blue": "Light blue — private",
    "dark_blue": "Dark blue — police",
    "brown": "Brown — commercial",
    "white": "White — no band read",
    "unknown": "Unknown",
}

# Resolution floors, in plate pixel width. Kept here rather than inline so the
# footage checker and this module cannot drift apart.
MIN_PX_FOR_COLOR = 20.0
MIN_PX_FOR_OCR = 100.0
MIN_SAMPLES = 3          # per-track colour votes before a colour is reported


# How much more saturated the band must be than the plate BODY before it counts
# as coloured, plus an absolute floor so a noisy dark crop cannot qualify on
# margin alone.
SAT_MARGIN_OVER_BODY = 20.0
SAT_ABSOLUTE_FLOOR = 30.0


def classify_band(h: float, s: float, v: float,
                  s_body: float = 0.0, v_body: float = 255.0) -> str:
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

    CAVEAT: the margin is currently anchored on a handful of hand-located plates.
    It should be re-fitted against the EALPR dataset, which has enough labelled
    Egyptian plates to set it from a distribution rather than an example — the
    same reservation classify.HEAVY_MIN_FRONTAL_AREA_M2 carries.
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
                 min_px_for_ocr: float = MIN_PX_FOR_OCR):
        self.model = None
        self.load_error: str | None = None
        self.ocr = ocr
        self.min_px_for_ocr = min_px_for_ocr
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
        self._texts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._boxes: dict[int, tuple[int, int, int, int]] = {}
        self._cache: dict[int, str] = {}
        self.detections = 0

    # --- geometry ---------------------------------------------------------------
    def _find_plate(self, crop: np.ndarray) -> tuple[int, int, int, int] | None:
        """Plate box within a vehicle crop, in crop coordinates."""
        if self.model is not None:
            r = self.model.predict(crop, verbose=False, conf=0.25)[0]
            if len(r.boxes) == 0:
                return None
            # Largest detection: a vehicle carries one plate on the face we see,
            # and the biggest box is the best-resolved candidate.
            b = max(r.boxes.xyxy.cpu().numpy(),
                    key=lambda z: (z[2] - z[0]) * (z[3] - z[1]))
            return tuple(int(v) for v in b)          # type: ignore[return-value]
        return self._find_plate_heuristic(crop)

    @staticmethod
    def _find_plate_heuristic(crop: np.ndarray) -> tuple[int, int, int, int] | None:
        """Fallback with no trained model: bright, wide, low-in-frame rectangle.

        This is a stopgap so the stage is testable before the detector is
        trained. It is genuinely weak — headlights and reflective bumper trim
        also qualify — and is not a substitute for the trained model.
        """
        h, w = crop.shape[:2]
        if h < 40 or w < 40:
            return None
        low = crop[int(h * 0.55):, :]                # plates sit low on the body
        grey = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
        thr = max(int(np.percentile(grey, 97)), 90)
        _, mask = cv2.threshold(grey, thr, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = (stats[i, k] for k in range(5))
            if bw < 12 or bh < 5 or bh > bw:
                continue
            if not (1.6 <= bw / bh <= 6.0):          # plate aspect
                continue
            if best is None or area > best[4]:
                best = (x, y, bw, bh, area)
        if best is None:
            return None
        x, y, bw, bh, _ = best
        return (x, y + int(h * 0.55), x + bw, y + bh + int(h * 0.55))

    # --- evidence ---------------------------------------------------------------
    def observe(self, tid: int, frame: np.ndarray, xyxy) -> None:
        """Record one frame's plate evidence for a tracked vehicle."""
        tid = int(tid)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(max(v, 0)) for v in xyxy)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return
        crop = frame[y1:y2, x1:x2]
        box = self._find_plate(crop)
        if box is None:
            return
        px1, py1, px2, py2 = box
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 4:
            return
        self.detections += 1
        self._widths[tid].append(float(pw))
        self._boxes[tid] = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)

        if pw < MIN_PX_FOR_COLOR:
            return
        plate = crop[py1:py2, px1:px2]
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
        self._cache.pop(tid, None)

        if self.ocr is not None and pw >= self.min_px_for_ocr:
            try:
                text, conf = self.ocr(plate)
            except Exception:                         # pragma: no cover
                return
            if text:
                self._texts[tid][text] += float(conf)

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
        pool = real or votes
        if sum(1 for _ in self._widths.get(tid, [])) < MIN_SAMPLES:
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

    def box_of(self, tid: int):
        return self._boxes.get(int(tid))

    def _ocr_note(self, p90: float) -> str:
        """Explain the OCR outcome, leading with resolution — the binding limit."""
        if not p90:
            return "no plates measured, so OCR feasibility is undetermined"
        if p90 < self.min_px_for_ocr:
            short = self.min_px_for_ocr / max(p90, 1e-6)
            return (
                f"plate p90 {p90:.0f}px is below the {self.min_px_for_ocr:.0f}px "
                f"floor for character recognition (~{short:.1f}x short); this is a "
                "camera-resolution limit, not a model limit — more training data "
                "cannot add pixels the sensor never captured "
                "(see tools/plate_footage_check.py)"
            )
        if self.ocr is None:
            return (f"plate p90 {p90:.0f}px clears the "
                    f"{self.min_px_for_ocr:.0f}px floor, but no OCR engine is "
                    "configured — this footage COULD be read")
        return f"plate p90 {p90:.0f}px >= {self.min_px_for_ocr:.0f}px floor — OCR ran"

    def summary(self, tracks) -> dict:
        """Report for analytics.json, including why OCR did or did not run."""
        rows, widths = [], []
        for tid in tracks:
            c = self.color_of(tid)
            px = self.plate_px(tid)
            if px:
                widths.append(px)
            text, tconf = self.text_of(tid)
            rows.append({
                "track": int(tid),
                "plate_color": c,
                "plate_color_display": COLOR_DISPLAY.get(c, c),
                "supports_classes": list(COLOR_TO_CODES.get(c, ())),
                "plate_px": round(px, 1),
                "plate_text": text,
                "plate_text_confidence": round(tconf, 2),
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
            "plate_px_p90": p90,
            "color_mix": dict(mix),
            "ocr_attempted": self.ocr is not None,
            "ocr_produced_text": ocr_ran,
            # The whole point of the resolution note: an empty OCR column must be
            # attributable, or it reads as a broken model and sends the next two
            # weeks into collecting more training data that cannot help.
            #
            # The resolution verdict is stated whether or not an engine is
            # configured. Reporting only "no engine" when the footage ALSO cannot
            # support OCR hides the fact that actually matters — someone would
            # plug in an engine, get nothing, and still not know why.
            "ocr_note": self._ocr_note(p90),
            "vehicles": rows,
        }
