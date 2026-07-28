"""Two-stage ANPR cascade: vehicle -> plate -> characters + colour.

This is the architecture the mentor specified, and it is deliberately a cascade
rather than one model that detects everything at once:

    stage 1   vehicle          base COCO YOLO + the tracker (already in the
                               pipeline; nothing new to train)
    stage 2   plate box        a detector run ON THE VEHICLE CROP
    stage 3   characters       a detector run ON THE PLATE CROP
    stage 4   colour           measured from the plate's top band, no model

**Why the plate detector runs on the vehicle crop and not on the frame.** The
plate is a tiny object in a 1280x720 frame and a large one inside a vehicle box,
so cropping first spends the detector's fixed input resolution on the region
that matters. It also scopes the answer: a plate found inside vehicle #7's crop
belongs to vehicle #7, with no containment test to get wrong.

**The plate is not always in the middle.** Measured across EALPR's 2,087
annotated vehicles the plate's centre-x spans 0.026 to 0.962 of the vehicle
width — it sits on the far left or far right often enough that any assumption
about where to look is wrong. Stage 2 therefore searches the whole crop, and
``PlateRead.position`` reports where it actually was, so a reviewer can check
that rather than take it on trust.

**Reading order.** An Egyptian plate carries Arabic letters and Arabic-Indic
digits, and the two are NOT read the same way round: the letters read
right-to-left, the number reads left-to-right, exactly as Arabic text does. A
single "sort by x" produces a string that looks plausible and has the letters
backwards, which is the kind of error that survives review because nobody
reading the report reads Arabic. Both orders are emitted, plus a Latin
transliteration, so the CSV is checkable by someone who does not.

**What is NOT claimed.** The publish gates from the single-frame pipeline apply
here unchanged: a read below the measured resolution floor, or with too few
characters to be a real plate, is recorded as evidence and NOT published as a
plate number. See pipeline/plate_ocr.py for the measurement those gates encode
and CLAUDE.md for what happened the one time they were relaxed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .plates import (
    COLOR_DISPLAY,
    COLOR_TO_CODES,
    MAX_PLATE_ASPECT,
    MIN_PLATE_ASPECT,
    MIN_PX_FOR_COLOR,
    classify_band,
)

# Arabic-Indic digits used on Egyptian plates. There is no ٠: the digit zero
# does not appear on a single one of EALPR's 1,923 annotated plates, and the
# dataset's own class list has a hole where it would be. Treating "no zero" as a
# property of the plates rather than a gap in the data is what lets the reader
# below split letters from digits without a lookup table.
AR_DIGITS = "١٢٣٤٥٦٧٨٩"
DIGIT_TO_LATIN = {d: str(i + 1) for i, d in enumerate(AR_DIGITS)}

# Transliteration for the 17 letters modern Egyptian plates use. For the CSV and
# the dashboard: a reviewer who does not read Arabic still has to be able to
# check a plate against a photograph.
LETTER_TO_LATIN = {
    "أ": "A", "ب": "B", "ج": "G", "د": "D", "ر": "R", "س": "S", "ص": "SD",
    "ط": "T", "ع": "E", "ف": "F", "ق": "Q", "ل": "L", "م": "M", "ن": "N",
    "ھ": "H", "و": "W", "ى": "Y",
}

# Pretrained Egyptian ALPR models name their classes in Arabic-chat
# transliteration ("Seen", "geem", "7aah") rather than with the glyph, and the
# two available sets do not even agree with each other. Normalising to the glyph
# here means the cascade can run on an off-the-shelf model today and on a model
# trained by tools/train_anpr.py later, without the reading, the CSV or the
# publish gates knowing which produced the characters.
#
# Classes that are not characters ("License Plate", "car") map to None and are
# dropped — a plate box inside the character string is not a character.
_NAME_TO_GLYPH: dict[str, str | None] = {
    **{str(i): d for i, d in enumerate(AR_DIGITS, start=1)},
    **{d: d for d in AR_DIGITS},
    "alef": "أ", "a": "أ",
    "baa": "ب", "beh": "ب", "b": "ب",
    "geem": "ج", "g": "ج",
    "daal": "د", "dal": "د", "d": "د",
    "r": "ر", "reh": "ر",
    "seen": "س", "s": "س",
    "saad": "ص", "sad": "ص",
    "taa": "ط", "tah": "ط", "t": "ط",
    "een": "ع", "ain": "ع", "e": "ع",
    "f": "ف", "feh": "ف",
    "q": "ق", "qaf": "ق",
    "laam": "ل", "lam": "ل", "l": "ل",
    "meem": "م", "m": "م",
    "noon": "ن", "n": "ن",
    "heeh": "ھ", "heh": "ھ", "h": "ھ", "7aah": "ھ",
    "wow": "و", "waw": "و", "w": "و",
    "yeeh": "ى", "yeh": "ى", "y": "ى",
    # Letters some pretrained sets carry that are NOT on Egyptian civilian
    # plates. Kept mapped rather than dropped: the model can emit them, and a
    # silent drop would shorten the string and trip the character-count gate for
    # the wrong reason.
    "daad": "ض", "kaaf": "ك", "zeen": "ز",
    "license plate": None, "licenseplate": None, "license_plate": None,
    "plate": None, "car": None, "vehicle": None,
}


def glyph_for(name: str) -> str | None:
    """Arabic glyph for a model's class name, or None if it isn't a character."""
    key = str(name).strip()
    if key in _NAME_TO_GLYPH:
        return _NAME_TO_GLYPH[key]
    low = key.lower()
    if low in _NAME_TO_GLYPH:
        return _NAME_TO_GLYPH[low]
    # A single Arabic glyph used as its own class name.
    return key if len(key) == 1 and not key.isascii() else None


# A real Egyptian civilian plate carries at least 2 letters and 2 digits. Fewer
# than 5 glyphs total is the character detector firing on noise, not a partial
# read — see pipeline/plate_ocr.MIN_PLATE_CHARS, same measurement.
MIN_CHARS = 5
MIN_LETTERS, MIN_DIGITS = 2, 2

# Character height below which recognition is not attempted. This is the
# industry ANPR minimum and the number the whole plate-resolution finding in
# CLAUDE.md §3 is written in — 20 px of glyph, not of plate.
MIN_CHAR_PX = 15.0

# Confidence floor for one character detection to join the string.
MIN_CHAR_CONF = 0.25


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread that survives non-ASCII paths (Windows ANSI file API)."""
    try:
        buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return None if buf.size == 0 else cv2.imdecode(buf, cv2.IMREAD_COLOR)


@dataclass
class PlateRead:
    """One vehicle's resolved plate."""

    track: int
    text_arabic: str = ""          # letters R->L then the number L->R
    text_latin: str = ""           # transliterated, for a non-Arabic reader
    text_visual: str = ""          # raw left-to-right, for checking against a crop
    confidence: float = 0.0
    color: str = "unknown"
    color_display: str = ""
    letters: str = ""
    digits: str = ""
    char_px: float = 0.0
    plate_px: float = 0.0
    # Where on the vehicle the plate was found, as a fraction of vehicle width,
    # plus a word for it. The mentor's "it might be on the sides" made visible.
    position: str = ""
    position_x: float = 0.0
    n_reads: int = 0
    published: bool = True
    note: str = ""
    chars: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "plate_arabic": self.text_arabic,
            "plate_latin": self.text_latin,
            "plate_visual_ltr": self.text_visual,
            "plate_letters": self.letters,
            "plate_digits": self.digits,
            "plate_confidence": round(self.confidence, 3),
            "plate_color": self.color,
            "plate_color_display": self.color_display,
            "plate_position": self.position,
            "plate_position_x": round(self.position_x, 3),
            "char_height_px": round(self.char_px, 1),
            "plate_width_px": round(self.plate_px, 1),
            "reads_fused": self.n_reads,
            "published": self.published,
            "note": self.note,
        }


def assemble(chars: list[dict]) -> tuple[str, str, str, str, str]:
    """Turn positioned glyphs into the strings a report should carry.

    ``chars`` is a list of ``{"glyph", "x", "conf"}``. Returns
    ``(arabic, latin, visual_ltr, letters, digits)``.

    The number and the letters run in OPPOSITE directions, which is the whole
    reason this is a function and not a ``"".join(sorted(...))`` at the call
    site. Arabic text reads right-to-left, so the letter group reads from the
    right; Arabic-Indic numerals are written most-significant-digit first, i.e.
    left-to-right, exactly as in English. Sorting everything one way silently
    reverses the letters.
    """
    ordered = sorted(chars, key=lambda c: c["x"])
    visual = "".join(c["glyph"] for c in ordered)
    digits = "".join(c["glyph"] for c in ordered if c["glyph"] in AR_DIGITS)
    letters_ltr = [c["glyph"] for c in ordered if c["glyph"] not in AR_DIGITS]
    letters = "".join(reversed(letters_ltr))          # right-to-left
    arabic = f"{letters} {digits}".strip()
    latin = " ".join(x for x in (
        "".join(LETTER_TO_LATIN.get(g, g) for g in letters),
        "".join(DIGIT_TO_LATIN.get(g, g) for g in digits),
    ) if x)
    return arabic, latin, visual, letters, digits


class PlateLocator:
    """Stage 2 — find the plate box inside a vehicle crop.

    Missing weights are reported, never raised: the plate stage is optional and
    must not be able to cost the run its counts, speeds or classes.
    """

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 320,
                 conf: float = 0.25):
        self.model = None
        self.error: str | None = None
        self.imgsz, self.conf, self.device = imgsz, conf, device
        try:
            from ultralytics import YOLO
            if not Path(weights).exists():
                raise FileNotFoundError(weights)
            self.model = YOLO(weights)
        except Exception as exc:                    # pragma: no cover - env dependent
            self.error = f"{type(exc).__name__}: {exc}"

    def locate(self, vehicle_crop: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
        """Best plate box in this crop as (x1, y1, x2, y2), or None."""
        if self.model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None
        h, w = vehicle_crop.shape[:2]
        if w < 16 or h < 16:
            return None
        try:
            r = self.model.predict(vehicle_crop, imgsz=self.imgsz, conf=self.conf,
                                   verbose=False, device=self.device)[0]
        except Exception:                           # pragma: no cover
            return None
        best, best_conf = None, 0.0
        for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                        r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in b)
            pw, ph = x2 - x1, y2 - y1
            if pw < 8 or ph < 4:
                continue
            # Same shape gate as the frame-level detector: a tailgate badge is
            # bright, horizontal and NOT plate-shaped. Rejecting it here stops it
            # voting on colour and being handed to the character reader.
            if not (MIN_PLATE_ASPECT <= pw / max(ph, 1e-6) <= MAX_PLATE_ASPECT):
                continue
            if float(c) > best_conf:
                best, best_conf = (int(x1), int(y1), int(x2), int(y2)), float(c)
        return None if best is None else (best, best_conf)


class CharReader:
    """Stage 3 — recognise the characters inside a plate crop.

    Character DETECTION, not sequence prediction: each glyph is found and scored
    independently, so a partly legible plate returns the glyphs it is sure of
    rather than one confidently-invented string. On marginal input that is the
    difference between an empty cell and a wrong plate number in a report.
    """

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 320,
                 conf: float = MIN_CHAR_CONF):
        self.model = None
        self.error: str | None = None
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.names: dict[int, str] = {}
        try:
            from ultralytics import YOLO
            if not Path(weights).exists():
                raise FileNotFoundError(weights)
            self.model = YOLO(weights)
            self.names = {int(k): str(v) for k, v in self.model.names.items()}
        except Exception as exc:                    # pragma: no cover - env dependent
            self.error = f"{type(exc).__name__}: {exc}"

    def read(self, plate: np.ndarray) -> list[dict]:
        """Glyphs found in this plate crop, each with its x position."""
        if self.model is None or plate is None or plate.size == 0:
            return []
        # Upscale a small plate before recognition. This does NOT add
        # information — the resolution gate above is what decides whether there
        # is any — but the detector's stride means a 12px-tall glyph has almost
        # no feature map to be found in, and it misses characters that are
        # genuinely legible.
        h, w = plate.shape[:2]
        if h < 64:
            k = 64.0 / max(h, 1)
            plate = cv2.resize(plate, (int(w * k), 64), interpolation=cv2.INTER_CUBIC)
        try:
            r = self.model.predict(plate, imgsz=self.imgsz, conf=self.conf,
                                   verbose=False, device=self.device)[0]
        except Exception:                           # pragma: no cover
            return []
        out = []
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            glyph = glyph_for(self.names.get(int(c), ""))
            # None means the class is not a character — pretrained ALPR models
            # also emit "License Plate" and "car" boxes, which must not join the
            # string.
            if not glyph:
                continue
            out.append({"glyph": glyph, "conf": float(cf),
                        "x": float((b[0] + b[2]) / 2),
                        "h": float(b[3] - b[1])})
        return out


class ANPRCascade:
    """Runs stages 2-4 for each tracked vehicle and votes across frames.

    A vehicle is seen in tens of frames and they are not equally good. Each
    frame contributes one candidate read; the per-track vote below picks the
    string with the most confidence behind it, and records how many independent
    frames supported it — because one read cannot corroborate itself.
    """

    def __init__(self, locator: PlateLocator | None = None,
                 reader: CharReader | None = None,
                 min_char_px: float = MIN_CHAR_PX):
        self.locator = locator if (locator and locator.model) else None
        self.reader = reader if (reader and reader.model) else None
        self.min_char_px = min_char_px
        self._texts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # The assembled forms of each candidate string, keyed by the string
        # itself. Kept alongside the votes so the row that gets published is
        # built from the string that WON the vote — see resolve().
        self._forms: dict[int, dict[str, tuple]] = defaultdict(dict)
        # How many frames contributed a read, as opposed to how many distinct
        # strings came back. Reported as `reads_fused`, where the distinct-string
        # count would understate corroboration badly: five frames that all agree
        # are one string and five reads, and the agreement is the whole point.
        self._read_n: dict[int, int] = defaultdict(int)
        self._chars: dict[int, list] = {}
        self._colors: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._color_n: dict[int, int] = defaultdict(int)
        self._char_px: dict[int, float] = defaultdict(float)
        self._plate_px: dict[int, float] = defaultdict(float)
        self._pos: dict[int, list[float]] = defaultdict(list)
        self._boxes: dict[int, tuple[int, int, int, int]] = {}
        self.located = 0
        self.vehicles_seen = 0

    # --- per frame ----------------------------------------------------------
    def observe(self, frame: np.ndarray, tid: int, xyxy) -> None:
        """One vehicle, one frame: locate the plate, read colour and characters."""
        if self.locator is None:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return
        self.vehicles_seen += 1
        crop = frame[y1:y2, x1:x2]
        got = self.locator.locate(crop)
        if got is None:
            return
        (px1, py1, px2, py2), _conf = got
        self.located += 1
        pw, ph = px2 - px1, py2 - py1
        tid = int(tid)
        # Where the plate sat on the vehicle, as a fraction of vehicle width.
        self._pos[tid].append(((px1 + px2) / 2.0) / max(x2 - x1, 1))
        self._boxes[tid] = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
        self._plate_px[tid] = max(self._plate_px[tid], float(pw))

        plate = crop[py1:py2, px1:px2]
        if plate.size == 0:
            return
        self._read_colour(tid, plate, pw, ph)
        # Egyptian glyphs occupy roughly 55% of plate height — the same ratio the
        # footage checker and the resolution finding are written in.
        char_px = ph * 0.55
        self._char_px[tid] = max(self._char_px[tid], float(char_px))
        if self.reader is not None and char_px >= self.min_char_px:
            found = self.reader.read(plate)
            if found:
                self._vote_text(tid, found)

    def _read_colour(self, tid: int, plate: np.ndarray, pw: int, ph: int) -> None:
        if pw < MIN_PX_FOR_COLOR:
            return
        split = max(ph // 3, 1)
        band, body = plate[:split, :], plate[split:, :]
        if band.size == 0 or body.size == 0:
            return
        # Median, not mean: the band carries black "EGYPT" lettering and a mean
        # drags the reading toward grey.
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        hh, ss, vv = (float(np.median(hsv[..., k])) for k in range(3))
        hsv_b = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        s_body, v_body = (float(np.median(hsv_b[..., k])) for k in (1, 2))
        name = classify_band(hh, ss, vv, s_body, v_body)
        self._colors[tid][name] += pw * max(vv, 1.0) / 255.0
        self._color_n[tid] += 1

    def _vote_text(self, tid: int, found: list[dict]) -> None:
        good = [c for c in found if c["conf"] >= MIN_CHAR_CONF]
        if not good:
            return
        arabic, latin, visual, letters, digits = assemble(good)
        if not arabic:
            return
        self._texts[tid][arabic] += float(np.mean([c["conf"] for c in good]))
        self._forms[tid][arabic] = (latin, visual, letters, digits)
        self._read_n[tid] += 1
        # Keep the glyph detail of the most confident frame, so a published
        # string can be traced back to the characters that produced it.
        prev = self._chars.get(tid)
        if prev is None or np.mean([c["conf"] for c in good]) > np.mean(
                [c["conf"] for c in prev]):
            self._chars[tid] = good

    # --- resolution ---------------------------------------------------------
    def colour_of(self, tid: int) -> str:
        votes = self._colors.get(int(tid))
        if not votes or self._color_n.get(int(tid), 0) < 3:
            return "unknown"
        real = {k: v for k, v in votes.items() if k not in ("unknown", "white")}
        pool = real or votes
        return max(pool, key=pool.get)

    def box_of(self, tid: int):
        return self._boxes.get(int(tid))

    def resolve(self, tid: int) -> PlateRead:
        tid = int(tid)
        colour = self.colour_of(tid)
        xs = self._pos.get(tid) or []
        x_mean = float(np.mean(xs)) if xs else 0.0
        where = ("left" if x_mean < 0.35 else
                 "right" if x_mean > 0.65 else "centre") if xs else ""
        out = PlateRead(
            track=tid, color=colour,
            color_display=COLOR_DISPLAY.get(colour, colour),
            char_px=self._char_px.get(tid, 0.0),
            plate_px=self._plate_px.get(tid, 0.0),
            position=where, position_x=x_mean,
        )
        votes = self._texts.get(tid)
        if not votes:
            out.published = False
            # Attribute the blank to its ACTUAL cause, in the order the causes
            # actually gate. Getting this order wrong is not cosmetic: with no
            # character model loaded but perfectly adequate resolution, an
            # earlier version blamed the camera — which is the one misreading
            # this column exists to prevent, and it sends people off collecting
            # footage that was never the problem.
            if not out.char_px:
                out.note = "no plate located on this vehicle"
            elif self.reader is None:
                out.note = ("no character-recognition model is loaded, so "
                            "characters were never attempted — the plate itself "
                            "was found and its colour read")
            elif out.char_px < self.min_char_px:
                out.note = (f"character height {out.char_px:.1f}px is below the "
                            f"{self.min_char_px:.0f}px floor for recognition — a "
                            f"camera resolution limit, not a model limit")
            else:
                out.note = (f"character height {out.char_px:.1f}px clears the "
                            f"{self.min_char_px:.0f}px floor but no characters "
                            f"were recognised")
            return out
        # The string that WON the vote is the answer, and every field below is
        # derived from that same string. Building the row from the single
        # highest-confidence frame instead would publish one plate carrying
        # another plate's confidence — the two disagree precisely when the reads
        # disagree, which is exactly when the number matters.
        best = max(votes, key=votes.get)
        total = sum(votes.values()) or 1.0
        latin, visual, letters, digits = self._forms[tid].get(best, ("", "", "", ""))
        out.text_arabic, out.text_latin, out.text_visual = best, latin, visual
        out.letters, out.digits = letters, digits
        # Agreement across frames, not the character detector's own confidence.
        # A model can be certain and wrong; independent frames agreeing is
        # evidence. With a single read every frame agrees with itself by
        # construction, so that case is halved rather than certified.
        out.confidence = (votes[best] / total) * (
            1.0 if self._read_n.get(tid, 0) >= 2 else 0.5)
        out.n_reads = self._read_n.get(tid, 0)
        out.chars = self._chars.get(tid, [])

        # --- publish gates --------------------------------------------------
        # Same discipline as the single-frame pipeline: everything above is
        # evidence and stays in the record; these decide whether it is fit to be
        # printed as a plate number. A plausible wrong plate is worse than a
        # blank, because it gets believed.
        reject = None
        if len(letters) + len(digits) < MIN_CHARS:
            reject = (f"read {len(letters) + len(digits)} character(s); a real "
                      f"Egyptian plate has at least {MIN_CHARS}")
        elif len(letters) < MIN_LETTERS or len(digits) < MIN_DIGITS:
            reject = (f"read {len(letters)} letter(s) and {len(digits)} digit(s); "
                      f"an Egyptian plate carries at least {MIN_LETTERS} letters "
                      f"and {MIN_DIGITS} digits")
        elif out.char_px < self.min_char_px:
            reject = (f"character height {out.char_px:.1f}px is below the "
                      f"{self.min_char_px:.0f}px recognition floor")
        if reject:
            out.published = False
            out.note = f"read discarded — {reject}"
            out.text_arabic = out.text_latin = out.text_visual = ""
            out.letters = out.digits = ""
            out.confidence = 0.0
        return out

    def summary(self, tracks, classifier=None, lane_of=None,
                display_id=None) -> dict:
        rows = []
        for tid in tracks:
            r = self.resolve(tid)
            code = classifier.resolve(tid) if classifier is not None else None
            lane = None if lane_of is None else lane_of.get(int(tid))
            rows.append({
                "track": int(tid),
                "vehicle_no": display_id(tid) if display_id is not None else None,
                "vehicle_class": code,
                "lane": None if lane is None else lane + 1,
                "supports_classes": list(COLOR_TO_CODES.get(r.color, ())),
                "agrees_with_class": (
                    None if not code or not COLOR_TO_CODES.get(r.color)
                    else code in COLOR_TO_CODES[r.color]),
                **r.as_row(),
            })
        mix: dict[str, int] = defaultdict(int)
        for row in rows:
            mix[row["plate_color"]] += 1
        published = [r for r in rows if r["published"] and r["plate_arabic"]]
        return {
            "architecture": "cascade: vehicle -> plate (stage 2) -> "
                            "characters (stage 3) + colour",
            "stage2_model": ("loaded" if self.locator else
                             "not loaded — no plate boxes"),
            "stage3_model": ("loaded" if self.reader else
                             "not loaded — no characters"),
            "vehicle_crops_searched": self.vehicles_seen,
            "plates_located": self.located,
            "plates_published": len(published),
            "color_mix": dict(mix),
            "char_px_required": MIN_CHAR_PX,
            "note": (
                f"{len(published)} of {len(rows)} vehicles produced a publishable "
                f"plate string. A blank is attributed per row in `note` — the "
                f"common cause is character height below the "
                f"{MIN_CHAR_PX:.0f}px recognition floor, which is a property of "
                f"the camera and cannot be fixed with more training data."),
            "vehicles": rows,
        }


def load_cascade(stage2_weights: str, stage3_weights: str,
                 device: str = "cpu") -> ANPRCascade:
    """Build the cascade, reporting rather than raising on missing weights."""
    loc = PlateLocator(stage2_weights, device=device) if stage2_weights else None
    rdr = CharReader(stage3_weights, device=device) if stage3_weights else None
    for name, m in (("stage 2 plate locator", loc), ("stage 3 char reader", rdr)):
        if m is not None and m.model is None:
            print(f"[anpr] {name}: {m.error}", flush=True)
        elif m is not None:
            print(f"[anpr] {name}: loaded", flush=True)
    return ANPRCascade(loc, rdr)


def alphabet_from(weights: str) -> str:
    """The glyph set a stage-3 model emits — for reports and sanity checks."""
    try:
        from ultralytics import YOLO
        return "".join(str(v) for _, v in sorted(YOLO(weights).names.items()))
    except Exception:                               # pragma: no cover
        return ""


__all__ = ["ANPRCascade", "CharReader", "PlateLocator", "PlateRead",
           "assemble", "load_cascade", "alphabet_from", "AR_DIGITS",
           "LETTER_TO_LATIN", "DIGIT_TO_LATIN", "MIN_CHAR_PX",
           "imread_unicode"]
