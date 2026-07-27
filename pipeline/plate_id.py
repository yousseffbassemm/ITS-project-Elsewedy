"""Plate FINGERPRINTING — measured, and NOT fit for deployment. See the verdict.

    +---------------------------------------------------------------------+
    |  VERDICT: this does not work. Do not use it to link vehicles across  |
    |  cameras. Measured false-match rate 53% at this camera's resolution  |
    |  and 32% at 100px, i.e. between one in two and one in three          |
    |  unrelated vehicles is wrongly linked. Kept in the tree because the  |
    |  measurement is the useful artefact, and because the same evaluation |
    |  harness is what any replacement must pass.                          |
    +---------------------------------------------------------------------+

Run `python -m tools.eval_plate_fingerprint` to reproduce.

**Why it was worth trying.** The characters on this camera's plates are
unrecoverable (docs/anpr-plan.md: 11.6px glyphs against a 20px floor), but the
operational goal behind "capture every plate" is usually not the string — it is
*this vehicle, seen here, is the one seen there*: journey times, corridor flow,
origin-destination. That question does not need the number, only a stable
identifier, so a perceptual signature of the plate looked like a way around the
resolution limit.

**Why it failed, and it is not the reason expected.** The first measurement said
0% false matches, which was wrong because the test was circular: both "sightings"
were degraded from the SAME photograph, so the hash was rewarded for keying on
that photograph's exposure, framing and background — none of which transfer to a
second camera. Re-measured with each sighting given its own exposure, gamma,
mounting angle and detector framing, the false-match rate went to 59%.

Three rounds of fixes followed — rank normalisation (invariant to gain, offset
and gamma), re-normalising to the plate's own body instead of the detector box,
and deskewing from the plate's minimum-area rectangle. They moved it to 53% and
no further. The diagnostic that settles it: at NATIVE ~190px resolution, two
sightings of the same plate still differ in ~19 of 79 bits. A hand-crafted hash
of this shape is simply not reproducible across realistic capture variation, at
any resolution — so this is a design failure, not a footage failure, and more
pixels would not rescue it.

**What would work instead.** Hand-crafted perceptual hashes are known to be
brittle to geometry; the established answer for cross-camera matching is a
LEARNED embedding trained with metric learning. And it should be run on the whole
VEHICLE rather than the plate: the vehicle occupies roughly 100x126px here
against the plate's 34x14, about 25x more pixels, and vehicle re-identification
is a standard task with public datasets (VeRi-776, VehicleID) that trains on a
free GPU. That is the path to the mentor's actual goal.

---

Original design notes follow, for anyone extending this.

Plate FINGERPRINTING — matching a vehicle without reading its plate.

The characters on this camera's plates are unrecoverable (docs/anpr-plan.md: 11.6
px glyphs against a 20 px floor). But the operational goal behind "capture every
plate" is usually not the string itself — it is *this vehicle, seen here, is the
one seen there*: journey times, corridor flow, origin-destination, dwell. That
question does not need the number. It needs a stable identifier.

So this module produces one: a compact perceptual signature of the plate that is
consistent across frames of the same plate and different between different
plates. It reads nothing and claims nothing about the registration.

**What survives 34 px, and what does not.** Individual glyph strokes are gone —
that is the whole finding. What remains is coarser and genuinely informative:

  * the colour band (already exploited for vehicle category)
  * the LAYOUT of ink: Egyptian plates put digits on one half and letters on the
    other, and the pattern of dark columns encodes how many glyphs there are,
    roughly how wide, and where the gaps fall. Downscaling blurs strokes together
    but preserves where ink is and is not.
  * low-frequency structure, which is exactly what a DCT perceptual hash keeps
    and what JPEG/H.264 preserve best

The fingerprint is therefore a DCT hash plus an explicit ink-column profile,
which are complementary: the hash captures 2-D structure, the profile captures
the horizontal glyph layout that matters most on a plate.

**This is a WEAK identifier and must be treated as one.** At this resolution the
number of distinguishable states is small, so collisions are not hypothetical.
tools/eval_plate_fingerprint.py measures the actual separation on real Egyptian
plates degraded to deployment resolution, and any deployment must set its
threshold from that measurement rather than from hope. A fingerprint match is
EVIDENCE for "same vehicle", to be combined with time, direction and vehicle
class — never a standalone identification.

**Privacy.** A stable per-vehicle identifier is personal data under Egypt's PDPL
151/2020 even though it is not a readable plate — that is precisely what makes it
useful and what makes it regulated. It is pseudonymous, not anonymous. See
docs/anpr-plan.md §7; agree a retention period before deploying it.
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical plate geometry for hashing. Every crop is resampled to this before
# anything else, so a 30 px and a 54 px view of one plate produce comparable
# signatures. Wider than tall at 8:3, close to a real plate's proportions, and
# small enough that upsampling a 34 px crop invents little.
CANON_W, CANON_H = 64, 24

# DCT hash grid. 8x8 low-frequency coefficients (minus the DC term) is the
# standard pHash size and is about right here: larger grids start encoding
# high-frequency detail this footage does not contain, which is noise.
DCT_GRID = 8

# Ink-profile bins. 16 columns across the plate resolves the glyph groups and the
# gap between the number and letter fields without pretending to resolve
# individual characters.
PROFILE_BINS = 16

FINGERPRINT_BITS = (DCT_GRID * DCT_GRID - 1) + PROFILE_BINS   # 63 + 16 = 79


def _rank_normalise(g: np.ndarray) -> np.ndarray:
    """Replace each pixel by its rank among all pixels, scaled to 0..1.

    Invariant to ANY monotonic intensity transform — gain, offset and gamma
    alike — because ranks are preserved by all of them. This matters more than
    it sounds: two cameras on a corridor have different exposure and gamma, and
    a hash thresholded on raw or merely mean-shifted intensities keys on the
    camera as much as on the plate. Measured, that alone drove the false-match
    rate to ~60%.
    """
    flat = g.ravel()
    order = flat.argsort()
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.arange(flat.size, dtype=np.float32)
    return (ranks / max(flat.size - 1, 1)).reshape(g.shape)


def _deskew(g: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """Level the plate using its own body as the reference.

    Measured to be the single largest source of instability: with everything
    else normalised, a +/-3.5 degree mounting difference between two cameras
    still flipped ~20 of 79 bits on the SAME plate at native resolution, because
    a DCT grid moves content between cells as soon as the image rotates. The
    plate body is a strong, near-rectangular bright region, so its minimum-area
    rectangle gives the angle directly.

    Bounded by ``max_angle``: beyond that the estimate is more likely to be a
    bad fit than a genuinely tilted plate, and rotating on a bad fit is worse
    than leaving it alone.
    """
    thr = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(thr)
    if pts is None or len(pts) < 20:
        return g
    (_c, _wh, angle) = cv2.minAreaRect(pts)
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) > max_angle or abs(angle) < 0.2:
        return g
    h, w = g.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _content_box(g: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the plate's own body, in its crop.

    The detector does not cut the same box twice — it over- and under-shoots by
    a few percent, and on a 34px plate a few percent is a whole character's
    width. Hashing the detector's box therefore hashes the detector's jitter.
    Re-normalising to the plate's BODY (the bright region carrying the ink)
    makes the signature depend on the plate instead.
    """
    h, w = g.shape[:2]
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=max(w / 40.0, 0.8))
    thr = cv2.threshold(blur.astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cols = thr.mean(axis=0)
    rows = thr.mean(axis=1)
    cx = np.where(cols > 0.35 * cols.max())[0] if cols.max() > 0 else []
    ry = np.where(rows > 0.35 * rows.max())[0] if rows.max() > 0 else []
    x0, x1 = (int(cx[0]), int(cx[-1]) + 1) if len(cx) > 4 else (0, w)
    y0, y1 = (int(ry[0]), int(ry[-1]) + 1) if len(ry) > 2 else (0, h)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _canonical(crop: np.ndarray) -> np.ndarray | None:
    """Grayscale, geometry- and intensity-normalised plate.

    Three normalisations, each removing a nuisance variable that a corridor
    deployment genuinely varies and that measurement showed dominating the
    signature:

      * CONTENT BOX  — the detector's framing jitter (see _content_box)
      * RANK         — exposure and gamma (see _rank_normalise)
      * HIGH-PASS    — uneven lighting across the plate's own width, which a
                       global normalisation preserves and a high-pass removes
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if w < 8 or h < 4:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = _deskew(g)
    x0, y0, x1, y1 = _content_box(g)
    g = g[y0:y1, x0:x1]
    if g.size == 0:
        return None
    g = cv2.resize(g, (CANON_W, CANON_H),
                   interpolation=cv2.INTER_AREA if g.shape[1] > CANON_W
                   else cv2.INTER_CUBIC)
    r = _rank_normalise(g.astype(np.float32)) * 255.0
    background = cv2.GaussianBlur(r, (0, 0), sigmaX=CANON_W / 8.0)
    flat = r - background
    sd = float(flat.std())
    return flat / sd if sd > 1e-6 else flat


def _dct_bits(flat: np.ndarray) -> np.ndarray:
    """Low-frequency DCT signature, thresholded at its own median."""
    d = cv2.dct(flat)[:DCT_GRID, :DCT_GRID].flatten()[1:]   # drop DC
    return d > np.median(d)


def _ink_bits(flat: np.ndarray) -> np.ndarray:
    """Where the ink is, horizontally.

    Column-mean darkness, binned and thresholded at the median, which makes the
    profile invariant to how dark the plate happens to be and keeps only the
    PATTERN — glyph groups and the gaps between them.
    """
    col = -flat.mean(axis=0)                       # ink is dark -> negate
    binned = col.reshape(PROFILE_BINS, -1).mean(axis=1)
    return binned > np.median(binned)


def fingerprint(crop: np.ndarray) -> np.ndarray | None:
    """79-bit signature of one plate crop, or None if the crop is unusable."""
    flat = _canonical(crop)
    if flat is None:
        return None
    return np.concatenate([_dct_bits(flat), _ink_bits(flat)])


def fuse(crops: list[np.ndarray]) -> np.ndarray | None:
    """One fingerprint per VEHICLE, by majority vote per bit across its crops.

    The same argument that makes multi-frame fusion work for pixels applies to
    bits: any single 34 px view sets some bits by luck, but a bit that is
    genuinely determined by the plate agrees across views. Voting also means a
    motion-blurred or half-occluded crop cannot decide the identity on its own.
    """
    bits = [b for b in (fingerprint(c) for c in crops) if b is not None]
    if not bits:
        return None
    return np.mean(np.stack(bits).astype(np.float32), axis=0) >= 0.5


def stability(crops: list[np.ndarray]) -> float:
    """How reproducible this vehicle's fingerprint is, in 0..1.

    The mean per-bit agreement across its own crops. Published alongside the
    fingerprint because it is the honest confidence: a vehicle seen once, badly,
    produces a signature that will not match itself next time, and a consumer
    needs to know that before trusting a corridor match.
    """
    bits = [b for b in (fingerprint(c) for c in crops) if b is not None]
    if len(bits) < 2:
        return 0.0
    stack = np.stack(bits).astype(np.float32)
    p = stack.mean(axis=0)
    return float(np.mean(np.maximum(p, 1.0 - p)))


def distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two fingerprints (0 = identical)."""
    return int(np.count_nonzero(np.asarray(a, bool) != np.asarray(b, bool)))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming distance rescaled to 1.0 = identical, 0.0 = every bit differs."""
    return 1.0 - distance(a, b) / float(len(a))


def to_hex(bits: np.ndarray | None) -> str:
    """Compact, CSV- and database-friendly form."""
    if bits is None:
        return ""
    return np.packbits(np.asarray(bits, bool)).tobytes().hex()


def from_hex(text: str) -> np.ndarray | None:
    if not text:
        return None
    raw = np.frombuffer(bytes.fromhex(text), dtype=np.uint8)
    return np.unpackbits(raw)[:FINGERPRINT_BITS].astype(bool)
