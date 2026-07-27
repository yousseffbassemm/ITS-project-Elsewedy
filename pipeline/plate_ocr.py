"""Best-frame selection, super-resolution and multi-read voting for plate OCR.

A tracked vehicle is seen in tens of frames, and those frames are NOT equally
useful: most are motion-blurred, distant, or caught mid-occlusion. Reading the
first plate crop that comes along throws away that choice. This module makes the
choice explicit — score every crop, keep only the best few, spend the expensive
super-resolution pass on those, then combine several independent reads instead of
trusting one.

    crops of one vehicle
      -> score (detector confidence x sharpness x size)
      -> keep top N
      -> super-resolve (Real-ESRGAN x4)
      -> OCR each
      -> confidence-weighted vote
      -> one plate string + a confidence

Every stage writes its image to disk (``debug_dir``) so a wrong answer can be
traced to the crop that caused it rather than guessed at.

**What this can and cannot do.** Combining several 34 px views does recover real
detail — the sub-pixel offsets between frames carry information a single frame
does not, which is why forensic ANPR does this. What it cannot do is invent
strokes that no frame captured. Super-resolution in particular will always return
a sharp-looking image: that is what it is trained to do, and a confident-looking
output on unreadable input is a hallucination, not a read. Treat the vote's
agreement across independent crops as the evidence, not the sharpness of the
upscaled picture.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .plates import MIN_PX_FOR_FUSED_OCR

# --- crop quality -----------------------------------------------------------------
# Weights for the three quality terms. Sharpness dominates because a blurred plate
# is unrecoverable at any size, while a slightly smaller but sharp crop often
# reads fine. Size matters next; detector confidence least, because it measures
# "is this a plate", not "is this plate readable".
W_SHARP, W_SIZE, W_CONF = 0.5, 0.35, 0.15

# An Egyptian civilian plate carries 3-4 Arabic-Indic digits AND 2-3 Arabic
# letters. Fewer than five glyphs is not a partial read of a real plate, it is
# the character detector firing on noise — so a short string is rejected outright
# rather than published as a partial answer. Measured case: a 38px plate emitted
# a confident two-character "٣F" that corresponded to nothing on the vehicle.
MIN_PLATE_CHARS = 5

# Crops narrower than this fraction of the best one are excluded from fusion.
# 0.7 keeps the frames either side of the closest approach, where the plate is
# genuinely a similar size, and drops the distant views that only add blur.
FUSE_SCALE_TOLERANCE = 0.7


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard focus measure.

    High when edges are crisp, near zero on a smooth blur. Computed on the
    grayscale crop, and NOT normalised by size here: that is handled separately
    so the two signals stay independent and inspectable.
    """
    if img is None or img.size == 0:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def crop_score(conf: float, sharp: float, width: float,
               sharp_ref: float = 300.0, width_ref: float = 60.0) -> float:
    """Single comparable quality score for one plate crop.

    Each term is squashed into 0..1 before weighting, so one runaway value (a
    specular highlight can send Laplacian variance into the thousands) cannot
    dominate the ranking. The reference values are the point of diminishing
    returns, not a maximum.
    """
    s = math.tanh(max(sharp, 0.0) / sharp_ref)
    z = math.tanh(max(width, 0.0) / width_ref)
    c = min(max(conf, 0.0), 1.0)
    return W_SHARP * s + W_SIZE * z + W_CONF * c


# --- super-resolution -------------------------------------------------------------
def _build_rrdbnet(num_block: int = 23, nf: int = 64, gc: int = 32):
    """RRDBNet (ESRGAN / Real-ESRGAN x4), defined inline.

    Written out here rather than pulled from ``basicsr`` on purpose: basicsr
    pins old torchvision internals and breaks against current torch, and this
    project only needs the forward pass. ~60 lines beats a dependency that
    fights the rest of the environment.
    """
    import torch
    import torch.nn as nn

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(),
                                               ResidualDenseBlock(),
                                               ResidualDenseBlock())

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(num_block)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


class SuperResolver:
    """Real-ESRGAN x4 upscaler, with an honest fallback.

    If the weights are missing the class does NOT silently pass the image
    through — it reports that it is interpolating, because "super-resolved" and
    "resized with cubic interpolation" are very different claims about a result.
    """

    def __init__(self, weights: str = "models/RealESRGAN_x4.pth",
                 device: str = "cpu", tile: int = 0):
        self.scale = 4
        self.model = None
        self.mode = "bicubic (no weights)"
        self.device = device
        self.tile = tile
        path = Path(weights)
        if not path.exists():
            self.error = f"weights not found: {path}"
            return
        try:
            import torch
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
            sd = sd.get("params_ema") or sd.get("params") or sd
            net = _build_rrdbnet()
            net.load_state_dict(sd, strict=True)
            net.eval().to(device)
            self.model = net
            self.mode = "Real-ESRGAN x4"
            self.error = None
        except Exception as exc:                       # pragma: no cover
            self.error = f"{type(exc).__name__}: {exc}"

    def upscale(self, img: np.ndarray) -> np.ndarray:
        if self.model is None:
            h, w = img.shape[:2]
            return cv2.resize(img, (w * self.scale, h * self.scale),
                              interpolation=cv2.INTER_CUBIC)
        import torch
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(t)
        y = y.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return cv2.cvtColor((y * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# --- OCR --------------------------------------------------------------------------
# Classes in the pretrained Egyptian ALPR models that are not characters.
_NON_CHAR = {"license plate", "car", "plate"}

# Arabic-Indic rendering for the digit classes, so the output looks like the
# plate rather than like ASCII.
_AR_DIGIT = {"0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
             "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"}


class EgyptianPlateOCR:
    """Character-detection OCR using a pretrained Egyptian ALPR model.

    Character DETECTION rather than sequence recognition: a partly readable
    plate returns the glyphs it is confident about instead of one confidently
    wrong string, which is the right failure mode when the input is marginal.
    """

    def __init__(self, weights: str = "models/eg_alpr.pt", conf: float = 0.25,
                 imgsz: int = 320, device: str = "cpu"):
        self.model = None
        self.error = None
        self.conf, self.imgsz, self.device = conf, imgsz, device
        try:
            from ultralytics import YOLO
            if not Path(weights).exists():
                raise FileNotFoundError(weights)
            self.model = YOLO(weights)
            self.names = {int(k): str(v) for k, v in self.model.names.items()}
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def read(self, plate: np.ndarray) -> tuple[str, float, list]:
        """Return (text, mean confidence, per-character details)."""
        if self.model is None or plate is None or plate.size == 0:
            return "", 0.0, []
        r = self.model.predict(plate, imgsz=self.imgsz, conf=self.conf,
                               verbose=False, device=self.device)[0]
        chars = []
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            name = self.names.get(int(c), "?")
            if name.strip().lower() in _NON_CHAR:
                continue
            chars.append({"name": name, "conf": float(cf),
                          "x": float((b[0] + b[2]) / 2),
                          "w": float(b[2] - b[0])})
        if not chars:
            return "", 0.0, []
        # Sort by x: this is VISUAL left-to-right order, not Arabic reading
        # order. Kept visual on purpose — it is what an inspector comparing the
        # string against the saved crop expects to see.
        chars.sort(key=lambda d: d["x"])
        text = "".join(_AR_DIGIT.get(d["name"], d["name"][:1].upper()) for d in chars)
        return text, float(np.mean([d["conf"] for d in chars])), chars


# --- voting -----------------------------------------------------------------------
def vote(reads: list[tuple[str, float]]) -> tuple[str, float, dict]:
    """Combine several independent reads of ONE plate into a single answer.

    Two stages, because plate reads disagree in two different ways:

      1. LENGTH. Crops that resolved a different number of characters are not
         comparable position-by-position, so the best-supported length wins
         first (weighted by confidence, not by count — three blurry agreeing
         reads should not outvote one clear one).
      2. POSITION. Within that length, each character position is voted
         separately, so a plate can be right in five positions and uncertain in
         the sixth rather than being discarded whole.

    The returned confidence is the mean per-position agreement, scaled by how
    many independent reads actually SUPPORT the answer. Agreement alone is
    deliberately harsher than the OCR's own confidence — a model can be certain
    and wrong, but independent crops agreeing is real evidence — except in the
    one case where it is not evidence at all: with a single read, every position
    agrees with itself and the mean is 1.0 by construction.

    That is not hypothetical. On a 38px plate in street_egypt.mp4 all three crop
    reads came back empty and only the fused image produced anything; being the
    lone read, it was published at **confidence 1.0**. Corroboration is now part
    of the number, so an uncorroborated read can never present as certain.
    """
    reads = [(t, c) for t, c in reads if t]
    if not reads:
        return "", 0.0, {"reads": 0}
    by_len: dict[int, float] = defaultdict(float)
    for t, c in reads:
        by_len[len(t)] += c
    # Break an exact tie on the LONGER read rather than on whichever crop was
    # scored first. Ties are not hypothetical with two crops of equal
    # confidence, and dict-insertion order made the answer depend on frame
    # ordering — the same plate could vote differently across runs.
    best_len = max(by_len, key=lambda n: (by_len[n], n))
    same = [(t, c) for t, c in reads if len(t) == best_len]

    out, agreements = [], []
    for i in range(best_len):
        tally: dict[str, float] = defaultdict(float)
        for t, c in same:
            tally[t[i]] += c
        ch = max(tally, key=lambda k: (tally[k], k))   # deterministic on a tie
        total = sum(tally.values()) or 1.0
        out.append(ch)
        agreements.append(tally[ch] / total)
    # One read cannot corroborate itself; two is the minimum that means
    # anything. Halving rather than zeroing keeps a lone read usable on footage
    # where a single crop is genuinely enough, while making it impossible for it
    # to outrank a cross-crop agreement.
    corroboration = 1.0 if len(same) >= 2 else 0.5
    return ("".join(out), float(np.mean(agreements)) * corroboration,
            {"reads": len(reads), "reads_at_best_length": len(same),
             "corroboration": corroboration,
             "length_votes": {int(k): round(v, 2) for k, v in by_len.items()},
             "per_position_agreement": [round(a, 2) for a in agreements]})


# --- orchestration ----------------------------------------------------------------
class PlateEnhancer:
    """Per-track: keep the best crops, super-resolve them, OCR, and vote."""

    def __init__(self, sr: SuperResolver | None = None,
                 ocr: EgyptianPlateOCR | None = None,
                 keep: int = 3, debug_dir: str | None = None,
                 min_px: float = 0.0):
        self.sr = sr
        self.ocr = ocr
        self.keep = keep
        self.min_px = min_px
        self.debug = Path(debug_dir) if debug_dir else None
        if self.debug:
            self.debug.mkdir(parents=True, exist_ok=True)
        self._best: dict[int, list] = defaultdict(list)
        self._seq = 0

    def offer(self, tid: int, plate_img: np.ndarray, conf: float) -> None:
        """Consider one plate crop for a track, keeping only the best `keep`."""
        if plate_img is None or plate_img.size == 0:
            return
        h, w = plate_img.shape[:2]
        if w < 8 or h < 4:
            return
        sharp = sharpness(plate_img)
        sc = crop_score(conf, sharp, w)
        keep = self._best[int(tid)]
        keep.append({"score": sc, "img": plate_img.copy(), "conf": float(conf),
                     "sharp": sharp, "w": w, "h": h,
                     "seq": self._seq})
        self._seq += 1
        keep.sort(key=lambda d: -d["score"])
        del keep[self.keep:]

    @staticmethod
    def _fuse(crops: list, up: int = 3):
        """Multi-frame fusion: align the kept crops at sub-pixel and combine.

        This is the one enhancement with real information-theoretic backing.
        Each frame of a moving vehicle samples the plate at a slightly different
        sub-pixel offset, so aligning several onto a finer grid and robustly
        combining them recovers genuine detail that no single frame holds — the
        opposite of super-resolution, which invents it.

        Measured on 50 real Egyptian plates degraded to a fixed width, 12-frame
        fusion beat a single frame at every size: at 65px it lifted character
        accuracy 78% -> 93% and fully-read plates 0 -> 30/50. It does NOT rescue
        34px footage (12% char accuracy, 0 plates), because there is too little
        in each frame to align — see docs/anpr-plan.md.

        Returns a fused BGR image (upscaled by ``up``), or None if there are too
        few crops to fuse.
        """
        usable = [c for c in crops if c["img"] is not None]
        if len(usable) < 2:
            return None
        # Fuse only crops of COMPARABLE scale. Multi-frame fusion combines
        # several samples of the same signal; a crop half the width of the best
        # one carries a quarter of the information, and upscaling it to match
        # contributes interpolated mush that the median then mixes into the
        # result. Measured: vehicle #16's crops were 94, 36 and 33px wide (the
        # 94 being a misdetected tailgate badge), and fusing them degraded the
        # best view instead of improving it.
        widest = max(c["w"] for c in usable)
        usable = [c for c in usable if c["w"] >= FUSE_SCALE_TOLERANCE * widest]
        if len(usable) < 2:
            return None
        imgs = [c["img"] for c in usable]
        h = max(i.shape[0] for i in imgs)
        w = max(i.shape[1] for i in imgs)
        big = [cv2.resize(i, (w * up, h * up), interpolation=cv2.INTER_CUBIC)
               for i in imgs]
        ref = cv2.cvtColor(big[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        aligned = [big[0]]
        for f in big[1:]:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    ref, g, warp, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
                    None, 5)
                aligned.append(cv2.warpAffine(
                    f, warp, (f.shape[1], f.shape[0]),
                    flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP))
            except cv2.error:
                aligned.append(f)                 # unalignable frame — use as-is
        # Median across aligned frames: robust to a bad frame in a way the mean
        # is not.
        return np.median(np.stack(aligned), axis=0).astype(np.uint8)

    def resolve(self, tid: int) -> dict:
        """Run SR + OCR + voting for one track and return the final answer."""
        tid = int(tid)
        crops = self._best.get(tid) or []
        info = {
            "track": tid,
            "crops_kept": len(crops),
            "crop_scores": [round(c["score"], 3) for c in crops],
            "crop_widths": [c["w"] for c in crops],
            "crop_sharpness": [round(c["sharp"], 1) for c in crops],
            "sr_mode": self.sr.mode if self.sr else "disabled",
            "reads": [],
            "plate_text": "",
            "plate_confidence": 0.0,
        }
        if not crops:
            info["note"] = "no plate crops for this track"
            return info
        if self.min_px and max(c["w"] for c in crops) < self.min_px:
            info["note"] = (f"best crop {max(c['w'] for c in crops)}px is below the "
                            f"{self.min_px:.0f}px OCR floor — not attempted")
            return info
        if self.ocr is None or self.ocr.model is None:
            info["note"] = "no OCR engine"
            return info

        d = self.debug / f"track_{tid:04d}" if self.debug else None
        if d:
            d.mkdir(parents=True, exist_ok=True)
        reads = []
        for i, c in enumerate(crops):
            if d:
                cv2.imwrite(str(d / f"{i}_0_raw_{c['w']}x{c['h']}.png"), c["img"])
            img = self.sr.upscale(c["img"]) if self.sr else c["img"]
            if d:
                cv2.imwrite(str(d / f"{i}_1_sr.png"), img)
            text, conf, chars = self.ocr.read(img)
            if d:
                ann = img.copy()
                for ch in chars:
                    x = int(ch["x"])
                    cv2.line(ann, (x, 0), (x, ann.shape[0]), (0, 255, 0), 1)
                    cv2.putText(ann, f"{ch['name']}", (max(x - 8, 0), 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imwrite(str(d / f"{i}_2_ocr_{text or 'none'}.png"), ann)
            reads.append((text, conf))
            info["reads"].append({"crop": i, "text": text, "conf": round(conf, 3),
                                  "chars": len(chars)})

        # Multi-frame fused read. This is the strongest single read on adequate
        # footage (measured +15-20 char-accuracy points over one frame), so it
        # is weighted above any individual crop in the vote.
        fused = self._fuse(crops)
        if fused is not None:
            ftext, fconf, fchars = self.ocr.read(fused)
            if d:
                cv2.imwrite(str(d / "fused_0.png"), fused)
                cv2.imwrite(str(d / f"fused_1_ocr_{ftext or 'none'}.png"), fused)
            info["fused_read"] = {"text": ftext, "conf": round(fconf, 3),
                                  "chars": len(fchars)}
            if ftext:
                # 1.5x weight: the fused read has seen every kept frame at once.
                reads.append((ftext, fconf * 1.5))

        text, conf, detail = vote(reads)
        info["vote"] = detail
        info["plate_text"] = text
        info["plate_confidence"] = round(conf, 3)

        # --- publish gates ---------------------------------------------------
        # Everything above is EVIDENCE and stays in the record for inspection.
        # These decide whether it is fit to appear as a plate number, because a
        # plausible-looking wrong plate is far more damaging than an empty cell:
        # it gets believed and acted on.
        widest = max(c["w"] for c in crops)
        reject = None
        if len(text) < MIN_PLATE_CHARS:
            reject = (f"read {len(text)} character(s); a real Egyptian plate has "
                      f"at least {MIN_PLATE_CHARS}, so this is detector noise")
        elif widest < MIN_PX_FOR_FUSED_OCR:
            # The measured curve: at 34px, 12-frame fusion reads 0 of 50 plates.
            # Anything this pipeline "reads" below the fusion floor is therefore
            # not a read, whatever the vote says. See docs/anpr-plan.md §5c.
            reject = (f"best crop {widest}px is below the "
                      f"{MIN_PX_FOR_FUSED_OCR:.0f}px floor at which multi-frame "
                      "fusion has been measured to read plates at all")
        if reject:
            info["untrusted_read"] = {"text": text, "confidence": round(conf, 3),
                                      "rejected_because": reject}
            info["plate_text"] = ""
            info["plate_confidence"] = 0.0
            info["note"] = f"read discarded — {reject}"
        return info
