"""Recover EALPR's character class map (class id -> Arabic glyph).

EALPR ships its character annotations as YOLO label files whose class ids are
integers with **no legend anywhere in the dataset**. Training an OCR model on
them without the legend produces a model that predicts `24` and cannot say what
`24` means, which is worse than useless — it looks like it works.

The legend is recoverable because the dataset carries the same information
twice, in two forms that were never meant to be cross-checked:

    EALPR- LP characters dataset/Characters Labeling/0001_....txt
        7 rows of `<class_id> <cx> <cy> <w> <h>`      <- ids, positioned
    EALPR- LP characters dataset/Characters/0001_...-س-0.png
        7 cropped glyph images, each NAMED with its glyph   <- glyphs, unpositioned

So each plate gives a set of ids and a set of glyphs that must correspond, but
in an unknown order. Cropping each labelled box out of the plate image and
matching it against the named crops by pixel correlation recovers the pairing
for that plate; doing it across ~2,000 plates and taking the majority makes the
map robust to individual mismatches.

**Why this is trustworthy.** The result is not asserted, it is measured. Every
plate votes independently, and the report prints the agreement for each class:
a genuine mapping shows near-unanimous votes, while a class that is actually
ambiguous shows a split and is flagged rather than quietly picked. Run it and
read the `agree` column before believing the output.

    python -m tools.ealpr_charmap                 # derive, verify, write JSON
    python -m tools.ealpr_charmap --sample 300    # quick pass over 300 plates

Output: data/plates/EALPR_charmap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EALPR = ROOT / "data" / "plates" / "EALPR"
PLATES = EALPR / "EALPR- Plates dataset"
CHARS = EALPR / "EALPR- LP characters dataset" / "Characters"
CHAR_LABELS = EALPR / "EALPR- LP characters dataset" / "Characters Labeling"
OUT = ROOT / "data" / "plates" / "EALPR_charmap.json"

# Size every glyph is resized to before correlating. Small on purpose: the
# comparison must survive the crops having been saved with slightly different
# padding than the boxes imply, and fine detail only adds misalignment noise.
MATCH_PX = 24


def _utf8_stdout() -> None:
    """Make stdout able to print Arabic.

    The Windows console defaults to cp1252, which cannot encode a single glyph
    in this dataset. Without this the derivation completes correctly and then
    dies printing its own results — the work is done and thrown away.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):        # not a real console — fine
            pass


def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    # Explicit utf-8 with errors ignored: a handful of these label files carry
    # stray high bytes, and the platform default (cp1252 on Windows) raises on
    # them, taking down the whole derivation over three malformed files.
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return rows


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread that survives non-ASCII paths.

    Every glyph crop in this dataset is NAMED with the Arabic character it
    contains, so the whole set is unreadable through cv2.imread on Windows — it
    passes the path to the ANSI file API, which cannot express those names and
    returns None for all 10,505 of them. That failure is silent (imread returns
    None rather than raising), so the derivation ran, matched nothing, and
    reported an empty map as though the dataset were the problem.
    """
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:                     # zero-byte file — imdecode asserts
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _norm(img: np.ndarray) -> np.ndarray | None:
    """Grayscale, resized, zero-mean unit-norm — ready for a dot-product score."""
    if img is None or img.size == 0:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Flattened to 1-D so the correlation below is a dot product (a scalar) and
    # not a 24x24 matrix product.
    g = cv2.resize(g, (MATCH_PX, MATCH_PX)).astype(np.float32).ravel()
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return None if n < 1e-6 else g / n


def _glyph_of(name: str) -> str | None:
    """Glyph out of `0001_license_plate_1-س-0.png`.

    Split from the RIGHT and take the middle field: the plate stem itself
    contains underscores and digits, and one filename in the set contains a
    hyphen before the glyph field, which splitting from the left mis-parses.
    """
    stem = Path(name).stem
    parts = stem.rsplit("-", 2)
    return parts[1] if len(parts) == 3 and parts[1] else None


def derive(sample: int | None = None) -> dict:
    if not CHAR_LABELS.exists():
        raise SystemExit(
            f"{CHAR_LABELS} not found. Clone EALPR into data/plates/EALPR first "
            "— see docs/anpr-plan.md.")

    # Group the named glyph crops by the plate they came from.
    by_plate: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in CHARS.glob("*.png"):
        g = _glyph_of(p.name)
        if g:
            by_plate[p.name.rsplit("-", 2)[0]].append((g, p))

    label_files = sorted(CHAR_LABELS.glob("*.txt"))
    if sample:
        label_files = label_files[:sample]

    votes: dict[int, Counter] = defaultdict(Counter)
    used = skipped = 0
    for lf in label_files:
        stem = lf.stem
        rows = _read_labels(lf)
        named = by_plate.get(stem, [])
        # Only use plates where the two records agree on how many characters
        # there are. A mismatch means one side is incomplete, and a forced
        # assignment there would inject a wrong pair into every class it touches.
        if not rows or len(rows) != len(named):
            skipped += 1
            continue
        img = None
        for ext in (".png", ".jpg"):
            cand = PLATES / f"{stem}{ext}"
            if cand.exists():
                img = imread_unicode(cand)
                break
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        boxed = []
        for cid, cx, cy, bw, bh in rows:
            x1, x2 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
            y1, y2 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
            v = _norm(img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)])
            if v is not None:
                boxed.append((cid, v))
        crops = [(g, _norm(imread_unicode(p))) for g, p in named]
        crops = [(g, v) for g, v in crops if v is not None]
        if len(boxed) != len(crops) or not boxed:
            skipped += 1
            continue

        # Greedy best-first assignment over the correlation matrix. Hungarian
        # would be optimal, but greedy on a handful of glyphs that are mostly
        # distinct agrees with it here and keeps scipy out of the dependencies —
        # and any pair it gets wrong is outvoted across 2,000 plates.
        score = np.array([[float(bv @ cv) for _, cv in crops] for _, bv in boxed])
        pairs, taken_r, taken_c = [], set(), set()
        for idx in np.argsort(-score, axis=None):
            r, c = divmod(int(idx), score.shape[1])
            if r in taken_r or c in taken_c:
                continue
            taken_r.add(r)
            taken_c.add(c)
            pairs.append((boxed[r][0], crops[c][0], float(score[r, c])))
        # A confident assignment correlates strongly. Below this the crop and
        # the box are probably not the same glyph, so the pair is dropped rather
        # than voted with.
        for cid, glyph, s in pairs:
            if s >= 0.5:
                votes[cid][glyph] += 1
        used += 1

    mapping, report = {}, []
    for cid in sorted(votes):
        c = votes[cid]
        glyph, n = c.most_common(1)[0]
        total = sum(c.values())
        mapping[cid] = glyph
        report.append({"id": cid, "glyph": glyph, "votes": n, "total": total,
                       "agree": round(n / total, 3),
                       "runners_up": dict(c.most_common(4)[1:])})
    return {"map": mapping, "report": report, "plates_used": used,
            "plates_skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=None,
                    help="only inspect the first N plates (quick check)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    _utf8_stdout()
    res = derive(args.sample)
    print(f"plates used {res['plates_used']}  skipped {res['plates_skipped']}")
    print(f"{'id':>3}  {'glyph':<6} {'agree':>6}  {'votes':>12}  runners-up")
    weak = []
    for r in res["report"]:
        print(f"{r['id']:>3}  {r['glyph']:<6} {r['agree']:>6.1%}  "
              f"{r['votes']:>5}/{r['total']:<6}  {r['runners_up']}")
        if r["agree"] < 0.9 or r["total"] < 5:
            weak.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(res['map'])} classes -> {out}")
    if weak:
        # Not a failure — a disclosure. A class the evidence does not settle must
        # be visible, because it becomes a silently wrong character in every
        # plate the OCR reads.
        print(f"[CHECK] {len(weak)} class(es) below 90% agreement or with thin "
              f"evidence: {[(w['id'], w['glyph'], w['agree']) for w in weak]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
