"""Stable vehicle identities on top of ByteTrack.

ByteTrack associates by motion + IoU only. When a vehicle is occluded (behind the
truck, or by another car) for longer than ``track_buffer``, its track is retired
and the SAME vehicle comes back with a brand-new id. Measured on this clip that
inflated the id count by 25%, and one car was split across seven ids
(73 -> 76 -> 77 -> 81 -> 88 -> 90 -> 95). Every downstream number suffers: a
vehicle can be counted twice, its speed samples are split across ids, and its
class vote is fragmented.

This module keeps ByteTrack as the tracker (it is fast and needs no GPU) and adds
a re-identification pass that maps raw tracker ids to CANONICAL ids. When a raw id
first appears, it is compared against recently-lost canonical tracks on four
independent gates, all of which must pass:

  * motion    — where the lost track would be now, at its last known velocity
  * appearance— HS colour histogram of the vehicle's own pixels
  * size      — box area must be within a factor of the remembered one
  * class     — a bus cannot re-appear as a motorcycle

Gates are deliberately conservative: a missed stitch costs one duplicate id,
whereas a wrong stitch welds two different vehicles together and corrupts both
their counts and their speeds. Anything ambiguous is left as a new vehicle.

No deep re-ID network is used — a colour histogram is a few microseconds and this
machine has no GPU.
"""
from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from .config import COCO_SCHEME, UNGROUPED

# Coarse class groups for the re-id gate live in config.ClassScheme, because a
# model fine-tuned on the mentor taxonomy uses different class ids for the same
# concepts and the gate has to follow it. This module-level helper is the COCO
# case, kept for callers that have no scheme to hand.
_CLASS_GROUP = dict(COCO_SCHEME.groups)


def _group(cls: int) -> int:
    return COCO_SCHEME.group(cls)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def _hist(frame: np.ndarray, xyxy) -> np.ndarray | None:
    """Normalised HS colour histogram of a detection's pixels."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    crop = frame[y1:y2, x1:x2]
    # Trim the border: box edges include road/background that would otherwise
    # dominate the histogram of a small, distant vehicle.
    mh, mw = crop.shape[0] // 6, crop.shape[1] // 6
    if crop.shape[0] - 2 * mh >= 4 and crop.shape[1] - 2 * mw >= 4:
        crop = crop[mh:crop.shape[0] - mh, mw:crop.shape[1] - mw]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


class _Track:
    __slots__ = ("cid", "last_frame", "pos", "vel", "hist", "area", "cls",
                 "_recent", "_votes", "_group_of")

    def __init__(self, cid: int, frame_idx: int, pos, area, cls, hist,
                 group_of=_group):
        self.cid = cid
        self.last_frame = frame_idx
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(2)
        self.hist = hist
        self.area = float(area)
        self.cls = int(cls)
        self._group_of = group_of
        self._votes: Counter = Counter([group_of(cls)])
        self._recent: deque = deque(maxlen=6)
        self._recent.append((frame_idx, self.pos.copy()))

    @property
    def group(self) -> int:
        """Dominant class group over the track's whole history."""
        return self._votes.most_common(1)[0][0]

    def update(self, frame_idx: int, pos, area, cls, hist):
        pos = np.asarray(pos, dtype=float)
        self._recent.append((frame_idx, pos.copy()))
        if len(self._recent) >= 2:
            (f0, p0), (f1, p1) = self._recent[0], self._recent[-1]
            if f1 > f0:
                self.vel = (p1 - p0) / (f1 - f0)
        self.last_frame = frame_idx
        self.pos = pos
        self.area = float(area)
        self.cls = int(cls)
        self._votes[self._group_of(cls)] += 1
        if hist is not None:
            # Rolling blend keeps the descriptor current as lighting/scale change
            # without letting one bad crop overwrite the vehicle's appearance.
            self.hist = hist if self.hist is None else (0.7 * self.hist + 0.3 * hist)

    def predict(self, frame_idx: int) -> np.ndarray:
        return self.pos + self.vel * (frame_idx - self.last_frame)


class IdStabilizer:
    """Maps raw ByteTrack ids to canonical ids that survive occlusion gaps."""

    def __init__(self, fps: float, frame_stride: int = 1, max_gap_sec: float = 3.0,
                 base_tol_px: float = 55.0, tol_growth_px: float = 2.0,
                 min_hist_corr: float = 0.35, max_area_ratio: float = 2.5,
                 strong_frac: float = 0.30, strong_cap_px: float = 40.0,
                 dup_iou: float = 0.60, dup_ground_frac: float = 0.35,
                 dup_frames: int = 2, dup_gap: int = 5,
                 display_delay: int | None = None, scheme=COCO_SCHEME):
        # Class groups come from the detector's own scheme: a fine-tuned model
        # numbers its classes differently, and gating on COCO ids there would
        # put private cars in the pedestrian group and motorcycles nowhere.
        self.scheme = scheme
        self.max_gap = max(int(round(fps * max_gap_sec)), 1)
        self.base_tol = base_tol_px
        self.tol_growth = tol_growth_px
        self.min_hist_corr = min_hist_corr
        self.max_area_ratio = max_area_ratio
        # "Strong" position evidence: within this fraction of the tolerance, but
        # never looser than strong_cap_px in absolute terms, so a long gap (which
        # widens the tolerance) cannot quietly turn into a weak match.
        self.strong_frac = strong_frac
        self.strong_cap_px = strong_cap_px
        self.dup_iou = dup_iou
        self.dup_ground_frac = dup_ground_frac
        # Two frames of agreement is enough. Both gates must pass together (heavy
        # box overlap AND coincident ground contact AND same class group), which
        # over a 90 s clip fired on exactly 10 pairs — every one confirmed by eye
        # as one vehicle wearing two boxes, zero false positives. Requiring 4
        # frames left half of them unmerged, so a vehicle briefly showed two
        # boxes with two ids: the visible "id flicking and coming back".
        self.dup_frames = dup_frames
        # Frame indices are RAW frame numbers, so consecutive processed frames
        # are `frame_stride` apart. A fixed 5-frame tolerance is already smaller
        # than the step at stride 8, which would reset the evidence counter every
        # time and make duplicates unmergeable. Scale it with the stride.
        self.dup_gap = max(dup_gap, 2 * max(int(frame_stride), 1))
        self._canon: dict[int, int] = {}      # raw tracker id -> canonical id
        self._tracks: dict[int, _Track] = {}  # canonical id -> state
        self._alias: dict[int, int] = {}      # merged canonical id -> survivor
        # (pair) -> (consecutive-ish hit count, last frame seen)
        self._dup_hits: dict[tuple[int, int], tuple[int, int]] = {}
        self._next_cid = 1
        self.stitches = 0                     # re-attachments across a gap
        self.merges = 0                       # duplicate boxes collapsed
        # Human-facing numbering. Internal canonical ids are allocation counters:
        # when a vehicle briefly wears two boxes, two ids are consumed before they
        # merge, so the FIRST car on screen was labelled "#3" and the truck behind
        # it "#4". Display numbers are handed out separately, in order of first
        # confirmed appearance, and only after a track has survived a few frames —
        # by which point duplicates have already merged, so no number is wasted on
        # a box that turns out to be the same vehicle.
        self._seen: dict[int, int] = {}
        self._display: dict[int, int] = {}
        self._next_display = 1
        # A number is only handed out once a track has clearly earned it. If a
        # duplicate box were numbered before merging away, its number would be
        # retired and the sequence would show a GAP (#1, #2, #4). Measured on
        # this clip: duplicate ids live at most 10 processed frames, while the
        # shortest genuine vehicle track is 39 — so ~0.6 s of presence separates
        # them cleanly. Scaled by stride, since these are processed frames.
        self.display_delay = (display_delay if display_delay is not None
                              else max(4, int(round(0.6 * fps / max(frame_stride, 1)))))

    def numbering_report(self) -> dict:
        """Self-check that on-screen numbers run 1..N with no gaps or reuse."""
        nums = sorted(self._display.values())
        return {
            "vehicles_numbered": len(nums),
            "contiguous_from_1": nums == list(range(1, len(nums) + 1)),
            "no_duplicate_numbers": len(nums) == len(set(nums)),
            "highest_number": nums[-1] if nums else 0,
        }

    def display_id(self, cid: int) -> int | None:
        """Sequential, human-meaningful number for a vehicle (1, 2, 3, ...)."""
        cid = self._resolve(int(cid))
        n = self._display.get(cid)
        if n is not None:
            return n
        if self._seen.get(cid, 0) >= self.display_delay:
            self._display[cid] = self._next_display
            self._next_display += 1
            return self._display[cid]
        return None

    def resolve_id(self, cid: int) -> int:
        """Public: the surviving canonical id for ``cid`` after any merges.

        Anything that BUCKETS data by track id while the video plays needs this
        at the end. A merge decided at frame 900 does not retroactively re-key
        rows already filed under the id that lost, so without a final resolve
        pass one vehicle appears as two — which is exactly how the harvested
        crop set ended up with more identities than there were vehicles.
        """
        return self._resolve(int(cid))

    def _resolve(self, cid: int) -> int:
        """Follow the alias chain to the surviving id (with path compression)."""
        root = cid
        while root in self._alias:
            root = self._alias[root]
        while cid in self._alias:
            self._alias[cid], cid = root, self._alias[cid]
        return root

    def _merge(self, keep: int, drop: int) -> None:
        """Collapse ``drop`` into ``keep``; both were one vehicle all along."""
        if keep == drop:
            return
        self._alias[drop] = keep
        self._tracks.pop(drop, None)
        # The two ids were always one vehicle: fold their evidence together, and
        # if both somehow reached a display number keep the earlier one.
        self._seen[keep] = self._seen.get(keep, 0) + self._seen.pop(drop, 0)
        dn = self._display.pop(drop, None)
        if dn is not None:
            kn = self._display.get(keep)
            self._display[keep] = dn if kn is None else min(kn, dn)
        for raw, cid in list(self._canon.items()):
            if cid == drop:
                self._canon[raw] = keep
        self.merges += 1

    def _suppress_duplicates(self, xyxy, cids, frame_idx) -> None:
        """Collapse two ids that are tracking the SAME vehicle.

        The detector sometimes emits two heavily-overlapping boxes for one
        vehicle and ByteTrack then gives each its own id, so the vehicle is
        counted twice. These are NOT occlusion gaps — the ids coexist — so the
        re-id path above can never fix them.

        A merge needs box overlap AND near-identical ground contact points
        (two different vehicles, one behind the other, overlap in the image but
        touch the road at clearly different points), sustained over several
        frames so a momentary pass-by is never merged.
        """
        n = len(cids)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self._resolve(int(cids[i])), self._resolve(int(cids[j]))
                if a == b:
                    continue
                ta, tb = self._tracks.get(a), self._tracks.get(b)
                if (ta is None or tb is None or ta.group != tb.group
                        or ta.group == UNGROUPED):
                    continue
                ba, bb = xyxy[i], xyxy[j]
                if _iou(ba, bb) < self.dup_iou:
                    continue
                ga = np.array([(ba[0] + ba[2]) / 2.0, ba[3]])
                gb = np.array([(bb[0] + bb[2]) / 2.0, bb[3]])
                mean_h = ((ba[3] - ba[1]) + (bb[3] - bb[1])) / 2.0
                if np.linalg.norm(ga - gb) > self.dup_ground_frac * max(mean_h, 1.0):
                    continue
                key = (min(a, b), max(a, b))
                # Evidence must be sustained, but detections flicker: one of the
                # two boxes routinely goes missing for a frame or two. Requiring
                # strictly consecutive hits never fires (a real duplicate pair
                # here overlapped on frames 68,71,72,73,75... and was missed).
                # Count hits that are CLOSE together instead, and reset only
                # after a real break, so a brief pass-by still cannot accumulate.
                hits, last = self._dup_hits.get(key, (0, frame_idx))
                hits = hits + 1 if frame_idx - last <= self.dup_gap else 1
                self._dup_hits[key] = (hits, frame_idx)
                if hits >= self.dup_frames:
                    # Keep the id created first — it has the longer history and
                    # may already have been counted at the line.
                    self._merge(min(a, b), max(a, b))
                    self._dup_hits.pop(key, None)
        for key in [k for k, (_, last) in self._dup_hits.items()
                    if frame_idx - last > self.dup_gap]:
            del self._dup_hits[key]

    def _match(self, frame_idx, pos, area, cls, hist, busy: set[int]) -> int | None:
        best, best_d = None, None
        for cid, tr in self._tracks.items():
            if cid in busy:
                continue                       # that vehicle is visible right now
            gap = frame_idx - tr.last_frame
            if not (0 < gap <= self.max_gap):
                continue
            grp = self.scheme.group(cls)
            # UNGROUPED means "this class says nothing about what the vehicle
            # is" (mentor F / an id the scheme does not know). Two unidentified
            # blobs agreeing on being unidentified is not evidence, so they must
            # not satisfy the gate by both being -1.
            if grp == UNGROUPED or tr.group != grp:
                continue                       # class group must agree
            ratio = max(area, 1.0) / max(tr.area, 1.0)
            if not (1.0 / self.max_area_ratio <= ratio <= self.max_area_ratio):
                continue                       # size must be plausible
            tol = self.base_tol + self.tol_growth * gap
            d = float(np.linalg.norm(tr.predict(frame_idx) - np.asarray(pos)))
            if d > tol:
                continue                       # not where it should be
            # Evidence combines rather than vetoes. A colour histogram drifts as
            # a vehicle recedes and shrinks, and requiring position AND appearance
            # to agree was rejecting matches whose predicted position was off by
            # as little as 2 px — motion evidence that strong settles it alone.
            # Appearance is only consulted when position is merely "acceptable".
            if d > min(self.strong_frac * tol, self.strong_cap_px):
                if hist is not None and tr.hist is not None:
                    corr = float(cv2.compareHist(tr.hist.astype(np.float32),
                                                 hist.astype(np.float32),
                                                 cv2.HISTCMP_CORREL))
                    if corr < self.min_hist_corr:
                        continue               # looks like a different vehicle
            if best_d is None or d < best_d:
                best, best_d = cid, d
        return best

    def assign(self, frame, xyxy: np.ndarray, raw_ids: np.ndarray,
               class_ids: np.ndarray, frame_idx: int) -> np.ndarray:
        """Return canonical ids parallel to ``raw_ids``."""
        if len(raw_ids) == 0:
            return np.asarray([], dtype=int)
        busy = {self._resolve(self._canon[int(r)])
                for r in raw_ids if int(r) in self._canon}
        out = np.empty(len(raw_ids), dtype=int)
        for i, raw in enumerate(raw_ids):
            raw = int(raw)
            box = xyxy[i]
            pos = np.array([(box[0] + box[2]) / 2.0, box[3]])   # bottom-centre
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            cls = int(class_ids[i])
            cid = self._canon.get(raw)
            if cid is None:
                hist = _hist(frame, box)
                cid = self._match(frame_idx, pos, area, cls, hist, busy)
                if cid is None:
                    cid = self._next_cid
                    self._next_cid += 1
                    self._tracks[cid] = _Track(cid, frame_idx, pos, area, cls,
                                               hist, self.scheme.group)
                else:
                    self.stitches += 1
                    self._tracks[cid].update(frame_idx, pos, area, cls, hist)
                self._canon[raw] = cid
                busy.add(cid)
            else:
                cid = self._resolve(cid)
                self._canon[raw] = cid   # keep the map flat so _evict can clean it
                # Refresh appearance only occasionally — it is the costly part.
                hist = _hist(frame, box) if (frame_idx % 5 == 0) else None
                tr = self._tracks.get(cid)
                if tr is None:
                    self._tracks[cid] = _Track(cid, frame_idx, pos, area, cls,
                                               hist, self.scheme.group)
                else:
                    tr.update(frame_idx, pos, area, cls, hist)
            out[i] = cid
        self._suppress_duplicates(xyxy, out, frame_idx)
        for i in range(len(out)):
            out[i] = self._resolve(int(out[i]))
        for cid in set(int(c) for c in out):
            self._seen[cid] = self._seen.get(cid, 0) + 1
        self._evict(frame_idx)
        return out

    def _evict(self, frame_idx: int) -> None:
        """Drop tracks too old to ever be stitched, and their raw-id mappings."""
        dead = [cid for cid, tr in self._tracks.items()
                if frame_idx - tr.last_frame > self.max_gap]
        if not dead:
            return
        dead_set = set(dead)
        for cid in dead:
            del self._tracks[cid]
        for raw in [r for r, c in self._canon.items() if c in dead_set]:
            del self._canon[raw]
