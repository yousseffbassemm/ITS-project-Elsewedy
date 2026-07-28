"""Orchestrator: run the full ITS pipeline on a video and emit artifacts.

Reads an input video, runs detection -> tracking -> counting -> speed ->
congestion, writes an annotated H.264 MP4 and an analytics.json, and reports
progress via an optional callback (used by the web backend).

CLI:
    python -m pipeline.process_video --input clip.mp4 --output-dir data/jobs/demo
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import supervision as sv

from .classify import MENTOR_CLASSES, VehicleClassifier
from .config import (
    COCO_SCHEME,
    CONGESTION_COLORS,
    PipelineConfig,
)
from .congestion import CongestionMonitor
from .counting import LineCounter
from .detect_track import VehicleDetector, visible_mask
from .lanes import LANE_COLORS_HEX, LaneModel
from .plates import PLATE_BGR, PlateReader
from .speed import SpeedEstimator
from .video_writer import FFmpegH264Writer

ELSEWEDY_RED = (40, 40, 210)   # BGR
ProgressCb = Optional[Callable[[float, str], None]]


def _labels(det: sv.Detections, speeds: dict[int, float], classifier,
            detector=None, scheme=COCO_SCHEME) -> list[str]:
    out = []
    for i in range(len(det)):
        cid = int(det.class_id[i])
        tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
        code = "P" if scheme.is_person(cid) else classifier.resolve(tid)
        # Show the sequential display number, not the internal id. Until a track
        # is confirmed it has no number yet, so label it by class alone rather
        # than flashing an id that may be about to merge away.
        num = detector.display_id(tid) if detector is not None else tid
        head = f"#{num} {code}" if num is not None else code
        out.append(f"{head} {speeds[tid]:.0f}km/h" if tid in speeds else head)
    return out


def _draw_plates(frame, det: sv.Detections, plates: PlateReader):
    """Outline each located plate in the colour category it resolved to.

    Drawn in the plate's OWN colour so the overlay explains itself: a red box is
    a red (truck) plate. Deliberately no text — at this scale the plate is ~30 px
    wide and a label would cover the vehicle it belongs to.
    """
    if det.tracker_id is None:
        return
    for tid in det.tracker_id:
        box = plates.box_of(int(tid))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = PLATE_BGR.get(plates.color_of(int(tid)), PLATE_BGR["unknown"])
        # 1 px, because the box is only a few pixels tall — a thicker line would
        # hide the very thing it is pointing at.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _draw_anpr(frame, det: sv.Detections, anpr):
    """Outline the plate the cascade located, in its resolved colour category.

    Deliberately no character text on the video. At this scale the plate is a
    few tens of pixels and a rendered string would cover the vehicle; worse,
    Arabic needs a shaping-aware renderer that OpenCV does not have, so
    cv2.putText draws it disconnected and left-to-right — a wrong plate number
    burned into the deliverable. The characters belong in the CSV, where they
    can be rendered properly and checked.
    """
    if det.tracker_id is None:
        return
    for tid in det.tracker_id:
        box = anpr.box_of(int(tid))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = PLATE_BGR.get(anpr.colour_of(int(tid)), PLATE_BGR["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _draw_counting_line(frame, counter: LineCounter):
    (x1, y1), (x2, y2) = counter.line_px()
    cv2.line(frame, (x1, y1), (x2, y2), ELSEWEDY_RED, 3, cv2.LINE_AA)
    ins, outs = counter.live_counts()
    tag = f"IN {ins}  |  OUT {outs}"
    tx, ty = x1 + 6, max(y1 - 10, 20)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 4), (20, 15, 40), -1)
    cv2.putText(frame, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(frame, counter: LineCounter, level: str, avg_speed: float,
              t_sec: float, calibrated: bool, show_in_out: bool = False):
    h, w = frame.shape[:2]
    panel_w = 340
    panel_h = 150 if show_in_out else 122
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 20, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (0, 0), (6, panel_h), ELSEWEDY_RED, -1)

    total = counter.total_vehicles()
    ins, outs = counter.live_counts()
    lines = [
        (f"Vehicles: {total}", (255, 255, 255)),
        *([(f"IN {ins}   OUT {outs}", (200, 220, 255))] if show_in_out else []),
        # No vehicle inside the speed-reliable band is "no measurement", not
        # "0 km/h" — showing a zero reads as stationary traffic.
        (("Avg speed: --" if not avg_speed else
          f"Avg speed: {avg_speed:.0f} km/h" + ("" if calibrated else "  (est.)")),
         (200, 220, 255)),
        (f"Congestion: {level}", CONGESTION_COLORS.get(level, (255, 255, 255))),
    ]
    y = 34
    for text, color in lines:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    cv2.LINE_AA)
        y += 30
    cv2.putText(frame, f"t={t_sec:5.1f}s", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1, cv2.LINE_AA)


def process_video(
    input_path: str,
    output_dir: str,
    cfg: Optional[PipelineConfig] = None,
    progress_cb: ProgressCb = None,
) -> dict:
    cfg = cfg or PipelineConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(cfg.frame_stride, 1)
    # Encode at the SOURCE frame rate and hold each processed frame for `stride`
    # frames, rather than encoding at fps/stride.
    #
    # Both give a real-time video, but fps/stride produces a container with an
    # odd, very low frame rate — stride 6 on 25 fps footage gave a 4.17 fps file,
    # which browsers play badly and which looks broken rather than merely coarse.
    # Repeating frames costs almost nothing: H.264 encodes an identical frame as
    # a skip, so the file barely grows.
    out_fps = fps
    frame_repeat = stride

    detector = VehicleDetector(cfg, fps, frame_size=(w, h))
    # The detector knows whether the loaded model speaks COCO or the mentor
    # taxonomy; every stage below takes its class meanings from that one place
    # rather than assuming COCO ids. Without this a fine-tuned 7-class model
    # counts private cars (its class 0) as PEDESTRIANS. See config.ClassScheme.
    scheme = detector.scheme
    lane_model = LaneModel(cfg, w, h)
    counter = LineCounter(cfg, w, h, lane_model, scheme=scheme)
    speed = SpeedEstimator(cfg, w, h, fps, stride, scheme=scheme)
    congestion = CongestionMonitor(cfg, w, h, scheme=scheme)

    # The classifier reuses the speed homography to turn pixels into metres,
    # which is what lets it tell a pickup from a lorry.
    # The classifier reuses the speed homography to turn pixels into metres, which
    # is what lets it tell a pickup from a lorry. If the model is fine-tuned on the
    # mentor taxonomy, its own class head is used instead and the heuristic is
    # bypassed entirely.
    # Optional second-stage crop classifier. Self-contained like the plate
    # stage: missing weights are reported and every other analytic is
    # unaffected, because a class label is not worth losing the counts over.
    crop_clf = None
    if cfg.vehicle_cls_model:
        from .classify import CropClassifier
        crop_clf = CropClassifier(cfg.vehicle_cls_model)
        print(f"[classes] crop classifier: "
              f"{'loaded ' + cfg.vehicle_cls_model if crop_clf.model else crop_clf.error}",
              flush=True)
    else:
        # Say so. A run that silently used the weaker class source and reported
        # its numbers as the model's is exactly how an hour gets lost: the CLI
        # did not read ITS_VEHICLE_CLS, produced no warning, and the class mix
        # came out identical to the heuristic with nothing to explain it.
        print("[classes] no crop classifier — using the COCO size heuristic, "
              "which cannot express C or V and has no microbus concept. "
              "Set --vehicle-cls or ITS_VEHICLE_CLS.", flush=True)
    classifier = VehicleClassifier(speed.transformer, detector.native_codes,
                                   crop_classifier=crop_clf,
                                   crop_every=cfg.vehicle_cls_every)
    vehicle_ids = detector.vehicle_ids

    # Plate stage. Optional and self-contained: if the weights are missing it
    # records why and every other analytic is unaffected.
    enhancer = None
    if cfg.plates and cfg.plate_enhance:
        from .plate_ocr import EgyptianPlateOCR, PlateEnhancer, SuperResolver
        sr = SuperResolver(cfg.plate_sr_model)
        ocr_engine = EgyptianPlateOCR(cfg.plate_alpr_model)
        enhancer = PlateEnhancer(
            sr=sr, ocr=ocr_engine, keep=cfg.plate_keep_crops,
            debug_dir=(cfg.plate_debug_dir or str(out / "plate_stages")),
            min_px=cfg.plate_enhance_min_px)
        print(f"[plates] super-resolution: {sr.mode}"
              f"{'' if not sr.error else ' — ' + sr.error}")
        print(f"[plates] OCR: {'loaded' if ocr_engine.model else ocr_engine.error}")

    # Single-frame OCR engine, for a run without --enhance. cfg.plate_ocr_model
    # (ITS_PLATE_OCR_MODEL) was previously read from the environment and then
    # never used, so the setting docs/anpr-plan.md §6 tells operators to set did
    # nothing at all and OCR could only ever run via the enhance path.
    frame_ocr = None
    if cfg.plates and cfg.plate_ocr_model and not cfg.plate_enhance:
        from .plate_ocr import EgyptianPlateOCR
        engine = EgyptianPlateOCR(cfg.plate_ocr_model)
        if engine.model is None:
            print(f"[plates] OCR weights not loaded: {engine.error}")
        else:
            print(f"[plates] OCR: loaded {cfg.plate_ocr_model}")
            # PlateReader wants a callable returning (text, confidence); the
            # engine also returns per-character detail the reader has no use for.
            frame_ocr = lambda crop: engine.read(crop)[:2]  # noqa: E731
    plates = (PlateReader(cfg.plate_model, ocr=frame_ocr,
                          min_px_for_ocr=cfg.plate_ocr_min_px,
                          enhancer=enhancer)
              if cfg.plates else None)

    # ANPR cascade: vehicle crop -> plate -> characters + colour. Independent of
    # the `plates` stage above, and self-contained in the same way — missing
    # weights are reported and every other analytic proceeds.
    anpr = None
    if cfg.anpr:
        from .anpr import load_cascade
        anpr = load_cascade(cfg.anpr_stage2_model, cfg.anpr_stage3_model)
        if anpr.locator is None:
            print("[anpr] no plate locator — the cascade cannot run. Train one "
                  "with tools/train_anpr.py, or point ITS_ANPR_STAGE2 at "
                  "models/plate_detect.pt.", flush=True)
            anpr = None
    anpr_seen: dict[int, int] = defaultdict(int)

    # Colour by TRACK, not by class. supervision's default is ColorLookup.CLASS,
    # and base YOLO flip-flops between car/truck/bus on the same vehicle from one
    # frame to the next, so a box visibly cycled through palette colours while the
    # vehicle just drove straight. Keying colour to the (now stable) vehicle id
    # gives each vehicle one constant colour for its whole life.
    box = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1,
                              text_position=sv.Position.TOP_LEFT,
                              color_lookup=sv.ColorLookup.TRACK)
    trace = sv.TraceAnnotator(thickness=2, trace_length=30,
                              color_lookup=sv.ColorLookup.TRACK)

    writer = FFmpegH264Writer(str(out / "annotated.mp4"), w, h, out_fps)

    # Per-second time series buffers.
    ts_counts: dict[int, list[int]] = defaultdict(list)
    ts_speeds: dict[int, list[float]] = defaultdict(list)
    ts_level: dict[int, list[str]] = defaultdict(list)

    if progress_cb:
        progress_cb(0.0, "starting")

    seen_tracks: set[int] = set()
    frame_idx = 0
    processed = 0
    t_sec = 0.0          # last processed timestamp; stays 0 if the video has no frames
    t0 = time.time()
    failed = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            t_sec = frame_idx / fps

            det = detector.track(frame, frame_idx)
            # feed size evidence to the classifier (vehicles only)
            if det.tracker_id is not None:
                confs = (det.confidence if det.confidence is not None
                         else np.ones(len(det)))
                for i in range(len(det)):
                    cid = int(det.class_id[i])
                    if cid in vehicle_ids:
                        classifier.observe(int(det.tracker_id[i]), cid,
                                           det.xyxy[i], float(confs[i]),
                                           frame=frame)
                # Plates are found once across the whole frame, then matched to
                # vehicles — see PlateReader._find_plates_frame for why per-crop
                # detection is wrong here.
                if plates is not None:
                    plates.observe_frame(frame, det, imgsz=cfg.plate_imgsz)
                # The ANPR cascade works the other way round on purpose: it
                # searches INSIDE each vehicle's crop, so the plate's owner is
                # known by construction and the detector's input resolution is
                # spent on the vehicle rather than on the whole road. Sampled
                # per track, like the crop classifier.
                if anpr is not None:
                    for i in range(len(det)):
                        if int(det.class_id[i]) not in vehicle_ids:
                            continue
                        tid = int(det.tracker_id[i])
                        n = anpr_seen[tid]
                        anpr_seen[tid] = n + 1
                        if n % max(cfg.anpr_every, 1) == 0:
                            anpr.observe(frame, tid, det.xyxy[i])
            if det.tracker_id is not None:
                seen_tracks.update(int(t) for t in det.tracker_id)
            counter.update(det)
            speeds = speed.update(det, frame_idx)
            avg_speed = speed.frame_avg()
            level = congestion.update(det, t_sec, avg_speed)

            # time series (per integer second)
            sec = int(t_sec)
            # Count only vehicles inside the ROI. The camera also sees the
            # OPPOSITE carriageway, and including that traffic inflated the
            # volume chart with vehicles this analysis is not about (they can
            # never cross the counting line, so totals were unaffected).
            ts_counts[sec].append(congestion.vehicles_in_roi)
            if avg_speed:
                ts_speeds[sec].append(avg_speed)
            ts_level[sec].append(level)

            # annotate — lane zones first (background), then vehicles on top
            lane_model.draw(frame, counter.live_lane_counts())
            if len(det) > 0 and det.tracker_id is not None:
                # Draw-time only: hide a second box on a vehicle the re-id layer
                # has not merged yet. The analysis above already used every
                # detection.
                vis = det[visible_mask(det)]
                if len(vis):
                    frame = trace.annotate(frame, vis)
                    frame = box.annotate(frame, vis)
                    frame = label.annotate(
                        frame, vis,
                        labels=_labels(vis, speeds, classifier, detector,
                                       scheme))
            if plates is not None and cfg.draw_plates and len(det):
                _draw_plates(frame, det, plates)
            if anpr is not None and cfg.draw_plates and len(det):
                _draw_anpr(frame, det, anpr)
            if cfg.draw_counting_line:
                _draw_counting_line(frame, counter)
            _draw_hud(frame, counter, level, avg_speed, t_sec, cfg.calibrated,
                      cfg.draw_in_out_hud)
            for _ in range(frame_repeat):
                writer.write(frame)

            processed += 1
            frame_idx += 1
            if progress_cb and total_frames and processed % 10 == 0:
                pct = min(99.0, 100.0 * frame_idx / total_frames)
                progress_cb(pct, f"processing frame {frame_idx}/{total_frames}")
    except BaseException:
        failed = True
        raise
    finally:
        cap.release()
        # On the failure path the loop's exception is the useful one — don't let
        # a secondary "ffmpeg exited non-zero" from the truncated stream mask it.
        try:
            writer.close()
        except Exception:
            if not failed:
                raise

    # Stride guard. ByteTrack has no notion of frame_stride — it treats
    # consecutive calls as consecutive frames, so at a coarse stride its motion
    # model is wrong by the stride factor and association starts failing.
    # Counting needs an unbroken id either side of the line, so a modest drop in
    # confirmation becomes a large drop in counts, silently.
    #
    # Measured on the RAW detection stream rather than on surviving tracks. An
    # earlier version of this guard measured displacement only for tracks that
    # had already associated successfully, and therefore reported "OK" on a run
    # that under-counted 18 vehicles as 5 — it could only see the successes.
    confirm = detector.confirm_rate
    if confirm < 0.95:
        stride_health = "BROKEN"
        stride_note = (
            f"only {confirm*100:.0f}% of detections received a track id at "
            f"frame_stride={stride}. Counting needs an unbroken id either side of "
            f"the line, so vehicles are being MISSED and totals are under-reported. "
            f"Re-run with a lower frame_stride (1-3 for this camera)."
        )
    elif confirm < 0.99:
        stride_health = "MARGINAL"
        stride_note = (
            f"{confirm*100:.1f}% of detections received a track id at "
            f"frame_stride={stride}; a few vehicles may be missed. Use a lower "
            "stride when the counts matter."
        )
    else:
        stride_health = "OK"
        stride_note = (f"{confirm*100:.1f}% of detections received a track id — "
                       "tracking is healthy at this stride.")
    if stride_health != "OK":
        print(f"[WARNING] frame_stride={stride}: {stride_note}", flush=True)

    # Calibration guard. Every scene-specific value — counting line, ROI, lane
    # dividers, speed homography — was MEASURED for samples/street_egypt.mp4, and
    # the web app applies them unchanged to whatever gets uploaded. On any other
    # camera the line can sit off the carriageway entirely, and the run then
    # reports a small number confidently instead of failing.
    #
    # The symptom used is: vehicles tracked ON the analysed carriageway that
    # never cross the counting line. Those are already inside the ROI, so if most
    # of them never reach the line, the line is in the wrong place.
    #
    # A low ROI pass rate is deliberately NOT used, even though it looks like the
    # obvious signal. This camera also sees the opposite carriageway and the ROI
    # gate exists to discard it — config.py records 99 of 126 ids belonging to
    # traffic that is not analysed. On the calibrated clip the pass rate is 18%,
    # so an ROI-based check flags the one video that is known to be correct.
    roi_rate = detector.roi_pass_rate
    tracked = len(seen_tracks)
    counted = counter.total_vehicles()
    cross_rate = counted / tracked if tracked else 1.0
    calib_problems = []
    if tracked >= 8 and cross_rate < 0.35:
        calib_problems.append(
            f"{tracked} vehicles were tracked but only {counted} crossed the "
            "counting line — line_start/line_end do not span this camera's road")
    calibration_health = "OK" if not calib_problems else "MISMATCHED"
    calibration_note = ("scene geometry looks consistent with this video"
                        if not calib_problems else
                        "; ".join(calib_problems) +
                        ". The counting line, ROI, lanes and speed calibration in "
                        "pipeline/config.py were measured for street_egypt.mp4. "
                        "Counts, lanes and speeds are NOT reliable for this clip "
                        "until they are re-calibrated — see README "
                        "'Calibrating for a real scene'.")
    if calib_problems:
        print(f"[WARNING] calibration: {calibration_note}", flush=True)

    if processed == 0:
        raise RuntimeError(
            f"No frames could be decoded from {Path(input_path).name}. The file is "
            "empty, truncated or uses a codec OpenCV cannot read."
        )

    # ---- assemble analytics ----
    secs = sorted(set(ts_counts) | set(ts_speeds) | set(ts_level))
    timeseries = {
        "t_sec": secs,
        "vehicle_count": [int(round(np.mean(ts_counts[s]))) if ts_counts[s] else 0
                          for s in secs],
        "avg_speed_kmh": [round(float(np.mean(ts_speeds[s])), 1) if ts_speeds.get(s) else 0.0
                          for s in secs],
        "congestion_level": [max(set(ts_level[s]), key=ts_level[s].count) if ts_level.get(s)
                             else "Free-flow" for s in secs],
    }
    peak_sec = secs[int(np.argmax(timeseries["vehicle_count"]))] if secs else 0

    # ---- per-lane analytics + throughput ----
    from .classify import display_name
    # Derive duration from the frames actually decoded, not the container's
    # metadata: CAP_PROP_FRAME_COUNT is wrong or zero for plenty of files, and it
    # scales every veh/h figure. frame_idx equals total_frames when the metadata
    # is honest, so this is equal-or-better rather than a different answer.
    duration = max(frame_idx / max(fps, 1e-6), 1e-6)
    speed_pv = speed.per_vehicle()
    total_counted = counter.total_vehicles()
    lane_agg = {i: {"count": 0, "codes": defaultdict(int), "speeds": []}
                for i in range(lane_model.n)}
    for tid in counter.vehicle_events:
        lane = counter.lane_of.get(tid)
        if lane is None:
            continue
        lane_agg[lane]["count"] += 1
        lane_agg[lane]["codes"][classifier.resolve(tid)] += 1
        if tid in speed_pv:
            lane_agg[lane]["speeds"].append(speed_pv[tid])
    lanes_out = []
    for i in range(lane_model.n):
        a = lane_agg[i]
        dom = max(a["codes"], key=a["codes"].get) if a["codes"] else None
        lanes_out.append({
            "lane": i + 1,
            "color": LANE_COLORS_HEX[i % len(LANE_COLORS_HEX)],
            "count": a["count"],
            "pct": round(100.0 * a["count"] / total_counted, 1) if total_counted else 0.0,
            "avg_speed_kmh": round(float(np.mean(a["speeds"])), 1) if a["speeds"] else 0.0,
            "veh_per_hour": int(round(a["count"] / duration * 3600)),
            "dominant_class": f"{dom} · {display_name(dom)}" if dom else "—",
        })
    lane_counts = [l["count"] for l in lanes_out]
    busiest = int(np.argmax(lane_counts)) + 1 if any(lane_counts) else None
    throughput = {
        "veh_per_hour": int(round(total_counted / duration * 3600)),
        "per_lane_veh_per_hour": [l["veh_per_hour"] for l in lanes_out],
        "busiest_lane": busiest,
        "lane_balance_pct": round(100.0 * (min(lane_counts) / max(lane_counts)), 0)
                             if any(lane_counts) and max(lane_counts) else 0.0,
    }

    counting_summary = counter.summary(classifier)
    analytics = {
        "video": {
            "filename": Path(input_path).name,
            # Where the source came from, so the dashboard's "Original" tab works
            # for jobs produced by the CLI (which have no uploaded input.* copy).
            "source_path": str(Path(input_path).resolve()),
            "fps": round(fps, 2),
            "width": w, "height": h,
            "duration_sec": round(duration, 1),
            "frames_total": total_frames or frame_idx,
            "frames_processed": processed,
            "frame_stride": stride,
            "wall_seconds": round(time.time() - t0, 1),
        },
        "counting": counting_summary,
        "tracking": {
            "tracker": "ByteTrack + re-identification",
            "stable_ids": cfg.stable_ids,
            "gap_reattachments": detector.stitches,
            "duplicate_merges": detector.merges,
            # Whether frame_stride is low enough for the tracker to work at all.
            # A silently under-counting run is the failure this catches.
            "stride_health": stride_health,
            "stride_note": stride_note,
            "track_confirm_rate": round(confirm, 4),
            # On-screen numbering self-check: numbers must run 1..N with no gaps
            # and no reuse, so the first vehicle on screen is #1, the next #2, ...
            **(detector.stabilizer.numbering_report() if detector.stabilizer else {}),
            "note": ("Ids persist across occlusions, so a vehicle that the "
                     "tracker loses keeps its original id instead of returning "
                     "as a new vehicle."),
        },
        "speed": speed.summary(classifier),
        "congestion": congestion.summary(),
        "lanes": lanes_out,
        "throughput": throughput,
        # Only vehicles that were actually counted, so the plate table lines up
        # with every other total in the report.
        **({"plates": plates.summary(counter.vehicle_events, classifier,
                                     counter.lane_of, detector.display_id)}
           if plates is not None else {}),
        **({"anpr": anpr.summary(counter.vehicle_events, classifier,
                                 counter.lane_of, detector.display_id)}
           if anpr is not None else {}),
        "timeseries": timeseries,
        "class_scheme": {c: n for c, (n, _) in MENTOR_CLASSES.items()},
        # Which signal actually produced the class labels on this run. The
        # dashboard used to state unconditionally that C and V "need a
        # fine-tuned model", which became false the moment one was wired in —
        # the page denied using the thing it was using. A report should describe
        # the run it came from, not the state of the project when it was written.
        "classification": {
            "source": ("crop classifier + size heuristic on the C/D boundary"
                       if classifier.crop_clf else "size heuristic (COCO)"),
            "model": cfg.vehicle_cls_model if classifier.crop_clf else None,
            "sampled_every": (cfg.vehicle_cls_every if classifier.crop_clf
                              else None),
            "note": (
                "A second-stage classifier trained on the 7-class taxonomy names "
                "the vehicle; where it lands on C-or-D the monocular frontal-area "
                "estimate decides which, because a pickup and a lorry differ "
                "mainly in size. Measured against the heuristic alone on 68 "
                "hand-labelled vehicles from this camera: 0.500 -> 0.765."
                if classifier.crop_clf else
                "Classes C (light truck) and V (van) cannot be expressed by base "
                "COCO, which also has no microbus concept, so they are inferred "
                "from a monocular size estimate and are approximate. Train a "
                "classifier (notebooks/vehicle_classes_colab.ipynb) and set "
                "ITS_VEHICLE_CLS to replace this."),
        },
        "class_distribution": counting_summary["by_class"],
        "peak_traffic_second": peak_sec,
        "calibration": {
            "calibrated": cfg.calibrated,
            # Whether the scene geometry actually fits THIS video, as opposed to
            # the clip it was measured on.
            "health": calibration_health,
            "health_note": calibration_note,
            "roi_pass_rate": round(roi_rate, 3),
            "vehicles_tracked": tracked,
            "vehicles_crossed_line": counted,
            "note": ("Speeds are calibrated." if cfg.calibrated else
                     "Speeds are ESTIMATES using default road geometry. Provide "
                     "source_points + target_size_m to calibrate."),
        },
        "config": {
            "model": cfg.model, "imgsz": cfg.imgsz, "conf": cfg.conf,
            "frame_stride": stride,
        },
    }
    # Atomic: the job store treats the presence of analytics.json as "finished",
    # so a reader must never be able to observe a partially written file.
    tmp = out / "analytics.json.tmp"
    tmp.write_text(json.dumps(analytics, indent=2))
    os.replace(tmp, out / "analytics.json")
    if progress_cb:
        progress_cb(100.0, "done")
    return analytics


def main():
    ap = argparse.ArgumentParser(description="ITS street-video analytics pipeline")
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output-dir", required=True, help="dir for annotated.mp4 + analytics.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--config", default=None, help="optional JSON config file")
    ap.add_argument("--vehicle-cls", default=None,
                    help="second-stage 7-class crop classifier weights "
                         "(defaults to $ITS_VEHICLE_CLS)")
    ap.add_argument("--vehicle-cls-every", type=int, default=None,
                    help="run the crop classifier on every Nth observation of "
                         "a track (default 5; 1 = every frame, slower)")
    ap.add_argument("--anpr", action="store_true",
                    help="enable the ANPR cascade: vehicle -> plate -> "
                         "characters + colour (see pipeline/anpr.py)")
    ap.add_argument("--anpr-stage2", default=None,
                    help="plate-on-vehicle weights (default $ITS_ANPR_STAGE2)")
    ap.add_argument("--anpr-stage3", default=None,
                    help="character weights (default $ITS_ANPR_STAGE3)")
    ap.add_argument("--anpr-every", type=int, default=None,
                    help="run the cascade on every Nth observation of a track "
                         "(default 3)")
    ap.add_argument("--plates", action="store_true",
                    help="enable the licence-plate stage (detection + colour)")
    ap.add_argument("--plate-model", default=None)
    ap.add_argument("--plate-ocr-model", default=None,
                    help="character-recognition weights for single-frame OCR "
                         "(ignored with --enhance, which uses plate_alpr_model)")
    ap.add_argument("--enhance", action="store_true",
                    help="best-crop selection + super-resolution + voted OCR")
    ap.add_argument("--keep-crops", type=int, default=None)
    ap.add_argument("--plate-debug-dir", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.config:
        data = json.loads(Path(args.config).read_text())
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    if args.model:
        cfg.model = args.model
    if args.imgsz:
        cfg.imgsz = args.imgsz
    if args.conf:
        cfg.conf = args.conf
    if args.stride:
        cfg.frame_stride = args.stride
    if args.plates:
        cfg.plates = True
    # The CLI honours the same env vars the web app does. Without this,
    # `ITS_VEHICLE_CLS=... python -m pipeline.process_video` looked like it
    # worked and silently ran the size heuristic instead.
    cfg.vehicle_cls_model = (args.vehicle_cls
                             or os.getenv("ITS_VEHICLE_CLS")
                             or cfg.vehicle_cls_model)
    if args.vehicle_cls_every:
        cfg.vehicle_cls_every = args.vehicle_cls_every
    # ANPR cascade. Same env vars the web app reads — CLAUDE.md §4 records what
    # it costs when the two entry points disagree about a setting.
    if args.anpr or os.getenv("ITS_ANPR", "").lower() in ("1", "true", "yes"):
        cfg.anpr = True
    cfg.anpr_stage2_model = (args.anpr_stage2 or os.getenv("ITS_ANPR_STAGE2")
                             or cfg.anpr_stage2_model)
    cfg.anpr_stage3_model = (args.anpr_stage3 or os.getenv("ITS_ANPR_STAGE3")
                             or cfg.anpr_stage3_model)
    if args.anpr_every:
        cfg.anpr_every = args.anpr_every
    if args.plate_model:
        cfg.plate_model = args.plate_model
    if args.plate_ocr_model:
        cfg.plates = True
        cfg.plate_ocr_model = args.plate_ocr_model
    if args.enhance:
        cfg.plates = True
        cfg.plate_enhance = True
    if args.keep_crops:
        cfg.plate_keep_crops = args.keep_crops
    if args.plate_debug_dir:
        cfg.plate_debug_dir = args.plate_debug_dir

    def cb(pct, msg):
        print(f"[{pct:5.1f}%] {msg}", flush=True)

    result = process_video(args.input, args.output_dir, cfg, cb)
    print(json.dumps(result["counting"], indent=2))
    print("Saved:", Path(args.output_dir) / "annotated.mp4")


if __name__ == "__main__":
    main()
