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
    CONGESTION_COLORS,
    PERSON_CLASS,
    VEHICLE_CLASSES,
    PipelineConfig,
)
from .congestion import CongestionMonitor
from .counting import LineCounter
from .detect_track import VehicleDetector, visible_mask
from .lanes import LANE_COLORS_HEX, LaneModel
from .speed import SpeedEstimator
from .video_writer import FFmpegH264Writer

ELSEWEDY_RED = (40, 40, 210)   # BGR
ProgressCb = Optional[Callable[[float, str], None]]


def _labels(det: sv.Detections, speeds: dict[int, float], classifier,
            detector=None) -> list[str]:
    out = []
    for i in range(len(det)):
        cid = int(det.class_id[i])
        tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
        code = "P" if cid == PERSON_CLASS else classifier.resolve(tid)
        # Show the sequential display number, not the internal id. Until a track
        # is confirmed it has no number yet, so label it by class alone rather
        # than flashing an id that may be about to merge away.
        num = detector.display_id(tid) if detector is not None else tid
        head = f"#{num} {code}" if num is not None else code
        out.append(f"{head} {speeds[tid]:.0f}km/h" if tid in speeds else head)
    return out


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
    out_fps = fps / stride

    detector = VehicleDetector(cfg, fps, frame_size=(w, h))
    lane_model = LaneModel(cfg, w, h)
    counter = LineCounter(cfg, w, h, lane_model)
    speed = SpeedEstimator(cfg, w, h, fps, stride)
    congestion = CongestionMonitor(cfg, w, h)

    # The classifier reuses the speed homography to turn pixels into metres,
    # which is what lets it tell a pickup from a lorry.
    # The classifier reuses the speed homography to turn pixels into metres, which
    # is what lets it tell a pickup from a lorry. If the model is fine-tuned on the
    # mentor taxonomy, its own class head is used instead and the heuristic is
    # bypassed entirely.
    classifier = VehicleClassifier(speed.transformer, detector.native_codes)
    vehicle_ids = detector.vehicle_ids

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
                                           det.xyxy[i], float(confs[i]))
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
                        labels=_labels(vis, speeds, classifier, detector))
            if cfg.draw_counting_line:
                _draw_counting_line(frame, counter)
            _draw_hud(frame, counter, level, avg_speed, t_sec, cfg.calibrated,
                      cfg.draw_in_out_hud)
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
        "counting": counter.summary(classifier),
        "tracking": {
            "tracker": "ByteTrack + re-identification",
            "stable_ids": cfg.stable_ids,
            "gap_reattachments": detector.stitches,
            "duplicate_merges": detector.merges,
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
        "timeseries": timeseries,
        "class_scheme": {c: n for c, (n, _) in MENTOR_CLASSES.items()},
        "class_distribution": counter.summary(classifier)["by_class"],
        "peak_traffic_second": peak_sec,
        "calibration": {
            "calibrated": cfg.calibrated,
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

    def cb(pct, msg):
        print(f"[{pct:5.1f}%] {msg}", flush=True)

    result = process_video(args.input, args.output_dir, cfg, cb)
    print(json.dumps(result["counting"], indent=2))
    print("Saved:", Path(args.output_dir) / "annotated.mp4")


if __name__ == "__main__":
    main()
