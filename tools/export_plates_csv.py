"""Export one row per tracked vehicle: plate colour, plate characters, and context.

This is the deliverable table — what was captured for every car the system saw.

    python -m tools.export_plates_csv data/jobs/plates_demo
    python -m tools.export_plates_csv data/jobs/plates_demo --out report.csv

Columns are deliberately explicit about PROVENANCE, because two of them can be
empty for completely different reasons and the difference matters:

    plate_text      empty because OCR was never attempted (plate too small)
                    is not the same as empty because OCR ran and failed.
    ocr_status      says which. Never leave a reader to guess.

`plate_color` carries the Egyptian category meaning rather than just a colour
word, because the colour is only interesting for what it implies about the
vehicle: red means truck, light blue means private car.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.classify import display_name          # noqa: E402
from pipeline.plates import COLOR_DISPLAY, MIN_PX_FOR_OCR  # noqa: E402

# Arabic name of each plate category, so the table is readable to the people who
# actually operate Egyptian traffic systems.
COLOR_ARABIC = {
    "blue": "ملاكي",             # private ("malaky")
    "red": "نقل",                # haulage / truck
    "orange": "تاكسي",           # taxi
    "green": "دبلوماسي",          # diplomatic
    "yellow": "جمارك",           # customs
    "white": "",
    "unknown": "",
}

FIELDS = [
    # `row_no` is a clean 1..N index over the vehicles in THIS table;
    # `vehicle_no` is the number drawn on the annotated video. They differ, and
    # both are wanted: the table should not appear to skip a vehicle, and the
    # reader must still be able to find that vehicle on the video.
    #
    # They diverge for two legitimate reasons. A vehicle can be numbered on
    # screen and never cross the counting line, so it is absent from this
    # (counted-only) table; and when the re-id layer merges two numbered ids the
    # higher number is retired, leaving a gap in the on-video sequence. The run's
    # own `tracking.contiguous_from_1` flag reports whether that happened.
    "row_no", "vehicle_no", "track_id", "vehicle_class", "vehicle_class_name",
    "plate_color", "plate_category", "plate_category_ar", "supports_classes",
    "agrees_with_class", "plate_width_px", "plate_width_best_px",
    # The number every ANPR specification is actually written in, and the one
    # that decides whether characters are recoverable at all.
    "char_height_px", "char_height_required_px",
    "plate_text", "plate_text_confidence", "ocr_status",
    "crops_used", "reads_agreeing", "lane", "speed_kmh",
]


def ocr_status(vehicle: dict, plates: dict) -> str:
    """Why this row's plate_text is what it is.

    Decided from what the RUN actually did, not from a fixed threshold. The
    previous version always quoted the 100 px single-frame floor, so an
    --enhance run — whose raw-crop floor is 12 px, because fusion and
    cross-crop voting are what test the read — reported "not attempted" for
    every plate it had in fact attempted three times over. That is the one
    misstatement this column exists to prevent.
    """
    text = vehicle.get("plate_text")
    if text:
        return "read"
    plate_px = float(vehicle.get("plate_px") or 0.0)
    if not plate_px:
        return "no plate located"

    # The enhanced pipeline records per-vehicle exactly what it did.
    enh = vehicle.get("enhance") or {}
    if enh.get("reads"):
        return (f"attempted on {len(enh['reads'])} best crop(s)"
                + (" + fused" if enh.get("fused_read") else "")
                + " — no confident reading")
    if enh.get("note"):
        return f"not attempted — {enh['note']}"

    if not plates.get("ocr_attempted"):
        return "not attempted — no OCR engine configured"
    # Fall back to the floor the run reports, not a constant compiled in here.
    floor = float(plates.get("ocr_min_px") or MIN_PX_FOR_OCR)
    if plate_px < floor:
        # The single most important cell in the table. An empty plate_text with
        # no explanation reads as a broken model, and that misreading is what
        # sends a team off to collect training data that cannot help.
        return (f"not attempted — plate {plate_px:.0f}px is below the "
                f"{floor:.0f}px floor (camera limit, not model limit)")
    return "attempted — no confident reading"


# Columns for a run of the ANPR cascade. A different table from the colour-only
# one below, because it answers a different question and padding one schema with
# the other's empty columns is how a reader ends up believing a blank cell means
# "no plate" when it means "this column does not apply to this run".
ANPR_FIELDS = [
    "row_no", "vehicle_no", "track_id", "vehicle_class", "vehicle_class_name",
    # The plate itself, three ways. `plate_arabic` is the real answer;
    # `plate_latin` exists so a reviewer who does not read Arabic can still
    # check a row against the video; `plate_visual_ltr` is the raw
    # left-to-right glyph order, which is what you see in an image viewer and
    # is NOT the reading order.
    "plate_arabic", "plate_latin", "plate_visual_ltr",
    "plate_letters", "plate_digits", "plate_confidence",
    "plate_color", "plate_category", "plate_category_ar",
    # Where the plate sat on the vehicle. The mentor's "it might be on the
    # sides" is a requirement, so it gets a column rather than a footnote.
    "plate_position", "plate_position_x",
    "char_height_px", "char_height_required_px", "plate_width_px",
    "reads_fused", "published", "status",
    "supports_classes", "agrees_with_class", "lane", "speed_kmh",
]


def anpr_rows_from(analytics: dict) -> list[dict]:
    """One row per counted vehicle, from an ANPR cascade run."""
    anpr = analytics["anpr"]
    speeds = {int(k): v for k, v in
              (analytics.get("speed", {}).get("per_vehicle") or {}).items()}
    required = float(anpr.get("char_px_required") or 15.0)
    vehicles = sorted(anpr.get("vehicles", []),
                      key=lambda v: (v.get("vehicle_no") is None,
                                     v.get("vehicle_no") or 0))
    out = []
    for n, v in enumerate(vehicles, start=1):
        tid = int(v["track"])
        color = v.get("plate_color", "unknown")
        out.append({
            "row_no": n,
            "vehicle_no": v.get("vehicle_no") or "",
            "track_id": tid,
            "vehicle_class": v.get("vehicle_class", ""),
            "vehicle_class_name": display_name(v["vehicle_class"])
                                  if v.get("vehicle_class") else "",
            "plate_arabic": v.get("plate_arabic", ""),
            "plate_latin": v.get("plate_latin", ""),
            "plate_visual_ltr": v.get("plate_visual_ltr", ""),
            "plate_letters": v.get("plate_letters", ""),
            "plate_digits": v.get("plate_digits", ""),
            "plate_confidence": v.get("plate_confidence") or "",
            "plate_color": color,
            "plate_category": COLOR_DISPLAY.get(color, color),
            "plate_category_ar": COLOR_ARABIC.get(color, ""),
            "plate_position": v.get("plate_position", ""),
            "plate_position_x": v.get("plate_position_x", ""),
            "char_height_px": v.get("char_height_px", ""),
            "char_height_required_px": required,
            "plate_width_px": v.get("plate_width_px", ""),
            "reads_fused": v.get("reads_fused", ""),
            "published": "yes" if v.get("published") else "no",
            # Never leave a blank plate unexplained: an empty cell with no
            # reason reads as a broken model, and that misreading is what sends
            # a team off collecting training data that cannot help.
            "status": v.get("note") or ("read" if v.get("plate_arabic") else ""),
            "supports_classes": "/".join(v.get("supports_classes") or []),
            "agrees_with_class": ("" if v.get("agrees_with_class") is None
                                  else ("yes" if v["agrees_with_class"] else "NO")),
            "lane": v.get("lane", ""),
            "speed_kmh": speeds.get(tid, ""),
        })
    return out


def rows_from(analytics: dict) -> list[dict]:
    plates = analytics.get("plates")
    if not plates:
        raise SystemExit(
            "This analytics.json has neither an `anpr` nor a `plates` block — the "
            "run was made\nwith no plate stage at all. Re-run with one of:\n"
            "  python -m pipeline.process_video --input <clip> --output-dir <dir> "
            "--anpr\n      (cascade: vehicle -> plate -> characters + colour)\n"
            "  python -m pipeline.process_video --input <clip> --output-dir <dir> "
            "--plates\n      (frame-level plate detection + colour only)")

    # Speed and lane are reported per vehicle elsewhere in the report; index them
    # so the plate table can carry the context a reader needs to act on a row.
    # `per_vehicle` is emitted by speed.summary(); reports produced before it
    # existed simply leave the speed column blank rather than failing.
    speeds = {int(k): v for k, v in
              (analytics.get("speed", {}).get("per_vehicle") or {}).items()}

    required = float(plates.get("char_px_required") or 20.0)
    # Order by the number a reader sees on the video, so the table and the
    # footage can be followed side by side.
    vehicles = sorted(plates.get("vehicles", []),
                      key=lambda v: (v.get("vehicle_no") is None,
                                     v.get("vehicle_no") or 0))
    out = []
    for n, v in enumerate(vehicles, start=1):
        tid = int(v["track"])
        color = v.get("plate_color", "unknown")
        px = float(v.get("plate_px") or 0.0)
        text = v.get("plate_text")
        out.append({
            "row_no": n,
            "vehicle_no": v.get("vehicle_no") or "",
            "track_id": tid,
            "vehicle_class": v.get("vehicle_class", ""),
            "vehicle_class_name": display_name(v["vehicle_class"])
                                  if v.get("vehicle_class") else "",
            "plate_color": color,
            "plate_category": COLOR_DISPLAY.get(color, color),
            "plate_category_ar": COLOR_ARABIC.get(color, ""),
            "supports_classes": "/".join(v.get("supports_classes") or []),
            "agrees_with_class": ("" if v.get("agrees_with_class") is None
                                  else ("yes" if v["agrees_with_class"] else "NO")),
            "plate_width_px": round(px, 1),
            "plate_width_best_px": v.get("plate_px_best", ""),
            "char_height_px": v.get("char_px_best", ""),
            "char_height_required_px": required,
            "plate_text": text or "",
            "plate_text_confidence": v.get("plate_text_confidence") or "",
            "ocr_status": ocr_status(v, plates),
            "crops_used": (v.get("enhance") or {}).get("crops_kept", ""),
            "reads_agreeing": ((v.get("enhance") or {}).get("vote") or {})
                              .get("reads_at_best_length", ""),
            "lane": v.get("lane", ""),
            "speed_kmh": speeds.get(tid, ""),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", help="a job directory containing analytics.json")
    ap.add_argument("--out", default=None, help="default: <job_dir>/plates.csv")
    args = ap.parse_args()
    # The summary below can carry Arabic; the Windows console is cp1252 and
    # would raise on it after the CSV had already been written correctly.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    job = Path(args.job_dir)
    src = job / "analytics.json" if job.is_dir() else job
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    analytics = json.loads(src.read_text(encoding="utf-8"))
    # A cascade run is the richer table and takes precedence; a colour-only run
    # falls back to the original schema.
    use_anpr = bool(analytics.get("anpr"))
    rows = anpr_rows_from(analytics) if use_anpr else rows_from(analytics)
    fields = ANPR_FIELDS if use_anpr else FIELDS

    dest = Path(args.out) if args.out else (job if job.is_dir() else job.parent) / "plates.csv"
    # utf-8-sig: Excel opens a plain utf-8 CSV as mojibake and the Arabic column
    # is the whole point of having it.
    with open(dest, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} vehicles -> {dest}")
    if use_anpr:
        a = analytics["anpr"]
        print(f"  architecture     : {a.get('architecture')}")
        print(f"  plates located   : {a.get('plates_located')} "
              f"(in {a.get('vehicle_crops_searched')} vehicle crops)")
        print(f"  plate colour mix : {a.get('color_mix')}")
        read = sum(1 for r in rows if r["plate_arabic"])
        print(f"  plates published : {read}/{len(rows)}")
        if not read:
            print(f"  why              : {a.get('note')}")
        return 0
    plates = analytics["plates"]
    print(f"  plate colour mix : {plates.get('color_mix')}")
    print(f"  plate width p90  : {plates.get('plate_px_p90')} px")
    read = sum(1 for r in rows if r["plate_text"])
    print(f"  characters read  : {read}/{len(rows)}")
    if not read:
        print(f"  why              : {plates.get('ocr_note')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
