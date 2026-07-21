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
    "light_blue": "ملاكي",       # private
    "red": "نقل",                # haulage / truck
    "orange": "تاكسي",           # taxi
    "brown": "تجاري",            # commercial
    "dark_blue": "شرطة",         # police
    "green": "دبلوماسي",          # diplomatic
    "yellow": "جمارك",           # customs
    "white": "",
    "unknown": "",
}

FIELDS = [
    "vehicle_no", "track_id", "vehicle_class", "vehicle_class_name",
    "plate_color", "plate_category", "plate_category_ar", "supports_classes",
    "agrees_with_class", "plate_width_px", "plate_text",
    "plate_text_confidence", "ocr_status", "lane", "speed_kmh",
]


def ocr_status(plate_px: float, text, ocr_attempted: bool) -> str:
    """Why this row's plate_text is what it is."""
    if text:
        return "read"
    if not plate_px:
        return "no plate located"
    if plate_px < MIN_PX_FOR_OCR:
        # The single most important cell in the table. An empty plate_text with
        # no explanation reads as a broken model, and that misreading is what
        # sends a team off to collect training data that cannot help.
        return (f"not attempted — plate {plate_px:.0f}px is below the "
                f"{MIN_PX_FOR_OCR:.0f}px floor (camera limit, not model limit)")
    if not ocr_attempted:
        return "not attempted — no OCR engine configured"
    return "attempted — no confident reading"


def rows_from(analytics: dict) -> list[dict]:
    plates = analytics.get("plates")
    if not plates:
        raise SystemExit(
            "This analytics.json has no `plates` block — the run was made without\n"
            "the plate stage. Re-run with --plates:\n"
            "  python -m pipeline.process_video --input <clip> "
            "--output-dir <dir> --plates")

    ocr_attempted = bool(plates.get("ocr_attempted"))
    # Speed and lane are reported per vehicle elsewhere in the report; index them
    # so the plate table can carry the context a reader needs to act on a row.
    speeds = {int(k): v for k, v in
              (analytics.get("speed", {}).get("per_vehicle") or {}).items()}

    out = []
    for n, v in enumerate(plates.get("vehicles", []), start=1):
        tid = int(v["track"])
        color = v.get("plate_color", "unknown")
        px = float(v.get("plate_px") or 0.0)
        text = v.get("plate_text")
        out.append({
            "vehicle_no": v.get("vehicle_no") or n,
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
            "plate_text": text or "",
            "plate_text_confidence": v.get("plate_text_confidence") or "",
            "ocr_status": ocr_status(px, text, ocr_attempted),
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

    job = Path(args.job_dir)
    src = job / "analytics.json" if job.is_dir() else job
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    analytics = json.loads(src.read_text())
    rows = rows_from(analytics)

    dest = Path(args.out) if args.out else (job if job.is_dir() else job.parent) / "plates.csv"
    # utf-8-sig: Excel opens a plain utf-8 CSV as mojibake and the Arabic column
    # is the whole point of having it.
    with open(dest, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} vehicles -> {dest}")
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
