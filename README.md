# ITS Traffic Analytics — Elsewedy

Upload a video of a street, get **traffic intelligence**: vehicle detection &
counting, speed estimation, and congestion analysis — presented on an
Elsewedy-themed web dashboard with an annotated result video.

Built for the Elsewedy Electric AI-department Intelligent-Transportation-Systems
(ITS) prototype. Methodology & business case: [`docs/methodology.md`](docs/methodology.md).

## What it does
- **Detection & counting** — YOLO + ByteTrack track every vehicle; a virtual line
  counts crossings per class and direction (in/out).
- **Stable vehicle IDs** — ByteTrack alone loses a vehicle behind an occlusion and
  brings it back as a *new* one (measured: 25% ID inflation, one car split across
  seven IDs). A re-identification layer re-attaches it to its original ID, and
  duplicate detections of one vehicle are collapsed. See
  [`docs/methodology.md`](docs/methodology.md) §3.2 for the before/after numbers.
- **Speed estimation** — a perspective (bird's-eye) transform maps the road plane
  to metres and derives per-vehicle km/h.
- **Congestion** — road occupancy + density + average speed → a level
  (Free-flow / Moderate / Heavy / Jam) over time, with peak periods.
- **Per-lane analytics** — 4 coloured lane zones whose boundaries are *measured*
  from the painted stripes (median-background + Hough fit), not eyeballed. The
  rightmost lane is the widest, running from the last stripe out to the barrier.
  Each vehicle is assigned to the lane its footprint occupies most (straddling
  handled). Yields per-lane count, avg speed, dominant class, **flow rate (veh/h)**
  and busiest lane.
- **Clean annotation** — each vehicle keeps one colour for its whole life (colour
  is keyed to the vehicle ID, not to the detected class, which flickers). The
  counting line is an internal reference and is not drawn; in/out totals still
  appear in the report. Toggle with `draw_counting_line` / `draw_in_out_hud`.
- **Licence plates (ANPR cascade)** — `--anpr` runs
  **vehicle → plate → characters + colour**, each arrow a separate model. The
  plate detector searches the whole vehicle crop rather than assuming a central
  plate, because across the training set the plate's centre spans 0.03–0.96 of
  vehicle width. Output is one CSV row per vehicle: Arabic plate, Latin
  transliteration, colour category, and where on the vehicle the plate was.
  Measured **80.9% exact plates end to end** on held-out data
  ([`docs/anpr-plan.md`](docs/anpr-plan.md)).
  **On this project's own CCTV it correctly reads nothing** — glyphs are 6–11 px
  against a 15 px floor — and says so per row rather than inventing a number.
  That is a camera limit, not a model limit; check any clip first with
  `python -m tools.plate_footage_check`.
- **Plate colour** works where characters do not (it needs ~20 px, not ~100 px).
  In Egypt the band encodes vehicle category — red = truck, blue = private,
  orange = taxi — so it is independent evidence for the hard A/C/V classes.
- **Dashboard** — drag-drop upload → live progress → annotated video + KPIs +
  charts (volume, vehicle mix, speed histogram, directional flow, lane analytics,
  congestion timeline). Full-screen, Elsewedy-branded. Export to PDF from the browser.

## Stack
Python 3.12 · Ultralytics YOLOv8 · supervision · OpenCV · FastAPI · vanilla
JS + Chart.js. CPU-only friendly (no GPU required).

## Setup

> This machine has no NVIDIA GPU, so everything runs on CPU. Use a **Python 3.12**
> environment — the system Python 3.14 is too new for the CV wheels.

```bash
# 1. create the venv (uv handles the Python download automatically)
uv venv .venv --python 3.12
# or: py -3.12 -m venv .venv   (if you have Python 3.12 installed)

# 2. install dependencies
uv pip install -r requirements.txt
# or: .venv\Scripts\python -m pip install -r requirements.txt

# 3. model weights — yolov8n.pt (fast, CPU) should sit in the project root.
#    Ultralytics normally auto-downloads it from GitHub. If GitHub is blocked,
#    pull it from the Hugging Face mirror:
curl -L -o yolov8n.pt \
  https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
```

## Run the web app
```powershell
# Calibrated settings for the Elsewedy highway clip (accuracy-first)
$env:ITS_MODEL="yolov8s.pt"; $env:ITS_IMGSZ="736"; $env:ITS_FRAME_STRIDE="1"; $env:ITS_CONF="0.30"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```
Open <http://127.0.0.1:8010>, drop a street video, wait for processing, view the
dashboard. Port 8010 avoids a service already on 8000. For faster/rougher runs use
`yolov8n.pt`, `ITS_IMGSZ=480`, `ITS_FRAME_STRIDE=2`.

Instead of exporting variables each time, copy `.env.example` to `.env` and edit it —
the app loads it at startup. Variables already set in the shell take precedence.

### Vehicle class scheme (mentor taxonomy)
**A** Private car · **C** Light truck · **D** Heavy truck · **E** Bus ·
**G** Motorcycle · **V** Van · **F** Unknown.

Base COCO has four vehicle classes against these seven, and cannot express
**C** or **V** at all — every van is a "car" and every pickup a "truck" to it.
A second-stage classifier over each tracked crop supplies the missing classes;
it trains on MIO-TCD, ~30k traffic-camera crops that carry `work_van` and
`pickup_truck` as real labels. Where the classifier lands on C-or-D, the
monocular frontal-area estimate decides which, because a pickup and a lorry look
alike from behind and differ mainly in size.

```powershell
$env:ITS_VEHICLE_CLS="models/vehicle_cls.pt"
```

## Run the pipeline directly (no web UI)
```bash
.venv\Scripts\python -m pipeline.process_video \
  --input samples\street_egypt.mp4 --output-dir data\jobs\demo \
  --anpr --vehicle-cls models\vehicle_cls.pt
```
Outputs `annotated.mp4` + `analytics.json`. Then the deliverable table:
```bash
.venv\Scripts\python -m tools.export_plates_csv data\jobs\demo
```

## Tests
```powershell
.venv\Scripts\python -m tests.test_pipeline
# optionally cross-check a produced report for internal consistency:
.venv\Scripts\python -m tests.test_pipeline data\jobs\hybrid\analytics.json
```
No pytest needed. Every test encodes a bug that was actually found and fixed —
counting that broke at `frame_stride` 2–4, speed that emitted nothing at high
stride, a speed histogram that dropped fast vehicles, tracker tunables measured
in raw frames that stopped working when the stride changed. Run it after touching
`pipeline/`.

## Calibrating for a real scene
Defaults work on any clip, but for **accurate** counts and speeds, tune
`pipeline/config.py` (or pass a `--config file.json`) to the specific camera:
- `line_start` / `line_end` — counting line; span the **full** carriageway, or
  vehicles in the outermost lane cross outside it and are silently missed.
- `roi` — the road polygon used for congestion/occupancy.
- `lane_dividers` — one line per lane boundary, placed on the painted stripes.
- `source_points` + `target_size_m` — 4 road-plane points and the real-world
  metres they span, so speeds are true km/h. Set `calibrated: true` once done.

All values are fractions of frame width/height, so they're resolution-independent.

**Don't calibrate speed by eye.** Setting the quad length so speeds "look right"
is circular and was wrong here by 1.79×. Use the road as a ruler: painted dashes
are evenly spaced, so after a correct rectification their pitch must come out
*constant*. Build the quad from two genuinely parallel road edges, check the pitch
is flat across the frame, then scale `target_size_m` so it equals the real dash
pitch (`dash_pitch_m`, default 12 m = 3 m dash + 9 m gap). A surveyed distance
between two ground marks is better still. See `docs/methodology.md` §3.4.

## Performance notes (CPU-only)
- Use `yolov8n.pt` and `imgsz` 480–640; raise `frame_stride` (2–4) for long clips.
- Processing runs offline in a background worker; the dashboard polls progress.
- Roughly ~0.25 s/frame for 4K on this CPU — a 12 s 4K clip ≈ 40 s at stride 2.

## Layout
```
pipeline/   detect_track · reid · counting · classify · lanes · speed · congestion
            anpr · plates · plate_ocr · video_writer · process_video · config
app/        FastAPI backend (main) + background job runner (jobs)
web/        index.html · results.html · theme.css · app.js · vendor/chart.min.js
data/jobs/  per-job artifacts (input, annotated.mp4, analytics.json, plates.csv)
docs/       methodology.md · finetuning-plan.md · anpr-plan.md
notebooks/  train_all_colab.ipynb — trains all three models on a Colab GPU
tools/      dataset prep (build_anpr_datasets · prep_vehicle_dataset · ealpr_charmap)
            training  (train_anpr · train_vehicle_classes)
            measuring (eval_anpr · eval_vehicle_cls · plate_footage_check)
            export    (export_plates_csv · harvest_dataset)
samples/    demo clips
```

## Training on a GPU (Colab)
This machine is CPU-only, so training happens on a free Colab GPU. Open
[`notebooks/train_all_colab.ipynb`](notebooks/train_all_colab.ipynb), pick a T4
runtime, and *Run all*. It downloads both public datasets itself, builds them,
trains all three models and measures them — roughly two hours.

The notebook clones this repo, so **push the branch first**. Training logic
lives in `tools/` rather than inside the notebook, so it stays lintable,
testable and reviewable; the notebook is a thin driver.

Two rules the project has already paid to learn, enforced by the eval tools:

* **Validate on the deployment camera, never on a val split.** A plate detector
  scored mAP50 0.985 in-domain and was a straight regression on real footage.
* **Check the footage before spending GPU time on OCR.** Training cannot add
  pixels the sensor never captured, and an unreadable plate looks exactly like
  an undertrained model. `python -m tools.plate_footage_check` measures it.

## Roadmap
Vehicle re-ID embedding for cross-camera matching, violations (red-light /
wrong-way), multi-camera corridor view, and a live/edge mode. See
`docs/methodology.md` and `docs/anpr-plan.md`.
