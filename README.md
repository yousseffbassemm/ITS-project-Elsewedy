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
- **Licence plates** — plate detection + **plate colour**, which in Egypt encodes
  vehicle category (red = truck, light blue = private, brown = commercial) and so
  gives independent evidence for the hard A/C/V classes. **Plate OCR is
  footage-limited**: ~100 px of plate width to read a single frame, or ~65 px
  with the multi-frame fusion pipeline (`--enhance`), against the ~34 px the
  calibrated clip provides — so it is a camera limit rather than a model one.
  Check any clip before training with `python -m tools.plate_footage_check`. See
  [`docs/anpr-plan.md`](docs/anpr-plan.md).
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
Detections are mapped to: **A** Private car · **C** Light truck · **D** Heavy truck ·
**E** Bus · **G** Motorcycle · **V** Van · **F** Unknown. A/D/E/G are produced
reliably from base YOLO; **C** and **V** need a model fine-tuned on the 7 classes
(v1 maps to the nearest reliable class — see `docs/methodology.md`).

## Run the pipeline directly (no web UI)
```bash
.venv\Scripts\python -m pipeline.process_video \
  --input samples\vehicles_12s.mp4 --output-dir data\jobs\demo \
  --model yolov8n.pt --imgsz 480 --stride 2
```
Outputs `annotated.mp4` + `analytics.json` in the output dir.

## Tests
```powershell
.venv\Scripts\python -m tests.test_pipeline
# optionally cross-check a produced report for internal consistency:
.venv\Scripts\python -m tests.test_pipeline data\jobs\street_v2\analytics.json
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
            plates · video_writer · process_video · config · bytetrack.yaml
app/        FastAPI backend (main) + background job runner (jobs)
web/        index.html · results.html · theme.css · app.js · vendor/chart.min.js
data/jobs/  per-job artifacts (input, annotated.mp4, analytics.json)
docs/       methodology.md · finetuning-plan.md · anpr-plan.md
notebooks/  train_plates_colab.ipynb (thin Colab driver; logic is in tools/)
tools/      harvest_dataset · plate_footage_check · train_plates
samples/    demo clips
```

## Training on a GPU (Colab)
This machine is CPU-only, so model training happens on a free Colab GPU. The
workflow is **edit in VS Code → push → Colab pulls**; the training logic lives in
`tools/train_plates.py` rather than inside the notebook, so it stays lintable and
reviewable. Open `notebooks/train_plates_colab.ipynb` in Colab and run it.

Before spending GPU time on OCR, run `tools/plate_footage_check` on the target
footage. Training cannot add pixels the sensor never captured, and an
unreadable plate looks exactly like an undertrained model.

## Roadmap
Fine-tune on Egyptian classes (tuk-tuk, microbus), plate OCR once adequate
footage exists, violations (red-light / wrong-way), multi-camera corridor view,
and a live/edge mode. See `docs/methodology.md` and `docs/anpr-plan.md`.
