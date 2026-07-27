# Project context for future sessions

Read this before changing anything. It records what has been **measured**, what
has been **settled**, and the traps that have already cost real time. Everything
here was paid for once; re-deriving it is waste.

Companion docs, which this file does not duplicate:
`README.md` (setup/run) · `docs/methodology.md` (how each analytic works and why)
· `docs/anpr-plan.md` (plates, in depth) · `docs/finetuning-plan.md` (7-class model).

---

## 1. What this is

Elsewedy Electric ITS prototype. Upload a street video → vehicle counting, speed,
lane analytics, congestion, licence-plate colour → annotated MP4 + a branded web
dashboard. CPU-only by constraint (no NVIDIA GPU on the dev machine); model
training happens on a free Colab T4.

Owner is a student presenting to an **AI-department mentor**. The mentor's ask is
"capture every plate: characters + colour". Section 3 is why half of that is not
possible from the available footage, and what was delivered instead.

```
pipeline/   detect_track · reid · counting · classify · lanes · speed · congestion
            plates · plate_ocr · plate_id · video_writer · process_video · config
app/        FastAPI backend (main) + background job runner (jobs)
web/        index.html · results.html · theme.css · app.js
tools/      harvest_dataset · train_vehicle_classes · export_plates_csv ·
            plate_footage_check · eval_plate_fingerprint · train_plates · ...
notebooks/  vehicle_classes_colab.ipynb (upload to Colab, T4, Run all)
```

## 2. Running it

```bash
.venv\Scripts\python -m tests.test_pipeline          # 161 checks, no pytest needed
.venv\Scripts\python -m tests.test_pipeline data/jobs/<id>/analytics.json
```

Web app: `preview_start` with the `its-web` entry in `.claude/launch.json`
(port 8010). `config.py` defaults already match the calibrated clip, so no env
vars are needed.

**Always run the test suite after touching `pipeline/`.** Every check encodes a
bug that actually happened.

## 3. Settled questions — do not re-litigate

### Plate characters cannot be read from `samples/street_egypt.mp4`

Not a model problem, not a data problem, not fixable by training. Measured four
independent ways:

| check | requirement | this camera |
|---|---|---|
| character height | 20 px (industry min), 15 px absolute floor | **11.6 px** |
| character stroke width | 2 px | 1.4 px |
| pixels per lane | 700 min / 1440 recommended | 381 |
| plate width | ~65 px with multi-frame fusion | 34–54 px |

Inverting the industry spec (20 px glyph → 73 px plate) lands on the same answer
as the project's own measured 65 px fusion floor, derived independently.

Chased down: the p90 of 34 px hides the best views, and one detection was 94 px —
it was the **CHEVROLET badge** on a pickup tailgate, not a plate. `docs/anpr-plan.md`
§5d has the image. The user has confirmed **no better footage will ever be
available**, so this is closed.

What IS delivered: **plate colour** for 15 of 18 vehicles (red = truck, blue =
private — Egypt colour-codes by category), in `data/jobs/<id>/plates.csv`.

### Plate fingerprinting does not work

`pipeline/plate_id.py` — perceptual hashing to match a vehicle across cameras
*without* reading the plate. **53% false-match rate.** Kept in the tree with the
verdict boxed at the top of the module because the evaluation harness
(`tools/eval_plate_fingerprint.py`) is the durable artefact and any replacement
must pass it. Do not ship it.

The recommended alternative, not yet built: a **learned embedding over the whole
vehicle** (≈25× more pixels than the plate), standard vehicle re-ID, public
datasets, trains on Colab.

### The 7-class classifier works and is wired in

`ITS_VEHICLE_CLS=models/vehicle_cls.pt`. Measured on 68 hand-labelled vehicles
from the deployment camera, against the old size heuristic:

| | heuristic | classifier |
|---|---|---|
| A private car | 0.920 | 0.920 |
| **E bus/microbus** | **0.042** | **0.833** |
| C light truck | 0.526 | 0.158 |
| **overall** | **0.500** | **0.676** |

Microbuses are ~⅓ of this road's traffic and COCO cannot express them at all —
that class alone is the win. **C regressed** because the Gulf dataset's `lgv`
folder (2,229 light-goods-vehicle images) was wrongly excluded as ambiguous; the
fix is in `notebooks/vehicle_classes_colab.ipynb` but **v2 has not been trained
yet** (two runs died — a disconnect and a PC shutdown).

## 4. Traps that have already bitten

**Validate on the deployment camera, never on a val split.** The EALPR plate
detector scored mAP50 **0.985** and was a straight regression on real footage
(`anpr-plan.md` §2d). The vehicle classifier scored **0.952** in-domain and
**0.676** on the real camera. The in-domain number is meaningless here.

**Roboflow Universe metadata lies.** `vehicle-class-cr0z4` advertises
`LCV, Truck, Bus, Car…` on the site *and through the API*; its actual folders are
lines of the Roboflow README (`'* collect & organize images'`, 1269 images).
**Always print the real folder listing before training on any dataset.**

**Beware circular evaluation.** The plate fingerprint first measured 0% false
matches — because both "sightings" were degraded from the same photograph. With
independent per-sighting exposure/angle/framing it went to 59%. The same mistake
was made once before on speed calibration (tuning the quad until speeds "looked
right", wrong by 1.79×).

**OCR on inadequate footage fabricates; it does not fail quietly.** Raising
fusion to 12 crops gave the character detector more chances to fire on noise:
**17 of 18 vehicles produced a candidate plate string, all garbage**, one at
confidence 1.0 from a single uncorroborated read. Three publish gates now block
this. Do not weaken them.

**`(clip, track)` is not a vehicle identity.** The harvester runs two passes that
number tracks independently, so track 2 exists in both and means two different
vehicles. This exact bug appeared in **three separate files** (`harvest_dataset`
contact sheets + summary, `train_vehicle_classes.prepare`). The key must always
include `carriageway`.

**Colab's editor collapses nested indentation** when text is typed by automation
(2 spaces → 1). Write scripts to a file, or use flat code (comprehensions,
lambdas, semicolons) with no nested blocks.

**Colab runtimes are reclaimed without warning** and take session storage with
them. Download trained weights immediately after `model.train()` returns.

## 5. Architecture notes

**`config.ClassScheme` is the single source of truth for class ids.** COCO and a
fine-tuned 7-class model use different id spaces for the same concepts. Before
this existed, a fine-tuned model counted every private car (its class 0) as a
**pedestrian** and dropped G and F from counts, speed and occupancy entirely.
Every stage takes its class meanings from `detector.scheme` — never hard-code
COCO ids in counting/speed/congestion/reid.

**The crop classifier overrides the size heuristic, it does not vote alongside
it** (`classify.VehicleClassifier.resolve`). The heuristic is a monocular size
estimate that cannot express C or V; mixing would let the weaker signal dilute
the stronger. It is sampled every 5th observation per track — a second inference
per vehicle per frame is expensive on CPU and vehicles are visible for tens of
frames.

**Optional stages must degrade, never raise.** Plates, OCR, super-resolution and
the crop classifier all report missing weights and let every other analytic
proceed. A class label is not worth losing the counts over.

**Reports must explain themselves.** `analytics.json` carries `stride_health`
(a coarse `frame_stride` silently under-counts), `calibration.health` (the scene
geometry was measured for one clip and the web app applies it to anything
uploaded) and `ocr_note` (an unattributed empty column reads as a broken model).

## 6. Privacy — non-negotiable

Vehicle crops and source clips are **personal data** under Egypt's PDPL 151/2020;
plates are legible in them. `.gitignore` blocks `data/dataset/`, `data/plates/`,
`samples/*.mp4`, `*.pt` and `*_cls.zip`. Never commit them, not even to a private
repo. Clear Colab outputs and uploaded footage after a training session.

## 7. Current state (end of the 2026-07-26/27 session)

Branch `plates-anpr`, **3 commits ahead of `origin/plates-anpr`, not pushed**
(`d3a85e5` predates this session; `87196a7` and `8cd0b3c` are its work).

Done: 17 bugs fixed across pipeline/app/tools; the class scheme rewired; plate
hallucination gates; harvest rebuilt (139 identities → 71 real vehicles, 18
analysed numbered to match the video, duplicate audit built in); all 71 vehicles
hand-labelled (`data/dataset/manifest.csv`, `label_source=claude-visual`, 16
flagged low-confidence and worth spot-checking); v1 classifier trained and wired
in (`models/vehicle_cls_v1.pt`).

Deliverable: `data/jobs/plates_final/plates.csv` — 18 vehicles, colour for 15,
zero fabricated plate strings, `char_height_px` vs requirement per row.

### Outstanding

1. **Train v2** — `notebooks/vehicle_classes_colab.ipynb`, T4, Run all (~45 min).
   Has the `lgv → C` fix; should lift C without costing the E win. Needs the
   Roboflow key as a Colab secret named `ROBOFLOW_API_KEY`.
2. **Re-run the pipeline with `ITS_VEHICLE_CLS`** set, and confirm the dashboard's
   class mix changes as expected (many current "A" should become "E").
3. **Vehicle re-ID embedding** — the viable answer to cross-camera matching.
4. Dashboard does not render the `plates` block at all; plate data reaches users
   only via `analytics.json` and the CSV.
5. Residual: `#1`/`a2` are the same SUV (0.916 correlation — *lower* than two
   genuinely different trucks at 0.977, so appearance alone cannot separate them;
   only timing can). Mild for classification, matters for re-ID.
