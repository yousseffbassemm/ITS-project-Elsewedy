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

### Vehicle class: the answer is a HYBRID, not a better model

`ITS_VEHICLE_CLS=models/vehicle_cls.pt` (this is v1 — see below, do not replace
it with v2). Measured per vehicle on 68 hand-labelled vehicles from the
deployment camera:

| | overall | A | C | E |
|---|---|---|---|---|
| size heuristic (was shipping) | 0.500 | 0.920 | 0.526 | 0.042 |
| v1 classifier alone | 0.647 | 0.920 | 0.053 | 0.833 |
| v2, `lgv`→C, 3× more C data | 0.529 | 0.680 | 0.368 | 0.500 |
| **HYBRID (shipped)** | **0.765** | 0.920 | 0.474 | 0.833 |

Two things to take from this.

**Microbuses are the win.** ~⅓ of this road's traffic, and COCO cannot express
them at all — 0.042 → 0.833.

**More data made it worse, and that is the interesting result.** v1's only real
weakness was C↔D. The obvious fix — fold the Gulf dataset's `lgv` folder (2,229
light-goods images) into C and retrain — did lift C (0.053 → 0.368) and cost
**12 points overall**, because a bigger, broader C started swallowing cars and
microbuses (`E→C` 10, `A→C` 6). v2 scored **0.93 in-domain**, better-looking than
its real performance. Do not resurrect it.

What works instead: let each signal make the call it is good at. The classifier
says *what kind* of vehicle it is; where that lands on C-or-D, the monocular
frontal-area estimate decides *which*, because a pickup and a lorry look alike
from behind and differ mainly in size. Implemented in
`classify.VehicleClassifier.resolve` — 6 lines, and it beats either component.

Caveat: C is only 19 vehicles, so each one moves that column by ~5 points. Treat
the C figures as directional.

**The live pipeline does NOT reproduce 0.765, and this is unresolved.** Running
the real pipeline on `street_egypt.mp4` with the hybrid enabled:

| | A | C | D | E |
|---|---|---|---|---|
| heuristic only | 11 | 6 | 1 | **0** |
| hybrid (shipped) | 10 | 5 | 1 | **2** |

E going 0 → 2 is real and is the first time this pipeline has ever reported a
bus/microbus. But 2 of 18 is well short of what 0.833 on that class implies, and
**two attempts to close the gap both failed**:

* `vehicle_cls_every=1` (5× the classifier calls) — E stayed at 2
* `MIN_CLS_CROP_PX=64`, declining on small crops — mix *identical* to baseline

Three quite different configurations all land on E=2. Sampling frequency and
crop-size filtering are both ruled out. Everything else is unaffected — vehicles,
speed, lane counts identical across all runs — so the change is correctly
confined to the class label.

Untested candidates, in the order worth trying: the harvested crops are the
LARGEST view per vehicle while the pipeline sees the full size distribution; the
per-track vote may be dominated by a few observations; or the hand-labels
overcount microbuses among the 18 that actually crossed the line (the 24-microbus
figure covers all 71 vehicles including the opposite carriageway). The direct
diagnostic — dump per-observation votes and crop sizes for each of the 18 and
compare against the labels vehicle by vehicle — has not been run.

Do not quote 0.765 to anyone without that caveat. It is an offline,
crop-level figure that has not reproduced end to end.

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
them — this cost three training runs (a disconnect, a PC shutdown, a closed tab)
before the fix. Do not rely on downloading weights at the end. Mount Drive and
pass `project='/content/drive/MyDrive/its_models'` to `model.train()`, which
writes `last.pt` and `best.pt` there every epoch; a disconnect then costs time
rather than work. Note the training itself survived every one of those three
losses — only our access to `/content` died.

**More training data is not automatically better.** See §3: tripling class C
lifted C and cost 12 points overall. Always re-measure the classes you were NOT
trying to fix.

**An env var read by the web app is not read by the CLI.** `ITS_VEHICLE_CLS` was
honoured only in `app/main._cfg()`, so
`ITS_VEHICLE_CLS=... python -m pipeline.process_video` ran the size heuristic,
printed nothing, and produced a class mix identical to a no-classifier run —
which looked like "the classifier did not help" rather than "the classifier never
loaded". Both entry points now agree, and a run always states its class source.
`ITS_PLATE_OCR_MODEL` had the same defect. Check both when adding a setting.

**Measured dead ends — do not re-try without new evidence.** `classify.py`
carries the crop-size degradation table (below 64px the classifier answers "G"
for everything). That table is real, but acting on it via `MIN_CLS_CROP_PX=64`
changed the live class mix by exactly nothing. The constant is kept because the
measurement is worth having; the *conclusion drawn from it* was wrong.

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

## 7. Current state (end of the 2026-07-27 session)

**Uncommitted:** `CLAUDE.md`, `pipeline/classify.py`, `pipeline/process_video.py`,
`tests/test_pipeline.py` — the hybrid C/D deferral, the crop-size floor, the
`--vehicle-cls` / `--vehicle-cls-every` CLI flags and their tests. 168 checks
pass. Commit before starting anything new.

**Models on disk** (all gitignored): `models/vehicle_cls_v1.pt` = the shipped
one, also copied to `models/vehicle_cls.pt`; `models/vehicle_cls_v2.pt` = the
rejected retrain, kept only as evidence. v2 also lives in the user's Google Drive
at `MyDrive/its_models/`.

**Latest reports:** `data/jobs/hybrid` (classifier + hybrid, the current best),
`data/jobs/_verify` (heuristic only, the comparison baseline),
`data/jobs/hybrid_e1` and `data/jobs/hybrid_min64` (the two failed experiments).

**Known-stale UI text:** `web/app.js` still prints *"Classes C & V need a model
fine-tuned on the 7-class scheme; v1 maps to the nearest reliable class"* — written
before the classifier existed. The dashboard now denies using the thing it is
using. It should name the class source actually in force.

### Previous session

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

### Outstanding, most useful first

1. **Why does the live pipeline only find 2 microbuses?** See §3. Two hypotheses
   already falsified. The untried diagnostic: dump per-observation classifier
   votes + crop sizes for each of the 18 counted vehicles and compare against
   `data/dataset/manifest.csv` labels vehicle by vehicle. Do this before any
   further tuning — the last two changes were guesses and both cost a 15-minute
   run to disprove.
2. **Fix the stale dashboard note** in `web/app.js` (see above).
3. **Decide on `MIN_CLS_CROP_PX`** — measurably does nothing; keep for the
   documented measurement or revert.
4. **Vehicle re-ID embedding** — the viable answer to cross-camera matching, and
   the honest replacement for the plate fingerprint that failed. Whole vehicle,
   ~25× more pixels than the plate, public datasets (VeRi-776, VehicleID),
   trains on Colab.
5. Dashboard does not render the `plates` block at all; plate data reaches users
   only via `analytics.json` and the CSV.
6. Residual harvest duplicate: `#1`/`a2` are the same SUV (0.916 correlation —
   *lower* than two genuinely different trucks at 0.977, so appearance alone
   cannot separate them; only timing can). Mild for classification, matters for
   re-ID.
7. `docs/img/resolution_proof.png` is public on GitHub and shows a legible plate
   (`3AE 6211`), which contradicts §6. From a stock 4K clip, not Elsewedy
   footage. Blurring the glyph region would keep the figure's point intact.
