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
lane analytics, congestion, licence plates → annotated MP4 + a branded web
dashboard + a per-vehicle CSV. CPU-only by constraint (no NVIDIA GPU on the dev
machine); **all model training happens on a free Colab T4** —
`notebooks/train_all_colab.ipynb` trains all three models in one run.

Owner is a student presenting to an **AI-department mentor**.

```
pipeline/   detect_track · reid · counting · classify · lanes · speed · congestion
            anpr · plates · plate_ocr · plate_id · video_writer · process_video · config
app/        FastAPI backend (main) + background job runner (jobs)
web/        index.html · results.html · theme.css · app.js
tools/      prep:    ealpr_charmap · build_anpr_datasets · prep_vehicle_dataset ·
                     harvest_dataset · build_plate_colour_set · prep_plate_dataset
            train:   train_anpr · train_vehicle_classes · train_plates
            measure: eval_anpr · eval_vehicle_cls · plate_footage_check ·
                     eval_plate_fingerprint
            export:  export_plates_csv
notebooks/  train_all_colab.ipynb  (upload to Colab, T4, Run all)
```

## 2. Running it

```bash
.venv\Scripts\python -m tests.test_pipeline          # 203 checks, no pytest needed
.venv\Scripts\python -m tests.test_pipeline data/jobs/<id>/analytics.json
```

Web app: `preview_start` with the `its-web` entry in `.claude/launch.json`
(port 8010). `config.py` defaults already match the calibrated clip.

**Always run the test suite after touching `pipeline/`.** Every check encodes a
bug that actually happened.

## 3. The ANPR cascade — the architecture the mentor asked for

`pipeline/anpr.py`, enabled with `--anpr`. Four stages, each a separate concern:

```
frame --[COCO YOLO + tracker]--> vehicle --[stage 2]--> plate --[stage 3]--> characters
                                                             \--> colour (no model)
```

Stage 1 needs no training — detection was never the weak part. Stages 2 and 3
train on **EALPR**, a public Egyptian benchmark (`tools/build_anpr_datasets.py`).

**Measured, on 313 held-out vehicles** (`tools/eval_anpr.py`):

| | exact plate | character accuracy |
|---|---|---|
| stage 2 (find the plate) | 93.3% recall @IoU 0.5 | median IoU 0.805 |
| stage 3 on ground-truth crops | 89.4% | 98.9% |
| **end to end** | **80.9%** | **92.1%** |

End-to-end is the honest number. Stage 3 alone assumes a perfect plate crop and
is always the flattering figure — do not quote it on its own.

Caveat worth stating out loud: that run used the pretrained `models/eg_alpr.pt`,
whose provenance is unknown and which **may well have trained on EALPR itself**.
Re-measure after training our own stage 3 from the notebook; that split is
disjoint by plate, so the number will be trustworthy.

### The plate is not always in the middle

The mentor called this out and the data agrees: across EALPR's 2,087 annotated
vehicles the plate's centre-x spans **0.026 to 0.962** of vehicle width. So
stage 2 searches the whole vehicle crop, and every CSV row reports where the
plate actually was (`plate_position`, `plate_position_x`) rather than leaving a
reader to assume.

### Reading order is not "sort by x"

An Egyptian plate carries Arabic letters and Arabic-Indic digits and **the two
read in opposite directions**: letters right-to-left, the number left-to-right.
A single positional sort produces a string that looks entirely plausible with
the letters backwards, and nobody reviewing an English report catches it.
`anpr.assemble()` emits all three forms — Arabic, Latin transliteration, and raw
visual order — so the CSV is checkable by someone who does not read Arabic.

### EALPR's character labels had no legend

The dataset ships YOLO label files whose class ids are bare integers, documented
nowhere. `tools/ealpr_charmap.py` recovers the legend by cross-matching each
labelled box against the glyph crops, which are *named* with their character.
**100% agreement on all 26 classes across 1,923 plates** — the alphabet is 17
letters plus digits ١-٩. There is no ٠: the digit zero does not appear on a
single Egyptian plate in the set, and EALPR's own class list has a hole where it
would be.

Do not train on those labels without running that tool first. A model that
predicts `24` and cannot say which character `24` is looks like it works.

## 4. Settled questions — do not re-litigate

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
it was the **CHEVROLET badge** on a pickup tailgate, not a plate.
`docs/anpr-plan.md` §5d has the image. The user has confirmed **no better footage
will ever be available**, and the mentor has independently reached the same
conclusion. This is closed.

The cascade confirms it end to end and behaves correctly: on `street_25s.mp4` it
searched 111 vehicle crops, located **59 plates**, read plate **colour** for 3 of
4 counted vehicles, and published **zero** character strings — each row
attributed to a glyph height of 6.1–11.0 px against the 15 px floor. Locating and
colouring the plate works on this footage; reading it does not, and the system
says which.

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

Measured per vehicle on 71 hand-labelled vehicles from the deployment camera,
reproducibly, with `tools/eval_vehicle_cls.py`:

| | overall | A | C | D | E |
|---|---|---|---|---|---|
| size heuristic alone | 0.479 | 0.880 | 0.526 | 1.000 | 0.042 |
| v1 classifier alone | 0.634 | 0.920 | 0.053 | 1.000 | 0.833 |
| **HYBRID (shipped)** | **0.803** | 0.920 | 0.684 | 1.000 | 0.833 |

**Microbuses are the win.** ~⅓ of this road's traffic, and COCO cannot express
them at all — 0.042 → 0.833.

The hybrid is 6 lines in `classify.VehicleClassifier.resolve`: the classifier
says *what kind* of vehicle it is; where that lands on C-or-D, the monocular
frontal-area estimate decides *which*, because a pickup and a lorry look alike
from behind and differ mainly in size. It beats either component.

Caveat: D, F and V have 1 vehicle each in this test set, so those columns are
directional, not measurements. The eval tool prints that warning itself.

**More data made it worse, once, and that is the interesting result.** v1's only
real weakness was C↔D. The obvious fix — fold a Gulf dataset's `lgv` folder
(2,229 light-goods images) into C and retrain — did lift C and cost **12 points
overall**, because a bigger, broader C started swallowing cars and microbuses.
v2 scored 0.93 in-domain, better-looking than its real performance. Do not
resurrect it. **Always re-measure the classes you were NOT trying to fix.**

### The classifier's real fix is more data of the RIGHT kind

v1 was trained on 194 crops of 71 vehicles from one clip. That is enough to
*measure* a classifier and nowhere near enough to *train* one, which is why the
live pipeline reports only 2 microbuses where the offline figure implies ~6.

`tools/prep_vehicle_dataset.py` builds the replacement from **MIO-TCD** —
519,164 crops from real traffic cameras, the same *kind* of image this pipeline
sees. Critically it carries the two classes COCO cannot express:

| MIO-TCD | mentor | why it matters |
|---|---|---|
| `work_van` (9,679) | **V** | COCO calls every one of these a car |
| `pickup_truck` (50,906) | **C** | COCO calls every one of these a truck |

Classes are **capped to 6,000 each** (raw is 260k cars vs 9.7k vans — trained on
that, the cheapest route to a high score is to answer "car", which is the
behaviour being replaced). Built: 30,616 train / 5,401 val, plus the project's
own hand-labelled deployment crops copied out as a `test/` split that is **never
trained on**.

Prefer a dataset of surveillance crops over a larger one of clean press
photography. The domain gap is what breaks classifiers here, not the sample count.

## 5. Traps that have already bitten

**Validate on the deployment camera, never on a val split.** The EALPR plate
detector scored mAP50 **0.985** and was a straight regression on real footage
(`anpr-plan.md` §2d). The vehicle classifier scored **0.952** in-domain and
**0.676** on the real camera. The in-domain number is meaningless here.

**A measurement that cannot be re-run is an opinion with a decimal point.** The
0.765 hybrid figure in an earlier version of this file was measured by hand and
no script could reproduce it; every later decision leaned on a number nobody
could re-derive. `tools/eval_vehicle_cls.py` and `tools/eval_anpr.py` now exist
so every figure quoted here is one command away. Add the harness with the claim.

**Roboflow Universe metadata lies.** `vehicle-class-cr0z4` advertises
`LCV, Truck, Bus, Car…` on the site *and through the API*; its actual folders are
lines of the Roboflow README (`'* collect & organize images'`, 1269 images).
**Always print the real folder listing before training on any dataset** — and
make sure the listing prints the folders you SKIPPED too, or it cannot reveal the
mismatch it exists to catch.

**OpenCV cannot open non-ASCII paths on Windows.** `cv2.imread` goes through the
ANSI file API and returns `None` — silently, not an exception. EALPR names every
glyph crop after the Arabic character it contains, so all 10,505 of them were
unreadable and the charmap derivation reported an empty result as though the
dataset were at fault. Use `anpr.imread_unicode`. The same class of problem hits
stdout: the Windows console is cp1252 and raises on Arabic, so a tool can do its
work correctly and then die printing it.

**Beware circular evaluation.** The plate fingerprint first measured 0% false
matches — because both "sightings" were degraded from the same photograph. With
independent per-sighting exposure/angle/framing it went to 59%. The same mistake
was made once before on speed calibration (tuning the quad until speeds "looked
right", wrong by 1.79×).

**OCR on inadequate footage fabricates; it does not fail quietly.** Raising
fusion to 12 crops gave the character detector more chances to fire on noise:
**17 of 18 vehicles produced a candidate plate string, all garbage**, one at
confidence 1.0 from a single uncorroborated read. Publish gates now block this in
both the fusion path and the cascade. Do not weaken them.

**One read cannot corroborate itself.** With a single read every character
position agrees with itself and the mean agreement is 1.0 by construction. Both
`plate_ocr.vote` and `ANPRCascade.resolve` halve a lone read's confidence.

**Publish the string that won the vote, not the best single frame's.** The
cascade briefly took its text from the highest-confidence frame and its
confidence from the cross-frame vote — which publishes one plate carrying
another plate's number, and the two only diverge exactly when the reads disagree.

**`(clip, track)` is not a vehicle identity.** The harvester runs two passes that
number tracks independently, so track 2 exists in both and means two different
vehicles. This exact bug appeared in **three separate files**. The key must
always include `carriageway`.

**Colab runtimes are reclaimed without warning** and take session storage with
them — this cost three training runs before the fix. Do not rely on downloading
weights at the end. Mount Drive and pass `project='/content/drive/MyDrive/its_models'`
to `model.train()`, which writes `last.pt` and `best.pt` there every epoch. Every
training tool here takes `--project` for exactly this.

**Colab's editor collapses nested indentation** when text is typed by automation
(2 spaces → 1). Write scripts to a file, or use flat code with no nested blocks.
Generating the `.ipynb` JSON directly (as `train_all_colab.ipynb` was) avoids it.

**An env var read by the web app is not read by the CLI.** `ITS_VEHICLE_CLS` was
honoured only in `app/main._cfg()`, so the CLI ran the size heuristic, printed
nothing, and produced a class mix identical to a no-classifier run — which looked
like "the classifier did not help" rather than "the classifier never loaded".
`ITS_PLATE_OCR_MODEL` had the same defect. `ITS_ANPR*` is wired into both. Check
both entry points when adding a setting.

**Measured dead ends — do not re-try without new evidence.** `classify.py`
carries the crop-size degradation table (below 64 px the classifier answers "G"
for everything). That table is real, but acting on it via `MIN_CLS_CROP_PX=64`
changed the live class mix by exactly nothing. The constant is kept because the
measurement is worth having; the *conclusion drawn from it* was wrong.

## 6. Architecture notes

**`config.ClassScheme` is the single source of truth for class ids.** COCO and a
fine-tuned 7-class model use different id spaces for the same concepts. Before
this existed, a fine-tuned model counted every private car (its class 0) as a
**pedestrian** and dropped G and F from counts, speed and occupancy entirely.
Every stage takes its class meanings from `detector.scheme` — never hard-code
COCO ids in counting/speed/congestion/reid.

**The crop classifier overrides the size heuristic, it does not vote alongside
it.** The heuristic is a monocular size estimate that cannot express C or V;
mixing would let the weaker signal dilute the stronger. Sampled every 5th
observation per track — a second inference per vehicle per frame is expensive on
CPU and vehicles are visible for tens of frames.

**The cascade searches inside the vehicle crop; the colour-only plate stage
searches the whole frame.** Both are correct for what they do. The frame-level
detector was trained on full scenes and returns nonsense on a tight vehicle crop
(it once returned a box 83% of vehicle width at 0.73 confidence). The cascade's
stage 2 is trained on vehicle photographs, so the crop is its native domain.

**Optional stages must degrade, never raise.** Plates, OCR, the cascade,
super-resolution and the crop classifier all report missing weights and let every
other analytic proceed. A class label is not worth losing the counts over.

**Reports must explain themselves.** `analytics.json` carries `stride_health`
(a coarse `frame_stride` silently under-counts), `calibration.health` (the scene
geometry was measured for one clip and the web app applies it to anything
uploaded), `classification.source` (which signal produced the labels on *this*
run) and a per-row `note` on every blank plate. An unattributed empty column
reads as a broken model, and that misreading is what sends a team off collecting
training data that cannot help.

**No Arabic on the annotated video.** OpenCV has no shaping-aware text renderer,
so `cv2.putText` draws Arabic disconnected and left-to-right — a wrong plate
number burned into the deliverable. Characters go in the CSV and the dashboard,
which render them properly; the video gets a coloured plate box only.

## 7. Privacy — non-negotiable

Vehicle crops, plate datasets and source clips are **personal data** under
Egypt's PDPL 151/2020; plates are legible in them. `.gitignore` blocks
`data/dataset/`, `data/plates/`, `data/vehicle_ext/`, `samples/*.mp4`, `*.pt` and
`*_cls.zip`. Never commit them, not even to a private repo. Clear Colab outputs
and uploaded footage after a training session.

This is why `tools/eval_vehicle_cls.py` cannot run on Colab — the deployment
crops are deliberately not in the repo, so the honest score is computed locally
after downloading the weights.

`docs/img/resolution_proof.png` is public on GitHub. Its plate is now redacted in
both the image and the caption; the figure's point is resolution, which the
remaining legible characters still carry.

## 8. Current state (end of the 2026-07-28 session)

**Branch `plates-anpr`, ahead of `origin/plates-anpr` and NOT pushed.** The Colab
notebook clones the repo, so pushing is a prerequisite for training.

**Built this session**
* `pipeline/anpr.py` — the cascade, wired into `process_video` (`--anpr`), the
  web app (`ITS_ANPR*`), the CSV exporter and the dashboard.
* `tools/ealpr_charmap.py` — recovers EALPR's character legend (100% agreement).
* `tools/build_anpr_datasets.py` — stage 2 (2,087 vehicles) + stage 3 (1,978
  plates), split by plate.
* `tools/prep_vehicle_dataset.py` — MIO-TCD → 30,616 balanced 7-class crops,
  deployment camera held out as `test/`.
* `tools/train_anpr.py`, `tools/eval_anpr.py`, `tools/eval_vehicle_cls.py`.
* `notebooks/train_all_colab.ipynb` — all three models, one run, ~2 h on a T4.
* Dashboard now renders the plate table; it previously omitted plates entirely.

**Models on disk** (all gitignored): `models/vehicle_cls.pt` (the deployed
classifier) and `models/vehicle_cls_v1.pt` — **byte-identical today, and both
are wanted**: the Colab run overwrites `vehicle_cls.pt`, and v1 is then the
baseline the new model has to beat. Do not delete it as a duplicate; run
`tools/eval_vehicle_cls.py --model` against each and compare.
`models/vehicle_cls_v2.pt` (rejected retrain, kept as evidence),
`models/plate_detect.pt` (stage 2 stand-in), `models/eg_alpr.pt` (stage 3
stand-in), `models/RealESRGAN_x4.pth` (the `--enhance` fusion path).
After the Colab run, `plate_on_vehicle.pt`, `plate_chars.pt` and a retrained
`vehicle_cls.pt` replace the stand-ins.

**Latest reports:** `data/jobs/anpr_check` (the cascade on `street_25s.mp4`),
`data/jobs/hybrid` (classifier + hybrid), `data/jobs/_verify` (heuristic
baseline), `data/jobs/plates_final` (the 18-vehicle colour deliverable).

### Outstanding, most useful first

1. **Push the branch, then run `notebooks/train_all_colab.ipynb`.** Everything
   is prepared; this is the one step that needs a GPU. Afterwards re-run both
   eval tools — the stage-3 number in §3 is not trustworthy until stage 3 is
   ours, and the §4 classifier table should move once it trains on 30k crops
   instead of 194.
2. **Why does the live pipeline only find 2 microbuses?** Two hypotheses already
   falsified (`vehicle_cls_every=1`; `MIN_CLS_CROP_PX=64`) — three quite
   different configurations all land on E=2. The untried diagnostic: dump
   per-observation classifier votes + crop sizes for each counted vehicle and
   compare against `data/dataset/manifest.csv` vehicle by vehicle. Retraining on
   MIO-TCD may simply dissolve this; re-measure before investigating further.
3. **Decide on `MIN_CLS_CROP_PX`** — measurably does nothing; keep for the
   documented measurement or revert.
4. **Vehicle re-ID embedding** — the viable answer to cross-camera matching, and
   the honest replacement for the plate fingerprint that failed. Whole vehicle,
   ~25× more pixels than the plate, public datasets (VeRi-776, VehicleID).
5. Residual harvest duplicate: `#1`/`a2` are the same SUV (0.916 correlation —
   *lower* than two genuinely different trucks at 0.977, so appearance alone
   cannot separate them; only timing can). Mild for classification, matters for
   re-ID.
