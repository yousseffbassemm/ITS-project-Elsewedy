# Plate Capture Plan — OCR and Plate Colour

**Goal:** for every tracked vehicle, capture its licence plate and derive two
things — the **characters** (OCR) and the **plate colour**.

**Headline finding:** these two have very different resolution requirements, and
on the current target footage only one of them is achievable. Plate colour works
today. OCR does not, and the reason is the camera, not the model. That distinction
drives everything below, so it is established first with measurements.

---

## 1. What the footage can support

Three capabilities, three very different floors:

| capability | needs | why |
|---|---|---|
| plate **detection** | ~20 px plate width | a findable bright rectangle |
| plate **colour** | ~20 px | dominant hue of a large flat band |
| plate **OCR** | **~100 px** | individual glyph strokes must be resolved |

Measured on `samples/street_egypt.mp4` (1280x720) — the clip the whole pipeline is
calibrated for — by three independent methods that agree:

| method | result |
|---|---|
| geometric (the pipeline's calibrated road-plane homography x 0.32 m plate) | **34 px** at the closest point in frame |
| empirical (p95 of observed vehicle box widths x plate/vehicle ratio) | **38 px** |
| direct measurement of a hand-located plate in the clip | **34 x 16 px** |

Against a ~100 px floor, that is **~2.7x short**. For contrast, the 4K stock clip
`vehicles_12s.mp4` measures **98 px** and its plates are legible by eye — a plate
reading `NZ 412GC` was read straight off it. That contrast is what validates the
threshold in both directions: the tool passes footage that is genuinely readable
and fails footage that genuinely is not.

Reproduce any of this:

```bash
python -m tools.plate_footage_check                              # all sample clips
python -m tools.plate_footage_check --videos yourclip.mp4        # any new footage
```

**Why this is not a training problem.** OCR must resolve strokes that were never
captured. At a 32 px plate width, H.264 chroma subsampling leaves the whole plate
roughly 16 px of chroma detail; the characters are gone before the file is
written. A model trained on a million plates still cannot read them, because the
information is not in the file. Egyptian plates are harder than average here —
they carry Arabic letters *and* numerals in the same width, and Arabic glyphs are
distinguished by fine strokes and dot placement.

> This is the single most important thing to agree before GPU time is spent. The
> failure mode it prevents: train OCR on a large online dataset, run it on the
> target clip, get nothing, conclude the model needs more data, and spend weeks
> collecting data that cannot help.

### What would make OCR work

Options, cheapest first:

1. **A dedicated ANPR camera** — the standard answer. ANPR cameras cover one or
   two lanes at a shallow angle with a long lens, and are a *separate* device from
   the traffic-overview camera. One camera cannot do both jobs: overview needs a
   wide field of view, OCR needs a narrow one. This is a normal ITS deployment
   pattern, not a workaround.
2. **Higher resolution on the same view** — 4K instead of 720p is a 3x linear
   gain, which lands right at the ~100 px floor. Workable but with no margin, and
   worse at night.
3. **Tighter framing** — same sensor, longer lens, covering less road. Trades
   coverage for plate size; a good option if a chokepoint can be picked.
4. **Multi-frame fusion** (see §5) — a research angle, not a substitute.

---

## 2. Plate colour — the part that works now

Egypt colour-codes the plate's top band by **vehicle category**:

| band colour | category | supports mentor class |
|---|---|---|
| light blue | private car ("malaky") | A |
| red | truck / tractor | C or D |
| orange | taxi | A (commercial use) |
| brown | commercial vehicle | C / V |
| dark blue | police | — |
| green | diplomatic | — |
| yellow | customs unpaid | — |

This is worth more than it first appears. `docs/finetuning-plan.md` identifies
**A vs V** and **C vs D** as the hardest splits — the ones base COCO cannot express
and a monocular size heuristic got wrong. Plate colour is *independent* evidence
for exactly those classes, and it comes from a signal that survives at 34 px where
the OCR signal does not.

It is used as **evidence, not override** (`plates.COLOR_TO_CODES`). A truck towing
a private car still shows its own red plate, and a repainted plate is not a
reclassification.

### The measurement problem, and the fix

The first implementation used absolute HSV thresholds and **rejected a real
plate**. The red band of the box truck in `street_egypt.mp4` measures **S=57**,
far below the S>=70 a clean plate photo gives, because at a 32 px plate width the
codec leaves the 4 px band about 2 px of chroma. Any floor high enough to be safe
rejects it; any floor low enough to accept it starts calling grey bumpers red.

The fix uses a reference the plate carries with it: **the white body below the
band**. Judging the band relative to that neutral patch cancels codec, white
balance and exposure together, because both regions suffered them equally. On the
same truck the band reads S=57 against a body at **S=16** — unambiguous despite the
low absolute value.

Verified with negative controls — the white truck body, its roof, and road asphalt
are all correctly given no colour. Regression tests in `tests/test_pipeline.py`.

> **Open item:** the margin is currently anchored on a handful of hand-located
> plates. It should be re-fitted against EALPR, which has enough labelled Egyptian
> plates to set it from a distribution rather than an example. This is the same
> reservation `classify.HEAVY_MIN_FRONTAL_AREA_M2` carries.

---

## 3. Models to train

Two models, trained separately because they answer different questions.

| stage | task | classes | dataset |
|---|---|---|---|
| 1 | plate **detection** on the vehicle | 1 (`plate`) | Roboflow *Egypt Car Plate*; Kaggle license-plate sets; CCPD for volume |
| 2 | **character detection** within the plate crop | 27 | **EALPR** — Egyptian, Arabic, day/night |

Stage 2 is character *detection*, not a text recogniser (CRNN/CTC). This is the
approach the published Egyptian ALPR work uses, and it degrades better: a partly
readable plate returns the characters it is confident about rather than one
confidently wrong string.

```bash
python -m tools.train_plates --stage detect --epochs 80  --device 0
python -m tools.train_plates --stage ocr    --epochs 120 --device 0
```

Two augmentation choices matter more than the rest and are set in
`tools/train_plates.py`:

- **`fliplr=0`** — mirroring a plate reverses reading order and maps some Arabic
  glyphs onto each other. This is the augmentation most likely to quietly ruin the
  OCR model, and it is on by default in YOLO.
- **`mosaic=0` for OCR** — stitching four crops together invents plates that do
  not exist and splits real ones.

### Datasets

Not vendored — large, and separately licensed. Check each licence before use.

- **EALPR** — ~2,000 images / ~10,800 annotated characters, 27 classes, Egyptian,
  varied locations and day/night. The key resource for stage 2.
- **Roboflow Universe — Egypt Car Plate** — plate-level boxes for stage 1.
- **CCPD** — ~290k Chinese plates. Useless for Arabic characters, valuable for
  stage 1 *detection*, which is script-agnostic.
- **Open Images** — has a `Vehicle registration plate` class; diverse, good for
  generalisation.

---

## 4. Acceptance criteria

Judge the two capabilities separately, because one of them is footage-limited.

**Plate detection** (works on current footage)
- recall >= 0.85 on vehicles within the ROI at plate width >= 25 px
- measured on a held-out Elsewedy clip the model never trained on

**Plate colour** (works on current footage)
- accuracy >= 0.85 on hand-labelled bands, and crucially a **low false-colour
  rate**: a wrong colour votes on vehicle class, so "unknown" is much cheaper
  than a confident wrong answer
- report a confusion matrix; watch red<->brown and light blue<->dark blue, which
  differ mainly in value and are the pairs the codec damages most

**Plate OCR** (footage-limited — state the condition with the result)
- on footage meeting the ~100 px floor: character accuracy >= 0.90, full-plate
  exact match >= 0.75
- on `street_egypt.mp4`: **expected to produce nothing**, and the report says why.
  Do not tune against this clip — it cannot be fixed from the model side.

---

## 5. Multi-frame fusion — worth a look, not a plan

The pipeline already has the prerequisite that most ANPR systems lack: **stable
track IDs across occlusions** (`pipeline/reid.py`). A vehicle is observed over
tens of frames at slightly different sub-pixel offsets, and fusing those views can
recover genuine detail — this is standard in forensic plate work.

Realistic expectation: 34 px might fuse toward 60-80 px effective. That reaches
"partial digits, sometimes", not reliable plate reads, and Arabic glyphs need more
resolution than Latin ones to disambiguate. Worth an experiment and a good result
to show; **not** a substitute for adequate footage, and it should not be presented
as one.

---

## 6. Deployment

The stage is already wired to degrade rather than fail: with no weights present,
`PlateReader` reports the reason and the rest of the analytics are unaffected.

```powershell
$env:ITS_PLATE_MODEL="models/plate_detect.pt"
$env:ITS_PLATE_OCR_MODEL="models/plate_ocr.pt"
```

`analytics.json` gains a `plates` block: per-vehicle colour, the classes that
colour supports, measured plate pixel width, and — always — an `ocr_note` stating
why OCR did or did not run. That note is deliberate. An unattributed empty OCR
column reads as a broken model.

---

## 7. Privacy

Plates are **personal data**, and Egypt's Personal Data Protection Law (151/2020)
applies. `.gitignore` already blocks `data/` for this reason.

- Do not commit plate crops or source clips, including to a private repo.
- Do not leave footage in a shared Colab session; clear notebook outputs before
  sharing.
- For analytics that only need counts, speed and class, prefer storing a **hash**
  of the plate rather than the plate itself — it still supports journey-time
  matching across cameras without retaining the identifier.
- Agree a retention period before the first deployment, not after.
