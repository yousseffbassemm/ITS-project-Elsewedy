# Fine-Tuning Plan — Unlocking the Full 7-Class Taxonomy (C & V)

**Goal:** train a detector that natively outputs the mentor's 7 classes so that
**C (Light truck)** and **V (Van)** — which base COCO YOLO cannot separate from
trucks/cars — are produced directly and reliably, alongside A, D, E, G.

**Why this is needed:** base YOLOv8 is trained on COCO (car / truck / bus /
motorcycle). A monocular size heuristic to split car↔van and light↔heavy truck was
tested and rejected (it labelled sedans as vans). Reliable separation requires a
model *trained on examples of each class*. This plan delivers that.

---

## 1. Target classes & labelling rules
A single, unambiguous rulebook is the #1 driver of accuracy. Define each class by
what the annotator sees:

| Code | Class        | Includes | Distinguishing cue |
|------|--------------|----------|--------------------|
| A | Private car   | sedan, hatchback, SUV, pickup (personal) | ≤5 seats, single body |
| C | Light truck   | pickup w/ cargo bed loaded, small box/flatbed ≤3.5 t | short 2-axle, cab+small bed |
| D | Heavy truck   | lorry, box truck, trailer, tanker, dumper | large box / 3+ axles / trailer |
| E | Bus           | city bus, coach, **microbus/minibus** | long passenger body, window row |
| G | Motorcycle    | motorcycle, scooter, (bicycle → G) | two-wheeler |
| V | Van           | panel van, minivan, closed cargo van | boxy, no rear side windows, 1 unit |
| F | Unknown       | occluded / ambiguous / other (cart, tuk-tuk) | fallback only |

> **The hard boundaries** — write these into the guide with 3–5 example crops each:
> A↔V (SUV vs panel van), C↔D (pickup-with-bed vs box truck), E↔V (microbus vs
> van). These three pairs are where annotators (and models) disagree; nail them.

## 2. Data collection
- **Primary (in-domain):** frames from Elsewedy highway CCTV — the target domain.
  Sample across **time-of-day, weather, and cameras** (day/dusk/night, clear/haze)
  to avoid a model that only works at 05:44 in clear light. Pull frames every
  1–2 s from many clips; dedup near-identical frames.
- **Supplement (for rare classes):** C and V are rare on an open highway, so
  oversample them from public sets — **UA-DETRAC**, **AI City Challenge**,
  **BDD100K**, **Roboflow Universe "vehicle types"** — remapped to our codes.
- **Balance target:** ≥ **800 instances/class**, and specifically **≥ 400 each for
  C and V** (the rare, high-value ones). Total ~4–6k labelled images.

## 3. Labelling workflow (bootstrap + human-in-the-loop)
1. **Auto pre-label** every frame with the current `yolov8s` pipeline → boxes for
   car/truck/bus/moto are ~free; the classifier's A/D/E/G become draft labels.
2. **Human correction** in **CVAT / Label Studio / Roboflow**: fix classes, and —
   the whole point — relabel the trucks into **C vs D** and the wide cars/vans into
   **A vs V**, following the rulebook.
3. **Active learning:** after a first model, run it on new footage, surface
   **low-confidence and C/V predictions** for review first (highest information
   gain per label). Iterate 2–3 rounds.
4. **QA:** double-label a 5% sample; measure inter-annotator agreement on the three
   hard pairs; reconcile the guide until agreement > 90%.

## 4. Dataset split
- **70 / 20 / 10** train / val / test, split **by clip and time window** (never let
  frames from the same 5-second window land in both train and test — that inflates
  metrics). Keep a **fully held-out Elsewedy clip** the model never sees in training
  as the real acceptance test.

## 5. Model & training
- **Model:** fine-tune **YOLOv8s** (or v8m/v11s if the GPU allows) from COCO weights
  — transfer learning converges fast and needs less data.
- **Config:** `imgsz=736` (matches our 720p pipeline), mosaic + HSV + scale
  augmentation, **class-balanced sampling / copy-paste oversampling for C & V**,
  ~100 epochs with early stopping on val mAP, cosine LR.
- **Compute:** this machine is CPU-only — train on a **free GPU (Colab / Kaggle)**
  or a small cloud GPU; only the ~30 MB `best.pt` comes back. (Locally we can still
  *run* inference, and export **OpenVINO** to use the Intel iGPU for a speed-up.)

```bash
# once the dataset (data.yaml with names: [A,C,D,E,G,V,F]) is ready:
yolo detect train model=yolov8s.pt data=data.yaml imgsz=736 epochs=100 \
     mosaic=1.0 hsv_v=0.4 patience=20 name=its7class
```

## 6. Evaluation & acceptance
- **Metrics:** overall **mAP@0.5** and **mAP@0.5:0.95**, but judge success on
  **per-class AP and recall — especially C and V** — plus a **confusion matrix** to
  watch the A↔V and C↔D leakage specifically.
- **Acceptance targets (starting bar):** mAP@0.5 ≥ 0.80 overall; **recall ≥ 0.70 for
  C and V** on the held-out Elsewedy clip; A/D/E/G ≥ 0.85.
- **Report** per-class metrics next to the current heuristic baseline to show the lift.

## 7. Deployment (drop-in)
Put `best.pt` in the project and set `ITS_MODEL=best.pt`. That is the whole
change: `classify.native_code_map` recognises a model that emits the mentor
taxonomy (the test is whether it knows **C** or **V**, which COCO cannot express)
and `config.scheme_for_codes` derives the class scheme every other stage uses, so
the size heuristic is bypassed automatically and nothing else needs editing.

> This used to be less true than it read. Counting, speed, congestion and the
> re-id class gate each hard-coded the COCO ids, so a 7-class model — whose class
> 0 is **A**, not *person* — had every private car counted as a **pedestrian**,
> and **G** and **F** dropped from the counts, the speed sample and the occupancy
> measure entirely. All four now take their class meanings from
> `config.ClassScheme`; `tests/test_pipeline.py` covers each class end to end.

Name the classes so the mapping is unambiguous — `A`/`C`/`D`/`E`/`G`/`V`/`F`, or
descriptive names (`private car`, `light truck`, `panel van`, …) which
`classify._NAME_TO_CODE` also accepts. A name outside that vocabulary makes the
model fall back to the COCO path, which will not fit it.

## 8. Milestones
| Phase | Output | Rough effort |
|-------|--------|--------------|
| 1 | Labelling guide + 500 seed images labelled | 3–4 days |
| 2 | v0 model + active-learning loop to ~4k images | 1–2 weeks |
| 3 | Trained model hitting acceptance targets | few days |
| 4 | `best.pt` deployed + OpenVINO export + report | 1–2 days |

## 9. Risks & mitigations
- **Class imbalance (C/V rare)** → oversample + public-data supplement +
  focal / class-weighted loss.
- **Domain shift (night/rain/other cameras)** → collect those conditions explicitly.
- **Annotation drift on hard pairs** → the rulebook + QA agreement loop above.
- **Microbus ambiguity (E vs V)** — very common in Egypt → dedicate example crops
  and a firm rule (passenger windows ⇒ E; panel/cargo ⇒ V).
