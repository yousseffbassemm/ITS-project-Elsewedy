# ITS Traffic Analytics — Methodology & Vision

**Elsewedy Electric · AI Department · Intelligent Transportation Systems prototype**

---

## 1. Problem & motivation

Egyptian urban corridors suffer heavy, poorly-instrumented congestion. Traffic
authorities and operators mostly rely on manual observation or fixed-timing
signals, with little quantitative visibility into *how many* vehicles pass, *how
fast* they move, and *when* a road tips into gridlock. Cairo alone loses billions
of EGP per year to congestion (fuel, time, emissions).

Elsewedy Electric already delivers the physical layer of smart infrastructure
(cables, smart meters, substations, city-scale electrical works). An ITS layer —
turning ordinary street-camera video into structured traffic metrics — is a
natural, high-value extension: it feeds signal optimisation, corridor planning,
and congestion-cost reduction using footage that already exists.

This prototype demonstrates that a **single street video** can be turned into
actionable traffic intelligence with commodity, CPU-only hardware and open models.

## 2. Objectives

1. **Detect & count** vehicles by type and direction of travel.
2. **Estimate speed** per vehicle (km/h).
3. **Quantify congestion** and its evolution over time.
4. Deliver all of the above through an **operator-facing web dashboard** where a
   user uploads a clip and reads the results — no ML expertise required.

## 3. Approach & pipeline

The system is an offline (batch) computer-vision pipeline decoupled from a web
layer, because inference here is CPU-only (no GPU) and therefore slower than
real-time.

```
video → [YOLOv8 detection] → [ByteTrack tracking] → per-frame tracked objects
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                          ▼
     [line-crossing counting]   [speed via homography]   [occupancy/congestion]
              └─────────────────────────┼─────────────────────────┘
                                        ▼
                        annotated.mp4  +  analytics.json  →  dashboard
```

### 3.1 Detection & classification — YOLOv8 → mentor taxonomy
Ultralytics YOLOv8s detects the relevant COCO classes (**car, motorcycle, bus,
truck, bicycle, person**), which are then mapped to the mentor's 7-class scheme:

| Code | Class          | Source |
|------|----------------|--------|
| A    | Private car    | COCO car |
| C    | Light truck    | *needs fine-tuned model* |
| D    | Heavy truck    | COCO truck |
| E    | Bus            | COCO bus |
| G    | Motorcycle     | COCO motorcycle / bicycle |
| V    | Van            | *needs fine-tuned model* |
| F    | Unknown        | fallback |

**A, D, E, G are produced reliably.** Base monocular COCO cannot separate a van
from a car, or a light from a heavy truck, without labelling e.g. sedans as vans
— a size heuristic on the homography width was tried and rejected for this
reason. **C** and **V** are therefore part of the published scheme but are left
to a model *fine-tuned on the 7 classes* (the recommended next step); v1 maps
each detection to its nearest reliable class. On this footage the traffic is
private cars (A) and box trucks (D), which map cleanly.

### 3.2 Tracking — ByteTrack + re-identification
ByteTrack assigns an ID to each vehicle across frames. It is used deliberately
instead of DeepSORT: ByteTrack has **no deep re-identification network**, so it
stays fast without a GPU. Stable IDs are what make *counting* (one crossing per
vehicle) and *speed* (displacement of the same ID) possible.

**ByteTrack alone is not sufficient here, and we measured it.** Over 900 frames
of `street_egypt.mp4` it produced 71 IDs for far fewer vehicles: **18 of them
(25%) were continuations** of a vehicle it had already seen, one car being split
across seven IDs (`73→76→77→81→88→90→95`). It associates on motion and IoU only,
so once a vehicle is occluded for longer than `track_buffer` its track is retired
and the same car returns as a new vehicle. That corrupts everything downstream —
and it *loses* counts too: a fresh ID has no previous side-of-line recorded, so a
vehicle whose ID changes exactly at the counting line is never counted.

Two layers fix it, both keeping ByteTrack as the tracker:

1. **Tuned ByteTrack** (`pipeline/bytetrack.yaml`) — `track_buffer` raised from 30
   to 75 frames (1.2 s → 3 s), since occlusions here routinely exceed a second.
2. **Re-identification** (`pipeline/reid.py`) — a new raw ID is compared against
   recently-lost tracks on four independent gates that must all pass: motion
   (where the lost track would now be at its last velocity), appearance (HS colour
   histogram of the vehicle's own pixels), size, and class *group*. Grouping
   matters: base YOLO flip-flops between car/truck/bus on one vehicle — one lorry
   was labelled bus in 61 frames and truck in 315 — so an exact class match would
   block legitimate re-attachments.

A third failure mode surfaced during verification and is handled separately.
Sometimes the **detector emits two overlapping boxes for one vehicle** and
ByteTrack IDs each, so the ids *coexist* rather than being separated by a gap —
the re-id path above can never merge those. Eight such pairs were found and
confirmed by eye. They are collapsed when boxes overlap (IoU) **and** their
ground-contact points nearly coincide (two different vehicles, one behind the
other, overlap in the image but touch the road at clearly different points),
sustained over several frames. The redundant detection row is then dropped, so
one vehicle cannot vote twice for its class or append two positions per frame.

Gates combine evidence rather than veto each other. Requiring position **and**
appearance **and** size to agree rejected matches whose predicted position was off
by as little as 2 px, because a colour histogram drifts as a vehicle recedes and
shrinks. Motion evidence that strong now settles a match alone, subject to an
absolute 40 px cap so a long gap — which widens the tolerance — cannot quietly
turn into a weak match. Beyond that cap, appearance must still corroborate.

A fourth issue was pure noise rather than error. This camera also watches the
**opposite carriageway**, whose traffic can never reach the counting line but was
still consuming IDs: 99 of 126 IDs on the 90 s clip belonged to vehicles we do not
analyse, which is why on-screen IDs looked erratic. ByteTrack still sees the whole
frame (it needs unbroken motion history), but IDs, counting and annotation are now
restricted to the analysed carriageway, with a margin so a vehicle is not dropped
and re-acquired at the boundary. IDs fell from 126 to 28 with the vehicle count
unchanged at 17.

Gates are deliberately conservative: a missed stitch costs one duplicate ID; a
wrong stitch welds two vehicles together and corrupts both. Measured effect:

Measured over the full 90 s clip:

| | before | after |
|---|---|---|
| IDs created | 126 | **28** |
| vehicles counted | 17 | 17 (unchanged) |
| ID switches across occlusion | 18¹ | **0** |
| duplicate-box pairs | 11¹ | **0** |
| implausible position jumps (false-merge signal) | 0 | **0** |

¹ measured on a 900-frame segment before the fixes.

Verified against ground truth, not just metrics. Every counted vehicle's crop was
inspected: all 17 are distinct vehicles. The candidate duplicate pairs were each
checked individually — two lookalike white vans turned out to be **visible
simultaneously for 109 frames, 169 px apart**, so they cannot be one vehicle, and
the zero-overlap pairs were a vehicle leaving the top of frame while another
entered the bottom, 550–750 px apart.

**What is not claimed.** These are zero *measured* errors on this footage, checked
vehicle by vehicle — not a guarantee for all video. Two identical microbuses in the
same lane, one fully occluded for several seconds, is genuinely ambiguous from a
single camera. The audit is re-runnable, which is the useful property.

### 3.3 Counting — line crossing
A virtual line is placed across the lanes. For each tracked vehicle we track
which side of the line its ground-contact point sits on; when that side flips, we
record one crossing, attributed to the vehicle's **class** and **direction**
(in/out). This yields totals like "42 cars in, 8 trucks out."

**A vehicle already past the line when the clip begins is deliberately not
counted.** It never crosses the section, so it is not flow past that section.
On `street_egypt.mp4` this is a black SUV sitting at y≈400 in frame 0, already
beyond the line at y=432, which then drives away: the pipeline reports **17**
where a naive manual tally of the clip gives 18. The 17 is the intended answer —
a counting line measures traffic crossing it, not vehicles visible at t=0.

The crossing test is **geometric, not positional**. When the side flips we
interpolate the exact crossing point and project it onto the line segment. An
earlier version instead checked whether the vehicle's post-crossing *position*
sat within ±40 px of the line, which silently failed the moment a vehicle moved
further than that between processed frames — i.e. at `frame_stride` 2–4, the very
setting the README recommends for faster runs, where a vehicle covers 74–111 px
per step. Measured: a vehicle crossing the line was counted at stride 1 and
**lost at strides 2, 3 and 4**. The current test depends only on the geometry, so
counts no longer depend on how coarsely the video is sampled.

*(Implementation note: counting is done directly rather than via
`supervision.LineZone`, which depends on a 2-D `np.cross` removed in NumPy 2.5.)*

### 3.4 Speed — perspective transform
Camera images foreshorten distance, so pixels ≠ metres. We define a homography
(`cv2.getPerspectiveTransform`) from four road-plane points to a real-world
rectangle in metres, transform each vehicle's ground point into that metric
space, and compute speed from metric displacement over time, smoothed across a
~1-second window and converted to km/h.

**Robustness:** speed is only measured in a reliable near/mid image band (the far
field near the vanishing point amplifies a few pixels into many metres); each
vehicle needs enough tracked history before a speed is emitted; and results are
aggregated **per vehicle** (median speed per track) so a lingering slow vehicle
can't dominate and jitter is damped.

**Calibration for this camera** (`street_egypt.mp4`) — and how we know it is
right. The earlier calibration was set by "speed plausibility": the quad length
was chosen so speeds looked reasonable. That is circular, and it was wrong.

The road itself provides a ruler. Painted dashes are evenly spaced, so after a
*correct* rectification their measured pitch must be **constant**. Under the old
quad it was not — 25.1, 18.4, 14.5, 11.7 m from far to near, a 114% spread that
shrank with distance, proving the quad was not a road-plane rectangle. Speeds
were therefore wrong in a *position-dependent* way, and inflated: the clip
reported ~100 km/h averages.

The quad is now built from the two **measured** carriageway edge lines. Those are
genuinely parallel on the road, so mapping the quad they bound to a rectangle
sends both vanishing points to infinity and rectifies the plane correctly. The
dash pitch then comes out **27.2 / 27.1 / 27.9 / 28.8 — a 5.9% spread**, which is
the validation. Setting that pitch to the 12 m highway standard (3 m dash + 9 m
gap) fixes the quad length at **43.6 m, not 78 m**: speeds had been inflated
≈1.79×. A second, independent check — measuring each vehicle in three separate
sub-bands of the image — now agrees to **8%**, where a bad homography diverges.

What remains assumed is only `dash_pitch_m`. Change that one number if this road
uses a different marking standard and the whole scale moves linearly; a surveyed
distance between two ground marks removes the assumption entirely. Everything
else is measured rather than chosen.

### 3.5 Per-lane analytics & throughput
The carriageway is split into **4 lanes** whose boundaries sit on the actual
painted stripes. These are **measured, not eyeballed**: a 60-frame median
background removes the traffic, a morphological top-hat isolates the markings,
and a Hough fit recovers each line as `x = a·y + b`. Rectifying the road plane
confirms the fits are consistent to ~1% across image rows.

That measurement corrected a real error. The previous dividers were placed by
eye and did not land on the stripes; they also made **L3** the widest zone. The
measured layout gives L1 25%, L2 20%, L3 23% and **L4 31%** of the carriageway —
the rightmost lane is the widest, because it runs from the last stripe out to the
barrier and absorbs the paved shoulder. Sampling where vehicles actually drive
confirms the layout: traffic clusters into exactly four bands, with nothing
travelling beyond the solid right-edge line, so the shoulder only ever adds a
vehicle to L4 rather than creating a phantom fifth lane.

The counting line was widened at the same time. It previously spanned x=0.15–0.87
while lane 1 sits at x≈0.00–0.23 at that row, so vehicles in the leftmost lane
crossed *outside* the line and were silently missed. It now spans the full
carriageway between the two measured edges.

Each counted vehicle is assigned to
the lane its **ground footprint occupies most** — sample points along the box's
bottom edge are projected into the strip and the majority lane wins, which is
exactly the "straddling → count it in the lane it covers most" rule. From the
per-lane counts we derive **flow rate (veh/h)** per lane and overall, the
**busiest lane**, and a **lane-balance** figure — standard traffic-engineering
KPIs for spotting lane under/over-use and sizing signal/lane changes. Coloured
lane zones with live counts are drawn on the annotated video.

### 3.6 Congestion — occupancy + speed
Per frame we measure the fraction of the road region-of-interest covered by
vehicle boxes (occupancy), the vehicle count in the ROI, and the average speed,
then classify the moment as **Free-flow / Moderate / Heavy / Jam**. Low average
speed with meaningful occupancy escalates the level. Aggregated over time this
gives a congestion timeline and the road's peak-congestion periods.

Two corrections make that headline trustworthy:

* **Occupancy is an image-space area fraction**, so a single lorry passing close
  to the camera can cover a third of the visible ROI on its own. Congestion is a
  property of traffic, not of one large nearby vehicle, so the level is capped by
  how many vehicles are actually present: 0–1 can never exceed Free-flow, 2 caps
  at Moderate, 3–4 at Heavy. Genuine queues are unaffected.
* **The headline level must be one the road sustained.** Reporting the worst level
  ever *touched* meant a clip that was Free-flow 96.8% of the time and Heavy for
  0.4% (a fraction of one second) headlined the dashboard as "Heavy".
  `worst_level` now requires at least a 1% share, sub-second peak periods are
  discarded, and the raw maximum is still published as `peak_instant_level` so
  nothing is hidden.

## 4. Metrics produced
- Total vehicles; breakdown by class and by direction (in/out).
- Per-vehicle and average speed; 85th-percentile speed; speed histogram; average
  speed per class.
- Congestion level per second; % of time at each level; peak periods.
- Vehicle-count and average-speed time series (per second).
- An annotated MP4 (boxes, IDs, class, live speed, counting line, HUD counters).

## 5. Egyptian-street considerations
- **Vehicle mix**: microbuses, tuk-tuks and animal carts are common. In v1 they
  map to the nearest COCO class (microbus→bus/truck, tuk-tuk→car/motorcycle). A
  fine-tuned model with native *tuk-tuk* and *microbus* classes is the clear next
  step for local accuracy.
- **Lane discipline**: informal lane use and dense weaving stress the tracker;
  ByteTrack handles occlusion reasonably, but very dense scenes benefit from a
  higher-resolution model and lower frame stride.
- **Camera variety**: fractional (resolution-independent) geometry lets the same
  config transfer across cameras; only the four calibration points change.

## 6. Business value for Elsewedy
- **Signal-timing insight** — directional counts + congestion timelines justify
  and tune adaptive signal plans.
- **Congestion-cost quantification** — turning "the road is busy" into measured
  vehicle-hours and speed drops supports ROI cases for infrastructure spend.
- **Leverages existing assets** — runs on ordinary footage and CPU hardware,
  complementing Elsewedy's smart-infrastructure portfolio without new sensors.
- **Extensible platform** — the same base supports enforcement (violations,
  ANPR), corridor/multi-camera views, and eventually live edge deployment.

## 7. Limitations (v1)
- CPU-only ⇒ offline batch, not live; small model trades some recall for speed.
- Speeds are estimates until the homography is calibrated to the real scene.
- No native Egyptian vehicle classes yet; no license-plate/violation logic.
- Single counting line and single ROI per video (multi-lane zones are future work).

## 8. Roadmap
1. Calibrate on the target Cairo street video (counting line + speed homography).
2. Fine-tune YOLO on Egyptian classes (tuk-tuk, microbus, cart).
3. OpenVINO export to use the Intel iGPU for a speed-up.
4. Violations & ANPR (red-light running, wrong-way, plates).
5. Multi-camera corridor dashboard; then a live/edge (Jetson) mode.

---

*Prototype validated on a highway reference clip: end-to-end detection →
tracking → counting → speed → congestion → annotated video + dashboard, all on
CPU. Ready to be pointed at the real Egyptian street footage for calibration.*
