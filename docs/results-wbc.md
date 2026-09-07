# White cell detection, first measured baseline

24 August 2026. `src/autoscope/segmentation.py`, scored by `eval/wbc_eval.py`.

White cells come first because they are the denominator. Parasites per 200 white
cells is what becomes a density, so an error here moves every number downstream.

## The honest number

Fitted on 100 patients, reported on the **50 patients the fit never saw**.
Stain normalisation on, parameters refitted with it (`seed_fraction=0.45`,
`close_px=24`):

| | current | before refit | first baseline |
|---|---|---|---|
| precision | **0.783** | 0.770 | 0.657 |
| recall | **0.803** | 0.795 | 0.699 |
| F1 | **0.793** | 0.782 | 0.677 |
| count bias | **+2.5%** | +3.3% | +6.4% |
| count MAE | **4.7 cells/image** | 5.4 | 7.0 |

144 images, 3046 annotated cells, all three columns on the same patients.

Normalisation also relaxed the bias/F1 trade. The raw sweep forced a choice
between F1 0.745 at +18% bias and F1 0.696 at +2%; the normalised grid offers
F1 0.815 at +0.5% on the fit set. We still passed over its best F1, 0.850 at
+5.9% bias, for the same reason as before: the count is the denominator.

Always quote the held-out column. On the fit set this detector reads +0.5%
bias against +2.5% held out, and that gap is what scoring yourself on your own
tuning data buys you. It is why the split exists.

## Why not the best F1

The parameter sweep had a better F1 on offer: **0.745**, at `SEED_FRACTION=0.75`
with a 24 px close, against 0.696 for what we shipped. We did not take it. It
over-counts white cells by 18%, and an 18% surplus in the denominator makes
every parasitaemia we report 18% too low, for every patient, in the same
direction.

A systematic error in the number a clinician acts on is worse than a slightly
worse localisation score. We optimised the count, not the metric.

## Two bugs this work found

**Annotations are in a rotated frame.** The ground truth header reads
`count,4032,3024` while the images are 3024 wide by 4032 tall. The coordinates
were recorded on a landscape view, and the mapping into the stored portrait
image is `x = width - coord2, y = coord1`. Seven candidate transforms were
scored by how much darker the pixels under the points were than the lit field;
the winner led by 131 L units against 47 for the runner up. Overlaid on real
images it puts a nucleus under 11 of 12 markers, against 1 of 12 for the plain
reading. `tests/test_annotations.py` pins it.

Without this the detector would have been scored against noise while producing
numbers that looked entirely reasonable.

**The watershed used a global distance threshold.** Seeds were cut at 0.55 of
the whole image's maximum distance transform. With one large clump at distance
30 and a small cell at 8, the small cell got no seed and vanished rather than
being split. Recall was 0.497. Computing the threshold per blob took it to
0.702.

## What to improve next

- **Cells at the rim.** The disc mask cuts nuclei touching the edge, and they
  are neither cleanly found nor cleanly excluded. Deciding to ignore a margin,
  and telling the density code how much area was ignored, is probably worth
  more than any threshold change.
- **Stain variation.** Nothing normalises staining before segmentation yet.
  Stage 2 exists in the plan and not in the code.
- **The +6.4% bias.** Correct it in the density, or fix its cause. Do not
  calibrate it away with a constant fitted on the NLM set: it will not transfer
  to Tanzania, and the generalisation claim in the report depends on that set
  staying untouched.
- **Precision at 0.66** means a third of detections are not on an annotated
  cell. Look at those before assuming they are all wrong; some may be real cells
  the annotator skipped.

## 2 September: the stain statistics were wrong, and fixing them is worth six points

`results-stain.md` has the bug: the Lab statistics behind the normalisation
were accumulated in float32 and off by up to 6.5 units per image. Every
parameter above was fitted on images normalised with that error. With the
statistics corrected and the reference re-measured, the old parameters
over-counted by 9.6% on the held-out patients (F1 0.755), because the
corrected reference moves the background `a - b` level by about 8 units and
the nucleus threshold of 55 no longer sat where it was meant to.

So the sweep was redone with `STAIN_MIN` in the grid (55, 60, 65, 70), same
100 fit patients, same seed, same selection rule (count bias within one
point, then best F1). Chosen: `STAIN_MIN = 60`, `SEED_FRACTION = 0.75`,
`CLOSE_KERNEL_FULL = 24`, fit-set F1 0.872 at +0.3%. Passed over: F1 0.879
at -6.7% bias, for the same reason as before.

Held-out, the same 50 patients and 144 images as every column above:

| | before the fix (24 Aug) | **after (2 Sept)** |
|---|---|---|
| precision | 0.783 | **0.844** |
| recall | 0.803 | **0.859** |
| F1 | 0.793 | **0.851** |
| count bias | +2.5% | **+1.7%** |
| count MAE | 4.7 cells/image | **3.65** |

A first re-sweep was run with 120 images by mistake, which pulled 20
held-out patients into the fit set and shrank the held-out set to 30. It was
caught from the patient count in the output, discarded
(`wbc_sweep_fixedstats_120_contaminated.log`), and the script's default is
now 100 so it cannot happen silently again. The number above comes from the
clean run (`wbc_sweep_fixedstats_100.log`, `wbc_eval_holdout_refit.json`).

## 3 September: the pink slides, and why a nucleus has to be dark as well

Five of the 50 held-out patients get fewer than 80% of their annotated white
cells, two of them catastrophically: TF82 19%, TF48 40%. Both are
overstained, pink slides (raw Lab mean *a* 164 and 158 against 135 to 151
on the rest). After normalisation, which clamps its gain at 0.5 and so
cannot fully pull them in, their nuclei sit at `a - b` 44 to 46 while the
gate is 60 and a normal slide's nuclei sit at 95 to 118. In the picture
(`docs/figures/wbc-look-TF82_15_1_9.png`) every annotated nucleus is an
obvious dark round blob; the detector finds none of them.

Measured on 98 fit patients, one image each
(`eval/results/nucleus_contrast_fit.json`): the field sits at L 200 on
every slide (p5 198), nuclei at L 29 to 96 (p5 to p95), so **darkness gives
a gap of at least 100 on every slide**, while nucleus `a - b` runs from 37
(TF91) to 140 and falls under the gate on the six pinkest fit slides too.

So a nucleus is now "purple, or dark and at least faintly purple":
`stain > STAIN_MIN or (L < L_DARK_MAX and stain > STAIN_MIN_DARK)`. The
`a - b` floor keeps dark dirt out. Prototype, two images per slide:

| slide | kind | purple only | dark or purple |
|---|---|---|---|
| TF82_15_1_9 | held-out, pink | 0/19, 0/21 | 17/19, 20/21 |
| TF48_UW31 | held-out, pink | 28/45, 6/39 | 42/45, 38/39 |
| TF91_15_9A | fit, weakest a - b | 0/16, 9/17 | 14/16, 23/17 |
| TF87_15105A | fit | 27/47, 38/49 | 49/47, 47/49 |
| three normal fit slides | | unchanged | unchanged, to the cell |

(found/annotated.) `L_DARK_MAX` is in the sweep grid with 0 meaning off, so
the choice stays measured; the held-out number follows from chain 2.

The sweep chose `L_DARK_MAX = 130`, `STAIN_MIN = 65`, `SEED_FRACTION = 0.75`,
`CLOSE_KERNEL_FULL = 24`: fit-set F1 0.915 at -0.1%, against 0.872 for the
best purple-only setting in the same grid. Held-out, the same 50 patients
and 144 images:

| | 24 Aug | 3 Sept, statistics fixed | **3 Sept, dark path** |
|---|---|---|---|
| precision | 0.783 | 0.844 | **0.901** |
| recall | 0.803 | 0.859 | **0.919** |
| F1 | 0.793 | 0.851 | **0.910** |
| count bias | +2.5% | +1.7% | +2.0% |
| count MAE | 4.7 cells/image | 3.65 | **2.29** |

The slides that were failing, found/annotated on the two evaluation images
per patient:

| slide | before (all fields) | now |
|---|---|---|
| TF82_15_1_9 | 0.19 | **0.91** (42/46) |
| TF48_UW31 | 0.40 | **1.00** (21/21) |
| TF2N_159R | 0.63 | **1.10** (46/42) |
| TF117_R1 | 0.71 | **1.07** (105/98) |
| TF47_UW29 | 0.77 | **0.93** (43/46) |
| TF70_Scr440 | 1.51 | **1.06** (56/53) |
| TF35_Sc521 | 1.89 | **1.55** (62/40) |
| TF1N_1522 | 1.50 | **1.34** (63/47) |

Patients with fewer than 80% of their cells: 0 of 50; with more
than 140%: 2. The denominator failure of section 13 in
`results-parasites.md` is closed on this data. What remains is the
over-count on slides like TF70, where small purple clumps pass the area
gate, and that is a shape question rather than a colour one.


## 6 September: the denominator is now the larger error

With the chain-3 parasite network (`results-parasites.md` section 15) the
parasite count on the 609 held-out fields is within -1.1% and the density
through the protocol is -8.1%, because the white cell count is **+6.1% on
the 1 000 to 10 000/µL band and +5.3% above it**, and +26 to +30% on
TF70_Scr440, TF1N_1522 and TF46_UW23 (`eval/density_bands.py`). Those are
the slides where the dark-nucleus rule admits small purple clumps through
the area gate: the 3 September table above already showed TF70 at 1.06,
TF35 at 1.55 and TF1N at 1.34 of their annotated count. The fix is a shape
gate on the clump, not a colour one, and it comes after the parasite
detector because it is now the smaller of the two numerator-denominator
errors only on the dense band.

## 6 September, afternoon: a blob has to be mostly dark, and the area floor drops to 1 200

`eval/wbc_fp_look.py`, 197 fit images, 2 per patient: 4 591 detections,
4 147 hits, 444 false alarms (9.7%). What the false alarms look like
against the hits (medians): brightness L 135 against 83, minimum L 102
against 56, `a - b` 81 against 104, solidity 0.81 against 0.93, and the
share of the blob's pixels under `L_DARK_MAX`, **34% against 94%**. They
are the bright purple clumps the colour rules admit through the area gate:
parasite clusters, platelet heaps, precipitate. A gate on that share
(`DARK_FRACTION_MIN`) at 0.3 removes 214 of the 444 for 31 of the 4 147
hits; at 0.5, 270 for 65.

Two sweeps, same 100 fit images, same selection rule (count bias within
one point, then best F1).

**The gate alone does not ship** (`wbc_sweep_dark_fraction.log`, 48
points). At every seed fraction it lifts precision by eight to ten points
with recall unchanged, and reads −6.6% to −10.8% count bias, because on
the fit set the false alarms were balancing the misses (recall 0.91).
The rule kept it off.

**The gate with the recall given back** (`wbc_sweep_gate_recall.log`, 37
points): the gate at 0.3 or 0.5 with `AREA_MIN_FULL` in (1 200, 1 500,
1 800) and `STAIN_MIN_DARK` in (20, 30), the shipped setting as control.
Chosen: gate 0.3, area floor **1 200 px²** (a small nucleus, or a lobe
fragment the watershed cut off), dark path purple floor 30, seed 0.75,
close 24: fit-set F1 0.921 at +0.5%, control 0.915 at −0.1%, passing over
0.933 at −6.4%.

Held-out, the same 50 patients and 144 images as every column since
August:

| | 3 Sept, dark path | **6 Sept, gate + area 1 200** |
|---|---|---|
| precision | 0.901 | **0.921** |
| recall | 0.919 | **0.931** |
| F1 | 0.910 | **0.926** |
| count bias | +2.0% | **+1.1%** |
| count MAE | 2.29 cells/image | **1.82** |

On every held-out field (`density_bands.py`, chain of 6 September
afternoon): the white cell bias falls from +6.1% to **+3.9%** on the
1 000 to 10 000/µL band and from +5.3% to +4.0% above it; TF70_Scr440
from +30% to +21%, TF1N_1522 from +26% to +7%, TF46_UW23 out of the
worst list. With the parasite network unchanged, the density error per
slide through the WHO protocol goes from 28.3% to **25.7%** and the 95%
limits from −49% / +65% to **−47% / +59%**; reading every field, 70% of
slides are within 25% (from 64%). What remains of the over-count is on
the slides where the parasites themselves cluster into nucleus-sized
purple masses (TF70 +21%), and that is the learned detector's problem as
much as this stage's.
