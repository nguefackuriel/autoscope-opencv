# Parasite detection: what was measured, in the order it happened

Branch `u/parasite-detector`, 26 and 27 August 2026. Every number here is
reproducible from `eval/`; the held-out ones use the same 50 patients the white
cell work never fitted on (`eval/results/wbc_sweep.json`, `fit_patients`).

## 1. The annotation was being read wrong, by 13 px

Before any detector: overlaying the parasite annotations on the pixels showed
the circle sitting consistently to one side of the chromatin dot. The `Circle`
shape stores two points, and we had read them as a drag across the parasite,
taking the midpoint as the centre. They are the **centre and a point on the
rim**.

Measured on 834 parasites in 25 fit images, distance from the most purple pixel
near the annotation to:

| reading | median | within 8 px |
|---|---|---|
| midpoint of the two points (old) | 12.8 px | 22% |
| **first point (centre)** | **7.9 px** | **51%** |
| second point | 22.7 px | 1% |

On a 10 px object, 13 px off is a miss. Fixed in `annotations.py`, pinned by
`test_parasite_centre_is_the_first_point_not_the_midpoint`. The white cell
results are unaffected: their tolerance was 55 px and they are `Point`s.

Two consequences. The annotator's radius (median 21 px) is a loose circle, not
the size of the dot, so it is only an upper bound. And the residual 8 px at the
median is annotation imprecision we cannot remove, so the matching tolerance
for scoring is 15 px, not the 12 we started with.

## 2. Candidate recall is the ceiling, and a threshold could not reach it

First candidate generator: threshold the stain map (`a - b` in Lab), connected
components, area gate. **Candidate recall 52%.** No classifier can find a
parasite that never became a candidate, so this was diagnosed before anything
else:

| why a parasite was missed | share |
|---|---|
| inside a blob larger than the area gate | 35% |
| near a white cell (exclusion zone) | 4% |
| candidate exists but 12 to 30 px away | 10% |
| no blob at all | 3% |

The missed parasites were not faint. Their stain was stronger than the found
ones (median 86 against 77). Where the field itself is heavily stained, a
global threshold merges everything into one region and the area gate drops
it.

Second generator: morphological **top-hat** on the stain map (stain minus its
opening with a 31 px disc, bigger than any parasite) removes the background,
and a parasite is a **local maximum** of what remains. Two parasites that touch
are two peaks.

| | threshold + components | top-hat peaks |
|---|---|---|
| candidate recall (100 fit images) | 52% | **88%** |
| candidates per image | 300 | 422 |
| true parasites among them | | 9.4% |

The remaining 10 to 12%, opened on 10 fit images (584 parasites):

| why | share |
|---|---|
| inside the white cell exclusion zone | 3.6% |
| outside the 0.96 disc inset or the feature margin | 3.3% |
| a candidate exists, 15 to 25 px away | 2.6% |
| no local maximum of its own | 0.2% |
| top-hat below `PEAK_MIN` | 0% |

Nothing is lost to the peak threshold. The rim loss is by design (the eyepiece
edge is dim and smeared; the white cell count skips it too, so the ratio is
consistent). The exclusion zone is the one worth a look later: some of those
are parasites next to a nucleus, some may be parasite clusters that the
quarter-scale nucleus mask mistakes for a small white cell.

## 3. The look-alikes

Measured on the fit images after stain normalisation (Lab, median):

| | L | a | b | a - b |
|---|---|---|---|---|
| parasite centre | 163 | 155 | 101 | 54 |
| field | 200 | 137 | 117 | 20 |
| unannotated purple blob | 165 | 151 | 104 | 46 |

The third row is the problem. Platelets, stain precipitate and fragments of
lysed cells are the same colour and nearly the same size as a parasite. The
candidate step finds them all; the classifier has to tell them apart, and
colour alone cannot.

## 4. Classifier A: nineteen features and a small network

Per candidate: peak height, stain and lightness in a 4 px core, the same in an
8 to 16 px ring, contrast against the local field, size at half peak,
elongation, solidity, distance to the nearest white cell, number of other
peaks within 20 px. A 19 -> 64 -> 32 -> 1 network, trained in numpy, written
as ONNX, run by `cv.dnn`. OpenCV 5 has no `ml` module, so this is how a
classifier stays inside OpenCV at run time.

Cross-validated **by patient** on the fit set (candidate level), 5 folds:

| threshold | precision | recall | F1 | count bias |
|---|---|---|---|---|
| 0.60 | 0.568 | 0.759 | 0.650 | +33% |
| 0.70 | 0.624 | 0.693 | 0.657 | +11% |
| 0.75 | 0.657 | 0.653 | 0.655 | -0.6% |

Folds ranged from 0.57 to 0.72: patient-to-patient variation is large.

**End to end on the 50 held-out patients** (99 images, 4731 parasites,
threshold 0.7, `eval/results/parasite_eval_holdout_mlp.json`):

| | value |
|---|---|
| precision | 0.656 |
| recall | 0.643 |
| F1 | 0.650 |
| count bias | -0.94 per image (**-2.0%**) |
| count MAE | 20.3 per image |
| parasites per white cell, bias vs truth | +17.4% |
| rank correlation of the ratio | 0.858 |

The count bias is small because false alarms and misses roughly cancel, which
is exactly the situation `parasite_eval.py` was written to expose: the MAE of
20 per image says the per-image number is not trustworthy yet. The ratio
bias (+17%) is larger than the count bias because the white cell count sits
under it and its own +2.5% bias compounds, and because images with few
parasites make the log ratio noisy.

This is the baseline. It is kept as `--model mlp` for the ablation.

## 5. Classifier B: a patch network

The same candidates, but instead of nineteen numbers the classifier sees the
32 x 32 BGR patch around the peak (cut at 40 x 40, randomly cropped, flipped
and turned during training). Three conv blocks (24, 48, 96 channels) to 4 x 4,
then a 96-wide head. Trained with torch in its own venv (`make setup-train`,
because the last torch wheel for this Mac needs NumPy 1), exported to ONNX,
checked against `cv.dnn` to 2.4e-7, and run by `cv.dnn` like the MLP. Pixel
normalisation is inside the graph.

Cross-validated **by patient**, 5 folds, 10 epochs, 42 237 patches from 100
fit images (candidate level):

| threshold | precision | recall | F1 | count bias |
|---|---|---|---|---|
| 0.60 | 0.593 | 0.857 | 0.701 | +45% |
| 0.70 | 0.648 | 0.790 | 0.712 | +22% |
| 0.75 | 0.682 | 0.747 | **0.713** | +9.5% |
| 0.80 | 0.723 | 0.688 | 0.705 | -4.8% |

Against the feature MLP at its best threshold (0.657), that is +5.6 F1 points,
and the bias crosses zero near 0.78, which is the threshold used end to end.
Folds still spread from 0.64 to 0.74: the patient is the unit of variation,
and 100 images is one image per patient.

End to end on the 50 held-out patients: see section 6.

## 6. End to end on the held-out patients

Same 50 patients, 99 images, 4731 annotated parasites, nobody fitted anything
on them. The threshold is the one chosen on the fit set (section 5), not
re-chosen here.

| | feature MLP (thr 0.70) | patch CNN (thr 0.78) |
|---|---|---|
| precision | 0.656 | **0.827** |
| recall | 0.643 | 0.655 |
| F1 | 0.650 | **0.732** |
| count bias | -2.0% | **-20.8%** |
| count MAE per image | 20.3 | 13.4 |
| parasites per white cell, bias | +17.4% | -16.5% |
| rank correlation of the ratio | 0.858 | **0.908** |

Two readings of the same table.

The patch network finds parasites more reliably: better precision by 17
points, better per-image error, better ranking of patients by density. That
is the number a screening tool needs, since the first job is to sort patients.

But it **under-counts by a fifth** at this threshold, where the cross-validated
bias on the fit set was near zero. The held-out patients carry more parasites
per image (48 against 40), and a denser field is where the candidate step and
the classifier both lose recall: parasites sit closer together, and a peak
with neighbours looks less like the training median. The MLP's -2% is not
better calibration, it is false alarms and misses cancelling, as its MAE of
20 shows.

The threshold sensitivity table (`parasite_eval_holdout_cnn.json`, `sweep`)
is recorded for information. It is not used to pick the threshold, because
choosing on the held-out set would make it a fit set.

What comes next, in order: more training images (one per patient so far, the
fit set has twelve), then the crowded-field recall, which is the same
problem that limited the white cell count.
