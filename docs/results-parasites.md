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
bias on the fit set was near zero. The first guess was crowding, and it was
wrong. Split by how many parasites the image holds:

| parasites in truth | images | recall | precision | count bias |
|---|---|---|---|---|
| 0 to 9 | 33 | 0.588 | 0.495 | +18.7% |
| 10 to 29 | 28 | 0.572 | 0.691 | -17.3% |
| 30 to 79 | 17 | 0.554 | 0.771 | -28.1% |
| 80 and more | 21 | 0.697 | 0.889 | -21.6% |

Recall is flat to slightly *better* on dense fields. What changes with density
is precision: on a nearly clean field half of what the network calls a
parasite is not one, and on a heavy field nine in ten are. The under-count
comes from a threshold that is too high everywhere (recall 0.55 to 0.70
against a candidate ceiling of 0.88), and the dense images hold most of the
parasites so they dominate the total. The over-count on sparse fields is the
harder problem: a screening tool must not invent parasites on a negative
slide. The MLP's -2% is not better calibration, it is false alarms and misses
cancelling, as its MAE of 20 shows.

The threshold sensitivity on the held-out set, recorded for information only.
It is not used to pick the threshold, because choosing on the held-out set
would make it a fit set:

| threshold | precision | recall | F1 | count bias |
|---|---|---|---|---|
| 0.60 | 0.680 | 0.773 | 0.724 | +13.7% |
| 0.65 | 0.722 | 0.752 | 0.737 | +4.2% |
| 0.70 | 0.764 | 0.725 | 0.744 | -5.2% |
| 0.75 | 0.804 | 0.684 | 0.739 | -14.9% |
| 0.80 | 0.847 | 0.631 | 0.724 | -25.5% |

So the network itself is fine across a wide band (F1 0.72 to 0.74 from 0.60
to 0.80); what moved between fit and held-out is where the bias crosses zero,
from about 0.78 to about 0.67. One image per patient is too few to calibrate a
threshold on, which is the first reason to train on more.

## 7. Three images per patient instead of one

Same network, same recipe, 128 432 patches from 298 fit images (3 per
patient), 3 folds by patient, 8 epochs. Threshold 0.80, the fit-set grid
point nearest zero bias (+4.1%).

Cross-validated F1 did not move: 0.712 at 0.75, against 0.713 before. The
held-out numbers did:

| | 1 image / patient | **3 images / patient** |
|---|---|---|
| precision | 0.827 | 0.823 |
| recall | 0.655 | **0.705** |
| F1 | 0.732 | **0.759** |
| count bias | -20.8% | **-14.4%** |
| count MAE per image | 13.4 | **10.9** |
| parasites per white cell, bias | -16.5% | **-7.9%** |
| rank correlation of the ratio | 0.908 | **0.930** |

Sensitivity on held-out (information only): F1 peaks at 0.763 at 0.75 and the
bias crosses zero near 0.72, so the fit-set threshold is still a little high
for these patients, but by less than before (0.78 against 0.67 last time).

By parasite density:

| parasites in truth | images | recall | precision | count bias |
|---|---|---|---|---|
| 0 to 9 | 33 | 0.676 | 0.542 | +24.7% |
| 10 to 29 | 28 | 0.618 | 0.675 | -8.4% |
| 30 to 79 | 17 | 0.604 | 0.776 | -22.1% |
| 80 and more | 21 | 0.744 | 0.880 | -15.5% |

This is the model shipped as `models/parasite_cnn.onnx`. The one-image model
is in git history (commit 217fbed) for the ablation.

What comes next: the false alarms on sparse fields, which is where a negative
slide would be called positive. The candidate features that describe context
(neighbours within 20 px, distance to the nearest white cell, local field
level) are not visible in a 32 px patch, so the next experiment feeds them to
the network alongside the patch (`--hybrid`).

## 8. The hybrid model: a negative result

Same patch network, with the 19 candidate features (neighbours within 20 px,
distance to the nearest white cell, local field level, ring statistics)
concatenated to the conv embedding before the head. Trained on the one image
per patient set so it compares with the first patch network.

| | patch CNN (1 img/patient) | hybrid (1 img/patient) |
|---|---|---|
| cross-validated F1, best threshold | 0.713 | 0.717 |
| held-out F1 | 0.732 | **0.707** |
| held-out precision | 0.827 | 0.771 |
| precision on fields with < 10 parasites | 0.495 | 0.495 |
| precision on fields with 30 to 79 | 0.771 | 0.543 |

Cross-validation said "no difference"; the held-out set said "worse". The
features that describe context also describe the patient (how crowded the
slide is, how stained), and a network that reads them learns the fit patients
rather than the parasite. The sparse-field precision, the thing this was built
for, did not move at all.

Kept as `--model hybrid` for the record, not shipped. The 3-image training of
it was stopped after the held-out result came in.

Lesson, same as the stain normalisation one in reverse: the cross-validated
candidate-level score is not sensitive enough to rank models that are within a
point of each other. The held-out, end-to-end number is the one to decide on,
and it costs ten minutes.

## 9. What the sparse-field false alarms actually are

Before building anything else for the sparse-field precision, we looked. On
26 fit images with fewer than 10 annotated parasites, the shipped network at
threshold 0.80 made 66 true and 98 false detections. The 40 highest-scoring
false ones, 64 px around each, with the score:

![false positives on sparse fields](figures/parasite-false-positives-sparse-fields.png)

and true positives from the same images for comparison:

![true positives on sparse fields](figures/parasite-true-positives-sparse-fields.png)

Three groups in the false ones.

**Objects that look exactly like the true positives.** A compact dark dot with
a faint halo, scored 0.9 and above (top left, bottom row). We cannot tell
them from parasites, and we suspect the annotators did not mark every parasite
on fields where there were few. If so, the measured precision on sparse
fields is a **lower bound**, and the "over-count" on negative-looking fields
is partly the ground truth's under-count. This cannot be settled without a
microscopist; it goes in the report as a caveat, stated exactly like this.

**Rods and streaks.** Elongated purple marks, several times longer than wide
(third row). These are debris and stain artefacts, and a parasite is never
that shape. The network scores them 0.83 to 0.91, so it has not learned the
shape from 32 px patches with 10 epochs. This group is recoverable.

**Diffuse blobs and doubles.** Soft purple smudges without a dense core, and
pairs of small dots (platelets often come in twos). Harder; some may be
parasites out of focus.

So the honest position on sparse fields: part of the gap is not ours to close,
part of it is shape, and shape is what a longer-trained or slightly larger
patch network should get. That is the next experiment, and it is cheap
because it needs no new data.

## 10. Twenty epochs instead of eight

Same network and data as section 7. Because three-fold training at 20 epochs
would take eight hours, the trainer got a validation-split mode
(`--val-patients 20`): 20 fit patients held aside for the threshold table,
one training on the other 80, and that model is the one exported. On the
validation patients the bias crossed zero at 0.80 (-1.8%), so 0.80 it is.

Held-out, same 50 patients:

| | 8 epochs (section 7) | **20 epochs** |
|---|---|---|
| precision | 0.823 | **0.841** |
| recall | 0.705 | 0.698 |
| F1 | 0.759 | **0.762** |
| count bias | -14.4% | -17.0% |
| count MAE per image | 10.9 | 11.1 |
| rank correlation of the ratio | 0.930 | **0.935** |
| precision, fields with < 10 parasites | 0.542 | **0.575** |
| precision, fields with 30 to 79 | 0.776 | **0.800** |

Small, consistent, and in the direction section 9 predicted: longer training
buys precision (the rod-and-streak group), at a slight cost in recall. The
two criteria set before the run, better F1 and better sparse-field precision,
are both met, so this is the shipped `models/parasite_cnn.onnx`. The 8-epoch
model stays in git history (commit 189e64f).

The calibration gap has not closed: zero bias sits at 0.80 on the validation
patients and near 0.72 on the held-out ones (F1 0.754 there, bias +3.6%).
That is a property of the patients, not of the epochs. The honest way to
report the count is with this gap stated, and the practical way to narrow it
is a threshold set per slide from the white cell density rather than one
global number. That is a later experiment.

## 11. The threshold was calibrated against the wrong reference

The plan was a per-slide threshold, since one cut-off clearly did not suit both
a nearly clean field and a heavy one. Measuring it first turned the task into a
different, simpler one.

### Where the bias comes from

On the twenty fit patients the shipped network never trained on (it kept them
back for its own threshold table, so its scores there are honest), the -12%
count bias splits in two:

| | |
|---|---|
| summed calibrated scores vs candidates that really are parasites | **-0.4%** |
| those candidates vs every annotated parasite | **-11.4%** |

The scoring is right. The candidate step loses the parasites, and no
threshold can find an object that never became a candidate.

### How 0.80 came to be wrong

The threshold table printed during training scores the classifier against **the
candidates it was trained on**. A parasite that never became a candidate is
absent from both sides of that ratio, so the bias it reports is not the bias a
user sees:

| threshold | count | bias vs candidates | bias vs every annotated parasite |
|---|---|---|---|
| 0.70 | 6 602 | +27.2% | +12.8% |
| **0.75** | 5 878 | +13.3% | **+0.4%** |
| 0.78 | 5 443 | +4.9% | -7.0% |
| **0.80** | 5 138 | **-1.0%** | **-12.2%** |

0.80 looked unbiased and under-counted by 12%. It is a wrong-denominator
mistake, and it is worth a sentence in the report because it is the kind of
error that survives review: every number on the row is real, and the column is
the wrong one.

`prf()` in `eval/parasite_cnn_train.py` now takes the measured candidate recall
and reports the bias against every annotated parasite, so the next threshold
cannot be chosen this way.

### Four methods, held-out, 609 images and 50 slides

Nothing here is fitted on these patients: the calibration and the threshold
both come from the twenty calibration patients.

| method | count bias | count MAE | slide density bias | mean abs. density error |
|---|---|---|---|---|
| fixed 0.80, what shipped | -14.4% | 10.26 | -15.3% | 48.6% |
| expected count (sum of calibrated scores) | -15.4% | 12.98 | -4.3% | 54.4% |
| expected count / candidate recall | -4.6% | 11.31 | +8.0% | 53.4% |
| per-slide threshold from a lookup | -7.3% | 9.65 | -6.7% | 44.0% |
| **fixed 0.75, chosen on the calibration patients** | **-2.1%** | **9.83** | **-1.5%** | **43.9%** |

**The per-slide threshold is not needed, and the honest reason is that it never
worked.** The lookup it fitted returned 0.74 to 0.78 for every kind of slide,
so it was a global threshold wearing a costume, and its whole gain came from
being lower than 0.80. The best threshold per calibration slide does vary a
lot, from 0.60 to 0.86, but it correlates with the provisional parasite
density at **-0.010**: there is nothing observable to look it up by. Fitting a
table on that would have been fitting noise, and it would have looked like it
worked, because lowering the threshold helps for an unrelated reason.

The scores are also **already calibrated**, which was worth checking rather
than assuming: candidates scoring 0.50 to 0.80 are parasites 30.6% of the time
and the fitted map says 31.4%. So summing probabilities adds nothing over a
threshold here, and dividing by the candidate recall fixes the count total but
widens the per-slide error, because recall varies from slide to slide much
more than the count does.

### What shipped

`parasites.DEFAULT_THRESHOLD = 0.75`, and the pipeline default follows it.
Held-out, against 0.80: count bias -14.4% to **-2.1%**, slide density bias
-15.3% to **-1.5%**, mean absolute density error 48.6% to **43.9%**.

Scored the same way as section 10 (99 images, 2 per patient), the detector at
0.75 reads: precision 0.790, recall 0.739, **F1 0.763**, count bias -6.4%,
MAE 9.96 per image. Against 0.80 the F1 is unchanged (0.762) and the trade has
simply moved along the same curve, 5 points of precision for 4 of recall, but
the count is no longer systematically low. On the full 609-image table the bias
is -2.0%.

The remaining ten points of candidate loss are the next piece of work, and this
measurement is what says so.

## 12. The stain statistics were wrong; fixed, retrained, re-thresholded

3 September 2026. `results-stain.md` has the bug: the Lab statistics behind
the normalisation were accumulated in float32 over 15 million pixels and were
off by up to 6.5 units, differently on every image. Every number above was
measured on images normalised with that error, and the patch network was
trained on them. So the whole chain was redone, in order, with nothing
touched by hand in between (`eval/results/chain_2026-09-02.log`):

1. white cell parameters refit with the stain threshold in the grid
   (`results-wbc.md`: held-out F1 0.793 to 0.851);
2. the patch network retrained with the same recipe on correctly normalised
   patches (131 456 patches, 10 840 parasites, 3 images per patient,
   20 epochs, 20 validation patients). On the validation patients the count
   bias crosses zero at 0.75 with F1 0.712, where the old network read 0.693
   and crossed zero at 0.80. `cv.dnn` reproduces torch to 3.6e-7;
3. every candidate rescored, and the threshold re-chosen on the 20
   calibration patients by the section 11 rule, count against every
   annotated parasite: **0.76** (+0.1%). Candidate recall there is 0.898,
   up from 0.886, because the refit white cell mask excludes less.

Held-out, the same 50 patients, 99 images, 4 731 annotated parasites:

| | August (0.75) | **3 September (0.76)** |
|---|---|---|
| precision | 0.790 | **0.799** |
| recall | 0.739 | **0.768** |
| F1 | 0.763 | **0.784** |
| count bias per image | -6.4% | **-3.9%** |
| count MAE per image | 9.96 | **8.09** |
| rank correlation of the parasite/white cell ratio | 0.935 | **0.947** |
| ratio bias / 95% limits (97 images) | | +4.0% / -63% to +191% |

On the full 609-image table, the four methods of section 11 at the new
threshold:

| method | count bias | count MAE | slide density bias | mean abs. density error | 95% limits |
|---|---|---|---|---|---|
| **fixed 0.76, shipped** | **-0.6%** | **9.05** | +2.3% | **43.3%** | -56% to +137% |
| expected count | -11.8% | 12.16 | +3.2% | 52.2% | |
| expected / candidate recall | -1.8% | 10.89 | +14.9% | 54.1% | |
| per-slide threshold | -5.7% | 8.98 | -4.8% | 44.3% | |

Two points of F1 and three of recall from a bug fix, with the network,
the candidate stage and the recipe unchanged. And the number that matters
did not move: **mean absolute density error per slide is 43.3%, from
43.9%.** The average count is right (-0.6%) and the spread between slides
is not. Through the WHO protocol with the real counter, 54% of slides land
within 25% of the expert density and 78% within 50%; the worst six are all
over-counts:

| slide | expert density | our error |
|---|---|---|
| TF82_15_1_9 | 14 744/µL | +200% |
| TF81_W8B5 | 1 373/µL | +142% |
| TF70_Scr440 | 3 552/µL | +122% |
| TF107_249 | 1 707/µL | +120% |
| TF48_UW31 | 43 586/µL | +87% |
| TF37_Sc563 | 4 500/µL | +80% |

That is the sparse-field false alarm problem of section 9 again, and it is
the same thing that makes clean slides read positive (`results-negatives.md`).
Those six slides are the next thing to open, one by one, before any more
modelling.

Per field, the pipeline now takes 1.34 s on the laptop against 2.65 s in
August (`bench/field_profile.py`): the stain normalisation went from 1.5 s to
0.08 s and the candidate stage stopped computing nineteen features the
network never read.

