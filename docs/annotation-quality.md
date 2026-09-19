# What the NLM annotations actually are, measured

Branch `b/parasite-detectors`, September 2026. Every number here comes from
`eval/annotation_gap.py` and the audits below, on **fit patients only**. The
held-out 50 were not read.

This exists because converting point and circle annotations into boxes is a
change of format, and before making it we needed to know what the annotations
are worth. If the click is systematically off, or its stated size bears no
relation to the object, then some of what looks like detector error is
annotation error and no model work removes it.

## 1. What the dataset ReadMe settles

`data/raw/ReadMe_new_2021.docx`, read and then checked against the data rather
than taken on trust.

**The EXIF orientation flag.** Images were captured in smartphone portrait mode
with orientation flag 6, and annotated by a tool that *ignores* the flag.
OpenCV reads it and rotates. That is the mechanism behind the coordinate
transform `annotations.py` had derived empirically.

Verified: the header reads `25,4032,3024`, `cv.imread` returns `(4032, 3024)`
and `IMREAD_IGNORE_ORIENTATION` returns `(3024, 4032)`; `coord1` reaches 3365
across 3834 sampled annotations, so it runs along the 4032 axis. The stored
mapping `x = width - coord2, y = coord1` is exactly the compensating 90 degree
rotation. White cell clicks land **145 L units** darker than the lit field,
against the 131 recorded when the transform was first derived.

**No change was needed.** The existing transform is right.

**Shape coverage.** The ReadMe says either label may be stored as a `Circle` or
a `Point`, which would mean the parser silently drops combinations it does not
handle. Audited across all **119 958** annotation lines in the set:

| label | shape | count | parsed |
|---|---|---|---|
| Parasite | Circle | 84 509 | yes |
| White_Blood_Cell | Point | 35 449 | yes |

Only those two combinations exist, and they match the ReadMe's own stated
totals exactly. Nothing is dropped. No change needed.

**Images with no annotation file.** 53 of 1883 (2.8%). This is by design, not
missing data: annotation stops once 200 white cells have been counted, because
that is enough to compute a parasitaemia. See section 5.

## 2. Where the annotation sits, against the pixels

`eval/annotation_gap.py`, 2209 parasites and 999 white cells over 20 fit
patients. Each annotation is refined against the pixels using the same OpenCV
operations the pipeline already uses: a top-hat contour for a parasite, a
watershed-split nucleus for a white cell. So this reports exactly what
"apply findContours and friends" would buy.

### Parasites

| | |
|---|---|
| refined | 99.7% |
| click to stain peak | median **7.0 px**, p90 12.3 |
| mean offset vector | (+0.06, −0.32) px, length **0.33** |
| systematic / random | **0.047** |
| annotated radius | median **20.8 px** |
| contour radius | median **4.7 px** |
| ratio contour / annotated | **0.23** |

### White cells

| | |
|---|---|
| click within the nucleus | **91.2%** |
| click to nucleus centroid | median **13.5 px**, p90 64.3 |
| mean offset vector length | **3.66 px** |
| nucleus radius | median **42 px** |

### What this means

**The offset is jitter, not bias.** Both mean displacement vectors are near
zero. A systematic shift — a transform bug, or a consistent annotator habit —
would show as a mean vector comparable to the median offset; instead the ratio
is 0.047. There is nothing left to correct, and no contour trick recovers where
a human meant to click. Respect it in the matching tolerance and leave it alone.

**The stated size is badly wrong, and that part is fixable.** The annotator's
circle is 4.4x the parasite in radius, roughly **19x in area**. This is the one
large, correctable error, and it is why `boxes.py` derives boxes from the stain
rather than drawing them from the circle. It is the single measurement that
justifies the whole approach.

## 3. The recall ceiling the annotation imposes

A perfect detector reporting the true object would still be scored a miss when
the click sits beyond the matching tolerance:

| | tolerance | beyond it |
|---|---|---|
| parasites | 15 px | **1.6%** |
| white cells | 55 px | **13.0%** |

Negligible for parasites. For white cells it means roughly a third of the gap
between the shipped 0.803 recall and 1.0 is annotation, not detector — worth
knowing before anyone spends a week trying to close it.

These tolerances are doing real work. A detector that localises to the stain
peak will sometimes land 12 px from the click and still be correct. **Do not
tighten them.**

## 4. A method error, recorded so it is not repeated

The first white cell pass contoured the nucleus mask *without* the watershed
split the pipeline applies. Two touching cells became one blob whose centroid
sits in the isthmus between them, far from either click. That reported p90
89 px and **22.9%** unreachable inside the 55 px tolerance.

That cannot be true: the shipped detector scores recall 0.803 on these same
patients, which caps the unreachable share at 19.7%. Splitting first gives p90
64.3 and 13.0%, which is consistent. The lesson is narrow and useful: anything
measuring against the pipeline has to apply the same steps the pipeline does.

## 5. Decisions taken

**Boxes are derived from the stain, never from the annotator's circle.**
Section 2 is the evidence.

**No box is ever invented.** An annotation the stain cannot explain (about
0.5%) makes its tile unusable both ways — keep the tile and a real parasite is
labelled background, invent a box and the ground truth has been changed. The
export drops the tile instead. That costs a patch of image and touches nothing
about the annotations. `boxes.py` still offers `fallback=True` for a caller
that genuinely needs one box per annotation, but the export does not use it.

This is a **format conversion so that box detectors can read the NLM
annotations, not a re-annotation of the data**. Altering ground truth would
require per-image verification and validation that this project is not in a
position to carry out, and it should not happen by default and in silence.

**Unannotated images are excluded from train, validation and test.** A
detection on an image with no ground truth cannot be scored right or wrong, so
those 53 images have no place in any set that produces a number. They are
excluded before sampling rather than skipped after it, because sampling first
returns fewer images than asked for and the shortfall lands entirely on the 37
patients that have unannotated fields.

For the record, they are not a degraded tail. Paired against an annotated field
from the same patient, they are marginally *better*:

| metric | annotated | unannotated |
|---|---|---|
| focus_score | 15.68 | **17.50** |
| illumination_uniformity | 0.965 | 0.971 |
| cell_density | 0.030 | 0.036 |
| debris_ratio | 0.693 | **0.675** |

They stopped being annotated because the count reached 200, not because
anything was wrong with them. They are simply unusable for measurement.

## 6. Still open

**Published numbers need re-measuring.** Two changes to sampling have landed
together, and both move which images an evaluation reads.

*Filtering before sampling.* Every sampling site in `eval/` now goes through
`splits.annotated_images()`, so an unscoreable image is excluded rather than
drawn and then skipped. `tests/test_splits.py` fails if any script globs images
directly.

*Per-patient seeding.* The loops used to share one `random.Random(seed)`, so
every draw depended on every draw before it. `splits.image_rng(seed, patient)`
keys the generator on the patient instead.

The effect is smaller than the raw diff suggests. On the held-out 50 at
`--per-patient 2`, the old and new samples share 77 of 100 images. Of the 23
that left, **exactly one is an unannotated image**; the other 22 are perfectly
good annotated images that the reshuffled draw did not pick. So the substantive
change is one image in a hundred, and the rest is the draw landing elsewhere.
Expect F1 to move by sampling noise, in the third decimal, not by bias.

What per-patient seeding buys, measured on the held-out 50:

| | draw changed | of those, with no unannotated image |
|---|---|---|
| shared RNG | 20 of 50 | **9** |
| per-patient RNG | 10 of 50 | **0** |

and the loop becomes order-independent: reversing the iteration order used to
change 49 of 50 samples, and now changes none. That makes the patient loop safe
to parallelise, which matters because these evals take about ten minutes and
splitting them across cores is the obvious speedup — today that would have
silently produced different numbers.

`eval_helpers.py` keeps a shared generator on purpose: it shuffles *patients*
to reach an image limit, which is a different job from sampling within one.

**Anything quoted from before these changes will not reproduce bit-for-bit.**
The held-out figures in `docs/results-parasites.md` and `docs/results-wbc.md`
should be re-run before the final report.

**Tile loss — found and fixed.** This section previously recorded that about
26% of derived boxes never reached a tile, and attributed it to the geometry of
the disc edge. That was wrong. It was a threshold: `MIN_FOV = 0.9` kept only
the tiles lying almost entirely inside the eyepiece disc, 10.2 of 29.6 per
image.

The same constant was used at inference, so the detector was never shown 34.8%
of the parasites it was then scored against, and recall could not exceed 0.652
however good the model was. The first YOLO26 validation sweep read F1 0.452 at
its best, against a ceiling it could not see.

`MIN_FOV = 0.2` covers 99.9% of parasites at 27.8 tiles per image. Both the
export and `detector_eval.py` now take it from `autoscope/tiling.py`, so they
cannot drift apart again, and a test fails if it is raised back above 0.5.
After the change, 2.2% of boxes are never covered rather than 26%.
