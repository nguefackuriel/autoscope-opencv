# Tanzania: another microscope, another camera, another stain, run once

12 September 2026. `src/autoscope/site.py`, `eval/tanzania_eval.py`,
results in `eval/results/tanzania_*.json`. The generalisation number of the
plan's step 5 (`docs/plan-2026-09-05-hospital-grade.md`).

## What the set is, and what it can and cannot say

Harvard Dataverse `doi:10.7910/DVN/O2WVWA`, CC0 (`docs/data-notes.md`):
thick smears from five health centres in the Tanga region, a compound
microscope and a 4K Sony IMX334 camera, 3 840 × 2 160 JPEGs, no eyepiece
disc. 1 139 images from infected patients and 1 071 from uninfected ones.
**The only label is the folder**: no parasite positions, no white cell
positions, no patient identifier. So there is no precision, no recall and
no density error to report, and no slide-level decision can be made,
because the images cannot be grouped into slides. What can be measured is
whether the pipeline, fitted entirely on the NLM phone-through-eyepiece
rig, tells an image from an infected patient from an image from an
uninfected one.

## The site calibration, on twenty images

The first ten images of each folder in sorted order are the site's
calibration slides (`tanzania_split.json`, fixed before anything was
counted); every other image is test material, run once.

**Scale.** Every threshold in the pipeline was measured with a white cell
nucleus about 80 to 100 px across. Here a nucleus is four to five times
larger. `site.nucleus_diameters` measures round single nuclei (the dark
objects after stain normalisation, lobes rejoined, kept when their size
and aspect say one nucleus) the same way on both rigs: **77.2 px on NLM**
(30 fit images, 604 nuclei) and **370.8 px on Tanzania** (the 20
calibration images), so the scale is **0.2082**: a Tanzania image is
resized to 800 × 450 before anything else runs. Measuring both with the
same function matters. Against the 102 px of `segmentation.py`, which is
the detector's blob including multi-lobed neutrophils, the scale would
have been 0.28; the ratio of the same-function medians is stable from a
darkness threshold of L < 90 to L < 130 (0.206 to 0.225).

**Stain.** The raw field is darker and bluer than NLM's (Lab L 162, a − b
−0.1 against 198 and +24 after normalisation); `photometry.normalise`
maps each image onto the NLM reference as it does on NLM, and after it the
field reads L 197, a − b +24 on every calibration image. No parameter is
fitted to the site.

**Looked at** (`docs/figures/tanzania-calibration-infected.png`): the top
detections on the ten infected calibration images, cut at NLM scale, are
chromatin dots with a crescent of blue cytoplasm, the ring form the
network was trained on, plus a few white cell lobes. Six of the ten
infected images give 6 to 10 detections, four give none; the ten
uninfected images give none.

**A bug the calibration found.** The first calibration counted at 0.80:
`parasites.DEFAULT_THRESHOLD` had stayed at chain 2's value when chain 3
chose 0.53 for its network on 6 September. Every published number was
measured at 0.53, because the chain passes the threshold explicitly, but
anything that used the default, the agent's live counter included, counted
at 0.80. Fixed, and pinned by a test to the canonical
`threshold_fixedstats.json`.

## The run

Commit `b8995e0`, clean tree, shipped pipeline (top-hat candidates, the
48 px patch network at 0.53, the white cell gate of 6 September), scale
0.2082. 1 129 infected and 1 061 uninfected images, 0.17 s per image.
**Run 1**: `tanzania_eval.py --run` refuses a second run without
`--again`, and a second run is a second look that the report must state.

| per image | infected (1 129) | uninfected (1 061) |
|---|---|---|
| images with no parasite reported | 30% | **99.3%** |
| images with at least one | **70.2%** | 0.7% (7 images, one detection each) |
| parasites reported, median / p90 / max | 2 / 7 / 17 | 0 / 0 / 1 |
| white cells found, median | 1 | 2 |

| a field is called positive at | sensitivity | specificity |
|---|---|---|
| ≥ 1 parasite | **0.702** | **0.993** |
| ≥ 2 | 0.575 | 1.000 |
| ≥ 3 | 0.445 | 1.000 |
| ≥ 5 | 0.252 | 1.000 |

Image-level rank AUC on the parasite count: **0.850**, held down by ties
at zero (30% of infected images report none, and a tie scores a half);
on the density per image, 0.776, because a field this size holds one or
two white cells and the ratio is noise.

## What it says

**The clean side transfers.** A clean Tanzania field reports a parasite
0.7% of the time. Per unit of field area at NLM scale that is 0.018 false
parasites per megapixel (7 on 1 061 fields of 0.36 megapixel; 95%
interval 0.007 to 0.038), against 0.010 on the frozen NLM clean half (32
on 572 fields of 5.6 megapixels; 0.007 to 0.014). The intervals overlap:
on seven events the two rates cannot be told apart, and both are about
one false parasite per hundred megapixels of blood. The false-alarm rate
of the shipped pipeline survives a different microscope, camera, stain
and country, once the scale is set from ten slides.

**The infected side is weaker and cannot be pinned down here.** 70% of
infected fields report at least one parasite. Whether the other 30% hold
parasites the pipeline missed, or hold none (a field from an infected
patient need not contain a parasite, and this set says only which patient
group the image came from), cannot be told without positions. The fields
are also soft at native resolution (`docs/figures`), which the quality
gate was not asked to judge here.

**At the slide level**, which is what a clinic reads, a slide is many
fields. The set has no slide identifiers, so no slide-level number is
given; a model of independent fields would flatter it, since fields from
one slide are not independent.

## What the report can claim

- the pipeline was moved to a second site with a two-number calibration
  (scale from the nuclei of ten slides, stain by the existing
  per-image normalisation), nothing refitted;
- on that site, 99.3% of clean fields report no parasite and 70% of
  fields from infected patients report at least one, AUC 0.85, run once;
- the false-alarm rate per unit area cannot be told from the NLM frozen
  clean half's (0.018 against 0.010 per megapixel, intervals overlapping).

What it cannot: sensitivity per parasite or per slide on Tanzania, or any
density number.
