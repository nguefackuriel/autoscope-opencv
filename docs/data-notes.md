# What the NLM thick smear data actually is

Written after downloading and measuring it, 19 August 2026. Read this before
writing anything that touches the files, and before repeating any claim from the
proposal about focus or field counts.

## Layout

```
data/raw/
  TF<n>_<code>/            150 patient folders, one slide each
    20170830_110101.jpg    successive fields, 4032 x 3024, phone through eyepiece
    ...                    5 to 22 images per patient, median about 12
  GT_updated/
    TF<n>_<code>/
      20170830_110101.txt  one annotation file per image
  ReadMe_new_2021.docx
```

Annotation format. First line is `count,width,height`. Then one line per object:

```
12-1,Parasite,No_Comment,Circle,2,2795,768.65,2798.25,790.3
```

Labels present across the whole set: **84,509 Parasite** and **35,449
White_Blood_Cell**. Both are there, which is what makes the density metric
possible. Parasites per 200 white cells can be computed from ground truth, so
the Bland-Altman comparison we promised in the proposal is doable on this data.

## Three findings that change what we can claim

### 1. Every frame is a round eyepiece view, and 54% of it is black

Centre brightness about 210, corners about 0.7. The lit disc has a radius of
1310 to 1415 pixels inside a 4032 x 3024 portrait frame, and `fov.detect()`
finds it in **100% of images**.

Two consequences.

Any statistic taken over the whole frame is mostly a statistic about the black
surround. The first version of `illumination_uniformity()` did exactly that and
scored every real image at about 0.05, which would have rejected the entire
dataset. Fixed in `autoscope/fov.py`, and every metric now runs inside the disc.

Cropping to the disc drops 41% of the pixels, and the circular mask keeps 72% of
what is left. Useful pixels are about **42% of the frame**, so processing the
raw frame does roughly 2.4 times the necessary work. That belongs in the cost
number and in the COOL benchmark: crop first, then time the pipeline.

### 2. There are no z-stacks, so `refocus()` cannot be real on this data

Filenames are timestamps roughly a minute apart. They are successive fields with
the stage moved between them, one capture each. Nothing here is the same field
at two focal heights.

So on this dataset `refocus()` renders defocus with a Gaussian blur and marks
the frame `synthetic=True`. The report has to say so plainly. What we can still
claim honestly:

- detecting an unusable field is validated on real captures, because real soft
  fields exist in the data (focus p5 is 10.1 against a median of 17);
- the refocus **action** is simulated, and any number about how well refocusing
  recovers a field carries that caveat.

If a z-stack dataset turns up, `VirtualStage` already takes `z_planes` and will
use real pixels without a code change.

### 3. About 9 images make one WHO count, not 100

Mean white cells per image is **23.0**, so the WHO 200-cell count needs about
**8.7 images**. Patients have 5 to 22 images. One patient folder is therefore
roughly one full WHO count, sometimes two.

This means the phrase "the agent reads fewer than 100 fields" does not map onto
this data. A WHO field at 100x oil is much smaller than a 12 megapixel phone
frame. State the saving in the units the data supports:

> the agent reaches the target precision after fewer white cells, and fewer
> images, than the fixed 200-cell protocol

not in microscope fields. Same claim, defensible units.

## Measured distributions

300 images, 2 per patient, all 150 patients. From `eval/results/focus_survey.json`.

| metric | p5 | median | p95 |
|---|---|---|---|
| focus (Laplacian variance) | 10.1 | 17.0 | 30.2 |
| lighting evenness | 0.915 | 0.965 | 0.983 |
| cell coverage | 0.019 | 0.040 | 0.081 |
| debris fraction | 0.561 | 0.685 | 0.800 |
| field of view radius, px | 1310 | 1336 | 1415 |
| white cells per image | 10.5 | 19.0 | 51.6 |
| parasites per image | 2.0 | 18.0 | 159.7 |

Thresholds in `quality.py` come from these percentiles. Note that focus scores
are only comparable within this rig: Laplacian variance scales with resolution,
magnification and stain, so re-run the survey before using these numbers on any
other dataset.

`debris_ratio` is the weak one. The median image already scores 0.69, because a
thick smear is lysed blood and is full of small particles by design. The
threshold of 0.85 flags only the extreme tail. Before the final report, fit it
against fields the counter actually got wrong rather than against the
distribution.

## Minor

A handful of JPEGs emit `Invalid SOS parameters for sequential JPEG` on read.
They still decode. No action needed, but do not treat the warning as a failure.

## The Tanzania set, downloaded 26 August 2026

`data/download_tanzania.py`, Harvard Dataverse `doi:10.7910/DVN/O2WVWA`, CC0.
Four archives, 2.98 GB, extracted under `data/raw/tanzania/`:

```
Thick_Infected/Thick_Infected/       1139 images   Image01.jpg ...
Thick_Uninfected/Thick_Uninfected/   1071 images
Thin_Infected/Thin_Infected/         1064 images
Thin_Uninfected/Thin_Uninfected/      270 images
```

All 3840 x 2160 JPEGs from a 4K Sony IMX334 on a compound microscope.

Three things that decide how it can be used.

**No per-parasite annotation.** The only label is the folder: infected or not.
So there is no precision or recall to measure here. What can be measured is
whether the parasite count separates infected from uninfected images
(an ROC over images), and whether the density ranking is sensible. Per-object
validation stays on NLM.

**No eyepiece disc.** These are full-frame camera captures. `fov.detect()`
correctly returns None and the whole frame is the field. Every mask-based
statistic falls back to the full frame, which is right here.

**Different scale and stain.** Looking at the pictures, white cells are about
600 px across against 100 px on NLM, so the magnification is roughly four to
six times higher in pixels per micron, and the Giemsa is much bluer. Nothing
in the pipeline transfers without (a) a pixel scale factor and (b) stain
normalisation onto the NLM reference. Measure the white cell diameter here
before setting either; do not eyeball it.
