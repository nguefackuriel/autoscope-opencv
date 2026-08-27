# How AutoScope reads a field

From one phone photo through the eyepiece to two numbers: parasites and
white cells. Everything here is OpenCV 5; the only learned part is a small
network that `cv.dnn` runs from an ONNX file. Each stage is one module, and
each module's docstring says what was measured to set it up.

```
photo ──> fov ──> photometry ──> segmentation ──> parasites ──> density
          find    normalise      white cells      parasites     count -> µL
          disc    stain          (denominator)    (numerator)   + error bar
```

## 0. Find the field of view (`fov.py`)

A phone held to an eyepiece sees a bright disc on black. On the NLM images
54% of the frame is that black. `fov.detect()` works on a 512 px thumbnail:
threshold at 25 on grey, close, largest contour, minimum enclosing circle.
`fov.crop()` returns the square around the disc and a circular mask at 96% of
the radius (the rim is dim and smeared).

Why first: any statistic over the whole frame is a statistic about the black,
and the crop removes about 58% of the pixels every later stage would touch.
A full-frame capture (Tanzania) has no disc; `detect()` returns None and the
whole frame is the field.

## 1. Normalise the stain (`photometry.py`)

Reinhard colour transfer in Lab: match the mean and spread of L, a and b,
measured **inside the disc only**, to a reference (the median over 150 NLM
patients). Per-channel gain is clamped to [0.5, 2] so a flat channel is not
amplified into noise. Lab is used because a slide stained longer moves mostly
in L and a while the shapes stay put; doing this in BGR would shift hue.

Measured effect on the white cell count, same held-out patients: F1 0.677 to
0.782, count bias halved. Not because it moves the level, but because it
rescales each channel's spread, which changes how a fixed threshold meets each
image's contrast.

## 2. White cells, the denominator (`segmentation.py`)

A nucleus is very dark and very purple: L 75 against 206 in the field, a +41,
b -48 (680 cells measured). The stain map is `a - b`: about 107 in a nucleus,
19 in the field; threshold 55. Colour rather than darkness alone, because a
thick smear has many dark things that are not nuclei.

Runs at quarter scale (a 100 px nucleus is still 25 px; sixteen times cheaper).

Two shape problems, two tools:

- **Lobed nuclei.** A neutrophil is 2 to 5 lobes on thin strands; each lobe
  would count as a cell. A morphological **close** with a 24 px ellipse rejoins
  them. The size was swept against ground truth, not chosen by eye.
- **Touching cells.** A **watershed** splits them. Seeds come from the distance
  transform at 45% of **each blob's own** maximum. A global maximum silently
  deletes every cell smaller than the largest one; that bug cost half the
  recall.

Then an area gate (1800 to 30 000 px²) and solidity ≥ 0.55.

Parameters were chosen by a rule, written in `eval/apply_sweep.py`: among
settings with count bias inside 1%, the best F1 wins. The grid offered
F1 0.850 at +5.9% bias and we passed, because this count is the denominator:
6% too many white cells makes every density 6% too low, for every patient,
in the same direction. Held-out: F1 0.793, bias +2.5%.

## 3. Parasites, the numerator (`parasites.py`)

A parasite's chromatin dot is about 10 px across and the same colour as a
platelet or a speck of stain (Lab a-b 54 against 46, L 163 against 165). So
colour gives a candidate list, and a classifier does the sorting.

**Candidates.** Morphological **top-hat** on the stain map: stain minus its
opening with a 31 px disc, larger than any parasite. That removes the
background, so a heavily stained field no longer merges everything into one
region (a plain threshold reached 52% candidate recall for exactly that
reason). A candidate is a **local maximum** of the top-hat above 14, with
non-maximum suppression at 7 px. Nuclei, dilated by 6 px, are excluded.
Candidate recall 88%, about 420 per image, one in ten a parasite.

**Classifier.** A 32 x 32 BGR patch around each peak goes through a small
convolutional network (three blocks of 24, 48, 96 channels, then a 96-wide
head). It is trained once with torch on the fit patients, exported to ONNX
with pixel normalisation folded into the graph, checked against `cv.dnn` to
1e-6, and at run time it is one `cv.dnn` forward pass per batch of patches.
A 19-feature MLP (also ONNX, trained in numpy) is kept as the cheaper
baseline. The threshold is chosen on the fit set where the count bias crosses
zero; the held-out set never chooses anything.

Held-out (50 patients, 4731 parasites): F1 0.759, precision 0.823, recall
0.705, rank correlation of the parasites-per-white-cell ratio 0.930.

## 4. How we score ourselves (`eval/`)

A count can be right by accident: a false alarm and a miss cancel. So every
detection is matched to an annotation (nearest neighbour, 55 px for white
cells, 15 px for parasites) before anything is counted, and the report gives
precision, recall, F1, count bias, MAE, and the per-image ratio against ground
truth with Bland-Altman style limits.

The fit set is 100 patients; the other 50 are never fitted on, and their
number is the only one quoted as a result. Cross-validation, where used, is
split by patient, because two candidates from one image share stain, focus
and background, and a model can learn the image instead of the object.

## 5. From counts to a density (`density.py`, `agent/`)

parasites / white cells × 8000 white cells per µL, with an exact Poisson
interval rather than a normal approximation, because low parasitaemia is the
case that matters and it is where the approximation breaks. The agent reads
fields until the interval is tight enough, or a negative slide has passed the
WHO threshold, or a cap is reached and a person is asked. `counter.py` is the
vision counter the agent calls; the evaluation swaps in an oracle counter to
separate agent mistakes from vision mistakes.

## The two readings of the annotations that nearly sank everything

Both are in `annotations.py`, both pinned by tests on real data.

- The ground truth files are in a **landscape** frame while the images are
  portrait: `x = width - coord2, y = coord1`. Found by scoring seven candidate
  transforms on how dark the pixels under white cell marks were.
- A parasite `Circle` is **centre + a point on the rim**, not a drag across
  the parasite. The midpoint reading put every parasite about 13 px off.

Without the first, the white cell numbers would have been measured against
noise; without the second, the parasite numbers. Both would have looked
plausible.
