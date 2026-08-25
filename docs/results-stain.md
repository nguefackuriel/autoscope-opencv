# Stain normalisation

24 August 2026. `src/autoscope/photometry.py`, measured by `eval/stain_survey.py`.

## The result

Normalising the stain before segmenting, measured on the 50 patients the
parameter fit never saw:

| | off | on |
|---|---|---|
| precision | 0.657 | **0.770** |
| recall | 0.699 | **0.795** |
| F1 | 0.677 | **0.782** |
| count bias | +6.4% | **+3.3%** |
| count MAE per image | 7.02 | **5.38** |

It is on by default.

## We nearly shipped it off, and the reason is worth keeping

Before building it we asked whether stain explains the counting bias, by
correlating each image's stain statistics against its count error. Across 144
held-out images:

| correlation of relative count error with | r |
|---|---|
| how stained the field is | +0.05 |
| how stained the nuclei are | -0.02 |
| separation between nucleus and field | -0.06 |
| where the fixed threshold falls in the image's own range | +0.04 |

Nothing. The nucleus level is already stable on this dataset, 113 to 129 in
`a - b`, and the threshold at 55 sits comfortably between field and nucleus in
every image. We read that as evidence normalisation could not help.

**It was the wrong test.** Those correlations are real and they are still
correct: the global stain *level* does not predict the error. But normalisation
does not only shift the level, it also rescales each channel's spread, which
changes how a fixed threshold meets each image's contrast. That effect is not a
monotonic function of the level, so no correlation against the level could have
seen it.

The lesson is cheap to state and was expensive to learn: a correlation between a
summary statistic and an error is weak evidence of no effect. The A/B run is the
evidence, it costs one evaluation, and it should have come first.

## What still drives the residual error

Crowding, and it survives normalisation. Correlation of count error with the number of cells actually present:
**r = -0.55**.

| | mean count error |
|---|---|
| sparse images, 17 cells or fewer | **+5.97** |
| crowded images, more than 17 cells | **-3.71** |

Two opposite errors. On a sparse slide a lobed nucleus splits into lobes and one
cell counts as several. On a crowded slide two touching cells merge and count as
one. The +6.4% headline is the average of the two, which is also why the
per-image error is large, 7.0 cells, while the average looks respectable.

The rim was the other suspect and it is not guilty either: 25.2% of matched
detections sit beyond 0.85 of the radius against 27.0% of unmatched ones, and
only 5.7% of missed cells are out there.

## An attempt that failed

Counting cells in a clump by dividing its area by what one nucleus covers.
It should handle both directions at once and it does not work: recall rose from
0.699 to 0.816, but precision fell from 0.657 to 0.527 and the bias went from
+6.4% to **+54.6%**. Reverted.

Recorded here rather than deleted, because the idea is reasonable enough that
somebody will suggest it again. It needs a better estimate of the single-cell
area than a median over surviving blobs, which debris drags down until one cell
becomes three.

## It also does what it was built for

Transfer to another rig. Measured over 150 NLM patients,
before and after mapping each slide onto the measured reference:

| statistic | spread before (IQR) | after | narrowed |
|---|---|---|---|
| mean L | 17.58 | 2.42 | **86%** |
| mean a | 10.50 | 1.18 | **89%** |
| mean b | 13.16 | 2.22 | **83%** |
| std L | 6.59 | 0.92 | **86%** |
| std a | 3.22 | 0.81 | **75%** |
| std b | 2.52 | 0.89 | **65%** |

The reference is the median of those 150 patients:
`mean=(194.7, 134.6, 118.8), std=(24.4, 10.1, 10.0)`. Measured, not chosen, and
pinned by a test so nobody edits it without re-running the survey.

Every threshold in this pipeline was fitted on one rig: 4032x3024 phone captures
down one eyepiece, Giemsa stained by one group. The Tanzania set is another
country, another microscope and a 4K Sony IMX334. A threshold in raw Lab units
has no reason to mean the same thing there. Normalising first is what makes the
generalisation claim testable instead of hopeful.

So `normalise_stain` is on by default everywhere.

## Next, for the remaining bias

The split-and-merge behaviour, on evidence:

- separate the sparse and crowded cases and measure them apart, since a single
  parameter set is being asked to serve two opposite failures;
- get a single-cell area estimate that debris cannot drag down, which is what
  sank the clump attempt;
- look at the unmatched detections before assuming they are wrong. Precision is
  0.657, and some of that third may be real cells the annotator passed over.
