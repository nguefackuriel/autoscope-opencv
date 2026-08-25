# Stain normalisation, and what it does not fix

24 August 2026. `src/autoscope/photometry.py`, measured by `eval/stain_survey.py`.

## The question we started from

The white cell detector over-counts by 6.4% on held-out patients. The obvious
suspect was stain: slides stained darker or lighter shift the colour a nucleus
sits at, so a fixed threshold would cut in the wrong place.

**It is not the cause.** Across 144 held-out images:

| correlation of relative count error with | r |
|---|---|
| how stained the field is | +0.05 |
| how stained the nuclei are | -0.02 |
| separation between nucleus and field | -0.06 |
| where the fixed threshold falls in the image's own range | +0.04 |

Nothing. The nucleus level is already stable on this dataset, 113 to 129 in
`a - b`, and the threshold at 55 sits comfortably between field and nucleus in
every image.

## What is the cause

Crowding. Correlation of count error with the number of cells actually present:
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

## What normalisation is actually for

Transfer to another rig, and that it does well. Measured over 150 NLM patients,
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

So `normalise_stain` is off by default on NLM, where it is measurably not
needed, and it is the first thing to switch on for any other source.

## Next, for the bias

Not stain. The split-and-merge behaviour, on evidence:

- separate the sparse and crowded cases and measure them apart, since a single
  parameter set is being asked to serve two opposite failures;
- get a single-cell area estimate that debris cannot drag down, which is what
  sank the clump attempt;
- look at the unmatched detections before assuming they are wrong. Precision is
  0.657, and some of that third may be real cells the annotator passed over.
