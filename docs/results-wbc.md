# White cell detection, first measured baseline

24 August 2026. `src/autoscope/segmentation.py`, scored by `eval/wbc_eval.py`.

White cells come first because they are the denominator. Parasites per 200 white
cells is what becomes a density, so an error here moves every number downstream.

## The honest number

Fitted on 100 patients, reported on the **50 patients the fit never saw**.
Stain normalisation is on, see `results-stain.md`:

| | held out | without normalisation |
|---|---|---|
| images | 144 | 144 |
| annotated cells | 3046 | 3046 |
| precision | **0.770** | 0.657 |
| recall | **0.795** | 0.699 |
| F1 | **0.782** | 0.677 |
| count bias | **+3.3%** | +6.4% |
| count MAE | **5.4 cells per image** | 7.0 |

Always quote the held-out column. On the fit set the same detector reads +1.8%
bias instead of +3.3%, and that gap is exactly what scoring yourself on your own
tuning data buys you. It is why the split exists.

The segmentation parameters were fitted before normalisation existed. Re-running
the sweep with it on will probably do better still, and has not been done.

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
