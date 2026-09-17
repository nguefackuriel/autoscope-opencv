# *P. vivax*: the detector does not transfer, and the score cannot tell you

17 September 2026. Plan step 6, run once, with the pipeline fixed:
the shipped top-hat + 48 px patch network (`models/parasite_cnn.onnx`) at its
shipped threshold of 0.53, the same one measured on the falciparum calibration
patients. Nothing was fitted here. `eval/vivax_eval.py --run`,
`eval/results/vivax_eval.json`.

## The headline

| | falciparum, 50 held-out patients | ***P. vivax*, 90 patients** |
|---|---|---|
| parasite F1 | 0.800 | **0.292** |
| precision | 0.813 | **0.305** |
| recall | 0.787 | **0.280** |
| detections found / annotated | 1.00 | 0.92 |

**The detector finds roughly the right number of objects and 69% of them are
the wrong objects.** That combination is worse than either failure alone. A
count that is 8% low looks like a working pipeline; a laboratory reading the
density would have no signal that anything is wrong.

Per patient the recall median is 0.234 (p10 0.034, p90 0.432). **26 of the 90
patients are under 10%.** Four are over 50%. So it is not that the detector
half-works everywhere: on most vivax slides it is close to blind, and the
handful it does better on do not rescue the average.

## The result that matters more than the recall

The plan said: *if the scores differ from falciparum, add a "non-falciparum
morphology, confirm species" flag.* That flag cannot be built this way.

| detections on vivax slides | n | median score |
|---|---|---|
| matching an annotated parasite | 7 193 | 0.685 |
| matching nothing | 16 354 | **0.710** |

The detections the annotator did not mark score **slightly higher** than the
ones they did. No threshold on the score separates them, and a per-slide
statistic built from the score distribution — mean, low percentile, spread —
has nothing to work with either. Whatever the network is responding to on a
vivax film, it is as confident about it as it is about a real parasite.

So the species caveat cannot be a measured flag. It has to be a deployment
statement: **AutoScope is a *P. falciparum* screening aid, and a slide from a
region where *P. vivax* circulates needs a human reader.** That is now on the
demo page and in the report.

## What this run can and cannot say

**It cannot say anything about density.** This set annotates no white blood
cells at all — 25 706 parasites and zero white cells across every file. There
is no true denominator, so there is no parasites/µL to compare against. Any
density error computed here would be our own white cell count judging our own
parasite count.

**60 of the 150 patient folders have no annotation file at all** (1 210 of the
3 014 images). They still produce 17 028 detections. Every rate above is over
the 90 annotated patients only; counting the unannotated images in the "found"
total would have made the detector look like it over-counts by 58% when on the
annotated patients it slightly under-counts. This is worth stating because the
first aggregate written did exactly that.

**It cannot say the annotation is complete.** The NLM protocol annotates to
about 200 white cells per slide by design (`ReadMe_new_2021.docx`, and see
`docs/data-notes.md`). If vivax films were annotated to the same stopping point,
some unmatched detections could be real parasites the annotator never reached.
That would raise the true precision and would not raise the recall, which is
measured against what *was* marked. The recall is the number to lean on.

## Three things that had to be read before anything was run

This set is not shaped like the falciparum set, in three ways, each of which
would have produced a confident wrong number:

1. **The label is `Parasitized`, not `Parasite`,** and the files live in
   `All_annotations/<patient>/` rather than beside the images in `GT_updated/`.
   `annotations.for_image` returns `None` for every image here. Using it would
   have reported 0 annotated parasites, a recall of 0/0, and no error.
2. **No white cell annotations**, as above.
3. **The rotation is the other way round.** The falciparum set maps the
   landscape annotation frame into the portrait image clockwise
   (`x = width − coord2`, `y = coord1`). This set is counter-clockwise
   (`x = coord2`, `y = height − coord1`).

The rotation was measured, not assumed, against random points inside the lit
disc — because the question is not "is this point darkish" (blood film is
darkish) but "is this point darker than somewhere nobody annotated":

| reading | median L | share darker than L 150 |
|---|---|---|
| counter-clockwise (used) | 102 | **72.8%** |
| clockwise (falciparum set's) | 204 | 5.5% |
| random inside the disc | 205 | 4.1% |

Clockwise is throwing darts. `--layout` re-measures this before every run and
refuses to proceed if it stops beating chance.

A note on the probe: the falciparum set's version of this check reads the single
pixel under the annotation, which works because it is checking white cell nuclei
about 100 px across. A parasite's chromatin dot is about 10 px and the annotated
centre is itself a few pixels off, so one pixel lands on film as often as on the
parasite and *both* readings come out looking like background. The darkest pixel
within 6 px is the right probe at this object size.

## What follows

Nothing in the vision plan. The judge recipe is closed (`results-parasites.md`
§16g) and this is not a threshold problem: a detector that is as confident about
the wrong objects as the right ones does not have a threshold that fixes it.
Making AutoScope work on vivax means vivax in training, which is a different
project and is out of scope for 26 October.

What follows is in the report and on the page: the species row of the ECAMM
panel is the one we do not have, and now we can say exactly how badly rather
than saying "not measured".
