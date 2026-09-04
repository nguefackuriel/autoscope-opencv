# Specificity: what the pipeline says about slides with no parasites

3 September 2026. `eval/negative_fields.py` and `eval/negative_eval.py`, on
the NLM uninfected thick smear set (`make data-uninfected`): 50 patients,
1 141 fields, same rig and same group as the development set, with the
corrected pipeline and threshold 0.76. Frozen test material, nothing fitted
on it. This is the first time the negative case has been measured at all:
every one of the 150 development patients is infected.

Run it with:

```
cd eval
../.venv/bin/python negative_fields.py     # count every clean field once (32 min)
../.venv/bin/python negative_eval.py       # replay the protocols, ROC against held-out
```

## The number

**Every clean slide reads positive.** Through the WHO 200-cell protocol
with the real counter, the 50 uninfected slides read at a median of
**544 parasites/µL** (quartiles 341 to 993, p90 2 451, max 5 698). Not one
is called negative by the stopping rule, because the rule needs zero
parasites in 200 white cells and no slide gets there. The rule-based agent
does no better: 585/µL median, 49 of 50 escalated to a person after running
out of slide.

Per field: 22% of clean fields report zero parasites; the mean is 3.3 false
parasites per field, the median 2, the 90th percentile 8, the worst 56.
Pooled, 25.4 false parasites per 200 white cells, which is a **noise floor
of about 1 000/µL**.

Against the 50 held-out infected patients (expert densities from 283/µL,
p10 1 732, median 9 077), per patient through the same protocol:

| cut-off | specificity | sensitivity |
|---|---|---|
| any parasite | 0.02 | 1.00 |
| 200 /µL | 0.08 | 1.00 |
| 500 /µL | 0.46 | 0.98 |
| 1 000 /µL | 0.74 | 0.94 |
| 2 000 /µL | 0.86 | 0.84 |

Rank AUC 0.935. So the count *ranks* infected above uninfected well, but
there is no cut-off that gives both sensitivity and specificity above 0.9,
and a screening tool needs both. The sensitivity column is flattered by
the held-out cohort: only three of its patients are under 1 000/µL, and
those are exactly the ones a noise floor of 1 000/µL hides.

## Where the false parasites come from

They are not spread evenly. Patients range from 6 to 99 false parasites
per 200 white cells (partial run before the statistics fix, 19 patients:
the same spread). The top decile of fields carries most of the count, which
is the sparse-field precision problem of `results-parasites.md` section 9:
stain precipitate, platelets and rods that the top-hat picks up and the
patch network passes at 0.76. On an infected field they are a minority of
the detections; on a clean field they are all of them.

The white cell count, for comparison, transfers cleanly: 1 163 found
against 1 176 annotated on the first two patients, -1.1%.

## What this changes

- The **negative call is not available** from this pipeline as it stands,
  and the report must say so rather than quote the sensitivity table alone.
- The **stopping rule for negatives** (zero parasites in 200 white cells)
  can never fire. A rule that works needs a per-field false positive model:
  "no more parasites than a clean field would show", with the noise floor
  measured on uninfected fields, and a much longer read for a negative
  call, which is what the WHO's 100-field negative already asks.
- The **fix is on the vision side**, and it has a precise target: the
  detections on clean fields. Every one of them is a false alarm by
  definition, so this set is also training signal for the classifier, as
  long as it stays split by patient and the test half stays frozen.
- The **P. vivax set** (`make data-vivax`) and the Tanzania uninfected
  images are the next two measurements of the same kind.

## 4 September: with clean fields in the training set

The fit half of the uninfected set (25 patients, `uninfected_split.json`)
supplied 75 clean fields of negative patches to the classifier; the other
25 patients are the frozen test half and are the only ones reported here.
Threshold 0.80, chosen on the calibration patients as before.

| frozen half, 25 patients, 572 fields | 3 Sept (all 50, 0.76) | **4 Sept (0.80)** |
|---|---|---|
| clean fields reporting zero parasites | 22% | **41%** |
| false parasites per field, mean / p90 / max | 3.3 / 8 / 56 | **1.3 / 3 / 17** |
| pooled false parasites per 200 white cells | 25.4 | **10.6** |
| noise floor | 1 014/µL | **425/µL** |
| clean slides, WHO protocol: median / p90 / max | 544 / 2 451 / 5 698 | **336 / 763 / 1 541** |
| rank AUC against the 50 held-out infected | 0.935 | **0.982** |
| at 500/µL: specificity / sensitivity | 0.46 / 0.98 | 0.76 / 0.98 |
| **at 1 000/µL: specificity / sensitivity** | 0.74 / 0.94 | **0.92 / 0.94** |
| at 2 000/µL | 0.86 / 0.84 | 1.00 / 0.84 |

(The 3 Sept column is over all 50 patients; on the same frozen half the
before-fix numbers are not separately available because that run was
superseded. The all-50 numbers for 4 Sept are in `negative_eval_all50.json`
for the record, median 347/µL, AUC 0.962, and are not a test number because
half of those patients trained the classifier.)

So a cut-off at 1 000/µL now gives sensitivity and specificity both above
0.9 on this data, where yesterday no cut-off did. What has not changed:
**no clean slide is called negative by the stopping rule**, because the rule
still asks for zero parasites in 200 white cells and a clean slide reads
about 10 false ones. The negative call needs a rule built on the measured
noise floor, and that is agent-side work: stop and call negative when the
count after N white cells is within what a clean field produces, with the
floor measured on the fit half and N long enough for the confidence a lab
wants. It is the next thing to build once the Bedrock agent work starts.

The remaining false alarms are the sparse-field kind: the 17-parasite worst
field is one field, the median clean field now reports one. Sensitivity is
still flattered by the held-out cohort's density (three patients under
1 000/µL); the *P. vivax* set and Tanzania are the next two measurements.

