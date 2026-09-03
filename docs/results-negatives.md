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
