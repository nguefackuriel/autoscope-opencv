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


## 6 September: chain 3, the noise floor falls twenty-fold, and the first negative call

Chain 3 (`results-parasites.md` section 15: every clean field of the fit
half and hard negatives from every fit image in the training set, 48 px
patches, threshold 0.53). Same frozen half, same 572 fields, same protocols.

| frozen half, 25 patients, 572 fields | 4 Sept (0.80) | **6 Sept (0.53)** |
|---|---|---|
| clean fields reporting zero parasites | 41% | **95%** |
| false parasites per field, mean / p90 / max | 1.3 / 3 / 17 | **0.06 / 0 / 3** |
| pooled false parasites per 200 white cells | 10.6 | **0.46** |
| noise floor | 425/µL | **18/µL** |
| clean slides, WHO protocol: median / p90 / max | 336 / 763 / 1 541 | **0 / 60 / 76** |
| rank AUC against the 50 held-out infected | 0.982 | **1.000** |
| at 100/µL: specificity / sensitivity | 0.04 / 1.00 | **1.00 / 1.00** |
| at 500/µL | 0.76 / 0.98 | 1.00 / 0.98 |
| at 1 000/µL | 0.92 / 0.94 | 1.00 / 0.92 |
| clean slides called negative by the old rule (0 in 200 white cells) | 0 / 25 | **15 / 25** |

The fit half, whose fields the classifier trained on, reads 0.12 false
parasites per field and a 36/µL floor: 863/µL on 4 September. It is no
longer worse than the frozen half, which says the 4 September gap between
the halves was the classifier, not the slides.

### The decision rule

`density.decide`, `agent.policy.DecisionRule`, 5 September. The stopping
rule asked for zero parasites in 200 white cells for a negative and a 20%
error bar for a positive. This one calls what a technician calls:

- **negative** once at least N white cells have gone by and the estimate
  sits at or below the **noise floor**, what a clean slide produces;
- **positive** once the lower end of the interval clears the floor;
- a positive is **settled** when its interval sits inside one density band
  (under 1 000, 1 000 to 10 000, 10 000 to 100 000, over 100 000/µL) and is
  within 30% of the estimate, or within the 20% target regardless;
- otherwise keep reading, and at the end of the slide **hand it to a
  person**, saying which of the three could not be settled.

The floor and N are chosen on the **fit half** (`negative_eval.py
--design --sweep`) and the frozen half is scored once with them. The floor
is the p95 of what the fit half's slides read at with every field counted:
**150/µL** (median 15, p90 69, max 223). On the fit half, N = 200 calls 23
of 25 clean slides negative and N = 500 calls 18, the rest running out of
slide before N white cells; no floor between 69 and 300/µL changes the
sensitivity (49 of 50 held-out infected called positive) and 800/µL loses
it. N = 500 is shipped because it detects lower (next section), and a
slide that is too short for it is a limit of this dataset, where a clean
slide has 12 to 33 captured fields, not of the rule: on a microscope the
stage moves on.

| frozen half and held-out, floor 150/µL | N = 500, shipped | N = 200 |
|---|---|---|
| clean slides called negative | **20 of 25**, the 5 others too short for 500 white cells, **0 called positive** | 24 of 25, 1 too short |
| infected slides called positive | **49 of 50**, 0 called negative | 49 of 50 |
| the one not called | TF150_R5, 283/µL, the only held-out patient under 1 000/µL: its lower bound never clears the floor and it goes to a person | same |
| by expert band | 1 000 to 10 000: 26/26; 10 000 to 100 000: 22/22; over: 1/1 | same |
| fields read, clean + infected | 490 + 448 | 208 + 448 |

So the first negative call: 20 of the 20 clean slides long enough to
decide, no clean slide called positive, no infected slide called negative,
two slides of 75 sent to a person. On the chain-2 tables the same rule at
its own floor (2 610/µL, the fit half was that dirty) read 0.80 / 0.76
with 12 infected slides sent to a person (`negative_eval_decision_dev.json`).

### Limit of detection by dilution

Six of the 150 NLM patients are under 1 000/µL, so the sensitivity a
screening tool is for cannot be read off the patients. `dilution_eval.py`
builds it the way a laboratory would: pseudo-slides of shuffled clean
fields from the frozen half and a few sparse infected fields from the
held-out patients (1 to 12 expert parasites each), mixed to a target
density, 40 per level, replayed through the decision rule. Both pools are
test material; nothing was fitted on them.

| target /µL | realised | fields | read | positive | negative | to a person |
|---|---|---|---|---|---|---|
| 0 | 0 | 26 | 20 | 0% | **100%** | 0% |
| 50 | 50 | 50 | 21 | 0% | 95% | 5% |
| 100 | 99 | 35 | 22 | 0% | 80% | 20% |
| 200 | 197 | 31 | 28 | 32% | 35% | 32% |
| 500 | 494 | 30 | 29 | **98%** | 2% | 0% |
| 1 000 | 977 | 28 | 28 | 100% | 0% | 0% |
| 2 000 | 1 959 | 30 | 28 | 100% | 0% | 0% |

**Limit of detection at 95% sensitivity: 500/µL** (N = 500; 1 000/µL at
N = 200, because the shorter read calls a 200/µL slide negative before its
parasites add up). On the chain-2 tables it was above 2 000/µL. A Level 1
microscopist finds far less than 500/µL; the floor is 18/µL on the frozen
half and 150 by the p95 rule, so the next factor of three to five is the
same false alarms, on the fit half's dirtiest slides, and the candidate
recall on light fields. What the harness does not model: a real light
slide spreads its parasites over every field, where here they sit in a few
fields among clean ones; the running total the rule reads is the same, the
clustering is not.

### With the white cell gate (6 September, afternoon)

Same network, white cells with the dark-fraction gate
(`results-wbc.md`). Frozen half: 94.8% of clean fields zero, 0.06 false
parasites per field, floor 19/µL, AUC 1.000; the fit half's p95 floor is
148/µL (from 150). The decision rule at N = 500 now calls **15 of the 15
clean slides long enough** negative and sends 10 to a person, up from 5,
for a reason that is not the vision: the gate removes white cell false
alarms, so a clean slide counts fewer white cells and more of them fall
short of 500 before the captured fields run out. On the fit half the
same rule decides 22 of 25 at N = 300 and 23 of 25 at N = 200
(`chain_2026-09-06_cnn.log`, design sweep). N stays at 500 because the limit of detection depends on
it (500/µL at N = 500, 1 000/µL at N = 200), and a slide with fewer than
500 white cells captured is a limit of this dataset, not of a microscope
with a stage. Sensitivity 49 of 50, unchanged; dilution unchanged, limit
of detection 500/µL, level 0 98% negative.
