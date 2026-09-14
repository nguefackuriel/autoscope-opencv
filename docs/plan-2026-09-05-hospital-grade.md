# What "a hospital can use it" means, measured against where we are

5 September 2026, after PR #5. Every number below is from `eval/results/`
on the 50 held-out NLM patients and the frozen half of the uninfected set,
unless it says otherwise. Nothing new was fitted to write this.

## The yardstick

A hospital microscopist is certified against the WHO external competency
assessment (ECAMM). Level 1, the top grade, needs **at least 90% accuracy
in detecting parasites (negatives and positives), 90% in identifying the
species, and at least 50% of parasite counts within 25% of the true
count**, on a panel of 20 negative slides, 20 positive slides at set
densities and 16 *P. falciparum* slides for counting
([Malaria Journal, 2019](https://malariajournal.biomedcentral.com/articles/10.1186/s12936-019-2996-3)).
Lower levels relax that to 80/80/40 and 70/70/30 (to be checked against
the WHO malaria microscopy QA manual before it goes in the report).

That panel is the right frame for us, because it scores a *decision* per
slide, not an F1 per object.

## Where AutoScope stands on that panel

**Updated 6 September after chain 3 and the decision rule** (steps 1, 2
and 3 below are done; `results-parasites.md` §15, `results-negatives.md`
and `results-agent.md`, 6 September):

| ECAMM criterion | Level 1 | 4 Sept | **6 Sept** |
|---|---|---|---|
| negatives called negative | ≥ 90% | 0 of 25 | **20 of 20 long enough to decide** (5 of 25 too short for 500 white cells, sent to a person; none called positive) |
| positives called positive | ≥ 90% | no call existed | **49 of 50**, none called negative (TF150_R5 at 283/µL sent to a person) |
| limit of detection, 95% sensitivity | | above 2 000/µL | **500/µL** by dilution |
| counts within 25% | ≥ 50% | 58% | **64%** (every field), 92% within 50%; **70% / 90% with the white cell gate, 6 Sept afternoon** |
| species | ≥ 90% | nothing | nothing |
| another microscope, camera, stain | | nothing | nothing |

The white cell gate of 6 September afternoon (`results-wbc.md`) took the
density error per slide from 28.3% to 25.7% on top of this. The table
that follows is the 4 September state the plan was written against.


| ECAMM criterion | Level 1 | AutoScope, 4 Sept | what is missing |
|---|---|---|---|
| detection: negatives called negative | ≥ 90% | **0 of 25** clean slides called negative by the stopping rule; every one reads 100 to 1 500/µL | a negative call built on the measured noise floor |
| detection: positives called positive | ≥ 90% | 50 of 50 held-out patients read positive, but only 1 of them is under 1 000/µL | sensitivity below 1 000/µL is unmeasured, and 6 of the 150 NLM patients are under 1 000/µL, so it cannot be measured on this data as it is |
| detection at a 1 000/µL cut-off | | sensitivity 0.94, specificity 0.92, AUC 0.982 | the cut-off is 10 to 20 times what a Level 1 microscopist detects |
| counts within 25% | ≥ 50% | **58%** reading every field, 54% through the 200-WBC protocol (3 Sept) | passes, with 95% limits of −55% to +103% |
| species | ≥ 90% | nothing | *P. vivax* set downloaded, untested |
| another microscope, camera, stain | | nothing measured | Tanzania set downloaded, untouched |

So on the ECAMM panel today: counting passes Level 1, detection fails
because there is no negative call, species does not exist.

## The three numbers that decide it

**1. False parasites on clean and sparse fields.** On the frozen clean
half the pipeline reports 1.34 false parasites per field (median 1, p90
3, max 17; 10.6 per 200 white cells). On the fit half, whose 75 training
fields were a small part of 569, it is **2.78 per field, 21.6 per 200
white cells**: the frozen half happens to be cleaner slides, so the
425/µL floor in `results-negatives.md` is the optimistic end and about
860/µL is the other. The same false alarms are the +27% parasite
over-count on the 1 000 to 10 000/µL band (4 118 found for 3 239
annotated over 26 slides, white cells +6%) and the four worst slides:

| slide | expert /µL | parasites found / annotated | white cells found / annotated | density error |
|---|---|---|---|---|
| TF70_Scr440 | 3 552 | 323 / 99 | 291 / 223 | +150% |
| TF107_249 | 1 707 | 106 / 51 | 225 / 239 | +121% |
| TF81_W8B5 | 1 373 | 90 / 40 | 238 / 233 | +120% |
| TF37_Sc563 | 4 500 | 287 / 135 | 275 / 240 | +86% |

All numerator. The denominator is fixed. A Level 1 detection limit means
false parasites per 200 white cells under about 2, from 10 to 20 now: a
five to ten fold reduction on clean fields, which no threshold gives
(section 11 of `results-parasites.md`) and no stopping rule hides.

**2. Under-count on dense slides.** TF124_S540 (12 203/µL) finds 150 of
360 parasites, TF2N_159R (15 930/µL) 213 of 458, TF82_15_1_9 251 of 446,
white cells within 5% on all three. Band over 10 000/µL: −5% overall,
−14% on the density because of the +6% white cells. Candidate recall is
0.91 (`parasite_candidates.json`), so part is the top-hat (peaks closer
than 7 px merge, 3.6% inside the white cell exclusion zone, 3.3% at the
rim) and part is the classifier on crowded patches.

**3. Sampling.** Through the 200-white-cell protocol the mean absolute
density error is 38.0%. Reading every field of the same slides with the
same counter it is **29.6%** (58% within 25%, 86% within 50%). The
protocol costs eight points on its own. The agent already reads more
where the error bar is wide; a rule that stops on a *decision* (band or
threshold) rather than on a 20% interval reads more on exactly the sparse
slides where both the false alarms and the sampling error live.

## What to build, in order

### 1. Sparse-field false alarms, without new data (this week) — done 6 Sept: false parasites per clean field 1.34 → 0.06, floor 425 → 18/µL, density error 38.0% → 28.3%

Same recipe, four changes, one chain:

- **all 569 clean fields** of the fit half as negatives, not 75;
- **hard negatives from the fit set**: score every candidate on the ~950
  fit images the training never used (1 247 available, 300 used), keep
  everything the current network scores above 0.2 as a negative, plus
  every annotated parasite as a positive. About 40 min of scoring;
- **48 px input cut at 56**, so the patch holds the ring of cytoplasm and
  the neighbour that tells a platelet pair from a parasite; rods are
  already 30 to 40 px long and do not fit in 32;
- 20 epochs, then the threshold re-chosen on the calibration patients.

Cap the set near 250 000 patches so the laptop trains it in about five
hours overnight (`caffeinate -i`; 34 GB RAM holds it if the tensors stay
uint8 until the batch). On a g5 GPU on AWS it is minutes, which is the
first concrete use of the account.

Measure: false parasites per field on the frozen half, the 1 000 to
10 000 band bias, TF70/107/81/37, and the held-out F1 so recall does not
pay for it. Target: under 0.5 per field, floor under 200/µL, band bias
under +10%.

### 2. A decision rule in the agent (one to two days, replay in seconds) — done 5 Sept: `DecisionRule`, floor p95 of the fit half, N = 500

The stopping rule asks for a 20% interval, which 35 of 50 slides can never
reach, and calls negative only at zero parasites, which no clean slide
reaches. Replace it with what a technician does:

- **negative**: after at least 500 white cells (the WHO negative reads
  100 fields), the count sits below the 95th percentile of what a clean
  slide produces at that many white cells, with the floor measured on the
  fit half's fields the classifier never trained on;
- **positive**: the lower bound of the interval sits above that floor;
- **density band**: stop when the interval sits inside one of
  under 1 000, 1 000 to 10 000, 10 000 to 100 000, over 100 000/µL;
- otherwise **ask a person**, saying which of the three it could not
  settle.

`should_stop` already returns a reason string; the bands and the floor are
two more branches, and `agent_eval.py` measures them on the cached field
table in a second. Output: for the first time a per-patient sensitivity
and specificity with an actual decision, fields read per band, and how
many slides go to a human. That is the ECAMM row.

### 3. Limit of detection by dilution (one day, measurement only) — done 5 Sept: `dilution_eval.py`, LOD 500/µL on 6 Sept

Six NLM patients under 1 000/µL is not a low-density cohort. Build one:
pseudo-slides from the frozen clean half's fields plus *k* held-out
infected fields with known expert counts, mixed to 50, 100, 200, 500 and
1 000/µL, 30 pseudo-slides per level, run through the rule of step 2.
Report sensitivity per level. It is how a laboratory validates a new
method, it fits nothing, and it is the only way this data yields a number
below 1 000/µL.

### 4. A learned detector (two weeks) — built 6–14 Sept, chains 3 to 7, closed: the proposer is settled (F1 0.85 to 0.87 with any judge, dense band ~11%), the judge is not; every mining recipe trades the sparse infected slides against the clean fields under one global threshold and every judge combination lands at 23.5 to 25.8% density error; the shipped pair stays; what would change the total is a threshold per density band from the agent's running estimate, or a wider view, after P. vivax, the product and AWS (`results-parasites.md` §16 to 16g)

A small fully convolutional network through `cv.dnn` on the normalised
crop at half resolution, trained from the point annotations as Gaussian
heat-maps, replacing top-hat plus patch network. It sees context, has no
0.91 candidate ceiling and no 7 px merge, and it is where the dense-slide
under-count and the last of the sparse-field false alarms both live. The
patch network stays as the control. Target: object F1 at least 0.85 on
the held-out patients, dense band bias within ±10%.

### 5. Site calibration for another microscope (one week, after 4)

A hospital will not own the NLM phone-through-eyepiece rig. Two
measurements turn a new site's slides into NLM scale: the white cell
nucleus diameter in pixels (about 100 on NLM, about 600 on Tanzania) sets
the resize factor; the Lab statistics of ten local slides set the stain
reference. Then Tanzania is run once, image-level AUC, as the
generalisation number. Any deployment starts with this ten-slide
calibration, and the report says so.

### 6. *P. vivax*, run once

Does the detector count vivax, and at what recall. If the scores differ
from falciparum, add a "non-falciparum morphology, confirm species" flag.
Species identification proper is out of scope for 26 October and the
report has to say it: that is the one ECAMM row we will not have.

### 7. The product

Verification page: every detection as a crop the technician accepts or
rejects, the count and density recomputed from the corrections, and the
corrections kept as training data. Report written from the trace by the
Bedrock agent (what was read, why it stopped, what it could not settle),
the trace as the audit trail. API and page on AWS, the COOL benchmark
with its three builds. Screening aid, not a diagnostic device, on every
page.

## What we will not be able to claim on 26 October

Species. Detection below the level step 3 shows. Any number on a
microscope other than NLM's until step 5 has run. A clinical validation:
everything here is on retrospective, de-identified research slides.

## Split

Uriel: 1, 2, 3, 4. Bruno: AWS, COOL three builds, Bedrock report and
agent, 7. Step 5 together. Step 1 is the one to start today: it moves
the negative call, the sparse-slide over-count and the ECAMM detection
row at once, and needs nothing that does not already exist on this
laptop.
