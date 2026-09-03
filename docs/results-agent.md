# Does the agent help?

Branch `u/parasite-detector`, 28 August 2026. The proposal's central claim was
that the stopping rule is not an optimisation but the product: without it there
is no reason to prefer this to a fixed protocol. This is the measurement.

Run it with:

```
cd eval
../.venv/bin/python agent_fields.py      # count every held-out field once (29 min)
../.venv/bin/python agent_eval.py        # replay the policies over that table
../.venv/bin/python agent_eval.py --sweep
```

## The design

Every field of every held-out patient was counted once (`agent_fields.py`,
609 annotated fields from 50 patients, 2.65 s per field). `agent_eval.py` then
replays sessions over that table using **the real `read_slide` loop, the real
policies and the real stopping rule**; only the source of the numbers is
cached. A field returns the same counts whoever asked for it, so no decision
changes, and a two-hour experiment becomes a one-second one.

Two baselines and two counters:

| | |
|---|---|
| `who-200` | the WHO thick film method: read fields until 200 white cells have gone by. Not a fixed field count, because how many fields that takes depends on the slide |
| `rule-based` | quality gate, then read until the error bar is inside 20%, or a negative slide has passed 200 white cells |
| source `truth` | the expert annotation. Isolates the agent from the vision |
| source `found` | what the pipeline found. What a user would get |

The reference answer for a patient is its own expert annotation over **all**
its fields.

## The headline

50 patients, 609 fields available, target error bar 20%, minimum 5 fields.

| arm | fields read | median | error bar | density bias | mean abs. error | flagged |
|---|---|---|---|---|---|---|
| who-200 / truth | 556 | 11 | 27% | -1.1% | 4.0% | 11 |
| **rule-based / truth** | **470** | **8** | 26% | **-0.1%** | **3.7%** | 37 |
| who-200 / found | 492 | 10 | 26% | -3.0% | 44.4% | 14 |
| rule-based / found | 461 | 8 | 27% | -0.6% | 43.5% | 34 |

(The two `found` rows are at the corrected classifier threshold of 0.75. At
the 0.80 this experiment first ran with, they read 492 and 469 fields with
density bias -16.2% and -14.1%: the policy barely moved, the answer did. See
`results-parasites.md` section 11.)

**The agent reads 15% fewer fields and is no less accurate.** With the expert
counter, 470 fields against 556, mean absolute density error 3.7% against
4.0%, bias -0.1% against -1.1%.

That average hides the shape of the result, which is the interesting part.

## Where the saving actually is

Split by the patient's true parasite load:

| slide | patients | WHO fields | agent fields | saving | agent error bar | agent abs. error |
|---|---|---|---|---|---|---|
| under 1 000 /µL | 1 | 4 | 5 | -25% | 129% | 0.0% |
| 1 000 to 10 000 /µL | 26 | 294 | 307 | **-4%** | 41% | 1.3% |
| over 10 000 /µL | 23 | 258 | **158** | **+39%** | 19% | 6.1% |

This is the behaviour the proposal argued for, and it is worth stating in the
right direction: **the agent does not read less, it reads the right amount.**
On heavy slides it stops at 158 fields where the protocol reads 258, because
the error bar closes fast when parasites are plentiful. On middling slides it
reads 4% *more*, because 200 white cells is not enough there and the fixed
protocol stops too early. A method that only ever saved would be a method that
stops too early on the slides that matter.

The single trace that shows it, from `docs/figures/agent-trace-example.txt`:

```
slide: TF15_C87P48
  1  (0,0) z=0   count_field()   focus=28 even=0.96 debris=0.71  parasites=79  wbc=18
  2  (1,0) z=0   count_field()   focus=26 even=0.96 debris=0.67  parasites=88  wbc=17
  3  (2,0) z=0   count_field()   focus=25 even=0.97 debris=0.71  parasites=87  wbc=14
  4  (3,0) z=0   count_field()   focus=22 even=0.96 debris=0.67  parasites=88  wbc=16
  5  (4,0) z=0   count_field()   focus=23 even=0.94 debris=0.67  parasites=71  wbc=20
  6  (5,0) z=0   stop            38,871/uL, error bar 20% is within the 20% target
read 5 fields, 413 parasites in 85 white cells
```

Five fields against the WHO protocol's eighteen on that slide: 3.6 times less
work, on a slide at 38 871 parasites/µL where the answer was never in doubt
after the third field.

## What this run does not show, and why

**The other half of the claim is untestable on this data.** The proposal also
said the agent keeps reading past the fixed protocol on light slides, which is
where a fixed method misses people. There is exactly **one** patient under
1 000 /µL in the held-out set, and slides here carry 5 to 22 fields. The agent
wanted to keep reading on 24 of the 50 patients and ran out of slide. That is
what the 37 "flagged" sessions are: not failures, but the loop correctly
refusing to answer and asking for a person. To measure this half we need
slides with more fields or a lightly infected cohort, and until then the claim
stays unproven rather than quietly dropped.

**Only 15 of 50 patients reach a 20% error bar at all**, even reading every
field. On this data a 20% target is out of reach for a slide of twelve fields.
At a 30% target, 29 of 50 reach it. The target belongs in the report as a
parameter with a measured cost, not as a constant.

**Refocus cannot be scored.** One capture per field means a policy can decide
to refocus, and we can log it, but no count changes as a result.

## Two things the trace exposed, both now fixed

**A log line that read as a bug.** `why_not()` rounded to whole units, so a
field at 10.95 against a limit of 11 logged "sharpness 11 needs 11". A judge
reading that concludes the code is broken. Now two decimals.

**The agent hunting for a focus that could not move.** With one capture per
field, sharpness never changes after a refocus, so the policy burned its four
attempts on every soft field. It now gives up as soon as a refocus leaves the
score unchanged, which is also the right behaviour on real hardware with a
stuck objective. Over the 50 slides: refocus calls fall from 100 to 25, log
steps from 662 to 587, and all 469 counted fields are identical.

## The honest split between agent and vision

The two `found` rows are the answer to "is the agent robust to a vision system
that makes mistakes". It reads 469 fields against 470 with the perfect
counter: **the agent's decisions barely move.** What moves is the answer, from
-0.1% bias to -14.1%, which is the parasite detector's known under-count
(see `results-parasites.md`, section 10) flowing straight through.

So: the agent works, and the number a user would get is limited by the vision,
not by the policy. That is the split the injected counter was built to expose,
and it says where the next week of work goes.

It also proved its worth immediately. Run first at the old threshold, this
table showed a -14.1% density bias with the real counter against -0.1% with
the expert one. The policy was identical in both, which pointed at the
counting rather than the deciding, and the threshold error in
`results-parasites.md` section 11 came out of following that. With the
threshold corrected the same table reads -0.6%.

## Parameters

`min_fields` was 10, a guessed constant in `density.should_stop`. On this data
a field carries about 19 white cells, so a floor of 10 fields is roughly the
WHO count itself, and the agent could never stop earlier than the baseline: the
comparison would have been vacuous by construction. It is now a policy
argument, defaulting to 5, and swept in `agent_sweep.json`:

| target | min fields | agent fields | abs. error |
|---|---|---|---|
| 15% | 3 | 492 | 3.4% |
| 20% | 3 | 455 | 4.3% |
| 20% | 5 | 470 | 3.7% |
| 20% | 8 | 500 | 2.7% |
| 30% | 3 | 393 | 7.3% |
| 30% | 5 | 421 | 5.7% |

The floor buys accuracy at a predictable price in fields, which is the trade a
lab should be allowed to set rather than one we bury in a constant.

## Re-measured on 3 September with the corrected pipeline

Same 50 patients, same 609 fields, same loop and policies. Three things
changed underneath: the stain statistics were fixed and the white cell and
parasite stages refit and retrained on them (`results-parasites.md` §12),
the sharpness score no longer counts noise (`data-notes.md`), and the
classifier threshold is 0.76. Fields are now 1.34 s each, from 2.65 s.

| arm | fields read | median | error bar | density bias | mean abs. error | flagged |
|---|---|---|---|---|---|---|
| who-200 / truth | 556 | 11 | 27% | -1.1% | 4.0% | 11 |
| **rule-based / truth** | **494** | **10** | 26% | **-0.6%** | **2.8%** | 37 |
| who-200 / found | 503 | 10 | 26% | +1.1% | 42.5% | 13 |
| rule-based / found | 490 | 9 | 24% | +1.9% | 43.5% | 36 |

The agent reads 11% fewer fields than the protocol (494 against 556) with a
smaller error (2.8% against 4.0%). It reads more than in August (470) for a
good reason: the old sharpness score was rejecting fields for their JPEG
grain, and the gate now passes 587 of 609 fields instead of skipping them.
Refocus calls fell from 25 to 1 for the same reason.

By density band, expert counts:

| slide | patients | WHO fields | agent fields | saving | agent error bar |
|---|---|---|---|---|---|
| under 1 000 /µL | 1 | 4 | 5 | -25% | 129% |
| 1 000 to 10 000 /µL | 26 | 294 | 318 | -8% | 41% |
| over 10 000 /µL | 23 | 258 | **171** | **+34%** | 19% |

Same shape as before: the saving is on the heavy slides, and on the middling
ones the agent keeps reading past the 200-cell protocol because the error
bar is still wide. The two `found` rows say what they said in August, more
sharply: with the real counter the agent's decisions barely move (490
against 494 fields) and the answer's error is 43.5% against 2.8%. The
vision, not the policy, is what stands between this and a useful number.

