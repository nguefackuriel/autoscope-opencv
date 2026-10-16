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

## 4 September, chain 2

Same replay after the dark-nucleus white cell rule and the classifier
retrained with clean fields (threshold 0.80). The expert-count arms are
unchanged by construction; the real-counter arms:

| arm | fields read | density bias | mean abs. error |
|---|---|---|---|
| who-200 / found | 518 | -5.0% | **38.4%** (was 42.5%) |
| rule-based / found | 491 | -3.9% | **38.3%** (was 43.5%) |

The agent's decisions did not move (491 against 490 fields); the answer got
better. Fields now take 1.26 s.


## 6 September: chain 3, and the decision rule as a third policy

Same replay after the chain-3 network (`results-parasites.md` section 15,
threshold 0.53, 1.54 s per field). The **decision** arm is
`policy.DecisionRule` (`results-negatives.md`, 6 September): negative on
the noise floor after 500 white cells, positive when the interval clears
it, settled when it sits in one band within 30%. With expert counts the
floor is 0, there being no false alarms to clear.

| arm | fields read | median | error bar | density bias | mean abs. error | flagged | positive | negative | band settled |
|---|---|---|---|---|---|---|---|---|---|
| who-200 / truth | 556 | 11 | 27% | -1.1% | 4.0% | 11 | | | |
| rule-based / truth | 494 | 10 | 26% | -0.6% | 2.8% | 37 | | | |
| **decision / truth** | **440** | 8 | 29% | -1.0% | 4.7% | 23 | 49 | 0 | 42 |
| who-200 / found | 518 | 10 | 28% | -8.9% | **29.1%** (was 38.4%) | 11 | | | |
| rule-based / found | 491 | 9 | 27% | -8.4% | **29.6%** (was 38.3%) | 36 | | | |
| **decision / found** | **448** | 9 | 29% | -8.6% | 29.6% | 24 | 49 | 0 | 40 |

Nine points off the density error with the real counter, from the vision.
The decision arm reads 11% fewer fields than the rule-based one with expert
counts (440 against 494) and 21% fewer than the protocol, for 4.7% error
against 2.8% and 4.0%; with the real counter, 448 fields for the same
29.6%. It calls all 49 positives positive and settles a band on 40 of
them; the 24 flagged are slides that ran out before the band was settled,
and TF150_R5 at 283/µL, which it will not call either way. The -8.6% bias
of the real-counter arms is now the white cell count (+6%), not the
parasites (-1.1%): `results-wbc.md`, 6 September.

How tight a positive has to be before a settled band is enough to stop
(`--sweep-settle`, expert counts):

| settle width | fields | error bar | mean abs. error | flagged |
|---|---|---|---|---|
| band alone | 282 | 37% | 10.9% | 6 |
| 50% | 361 | 37% | 7.7% | 13 |
| **30%, shipped** | **440** | 29% | **4.7%** | 23 |
| 20% | 494 | 26% | 2.8% | 37 |

Band alone stopped every dense slide after five fields with the band right
and the number wrong; 20% is the old rule again. 30% is the compromise
shipped: a clinic monitoring a patient on day 0, 2 and 3 needs the number
as well as the call.

### With the white cell gate (6 September, afternoon)

Same replay after the white cell gate (`results-wbc.md`), network
unchanged. Expert-count arms identical by construction; real counter:
who-200 26.5% (from 29.1%), rule-based 27.2% (from 29.6%), decision 27.4%
(from 29.6%) mean absolute density error, decision arm 450 fields, 49
positive, 40 banded.

### heat-cnn (6 September, evening)

Same replay with the heat map proposing and the patch network judging
(`results-parasites.md` 16b), threshold 0.42: WHO 25.7%, rule-based
27.1%, decision 27.3% mean absolute density error with the real counter,
against 26.5 / 27.2 / 27.4 for the plain pair; the density bias of the
real-counter arms goes from −8.4% to +1.6%, and the decision arm reads
451 fields, 49 positive, 40 banded. The heat map alone: 43.8 / 45.7 /
45.5%, with +5 to +6.5% bias (`agent_eval_heatmap.json`).


## The agent arm, and how to reproduce it without paying for it

18 September 2026. `BedrockPolicy` is now a fifth arm of this comparison, run
by the same `read_slide` over the same cached field table as the other four.
Two things about how it is scored, because both are choices and neither is
obvious.

**The agent chooses when to stop. `density.decide()` says what the counts
mean.** The model's `stop` tool answers with a `call` of positive, negative or
settled, and that answer is *not* what appears in the call column. The call is
derived from the counts it had accumulated, by the same function that serves
the rule. Otherwise the column would mean "the interval cleared the measured
noise floor" for one arm and "the model asserted a word" for another, and
comparing them would be a category error: a model that said positive on every
slide would score 49/50 and look like it had beaten the rule.

It is also the safety property claimed elsewhere in this repo — *it picks where
to look and when to stop, never the diagnosis* — and this is the line of code
where that is true or false.

The model's own answer is kept as `agent_call` and scored as an agreement rate.
How often a model's self-assessment matches what its own counts say is a number
worth having, and it costs nothing to collect.

**Every arm now carries `derived_call` and `derived_band`,** computed from the
counts where that arm stopped. The older `call`/`band` columns are unchanged
and still blank for who-200 and rule-based: they are policy-set, they are what
the table above publishes, and redefining a published column in place would
move numbers with nothing to show for it.

**Reproducing it costs nothing.** `eval/results/agent_bedrock_calls.jsonl`
holds every model call the experiment made, keyed by a hash of what the model
saw *and* of the prompt and tool definitions that asked it. A judge runs

```
python eval/agent_eval.py --policies who,rule,decision,bedrock
```

with no AWS account and no `anthropic` package installed, and gets the same
table. There is a test asserting that a replayed run never imports the SDK.

Two consequences worth stating rather than discovering later. **A cache miss is
an error, not a quiet live call** — a run advertised as replayed must not be
partly live, and a miss almost always means the observation stream diverged, so
falling through would silently re-run and re-pay for the whole experiment.
**And the noise floor is inside the observation**, so re-measuring it after a
chain run invalidates the recording by design; re-record, or run the agent at
the recorded floor and say which.

**The recording makes the agent deterministic, and a live model is not.** The
claim this artefact supports is "this recorded session, which anyone can replay
exactly, reads N fields", not "any live run reads N fields". Sampling variance
across live runs is unmeasured and this document does not imply otherwise.

## 11 October: the agent measured against the rule, on all fifty patients

The comparison this project has owed since the proposal. Claude Sonnet 4.6 on
Bedrock, one call per loop step, 900 calls recorded to
`eval/results/agent_bedrock_calls.jsonl` and replayable at zero cost — the
replay below served 906 calls from disk and made none over the network.

With the real counter, which is what a user would get:

| arm | fields read | mean abs. density error | escalated | band settled |
|---|---|---|---|---|
| who-200 | 524 | 26.5% | 7 | 39 |
| rule-based | 491 | 27.2% | 36 | 39 |
| decision | 450 | 27.4% | 25 | 40 |
| **bedrock** | **393** | **26.8%** | **14** | 39 |

**The agent reads 393 fields where the decision rule reads 450 and the WHO
protocol reads 524 — 13% fewer than the rule, 25% fewer than the protocol — and
the density error is the same.** 26.8% against 26.5% and 27.4%: those three are
within noise of each other, and the error is the vision's, not the policy's.

It also escalates less: 14 slides to a person against the rule's 25 and
rule-based's 36.

So the answer to the question the plan posed in September — *if the model
cannot beat the rule, the report says so* — is that it does beat it, on the
axis the proposal claimed: fewer fields at the same answer. That is the
saving, and it is measured rather than asserted.

### The agent's own verdict agrees with the counts, 36 times out of 36

The model's `stop` tool reports what it thinks the slide is. That is never what
the results table prints — the call is derived by `density.decide()` from the
counts, so every arm is scored by the same function. But the two can be
compared, and on the 36 slides where the agent gave a verdict it matched the
derived call every time.

That is worth stating precisely: it does not mean the model is a good
diagnostician. It means the model's self-report is consistent with the
arithmetic it was shown, which is the most this design lets it be responsible
for.

### Where it is worse, and this is the more interesting half

| arm | fields | mean abs. density error |
|---|---|---|
| decision / **expert counts** | 440 | **4.7%** |
| bedrock / **expert counts** | 395 | **8.1%** |

Given perfect counts, the agent stops too early: 8.1% error against the rule's
4.7%. With the real counter that penalty vanishes, because the vision's error
(around 27%) swamps it.

The honest reading is that **the agent's earlier stopping is free only because
our detector is not good enough for it to cost anything.** If the vision
improved to the point where sampling error mattered, this arm would need
re-tuning. The report should say that rather than quote 393 fields as a pure
win.

### Reproducing it

```
python eval/agent_eval.py --policies who,rule,decision,bedrock
```

No AWS account, no `anthropic` package, no cost. The recording is in the
repository and a cache miss is an error rather than a silent live call, so a
number that comes back is the number that was recorded.

**Model caveat.** The agent was designed against Claude Opus 5 and these
numbers are Sonnet 4.6, which is the most capable Anthropic model this AWS
account can reach — Opus 5, Sonnet 5, Opus 4.8 and 4.7 all answer "not
available for this account". A different model would need a new recording.

## 11 October, second half: does the agent need a frontier model?

`ConversePolicy` runs the identical loop and the identical five moves through
Bedrock's Converse API, which is model-agnostic, so the only thing that
differs between arms is the model. Three recorded, 3 005 calls, all replayable
at zero cost.

With the real counter:

| arm | fields read | density error | escalated | model calls |
|---|---|---|---|---|
| who-200 | 524 | 26.5% | 7 | — |
| rule-based | 491 | 27.2% | 36 | — |
| decision | 450 | 27.4% | 25 | — |
| **Claude Sonnet 4.6** | **393** | **26.8%** | 14 | 453 |
| **GPT-OSS 120B** | **390** | 29.9% | 15 | 442 |
| Llama 4 Maverick | 575 | 25.2% | **46** | 604 |

### An open-weight model does the job

**GPT-OSS 120B reads 390 fields against Sonnet's 393** — the same saving over
the rule, from a model with open weights. It pays about three points of
density error for it (29.9% against 26.8%), which is inside the band where the
vision's own error lives but is not nothing.

With expert counts, where the vision noise is removed, GPT-OSS is the better
stopper of the two: 6.9% against Sonnet's 8.1%, on 410 fields against 395.

So the answer to "does this need a frontier model" is **no**. The design is the
agent, not the model. That matters for a tool meant for district hospitals:
the arm that works is the one you can self-host.

### A model that cannot be forced to pick a move is not an agent

Llama 4 Maverick rejects `toolConfig.toolChoice.any` outright — Bedrock
answers *"This model doesn't support the toolConfig.toolChoice.any field"*. So
that arm runs unforced, and a turn that returns prose instead of a tool call is
recorded as the model declining to commit.

It declined on **46 of 50 slides**. It reads 575 fields, more than the WHO
protocol's 524, because it mostly never decides to stop. Its 25.2% density
error is the best in the table and means nothing: a policy that reads every
field and hands the slide to a person has not made a decision, it has avoided
one.

That is the useful negative here, and it is a property of the API surface
rather than of the model's intelligence: **forced tool choice is a hard
requirement for this architecture**, and it should be checked before a model
is considered at all. Our own first check missed it, because a `toolConfig`
without `toolChoice` is accepted by every model.

### What this costs

| model | calls | input tokens |
|---|---|---|
| Claude Sonnet 4.6 | 900 | 779k |
| GPT-OSS 120B | 901 | 407k |
| Llama 4 Maverick | 1 204 | 742k |

GPT-OSS uses roughly half the input tokens for the same work, because the
Converse path sends no `strict` schema and the model is terser.

### The models this account cannot reach

Opus 5, Sonnet 5, Opus 4.8, Opus 4.7, GPT-6.1 Sol, GPT-6, GPT-5.6, Mistral
Large, DeepSeek v3 and Qwen 3 all answer "not available for this account".
Every number above comes from what was actually reachable, and the report
should say so rather than imply a free choice of model.

## 16 October: twelve more models through the same loop

Every model runs the identical loop, the identical five moves and the identical
prompt through Bedrock's Converse API. The only thing that changes between arms
is the model id. Fifteen models are now recorded, 15 396 calls in
`eval/results/agent_bedrock_calls.jsonl`, all replayable at no cost:

```
.venv/bin/python eval/agent_eval.py --policies who,rule,decision,bedrock,gptoss,\
llama,haiku,deepseek,devstral,ministral,nova,gemma,minimax,nemotron,qwen80,\
qwen32,qwencoder,qwenvl
```

| model | fields read | density error | slides counted | escalated | chose stop | skipped | oracle error |
|---|---|---|---|---|---|---|---|
| Qwen3-Next 80B | 81 | 18.3% | 16 | 45 | 19 | 663 | 8.0% |
| Qwen3-VL 235B | 239 | 36.6% | 50 | 1 | 98 | 26 | 18.5% |
| Mistral Devstral 2 123B | 264 | 32.4% | 50 | 6 | 88 | 27 | 14.7% |
| MiniMax M2.5 | 303 | 29.2% | 50 | 18 | 64 | 35 | 14.6% |
| Qwen3-Coder-Next | 303 | 49.3% | 46 | 23 | 60 | 220 | 13.7% |
| GPT-OSS 120B | 390 | 29.9% | 50 | 15 | 64 | 36 | 6.9% |
| **Claude Sonnet 4.6** | **393** | 26.8% | 50 | 14 | 71 | 39 | 8.1% |
| Qwen3 32B | 479 | 26.5% | 50 | 29 | 37 | 40 | 2.3% |
| Claude Haiku 4.5 | 524 | 25.8% | 50 | 20 | 58 | 42 | 3.7% |
| DeepSeek V3.2 | 557 | 25.5% | 50 | 39 | 20 | 41 | 1.4% |
| Nvidia Nemotron Nano 3 30B | 568 | 25.0% | 50 | 50 | 0 | 22 | 1.2% |
| Llama 4 Maverick | 575 | 25.2% | 50 | 46 | 10 | 42 | 0.8% |
| Amazon Nova Pro | 587 | 24.8% | 50 | 50 | 0 | 42 | 0.6% |
| Gemma 3 27B | 587 | 24.8% | 50 | 50 | 0 | 42 | 0.6% |
| Ministral 3 14B | 587 | 24.8% | 50 | 50 | 2 | 42 | 0.6% |

The WHO protocol reads 524 fields for 26.5% error. Only six of the fifteen read
fewer, and three of those six pay for it with an error above 29%.

### What separates them is willingness to stop, not size

`eval/agent_moves.py` counts the move each model chose, from the cache:

```
.venv/bin/python eval/agent_moves.py
```

Nova Pro, Gemma 3 27B and Nemotron Nano 3 30B never chose `stop` once, in
1 218, 1 218 and 1 171 calls. Ministral 3 chose it twice in 1 215. All four read
every field the slide has and then send the slide to a person because the
fields ran out, which is why they sit at the bottom of the table with 50
escalations out of 50 slides. Their 24.8% density error is the best number in
the column and means nothing: a policy that reads everything and decides nothing
has not been measured on the thing we are measuring.

Forced tool choice is accepted by all of them. It makes the model return a move.
It does not make the model willing to pick the move that ends the slide.

At the other end, Qwen3-VL 235B chose `stop` 98 times out of 610 calls and reads
239 fields, less than half the WHO protocol, for 36.6% error. Devstral 2 and
MiniMax M2.5 behave the same way. Stopping early is easy; stopping at the right
time is the task.

### Two models refuse to count

Qwen3-Next 80B answered `skip` 663 times out of 1 185 calls. It counted on 16 of
50 slides and read 81 fields in total. Its 18.3% error is over those 16 slides
and is not comparable to any other row. Qwen3-Coder-Next skipped 220 times and
counted on 46 slides.

A model that declines the work scores well on every column except the one that
matters. The slides-counted column exists so that this is visible rather than
flattering.

### Size does not predict any of it

Haiku 4.5 is the smallest Anthropic model on the account and reads 524 fields at
25.8% error with 58 stops, which is a usable policy. Nova Pro and Gemma 3 27B
sit at opposite ends of the size range and behave identically, because neither
stops. The two 235B-class models, Qwen3-VL and Llama 4 Maverick, fail in
opposite directions. The ranking follows a behaviour, not a parameter count.

### For the entry

This is the measured answer to "does the agent need a frontier model". It does
not. It needs a model that will end a task. That is a property worth checking in
one cheap run before a model is considered at all, and it is not on any model
card.

## 16 October, second half: a fixed eight fields

`FixedProtocol(8)` and `FixedProtocol(12)` are arms now, so the comparison the
report had been making in prose is in the table:

| policy | fields | density error | escalated | oracle error |
|---|---|---|---|---|
| WHO 200 | 524 | 26.5% | 7 | 4.0% |
| **fixed 8** | **390** | **26.8%** | **8** | **7.1%** |
| fixed 12 | 531 | 26.7% | 27 | 1.9% |
| rule-based | 491 | 27.2% | 36 | 2.8% |
| decision | 450 | 27.4% | 24 | 4.7% |
| agent, Sonnet 4.6 | 393 | 26.8% | 14 | 8.1% |

Fixed 8 matches the agent on fields and on error, sends fewer slides to a
person, and is better on expert counts. It needs no model and no prompt. On
totals alone the agent is not worth its complexity, and the report now says so
in those words.

### What the agent does that the fixed policy cannot

The totals hide the allocation. Fields read against the slide's expert density,
over the 50 held-out patients:

| policy | ten densest slides | ten sparsest | range | r with log density |
|---|---|---|---|---|
| fixed 8 | 8 | 8 | 5 to 8 | +0.12 |
| agent, Sonnet 4.6 | 3 | 9.5 | 1 to 14 | **-0.70** |

The fixed policy reads eight fields whatever the slide is. The agent reads
three on a slide where the first field already settles the band, and nine or
ten where it does not. Same total, same error, spent differently.

That is the honest size of the claim. The agent allocates, gives a reason for
every move that a reviewer can read next to the field it was made on, and the
whole comparison replays from the cache. It does not beat a fixed eight on any
number in the first table.

Reproduce:

    .venv/bin/python eval/agent_eval.py --policies all
