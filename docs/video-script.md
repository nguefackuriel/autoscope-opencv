# The submission video: script and shot list

Three minutes. The rubric scores user experience and presentation, but the
thing a judge is really deciding is whether to believe us. Everything below is
chosen so that a sceptical viewer can check it afterwards.

**The rule for this video: never show a number that is not in the repository.**
If it is on screen it is in `docs/results-*.md` or `bench/results/`, and the
video says where.

## Shape

| | | |
|---|---|---|
| 0:00–0:25 | the problem | why a microscope, and why counting |
| 0:25–0:55 | what AutoScope is | the phone, the pipeline, the agent |
| 0:55–1:45 | **the live demo** | the page, a real session, the audit trail |
| 1:45–2:25 | the numbers | held-out, ECAMM, Tanzania |
| 2:25–2:50 | what does not work | vivax, COOL, species |
| 2:50–3:00 | close | the URL and the repository |

## 0:00-0:25, the problem

*Shot: the WHO report page with the figures visible, then a thick film down a
microscope.*

> Malaria killed 610,000 people in 2024, and 282 million got sick. Ninety-four
> percent of those cases were in Africa.
>
> Rapid tests are losing ground. Some parasites have stopped making the protein
> the tests look for. WHO found them in 42 countries. And artemisinin
> resistance is detected by *counting* parasites on day 0, 2 and 3. Rapid tests
> cannot count.
>
> So the work goes back to the microscope. A technician reads a hundred fields
> under oil before calling a slide negative. Twenty to thirty minutes, alone,
> and the mistakes come with fatigue.

Numbers from `docs/reference/WHO-World-Malaria-Report-2025.pdf`, read from the
report rather than recalled.

## 0:25-0:55, what it is

*Shot: `docs/figures/architecture.svg` building up, one stage at a time.*

> AutoScope turns the microscope they already own into a scanner. A phone films
> through the eyepiece. In the cloud, OpenCV 5 evens out the lighting and the
> stain, finds the white cells and the parasites, and turns the counts into
> parasites per microlitre with a confidence interval.
>
> Then an agent reads those numbers and decides what to do next. Refocus. Skip
> a dirty field. Keep reading. Stop. Or hand the slide to a person.
>
> Stages one to four and six run on AWS Batch on Graviton4. The agent is Claude
> on Bedrock. Step Functions holds the limits on fields, time and money outside
> the model, so no model can talk its way past them.

## 0:55-1:45, the demo, which is the part that matters

*Shot: screen recording of <https://d1k2o3z37c1maf.cloudfront.net>, no cuts.*

Open the page. Click the live session link. Let the table fill.

> This is a real session, read on AWS. Eight fields on a Graviton Spot fleet.

Scroll the log slowly. Let the viewer read the agent's own reasons.

> Field one: thirty white cells, the interval is very wide, keep reading.
> Field four: both bounds now sit inside one density band.
> Field seven: a hundred and thirty-three white cells, the band is settled.
> Stop.
>
> The agent is not pattern-matching a threshold. It is watching an error bar
> narrow and stopping when the answer stops changing, which is what a
> technician does.

Point at the interval bar.

> Eight thousand six hundred parasites per microlitre, interval seven thousand
> two hundred and fifty to ten thousand one hundred. The grey line is this
> pipeline's measured noise floor on clean slides. A positive is an interval
> that clears that line, not a verdict the page asserts.

Then click a detection in the verification grid and let the number move.

> And a technician can disagree. Reject a detection and the count and the
> interval recompute. Those corrections are what a deployment keeps as training
> data.

Last beat, and say it plainly:

> The rule-based control reads eleven fields on this patient. The agent reads
> eight. One slide is a data point, not a result. The comparison over all
> fifty held-out patients is in the repository.

## 1:45-2:25, the numbers

*Shot: the ECAMM table from `docs/plan-2026-09-05-hospital-grade.md`.*

> We measured against the WHO competency assessment, the panel a human
> microscopist is certified on. Fifty held-out patients the fit never saw.
>
> Parasite F1, 0.79. Limit of detection by dilution, 500 per microlitre.
> Twenty of twenty clean slides long enough to decide called negative, and
> forty-nine of fifty infected slides called positive, with none wrong.
> Counting passes Level 1.
>
> Then a Tanzanian set from a different microscope, a different camera and a
> different stain, run once: 99.3% of clean fields report nothing, 70% of
> infected fields find something, after a ten-slide site calibration.

## 2:25-2:50, what does not work

Do not skip this. It is the most persuasive forty seconds in the video.

> Three things we could not make work, and we are publishing all three.
>
> Species. On *P. vivax* the detector reads F1 0.29 against 0.80 on
> falciparum, and its confidence score does not separate the detections it
> gets right from the ones it gets wrong, so there is no automatic warning we
> can give. A slide from a vivax region needs a human reader.
>
> COOL. On a ten-function benchmark the Marketplace image is 1.59× faster than
> the wheel we ship, on the same machine. On our actual pipeline it is 3.8×
> *slower*: 143 slides an hour against 547. Six of the seven stages are
> faster; one is nine and a half times slower and swallows the rest. So we
> forced both builds down to a single thread, and they agreed to within six
> percent on every call. The kernels are the same speed. The stock build takes
> fifteen to twenty-five times back off the thirty-two cores, and the COOL
> image takes none, while `cvtColor` parallelises fine on both, which is what
> makes this a measurement and not a guess. Function benchmarks do not predict
> pipeline throughput, and we can name the call.
>
> And a learned detector that we trained, measured, and did not ship.
>
> The COOL result is a bug report with a two-line reproduction, and it is
> the half of that work we are proudest of.

## 2:50-3:00, close

*Shot: the page, then the repository.*

> AutoScope. The demo is live, every number is in the repository with the
> script that produced it, and it is a screening aid, not a diagnostic
> device. Every result is reviewed by a person.

## Production notes

- **Screen recording at 1440p**, the page in dark mode; it reads better
  compressed and the interval bar has more contrast.
- **No background music under the demo section.** The log scrolling is the
  evidence; let it be quiet.
- Record the demo **in one take**. A cut in the middle of a live session
  invites the question of what was cut.
- Subtitles throughout. Judges may watch muted, and the parasite counts are
  hard to hear.
- If the Marketplace COOL AMI arrives before filming, the COOL paragraph gains
  a fourth arm and should be re-recorded; the conclusion probably does not
  change.
- **Who speaks:** Uriel on the pipeline and the numbers, Alex on AWS, COOL and
  the agent, matching how the work was actually split.
