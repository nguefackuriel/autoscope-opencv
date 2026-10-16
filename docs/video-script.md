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
| 1:45–2:10 | the numbers | held-out, specificity, Tanzania |
| 2:10–2:25 | what the agent is worth | eighteen policies, fifteen models |
| 2:25–2:50 | what does not work | vivax, COOL |
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

*Shot: `docs/proposal/architecture.svg` building up, one stage at a time.*

> AutoScope turns the microscope they already own into a scanner. A phone films
> through the eyepiece. In the cloud, OpenCV 5 evens out the lighting and the
> stain, finds the white cells and the parasites, and turns the counts into
> parasites per microlitre with a confidence interval.
>
> Then an agent reads those numbers and decides what to do next. Refocus. Skip
> a dirty field. Keep reading. Stop. Or hand the slide to a person.
>
> All six stages are OpenCV, and they run on AWS Batch on Graviton4. The agent
> sits outside them: everything it sees is OpenCV output, and the only thing it
> returns is which field to read next. Step Functions holds the limits on
> fields, time and money outside the model, so no model can talk its way past
> them. The call itself comes from a fixed function reading the counts. The
> agent has no move that issues a diagnosis.

## 0:55-1:45, the demo, which is the part that matters

*Shot: screen recording of <https://d1k2o3z37c1maf.cloudfront.net>, no cuts.*

Open the page. It loads a real session by itself. Let the table fill.

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

Then press "Replay a recorded session" and click a detection in the
verification grid. The per-detection crops ship with that recorded session;
a session read live keeps the field pictures and the counts, not the crops.

> And a technician can disagree. Reject a detection and the count and the
> interval recompute. Those corrections are what a deployment keeps as training
> data.

Last beat, and say it plainly:

> One slide is a data point, not a result. The comparison over all fifty
> held-out patients is in the repository, and it does not say what you would
> expect.

## 1:45-2:10, the numbers

*Shot: the targets table on page 6 of the report, the one with two MISSED
chips in it.*

> Fifty held-out patients the fit never saw.
>
> Parasite F1, 0.79. Limit of detection by dilution, 500 parasites per
> microlitre. Forty-nine of fifty infected slides called positive and not one
> called negative. On twenty-five clean slides, none called positive, three
> sent to a person, so specificity is 0.88. We set ourselves 0.90 in
> September and we missed it.
>
> Then a Tanzanian set from a different microscope, a different camera and a
> different stain, scored once: 99.3% of clean fields report nothing, image
> level AUC 0.85, after a ten-slide site calibration.

## 2:10-2:25, what the agent is actually worth

*Shot: the arm table in section 6 of the report.*

> We ran eighteen stopping policies through the same loop, the same prompt and
> the same counter. Fifteen of them are models on Bedrock. Fifteen thousand
> recorded calls ship with the repository, so anyone can replay the whole
> comparison for nothing.
>
> Two things came out of it, and neither is the thing we expected.
>
> First: a fixed eight fields per slide does as well as the agent on the
> totals. Three hundred and ninety fields against three hundred and ninety
> three, the same error, fewer slides sent to a person. No model needed. We
> say that in the report in those words.
>
> What the agent does instead is spend the budget unevenly. Three fields on
> the ten densest slides, nine or ten on the ten sparsest, and the correlation
> with the slide's real density is minus zero point seven. The fixed policy
> reads eight whatever the slide is. Same cost, same error, put where the
> evidence is thin, with a reason attached to every move.
>
> Second: what separates the models is not size. It is whether they are
> willing to stop. Three of the fifteen never chose stop once in more than a
> thousand calls each. They read every field a slide has and then hand it over.
> Forced tool choice makes a model answer. It does not make it finish.

## 2:25-2:50, what does not work

Do not skip this. It is the most persuasive forty seconds in the video.

> Two things we could not make work, and we are publishing both.
>
> Species. On *P. vivax* the detector reads F1 0.29 against 0.80 on
> falciparum, and its confidence score does not separate the detections it
> gets right from the ones it gets wrong, so there is no automatic warning we
> can give. A slide from a vivax region needs a human reader.
>
> COOL. On a ten-function benchmark the Marketplace image is 1.59 times faster
> than the wheel we ship, on the same machine. On our pipeline it was 3.8 times
> slower: 143 slides an hour against 547. Six of the seven stages are faster.
> One is nine and a half times slower and swallows the rest.
>
> So we forced both builds down to a single thread, and they agreed to within
> six percent on every call. The kernels are the same speed. The stock build
> takes fifteen to twenty-five times back off the thirty-two cores. The COOL
> image takes none, while `cvtColor` parallelises fine on both, which is what
> makes this a measurement and not a guess.
>
> The fix is forty lines. We split the image into bands, one per core, each
> with a halo as wide as the kernel, and run the morphology band by band. The
> result is checked against the unsplit call before it is timed. With that on,
> the COOL image reads 532 slides an hour against the wheel's 547.
>
> Then we ran the whole pipeline twice on the same twelve fields, once on each
> build, and kept the counts. Six hundred and five parasites and every white
> cell, identical. The COOL image runs AutoScope at 0.97 times the wheel and
> returns the same answers. That defect is a bug report with a reproduction,
> and it is the half of this work we are proudest of.

## 2:50-3:00, close

*Shot: the page, then the repository.*

> AutoScope. The demo is live, every number is in the repository with the
> script that produced it, and it is a screening aid, not a diagnostic
> device. Every result is reviewed by a person.

## The footage of the agent reading

`docs/figures/session/four-sessions.gif`: 49 seconds, the four live AWS sessions
back to back, which is the length of the 0:55-1:45 beat. Each session's last
frame is held for 4.2 seconds so the decision can be read. The order is chosen so
the contrast lands:

| | fields | what it shows |
|---|---|---|
| TF41_UT5 | 3 | 164 parasites in the first field; stops at 90,435/uL |
| TF101_233 | 8 | the ordinary case, settles at 8,601/uL |
| TF22_C104P65 | 11 | reads everything it has and still straddles a band at 700 |
| TF150_R5 | 5 | runs out of slide at 288/uL and goes to a person |

The frames are real: the NLM capture, `count_field`'s own detections at the
shipped threshold, and the move and reason the model gave on Bedrock. Nothing is
staged and nothing is re-run for the camera.

Rebuild it, or re-pace it for the edit, without re-rendering anything:

    .venv/bin/python eval/render_session.py \
      --session TF41_UT5,TF101_233,TF22_C104P65,TF150_R5 \
      --gif --gif-only --seconds 1.4 --gif-width 960

`--gif-only` reads the PNGs already in `docs/figures/session/`, so changing the
pace costs a second instead of the five minutes it takes to run `count_field`
over 27 twelve-megapixel captures. Drop `--gif-only` and add
`--from-api https://fhemydiqba.execute-api.us-east-1.amazonaws.com` to redraw
them from the live traces. Per-session gifs land beside it as
`<session>-agent.gif` and are not committed, because that command makes them.

## Production notes

- **Screen recording at 1440p**, the page in dark mode; it reads better
  compressed and the interval bar has more contrast.
- **No background music under the demo section.** The log scrolling is the
  evidence; let it be quiet.
- Record the demo **in one take**. A cut in the middle of a live session
  invites the question of what was cut.
- Subtitles throughout. Judges may watch muted, and the parasite counts are
  hard to hear.
- The COOL beat is the long one and it is worth the time. It is the only part
  of the video where we find a defect in someone else's shipped product, fix
  it, and prove the fix returns the same answers.
- **Who speaks:** Uriel on the pipeline and the numbers, Alex on AWS, COOL and
  the agent, matching how the work was actually split.
