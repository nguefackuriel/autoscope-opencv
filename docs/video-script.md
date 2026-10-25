# The submission video: script and shot list

**Five minutes, which is the limit.** The rules ask the video to show the team,
the working application, its architecture and the principal results. The rubric
scores presentation, but what a judge is really deciding is whether to believe
us, so everything below is chosen so a sceptical viewer can check it afterwards.

**Two checks run on this file.** `python docs/check_script_pace.py` counts the
spoken words against the five minute limit. `python docs/check_style.py` reads
the same lines against the house style in `docs/style-rules.md`, and it reads
the report and the demo page too. Both must pass before recording.

**Who speaks** is marked on each line. Uriel takes the problem, the pipeline
and the numbers. Alex Bruno takes the agent, the demo and COOL. Both of you
appear at the end, because the rules ask for the team.

**Budget: 793 spoken words**, which is five minutes at a normal pace with
nothing to spare. The first draft was 1,183 words, close to eight minutes, in
a document whose first line said three. Count before recording, not after.

**The rule for this video: never show a number that is not in the repository.**
If it is on screen it is in `docs/results-*.md` or `bench/results/`, and the
video says where.

## Shape

| | | | words |
|---|---|---|---|
| 0:00-0:30 | the problem | why a microscope, and why counting | 77 |
| 0:30-1:14 | what AutoScope is | the phone, the pipeline, the agent, the architecture | 118 |
| 1:14-2:05 | **the live demo** | the page, a real session, the audit trail | 169 |
| 2:05-2:35 | the numbers | held-out, specificity, Tanzania | 77 |
| 2:35-3:30 | what the agent is worth | eighteen policies, fifteen models | 126 |
| 3:30-4:33 | what does not work | vivax, COOL, and the fix we sent upstream | 183 |
| 4:33-5:00 | close | the team, the URL, the repository | 62 |

## 0:00-0:30, the problem (79 words)

*Shot: the WHO report page with the figures visible, then a thick film down a
microscope.*

> **URIEL.** Malaria killed 610,000 people last year, and 94 percent of the
> cases were in Africa.
>
> In 42 countries the parasite no longer produces the protein that rapid tests
> detect, so those tests are failing. Drug resistance is measured from a
> parasite count, which a rapid test cannot give.
>
> The count comes from a microscope. A hundred fields are examined under oil
> before a slide can be called negative. That takes about half an hour, and
> accuracy falls as the reader tires.

Numbers from `docs/reference/WHO-World-Malaria-Report-2025.pdf`, read from the
report rather than recalled.

## 0:30-1:14, what it is (114 words)

*Shot: `docs/proposal/architecture.svg` building up, one stage at a time.*

> **URIEL.** AutoScope uses the microscope a laboratory already owns, with a
> phone at the eyepiece. The phone uploads each field, and six OpenCV 5 stages
> run on Graviton. They correct the lighting and the stain, then locate the
> white cells and the parasites. Those counts become parasites per microlitre,
> with a confidence interval.
>
> **ALEX BRUNO.** An agent reads those numbers and selects one of five moves.
> It may refocus a soft field or skip a dirty one. It may keep reading, or
> stop. The fifth move sends the slide to a person.
>
> The call itself comes from a fixed function that reads the counts, and the
> limits on fields and time are held in Step Functions, outside the model.

## 1:14-2:05, the demo, which is the part that matters (134 words)

*Shot: screen recording of <https://d1k2o3z37c1maf.cloudfront.net>, no cuts.*

Open the page. It loads a real session by itself. Let the table fill.

> **ALEX BRUNO.** This slide was read on AWS, across eight fields on Graviton
> spot.

Scroll the log slowly. Let the viewer read the agent's own reasons. Do not
paraphrase them into a clean convergence: the interval goes inside the band,
comes back out one field later, and only settles on the eighth. That is on
screen and it is the better story.

> The log carries the reason for every move. Sixteen white cells on the first
> field, and the interval is wide, so the agent continues. Midway it falls
> inside one density band, and a later field widens it past the edge again. The
> agent stops at the eighth field, on a hundred and thirty-three cells.
>
> That stopping point was its own choice. The agent watched the interval
> narrow and stopped once the estimate settled.

Point at the interval bar.

> Eight thousand six hundred per microlitre. The grey line marks the noise
> floor measured on clean slides, and a slide is positive when the whole
> interval sits above it.

Then press "Replay a recorded session". Say what changes, because the numbers
move on screen: it is the same slide replayed from the cached field table, and
the rule reads eleven fields where the agent read eight. Then click two
detections in the verification grid and let the density move.

> The reader may disagree. These thirty-three detections scored closest to the
> threshold, so they are the least certain. Selecting one removes it, and the
> interval moves.

Last beat, and say it plainly:

> A single slide settles nothing. The fifty-patient comparison is in the
> repository.

## 2:05-2:35, the numbers (79 words)

*Shot: the targets table on page 8 of the report, the one with two MISSED
chips in it.*

> **URIEL.** Fifty patients were held out of training. Parasite F1 is 0.79.
> The limit of detection by dilution is 500 parasites per microlitre.
>
> Of fifty infected slides, forty-nine were called positive and none negative.
> Of twenty-five clean slides, none were called positive and three were passed
> to a person, a specificity of 0.88. The target we set in September was 0.90.
>
> A Tanzanian set was scored once, on a different microscope and stain.
> Image-level AUC is 0.85.

## 2:35-3:30, what the agent is actually worth (146 words)

*Shot: the arm table in section 7 of the report, page 10.*

> **ALEX BRUNO.** Eighteen stopping policies were run through the same loop
> and counter, fifteen of them models on Bedrock. Every call was recorded, so
> the comparison replays free.
>
> A fixed policy that reads eight fields does as well as the agent. The
> density error matches and fewer slides go to a person, which the report
> states.
>
> The agent allocates effort by slide. It reads three fields on the ten densest
> slides and nine or ten on the sparsest, where the fixed policy reads eight in
> both cases.
>
> Willingness to stop separated the models. Three of the fifteen chose stop
> zero times, across a thousand calls each.

### The provenance shot, 10 seconds inside the COOL beat

The demo beat films TF101_233, which ran before the build line was recorded,
so its "Read by" row says it is not recorded. Cut to
`?session=TF124_S540` for this one line and hold on that row:

> Every field in this session records the binary that read it. OpenCV 5 on
> aarch64, with carotene and KleidiCV twenty-six oh three. The Batch job wrote
> it.

## 3:30-4:33, what does not work (166 words)

> **URIEL.** Two things did not work, and both are published.
>
> On *vivax* the detector reaches F1 0.29, against 0.79 on falciparum. The
> confidence score stays high on incorrect detections, so the failure cannot be
> flagged. Where vivax circulates, a person must read the slide.
>
> **ALEX BRUNO.** On a ten-function benchmark the COOL image runs 1.59 times
> faster than our wheel. On our pipeline it ran 3.8 times slower.
>
> Both builds were then pinned to a single thread. The results agree within
> three percent, so the kernels run at the same speed. Given every core, the
> wheel gains fifteen to twenty-six times and the image gains nothing, while
> `cvtColor` gains about twenty times on both. The morphology path in the image
> uses one core of thirty-two.
>
> The workaround is forty lines. The image is divided into strips, one per
> core, and the call runs on each. That produces 532 slides an hour against the
> wheel's 547, with every count on the same twelve fields identical.
>
> The report was sent to OpenCV and assigned to two maintainers.

## 4:33-5:00, close (68 words)

*Shot: the two of us on camera, then the page, then the repository.* The rules
ask the video to show the team, and nothing else in it does.

> **URIEL.** I am Uriel Nguefack Yefou.
>
> **ALEX BRUNO.** And I am Alex Bruno Kenfack Anafack.
>
> **URIEL.** AutoScope runs at that address without a login. Every number shown
> here is in the repository, beside the code that produced it.
>
> AutoScope is a screening aid. It has never been used on a clinical sample
> and it detects falciparum only. Two of our seven targets were missed. A
> person reviews every result.

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

## Check the pace before you record

    python docs/check_script_pace.py

Five minutes is the limit. 750 words is a comfortable five minutes, 900 is a
fast one, and anything above that is a video that does not fit. The script
prints the words and the seconds each beat needs, so you can see where you are
over before you are in front of a microphone, not after.

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
