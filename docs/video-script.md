# The submission video: script and shot list

**Five minutes, which is the limit.** The rules ask the video to show the team,
the working application, its architecture and the principal results. The rubric
scores presentation, but what a judge is really deciding is whether to believe
us, so everything below is chosen so a sceptical viewer can check it afterwards.

**Budget: about 780 spoken words.** The word count of each beat is in the
table, and `python docs/check_script_pace.py` adds them up. Where it stands
today: 799 words, which is 307 seconds at 156 words a minute and 300 at 160.
So it fits a five minute video at a normal pace with nothing to spare. If the
read runs long, the seven seconds come out of the 0:30 beat, which is the most
compressible.

The first draft of this was 1,183 words, nearly eight minutes, in a document
whose first line said three. Count before you record, not after.

**The rule for this video: never show a number that is not in the repository.**
If it is on screen it is in `docs/results-*.md` or `bench/results/`, and the
video says where.

## Shape

| | | | words |
|---|---|---|---|
| 0:00-0:30 | the problem | why a microscope, and why counting | 77 |
| 0:30-1:10 | what AutoScope is | the phone, the pipeline, the agent, the architecture | 118 |
| 1:10-2:10 | **the live demo** | the page, a real session, the audit trail | 169 |
| 2:10-2:40 | the numbers | held-out, specificity, Tanzania | 77 |
| 2:40-3:30 | what the agent is worth | eighteen policies, fifteen models | 126 |
| 3:30-4:35 | what does not work | vivax, COOL, and the fix we sent upstream | 183 |
| 4:35-5:00 | close | the team, the URL, the repository | 62 |

## 0:00-0:30, the problem (75 words)

*Shot: the WHO report page with the figures visible, then a thick film down a
microscope.*

> Malaria killed 610,000 people in 2024, 94 percent of them in Africa.
>
> Rapid tests are losing ground: in 42 countries, parasites have stopped making
> the protein they look for. And drug resistance is detected by counting
> parasites. A rapid test cannot count.
>
> So the work goes back to the microscope. A hundred fields under oil before a
> slide can be called negative. Half an hour, alone, and people make more
> mistakes as they get tired.

Numbers from `docs/reference/WHO-World-Malaria-Report-2025.pdf`, read from the
report rather than recalled.

## 0:30-1:10, what it is (118 words)

*Shot: `docs/proposal/architecture.svg` building up, one stage at a time.*

> AutoScope turns the microscope they already own into a scanner. A phone
> shoots the fields through the eyepiece and uploads them. Six OpenCV 5 stages
> on Graviton4 even out the lighting and the stain, find the white cells and
> the parasites, and turn the counts into parasites per microlitre with a
> confidence interval.
>
> An agent reads those numbers and picks the next move: refocus, skip a dirty
> field, keep reading, stop, or hand the slide to a person.
>
> It sits outside the pipeline and returns one of those five moves. Step
> Functions holds the limits. The call comes from a fixed function reading the
> counts, so the agent has no move that issues a diagnosis.

## 1:10-2:10, the demo, which is the part that matters (169 words)

*Shot: screen recording of <https://d1k2o3z37c1maf.cloudfront.net>, no cuts.*

Open the page. It loads a real session by itself. Let the table fill.

> This is a real session, read on AWS. Eight fields on a Graviton Spot fleet.

Scroll the log slowly. Let the viewer read the agent's own reasons.

> Field one: thirty white cells, the interval is very wide, keep reading.
> Field four: both bounds now sit inside one density band.
> Field seven: a hundred and thirty-three white cells, the band is settled.
> Stop.
>
> It is watching an error bar narrow, and it stops when the answer stops
> moving. That is what a technician does.

Point at the interval bar.

> Eight thousand six hundred parasites per microlitre, interval seven thousand
> two fifty to ten thousand one thirty. The grey line is this pipeline's
> measured noise floor on clean slides. The call is positive when the whole
> interval sits above it.

Then press "Replay a recorded session". Say what changes, because the numbers
move on screen: it is the same slide replayed from the cached field table, and
the rule reads eleven fields where the agent read eight. Then click two
detections in the verification grid and let the density move.

> A technician can disagree. These thirty-three are the detections nearest the
> threshold, the ones it was least sure about. Reject one and the count and the
> interval recompute. A deployment would keep those corrections.

Last beat, and say it plainly:

> One slide proves nothing on its own. The comparison over all fifty held-out
> patients is in the repository.

## 2:10-2:40, the numbers (77 words)

*Shot: the targets table on page 8 of the report, the one with two MISSED
chips in it.*

> Fifty held-out patients the fit never saw. Parasite F1, 0.79. Limit of
> detection by dilution, 500 parasites per microlitre.
>
> Forty-nine of fifty infected slides called positive, none called negative. On
> twenty-five clean slides, none called positive and three sent to a person, so
> specificity is 0.88. We set ourselves 0.90 in September and we missed it.
>
> Then a Tanzanian set, different microscope, camera and stain, scored once:
> image level AUC 0.85 after a calibration on ten slides.

## 2:40-3:30, what the agent is actually worth (126 words)

*Shot: the arm table in section 7 of the report, page 10.*

> Eighteen stopping policies through the same loop and the same counter.
> Fifteen are models on Bedrock, every call recorded, so the comparison replays
> for nothing.
>
> A fixed eight fields does as well as the agent on the totals, with fewer
> slides sent to a person. No model needed, and the report says that.
>
> What the agent does instead is spend the budget unevenly: three fields on the
> ten densest slides, nine or ten on the sparsest.
>
> And what separates the models is whether they will stop. Three of the fifteen
> never chose stop once, in a thousand calls each.

### The provenance shot, 10 seconds inside the COOL beat

The demo beat films TF101_233, which ran before the build line was recorded,
so its "Read by" row says it is not recorded. Cut to
`?session=TF124_S540` for this one line and hold on that row:

> Every field this one read says which binary read it. OpenCV 5 on aarch64,
> carotene and KleidiCV twenty-six point oh three. The Batch job wrote that
> itself.

## 3:30-4:35, what does not work (183 words)

> Two things we could not make work, and we publish both.
>
> Species. On *P. vivax* the detector reads F1 0.29 against 0.79 on falciparum,
> and its confidence score cannot tell you it is failing. A slide from a vivax
> region needs a human reader.
>
> COOL. On a ten-function benchmark the Marketplace image is 1.59 times faster
> than the wheel we ship. On our pipeline it was 3.8 times slower. So we forced
> both builds to a single thread. They agree to within three
> percent, so the kernels are the same speed. Then we gave the cores back. The
> wheel takes fifteen to twenty-six times off those calls. The image takes
> none, while `cvtColor` speeds up twenty times on both. Its filter and
> morphology run on one core out of thirty-two.
>
> The fix is forty lines: cut the image into bands, one per core, and run the
> call band by band. That takes it to 532 slides an hour against the wheel's
> 547, on the same twelve fields, every count identical.
>
> We reported it to OpenCV, issue three zero three seven three.

## 4:35-5:00, close (62 words)

*Shot: the two of us on camera, then the page, then the repository.* The rules
ask the video to show the team, and nothing else in it does.

> We are Uriel Nguefack Yefou and Alex Bruno Kenfack Anafack.
>
> AutoScope is live at that address, no login. Every number you have seen is in
> the repository beside the script that produced it.
>
> It is a screening aid, not a diagnostic device. Never run on a clinical
> sample, falciparum only, and two of our seven targets missed. A person
> reviews every result.

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
