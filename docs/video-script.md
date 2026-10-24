# The submission video: script and shot list

**Five minutes, which is the limit.** The rules ask the video to show the team,
the working application, its architecture and the principal results. The rubric
scores presentation, but what a judge is really deciding is whether to believe
us, so everything below is chosen so a sceptical viewer can check it afterwards.

**Say it, do not read it.** The lines below are written the way people talk,
with contractions and short sentences, because the first version of this script
read like a brochure. If a sentence feels stiff in your mouth, change it. The
numbers are the only part that must not move: every one of them is in a result
file.

**Who speaks** is marked on each line. Uriel takes the problem, the pipeline
and the numbers. Alex Bruno takes the agent, the demo and COOL. Both of you on
camera at the end, because the rules ask for it.

**Budget: 790 spoken words**, which is 300 seconds at a normal pace and nothing
to spare. `python docs/check_script_pace.py` counts them. The first draft was
1,183 words, nearly eight minutes, in a document whose first line said three.
Count before you record, not after.

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

> **URIEL.** Last year malaria killed 610,000 people, nearly all in Africa.
>
> The rapid tests are starting to fail. In 42 countries the parasite stopped
> making the protein they look for. And to know if a drug still works, you have
> to count parasites. A rapid test can't count.
>
> So you go back to the microscope. A hundred fields under oil before you can
> call a slide negative. Half an hour, on your own, and tired people miss
> things.

Numbers from `docs/reference/WHO-World-Malaria-Report-2025.pdf`, read from the
report rather than recalled.

## 0:30-1:14, what it is (114 words)

*Shot: `docs/proposal/architecture.svg` building up, one stage at a time.*

> **URIEL.** We didn't build a new microscope. We take the one the lab has,
> and a phone at the eyepiece.
>
> The phone sends the fields up. Six OpenCV 5 stages on Graviton even out the
> light and the stain, find the white cells and the parasites, and turn that
> into parasites per microlitre with an error bar.
>
> **ALEX BRUNO.** Then an agent looks at those numbers and picks what to do
> next. Refocus. Skip a dirty field. Keep going. Stop. Or hand the slide to a
> person.
>
> That's all it does. It never says what's wrong with the patient. The call
> comes from a fixed function reading the counts, and the limits live outside
> the model.

## 1:14-2:05, the demo, which is the part that matters (134 words)

*Shot: screen recording of <https://d1k2o3z37c1maf.cloudfront.net>, no cuts.*

Open the page. It loads a real session by itself. Let the table fill.

> **ALEX BRUNO.** A real slide, read on AWS. Eight fields on a Graviton spot
> fleet.

Scroll the log slowly. Let the viewer read the agent's own reasons. Do not
paraphrase them into a clean convergence: the interval goes inside the band,
comes back out one field later, and only settles on the eighth. That is on
screen and it is the better story.

> Watch the reasons. First field, sixteen white cells, the error bar is
> enormous, keep reading. Midway, both ends of the interval drop inside one
> band. Then a later field pushes it back out, and it keeps going. Eight
> fields, a hundred and thirty-three white cells, and it stops.
>
> Nobody told it eight. It watched the error bar close, and stopped when the
> answer stopped moving.

Point at the interval bar.

> Eight thousand six hundred per microlitre. That grey line is the noise floor
> we measured on clean slides. If the whole interval clears it, the slide is
> positive.

Then press "Replay a recorded session". Say what changes, because the numbers
move on screen: it is the same slide replayed from the cached field table, and
the rule reads eleven fields where the agent read eight. Then click two
detections in the verification grid and let the density move.

> And you can argue with it. These thirty-three are the ones it was least sure
> about. Click one you don't like. The count drops, the interval moves.

Last beat, and say it plainly:

> One slide proves nothing. The fifty-patient comparison is in the repo.

## 2:05-2:35, the numbers (79 words)

*Shot: the targets table on page 8 of the report, the one with two MISSED
chips in it.*

> **URIEL.** Fifty patients the model never saw while we built it. F1 of 0.79
> on the parasites. By dilution, we find 500 parasites per microlitre.
>
> Forty-nine of the fifty infected slides came back positive. Not one came back
> negative. Of twenty-five clean slides, none were called positive and three
> went to a person. That's 0.88 specificity. In September we said we'd hit
> 0.90. We missed.
>
> A Tanzanian set, different scope, different camera, different stain. We ran
> it once. AUC 0.85.

## 2:35-3:30, what the agent is actually worth (146 words)

*Shot: the arm table in section 7 of the report, page 10.*

> **ALEX BRUNO.** Here's the part we didn't expect.
>
> Eighteen stopping policies through the same loop, fifteen of them models on
> Bedrock. Every call is recorded, so you can replay it all for free.
>
> A dumb rule that reads eight fields does just as well. Same error, fewer
> slides pushed to a person, no model at all. We say so in the report.
>
> So what's it for? It puts the effort where it's needed. Three fields on the
> heaviest slides, nine or ten on the thin ones. The dumb rule reads eight
> either way.
>
> And what separates the models isn't size. It's whether they'll stop. Three of
> the fifteen never once chose to stop.

### The provenance shot, 10 seconds inside the COOL beat

The demo beat films TF101_233, which ran before the build line was recorded,
so its "Read by" row says it is not recorded. Cut to
`?session=TF124_S540` for this one line and hold on that row:

> Every field here records the binary that read it. OpenCV 5 on aarch64,
> carotene and KleidiCV twenty-six oh three. The job wrote that line itself.

## 3:30-4:33, what does not work (166 words)

> **URIEL.** Two things beat us, and we're publishing both.
>
> Species. On *vivax* the detector gets F1 0.29, against 0.79 on falciparum.
> And its confidence doesn't drop when it's wrong, so we can't even warn you.
> Where vivax circulates, a person has to read the slide.
>
> **ALEX BRUNO.** COOL. On ten functions the Marketplace image is 1.59 times
> faster than our wheel. On our pipeline it came out 3.8 times slower. That
> took us a while to believe.
>
> So we pinned both builds to one thread. Within three percent of each other.
> Same kernels, same speed. Then we gave the cores back.
> The wheel gets fifteen to twenty-six times faster. The image gets nothing,
> while `cvtColor` speeds up twenty times on both. Its morphology is running on
> one core out of thirty-two.
>
> The fix was forty lines. Cut the image into strips, one per core. That gives
> 532 slides an hour against the wheel's 547, same twelve fields, every count
> identical.
>
> We sent it to OpenCV. It's assigned to two of their maintainers.

## 4:33-5:00, close (68 words)

*Shot: the two of us on camera, then the page, then the repository.* The rules
ask the video to show the team, and nothing else in it does.

> **URIEL.** I'm Uriel Nguefack Yefou.
>
> **ALEX BRUNO.** And I'm Alex Bruno Kenfack Anafack.
>
> **URIEL.** AutoScope is live at that address. No login. Every number you have
> seen is in the repository, next to the code that made it.
>
> It's a screening aid, not a diagnostic. It's never touched a clinical sample,
> it only knows falciparum, and of our seven targets we missed two. A person
> reads every result.

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
