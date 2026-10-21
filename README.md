# AutoScope

Agent-driven malaria slide reading. OpenCV 5 does the image work on AWS
Graviton, and an agent decides which field to look at next and when to stop.

The container runs the OpenCV 5 pip wheel, which on aarch64 is built with
carotene and KleidiCV 26.03: Arm-accelerated OpenCV on Graviton4. The COOL
Marketplace image is measured as a fourth arm, and with the tiled-morphology
workaround it reads the same counts at 0.97x the wheel's speed. Section 8 of
the report has all four builds with the `cv.getBuildInformation()` line of each.

Entry for the OpenCV AI Competition 2026. Both award paths: Best Use of COOL and
Agentic Vision.

## If you are judging this

| | |
|---|---|
| **Live demo** | <https://d1k2o3z37c1maf.cloudfront.net>, public, no credentials. It opens on a session read live on AWS and plays every field the agent looked at, with the detections drawn and the move it made on each. Four sessions are linked, 3 to 11 fields depending on the slide. |
| **Technical report** | `docs/report/AutoScope-Technical-Report.pdf`, committed. Rebuild it with `.venv/bin/python docs/report/build_report.py --build-pdf`. Every figure in it is read from a committed result file. None is typed by hand. |
| **Architecture** | `docs/proposal/architecture.svg` |
| **Reported upstream** | [opencv/opencv#30373](https://github.com/opencv/opencv/issues/30373): the Marketplace COOL image runs its filter and morphology HAL paths on one core at 12 MP. Triaged by an OpenCV maintainer and assigned to the HAL maintainers. Diagnosed to the call, worked around in forty lines, and `bench/repro_hal_threads.py` reproduces it in one file with no AWS. |
| **Running it on AWS** | `docs/operations.md`: what is deployed, what is logged, every spend limit and where it is, and how to tear it down. |
| **Agent trace** | `docs/figures/agent-trace-example.txt`, and every move of every slide in `eval/results/agent_eval.json` |

Three claims and the one command that checks each. None of them needs an AWS
account except the benchmark:

```
make setup && make test                       # 152 tests, x86 and arm64 in CI
.venv/bin/python eval/agent_eval.py --policies all   # every arm in the report's
                                              # table, replayed from 15 396 recorded
                                              # model calls across fifteen models: free
SKIP_TRAIN=1 bash eval/chain.sh               # every held-out and specificity number (~2 h)
bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock   # one COOL arm; the instance terminates itself
```

**What we are not claiming.** AutoScope is a screening aid, not a diagnostic
device, and has never been run on a clinical sample. The stage is virtual: the
agent's moves select among fields that were already captured, so the comparison
against the rules is sound but "this drives a microscope" is not supported.
Falciparum only: on *P. vivax* the detector reads F1 0.29 and its confidence
score cannot tell you it is failing. Two of the seven targets we set ourselves
in September were missed, and section 6 of the report says which and why.

## Setup

Needs Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

```
make setup     # venv + pinned dependencies
make test      # unit tests
make bench     # time the OpenCV calls on this machine
make chain     # retrain the parasite network and re-measure every number (about 7 h)
               # SKIP_TRAIN=1 MODEL=yolo WEIGHTS=<best.pt> PY=../.venv-detect/bin/python bash eval/chain.sh
               #   measures a box detector trained elsewhere (docs/detectors.md)
```

Dependencies are pinned in `requirements.txt`. The competition asks for pinned
dependencies in the final submission, so we pin from the start and bump on
purpose, never by accident.

`make setup` installs the package editable, which is what normally puts `src/`
on `sys.path`. Remote runners that execute a code snapshot without installing
it -- a remote runner does this -- have no such install, so every entry point in
`eval/`, `train/` and `bench/` carries a short `--- repo paths` preamble that
derives the paths from `__file__` alone. It costs nothing when the package is
installed, and it is the reason those scripts run anywhere. If you add a script
that imports `autoscope` or a sibling module, copy the preamble into it.

## Data

Two sets. We fit on the first and only ever test on the second.

```
make data          # NLM thick smears, 3.9 GB, what we develop against
make data-tanzania # Tanzania, 3.0 GB, held out for the generalisation test
```

**NLM thick blood smears** (`data/download.py`). 150 patients, 1883 images,
de-identified, labelled by expert readers at the Mahidol-Oxford Tropical
Medicine Research Unit. 84 509 parasite and 35 449 white cell annotations. The
images were taken with a phone held to a microscope eyepiece, which is how
AutoScope captures too.

**Tanzania** (`data/download_tanzania.py`). Collected in five health centres of
the Tanga region on a 40X-2500X compound microscope with a 4K Sony IMX334
camera. CC0, so no restrictions at all. Different country, different rig,
different camera. Use `--thick-only` to skip 1.2 GB of thin smears.

Nothing in `data/` is committed. Both scripts print the real folder layout after
extracting; read that output before writing anything that touches the files.
What each set actually contains, and what that allows us to claim, is in
`docs/data-notes.md`. Short version: NLM has per-object annotations, Tanzania
has only an infected/uninfected label per image, so per-parasite precision and
recall are measured on NLM and Tanzania answers a different question.

The Tanzania set is the held-out test and we keep it that way. No threshold gets
fitted on it, no model gets trained on it, and nobody looks at its numbers until
the pipeline is frozen. A held-out set you have already peeked at is not a
held-out set, and the generalisation claim in the report depends on this.

Two more NLM sets, same group, fetched by `data/download_nlm_extra.py`:

```
make data-uninfected  # thick smears from uninfected patients, 2.5 GB
make data-vivax       # thick smears from 150 P. vivax patients, 5.0 GB
```

Every patient in the 150-patient set is infected, so nothing measured so far
says what the pipeline does on a clean slide. The uninfected set is where
specificity comes from; the P. vivax set is the other-species failure case.
Both are frozen test material like Tanzania: run once, at the end, nothing
fitted on them.

A third option exists if we want more: the Lacuna Malaria datasets from Ghana
and Uganda, `doi:10.7910/DVN/VEADSE`, CC BY 4.0, 7.2 GB in rar archives.

## Layout

```
docs/proposal/                what we submitted, and what the data has since
                              contradicted
docs/reference/               the WHO reports our numbers come from
docs/data-notes.md            what the data actually is. Read before coding.
docs/plan-2026-09-02.md       where we stand, what is wrong, the plan to 26 Oct
src/autoscope/
  fov.py                      find the round eyepiece view. Runs first.
  quality.py         stage 1  is this field worth counting, and frame_motion
                              for whether the stage has settled between views
  photometry.py      stage 2  even out lighting and stain
  annotations.py              read the ground truth, in the image's own frame
  segmentation.py    stage 4  find the white cells (the denominator)
  parasites.py       stage 5  find the parasites (the numerator): top-hat
                              peaks, then an ONNX classifier through cv.dnn
  site.py                     calibration for another microscope: the scale
                              from the nuclei of ten slides (results-tanzania.md)
  density.py         stage 6  count to parasites/uL with an error bar, and the
                              decision rule: negative on the noise floor, positive
                              when the interval clears it, settled in one band
  virtual_stage.py            the stage the agent moves around
  boxes.py                    boxes from the stain for training, and a box
                              detector (YOLO26, Faster R-CNN) run tile by tile
src/autoscope/agent/          the seven tools, five policies, the loop
                              policy.py   fixed, WHO-200, rule-based, decision
                              bedrock.py  the agent: Claude on Bedrock, the five
                                          moves as tools, behind the same Policy
                              moves.py    the prompt and the moves, no OpenCV, so
                                          the Lambda ships 300 kB not 60 MB
                              replay.py   every model call the experiment made,
                                          on disk: recorded once, replayed for
                                          ever, so the agent comparison costs
                                          nothing and needs no AWS account
src/autoscope/cloud/          what runs on AWS: read_field.py is one Batch job,
                              one field, stages 1 to 4 and 6 on Graviton
models/                       the ONNX classifiers cv.dnn loads (committed;
                              `make train-mlp` / `make train-cnn` rebuild them;
                              parasite_cnn_chain2_32px.onnx is the 32 px control)
bench/cool_bench.py           COOL against stock, medians and IQR
bench/field_profile.py        the real pipeline, stage by stage, on real fields,
                              and slides/hour
bench/cost_model.py           dollars per 1,000 slides, from measured field time
                              and real prices. Refuses to invent a price.
bench/Dockerfile.kleidicv-off the honest Arm control: OpenCV 5 built from source
                              with KleidiCV off, since the pip wheel has it on
eval/                         surveys, sweeps, and held-out scoring. Every
                              number in docs/results-*.md comes from here.
infra/terraform/              the stack of proposal section 6: S3, EventBridge,
                              Step Functions, Batch on Graviton Spot, Bedrock,
                              DynamoDB, API Gateway, CloudFront, Budgets
infra/lambda/                 the agent and the API, and build.sh which makes
                              their packages. Read docs/aws-setup.md first.
```

## Running the benchmark

Same script, three machines, then diff the JSON:

```
python bench/cool_bench.py --runs 100 --label c8g-cool          # Graviton4 + COOL AMI
python bench/cool_bench.py --runs 100 --label c8g-stock         # Graviton4, the pip wheel
python bench/cool_bench.py --runs 100 --label c8g-no-kleidicv   # Graviton4, KleidiCV off
python bench/cool_bench.py --runs 100 --label c7i-stock         # x86, the pip wheel
```

Four arms, not three, and the reason is in the next paragraph. The third is
built by `bench/Dockerfile.kleidicv-off`, which fails the build if the
acceleration is still linked: a control that quietly has KleidiCV in it would
report a 1.0x speedup as a finding.

Then the two numbers a lab reads, on each machine:

```
python bench/field_profile.py --label c8g-cool    # slides/hour, stage by stage
python bench/cost_model.py --fetch --markdown     # $/1000 slides, all arms
```

Every result file records `kleidicv_in_build` and the CPU baseline, so a reader
can tell which build produced the numbers. The COOL claim is worth exactly as
much as that line. Mind that the pinned pip wheel on aarch64 already ships KleidiCV
(CI prints `Custom HAL: KleidiCV (ver 26.03)`), so "stock" on Arm means a
build with KleidiCV switched off, not the wheel.

Note for local work: both our laptops are x86, so nothing here can time Arm.
Correctness on Arm can be checked with `docker buildx --platform linux/arm64`,
but the timings only mean something on a real `c8g`.

## Honesty rules we are holding ourselves to

- `refocus()` on the Virtual Stage is only real when the data has more than one
  capture of the same field. Otherwise the frame is marked `synthetic=True` and
  any published number has to say so.
- No best-of timings. Median and interquartile range over at least 100 runs.
- Thresholds in `quality.py` are starting values. Re-fit them in `eval/` against
  the data before the final report.
- AutoScope is a screening aid for a trained technician, not a diagnostic
  device. Every result stays checkable and overridable by a person.
