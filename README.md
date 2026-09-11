# AutoScope

Agent-driven malaria slide reading. OpenCV 5 does the image work on AWS Graviton
with COOL, and an agent decides which field to look at next and when to stop.

Entry for the OpenCV AI Competition 2026. Both award paths: Best Use of COOL and
Agentic Vision.

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
  quality.py         stage 1  is this field worth counting
  photometry.py      stage 2  even out lighting and stain
  registration.py    stage 3  line up views, measure overlap    (to write)
  annotations.py              read the ground truth, in the image's own frame
  segmentation.py    stage 4  find the white cells (the denominator)
  parasites.py       stage 5  find the parasites (the numerator): top-hat
                              peaks, then an ONNX classifier through cv.dnn
  density.py         stage 6  count to parasites/uL with an error bar, and the
                              decision rule: negative on the noise floor, positive
                              when the interval clears it, settled in one band
  virtual_stage.py            the stage the agent moves around
  boxes.py                    boxes from the stain for training, and a box
                              detector (YOLO26, Faster R-CNN) run tile by tile
src/autoscope/agent/          the seven tools, four policies, the loop
models/                       the ONNX classifiers cv.dnn loads (committed;
                              `make train-mlp` / `make train-cnn` rebuild them;
                              parasite_cnn_chain2_32px.onnx is the 32 px control)
bench/cool_bench.py           COOL against stock, medians and IQR
bench/field_profile.py        the real pipeline, stage by stage, on real fields
eval/                         surveys, sweeps, and held-out scoring. Every
                              number in docs/results-*.md comes from here.
infra/                        Terraform for AWS                 (to write)
```

## Running the benchmark

Same script, three machines, then diff the JSON:

```
python bench/cool_bench.py --runs 100 --label c8g-cool     # Graviton4 + COOL AMI
python bench/cool_bench.py --runs 100 --label c8g-stock    # Graviton4, stock OpenCV 5
python bench/cool_bench.py --runs 100 --label c7i-stock    # x86, stock OpenCV 5
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
