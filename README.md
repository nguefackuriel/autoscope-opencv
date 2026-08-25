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

The Tanzania set is the held-out test and we keep it that way. No threshold gets
fitted on it, no model gets trained on it, and nobody looks at its numbers until
the pipeline is frozen. A held-out set you have already peeked at is not a
held-out set, and the generalisation claim in the report depends on this.

A third option exists if we want more: the Lacuna Malaria datasets from Ghana
and Uganda, `doi:10.7910/DVN/VEADSE`, CC BY 4.0, 7.2 GB in rar archives.

## Layout

```
docs/proposal/                what we submitted, and what the data has since
                              contradicted
docs/reference/               the WHO reports our numbers come from
docs/data-notes.md            what the data actually is. Read before coding.
src/autoscope/
  fov.py                      find the round eyepiece view. Runs first.
  quality.py         stage 1  is this field worth counting
  photometry.py      stage 2  even out lighting and stain      (to write)
  registration.py    stage 3  line up views, measure overlap    (to write)
  segmentation.py    stage 4  find the cells                    (to write)
  classify.py        stage 5  ONNX model through cv.dnn         (to write)
  density.py         stage 6  count to parasites/uL with an error bar
  virtual_stage.py            the stage the agent moves around
src/autoscope/agent/          the seven tools, three policies, the loop
bench/cool_bench.py           COOL against stock, medians and IQR
eval/                         metrics and failure cases         (to write)
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
much as that line.

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
