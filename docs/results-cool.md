# COOL on Graviton: what we measured, including the part that does not help us

First real AWS numbers on this project, 10 October 2026. Every figure below
came off an EC2 instance that `bench/run_on_ec2.sh` launched, measured and
terminated; the result files are committed in `bench/results/` because a
benchmark whose evidence only ever existed on a terminated instance is an
assertion, not a measurement.

Method: `bench/cool_bench.py`, 100 runs per function on a 1024 px synthetic
tile, median and interquartile range, never a best-of. Same script, same
OpenCV version (5.0.0), same Python, three machines.

## Arm against x86, both with the stock wheel

| function | c8g.8xlarge (Arm) | c7i.8xlarge (x86) | |
|---|---|---|---|
| `resize` | 0.130 ms | 0.085 ms | x86 1.53× |
| `medianBlur` | 0.279 ms | 3.653 ms | **Arm 13.10×** |
| `GaussianBlur` | 1.028 ms | 0.766 ms | x86 1.34× |
| `adaptiveThreshold` | 9.206 ms | 6.550 ms | x86 1.41× |
| `morphologyEx_open` | 0.720 ms | 0.341 ms | x86 2.11× |
| `findContours` | 29.555 ms | 25.913 ms | x86 1.14× |
| `calcHist` | 0.513 ms | 0.322 ms | x86 1.60× |
| `warpPerspective` | 0.177 ms | 0.308 ms | Arm 1.73× |
| `Laplacian` | 1.619 ms | 1.022 ms | x86 1.58× |
| `cvtColor_BGR2Lab` | 0.184 ms | 0.511 ms | Arm 2.78× |
| **total** | **43.41 ms** | **39.47 ms** | **x86 1.10×** |

**x86 wins on seven of ten functions and on the total.** It has to be said
plainly, because a judge who re-runs this will find the same thing: on our
function mix, Graviton4 running the stock wheel is about 10% slower than the
x86 equivalent.

Arm's one crushing win is `medianBlur`, 13×. That is an accelerated path and
it is real. It does not rescue the total, because the two functions that
dominate — `findContours` at 30 ms and `adaptiveThreshold` at 9 ms, together
85% of the measured time — both go to x86.

## The prices, fetched from the account rather than quoted

`bench/cost_model.py --fetch`, us-east-1, 10 October 2026:

| | on demand | Spot (7-day median) |
|---|---|---|
| c8g.8xlarge | **$1.2762/h** | **$0.5115/h** |
| c7i.8xlarge | $1.4280/h | $0.5983/h |

Graviton is 10.6% cheaper per hour on demand and 14.5% cheaper on Spot.
Against 10% less throughput, **cost per unit of work is a draw** — about 2% in
Graviton's favour on demand, 6% on Spot. Within the noise of equal.

## Why this is not the COOL result

It would be easy to read the table above as "COOL does not help". It says
nothing about COOL at all. It compares **two processors**, and the COOL claim
is about **two builds on the same processor**.

The confusion is built into the packaging, and it is the single most important
thing to get right in the report: **the stock pip wheel on aarch64 already
ships KleidiCV**, which is the acceleration COOL is built around. Our own
result file records it — `"kleidicv_in_build": true` on the c8g row above. So
"COOL against the pip wheel" would compare two KleidiCV builds and measure
close to nothing, and the 13× on `medianBlur` is KleidiCV already working in
the arm we labelled "stock".

The comparison that isolates the acceleration is Graviton4 with KleidiCV
**compiled out**, which `bench/run_on_ec2.sh` builds from source
(`KLEIDICV=off`) and which refuses to report a number if the library is still
linked. It is the next section, and it is the COOL result.

The fourth arm, the Marketplace COOL AMI itself, is not measured: the
subscription had not come through. Three honest arms are worth more than a
silence, and the report says which three.

## The acceleration, isolated: KleidiCV on against off, same c8g

Same instance type, same OpenCV 5.0.0, same script. One build has the Arm
acceleration, the other has it compiled out and refuses to run if the library
is still linked.

| function | KleidiCV ON | OFF | |
|---|---|---|---|
| `medianBlur` | 0.279 ms | 5.292 ms | **18.98×** |
| `Laplacian` | 1.619 ms | 2.213 ms | 1.37× |
| `cvtColor_BGR2Lab` | 0.184 ms | 0.230 ms | 1.25× |
| `resize` | 0.130 ms | 0.146 ms | 1.13× |
| `findContours` | 29.555 ms | 30.284 ms | 1.02× |
| `adaptiveThreshold` | 9.206 ms | 8.998 ms | 0.98× |
| `calcHist` | 0.513 ms | 0.500 ms | 0.97× |
| `warpPerspective` | 0.177 ms | 0.172 ms | 0.97× |
| `morphologyEx_open` | 0.720 ms | 0.692 ms | 0.96× |
| `GaussianBlur` | 1.028 ms | 0.696 ms | **0.68× — slower** |
| **total** | **43.41 ms** | **49.22 ms** | **1.13×** |

**1.13× overall**, which sits just below COOL's published "average over 1.2×"
and is a real, reproducible number on our own function list.

Two things about it are more interesting than the headline.

**It is one function.** `medianBlur` goes from 5.29 ms to 0.28 ms and saves
5.0 ms of the 5.8 ms the acceleration saves in total. Everything else is noise
or close to it: `findContours`, which is 68% of the measured time, moves 1.02×.

**`GaussianBlur` is 32% slower with KleidiCV on.** Not a measurement artefact —
the interquartile ranges are 0.050 and 0.004 ms, nowhere near the 0.33 ms gap.
On this tile size and sigma, the accelerated path loses.

## The number does not transfer to our pipeline, and here is why

A synthetic benchmark weights every function equally. Our pipeline does not.
Counting the calls in `src/autoscope/`:

| function | calls in the pipeline | KleidiCV |
|---|---|---|
| `cvtColor` | 13 | 1.25× |
| `resize` | 11 | 1.13× |
| `GaussianBlur` | **6** | **0.68×** |
| `morphologyEx` | 6 | 0.96× |
| `findContours` | 4 | 1.02× |
| `adaptiveThreshold` | 1 | 0.98× |
| `Laplacian` | 1 | 1.37× |
| **`medianBlur`** | **1** | **18.98×** |
| `calcHist`, `warpPerspective` | 0 | — |

**The one function that carries the entire 1.13× is the one we call once**, and
the function we call six times is the one that gets slower. The two functions
we lean on hardest, `cvtColor` and `resize`, gain 1.25× and 1.13×.

So the honest reading is that the 1.13× is an artefact of the benchmark's
function mix, not a prediction for AutoScope. The real number is whole-pipeline
— `bench/field_profile.py` on actual NLM fields — and it has to be measured,
not extrapolated. Expect it to land between 1.0× and 1.1×, and expect the
`imgproc` ceiling to cap it further: the patch network through `cv.dnn` is 45%
of a field, so at best half of any imgproc gain reaches the slide.

## The whole pipeline on real fields, which is the number a laboratory buys

`bench/field_profile.py`, twelve real NLM fields from twelve different held-out
patients, shipped to the instance as a fixed bundle from S3 so every arm
profiles the same images. Graviton4, 32 threads, KleidiCV on:

| stage | seconds |
|---|---|
| `imread` | 0.057 |
| `assess_field` | 0.055 |
| prepare (disc + stain) | 0.024 |
| white cells | 0.070 |
| parasite candidates | 0.075 |
| patches | 0.002 |
| patch network (`cv.dnn`) | 0.050 |
| **per field, end to end** | **0.222** |
| **slides per hour** | **547.4** |

For scale, the same twelve fields on the development laptop read 2.104 s per
field and 57.8 slides per hour. A c8g.8xlarge is 9.5× quicker, which is what
32 Graviton4 cores against eight laptop threads buys.

**This revises the COOL ceiling upward.** On the laptop the patch network was
45% of a field, so at most 55% of the work could be accelerated. On Graviton it
is 0.050 s of 0.222 s — **22.5%** — leaving about 77% in `imgproc`. The ceiling
is higher here than the September plan assumed in either direction.

## The same pipeline with KleidiCV compiled out, and the result that decides it

| stage | KleidiCV ON | OFF | |
|---|---|---|---|
| `imread` | 0.057 s | 0.061 s | 1.07× |
| `assess_field` | 0.055 s | 0.078 s | 1.42× |
| prepare (disc + stain) | 0.024 s | 0.017 s | 0.73× |
| white cells | 0.070 s | 0.070 s | 0.99× |
| parasite candidates | 0.075 s | 0.081 s | 1.07× |
| patch network (`cv.dnn`) | 0.050 s | 0.044 s | 0.89× |
| **per field** | **0.222 s** | **0.214 s** | **0.97×** |
| **slides per hour** | 547.4 | **566.9** | |

**On the whole pipeline, KleidiCV is 3.5% slower, not faster.** The
microbenchmark promised 1.13×. The pipeline delivers 0.97×.

That is the headline COOL finding of this project and it is negative. It
should be reported as such.

### Why the two disagree, and why that was predictable

Everything needed to predict this was in the per-function table. The 1.13×
rested on `medianBlur`, which this pipeline calls once; `GaussianBlur`, which
it calls six times, is 32% slower accelerated. A benchmark that weights ten
functions equally cannot stand in for a pipeline that does not.

### Two limits on the strength of this claim

**It is not purely a KleidiCV comparison.** The ON arm is the official pip
wheel; the OFF arm is our own source build. They differ in compiler flags and
module list as well as in KleidiCV, so some of the 3.5% belongs to the build
and not to the library. Isolating it properly would mean building both arms
from source with the flag as the only difference, which is another forty
minutes of compile per arm.

**n = 12 fields, one run, no repeats.** There is no variance estimate behind
3.5%, and it is small enough that noise cannot be excluded. The defensible
statement is not "KleidiCV is 3.5% slower"; it is **"on this pipeline KleidiCV
shows no measurable speedup, and certainly not the 1.13× the microbenchmark
suggests"**.

## What a reader should take from this

Two claims, one positive and one negative, and the negative one is the more
useful.

*On a ten-function microbenchmark on Graviton4, the KleidiCV build is 1.13×
faster than the same OpenCV with the acceleration compiled out — carried almost
entirely by `medianBlur`.*

*On AutoScope's actual pipeline, reading real fields, that speedup does not
appear. 547 slides an hour with it, 567 without.*

The second is the one worth publishing. Function-level acceleration benchmarks
do not predict pipeline throughput, and we can show exactly why on our own
numbers: the accelerated function is called once, the deaccelerated one six
times. Any team choosing hardware on the strength of a function table should
see this.

The claim that is **not** available from this data is *Graviton is faster than
x86 for malaria microscopy*. On the contrary. If AutoScope runs on Graviton it
is because the cost per slide is a wash and the acceleration is real on the
calls that matter, not because the chip is quicker.

One honest limit on all of it: `cool_bench.py` times isolated calls on a
synthetic tile. The number a laboratory pays for is slides per hour on real
fields, which is `bench/field_profile.py` and needs the NLM images on the
instance. Measured on the laptop, the patch network is 45% of a field, so the
`imgproc` share any of this can accelerate is about 55% — not the 85% the
September plan assumed. Whatever speedup the control arm shows, roughly half
of it is the most the whole pipeline can inherit.

## The fourth arm: the COOL Marketplace image itself

Everything above compares builds we assembled. The product AWS and Arm actually
ship is a Marketplace AMI, and the subscription came through on 12 October with
a seven-day trial, so this is the image rather than our approximation of it.

### It is not "the wheel plus KleidiCV"

Recorded from `cv.getBuildInformation()` on the running instance, into every
result file in `bench/results/`:

| | the wheel we ship | the COOL image |
|---|---|---|
| OpenCV | `5.0.0-dirty` | `4.13.0-153-g9a6e0d1bff-dirty` (reports 4.14.0-pre) |
| Custom HAL | carotene 0.0.1 + KleidiCV **26.03** | carotene 0.0.1 + KleidiCV **0.7.0** |
| Parallel framework | pthreads | TBB 2022.1 |
| BLAS/LAPACK | none | Arm Performance Libraries 25.07.1 |
| Baseline | NEON FP16 | **SVE** NEON FP16 NEON_DOTPROD NEON_FP16 NEON_BF16 |
| Build date | 2026-07-01 | 2026-04-21 |
| OS | Amazon Linux 2023 | Ubuntu 24.04.3 |

Two things worth noticing before any timing. Carotene is in **both**, so it is
not one of the differences. And the pip wheel's KleidiCV is *newer* than the
Marketplace image's — 26.03 against 0.7.0, built ten weeks later. The COOL
image is a major OpenCV version behind and carries the older acceleration
library of the two.

Four things still differ at once, so no measurement below can be attributed to
KleidiCV alone, and this doc does not try to. A result file that said only
`kleidicv: true` would have let this section imply one library caused
everything, which is why `build_lines` now exists — and it is what caught the
carotene and version-ordering errors in the first draft of this table.

### On the microbenchmark it is the best arm by a distance

Geometric mean of the ten per-function speedups, same c8g.8xlarge:

| arm | vs the shipped wheel | vs KleidiCV compiled out |
|---|---|---|
| OpenCV 5, Arm HAL off | 0.73× | 1.00× |
| OpenCV 5 wheel (shipped) | 1.00× | 1.36× |
| **COOL image** | **1.59×** | **2.17×** |
| c7i.8xlarge, x86 wheel | 0.88× | 1.20× |

It also repairs the regression that made the stock KleidiCV arm a wash:
`GaussianBlur` was **0.68×** with the wheel's KleidiCV and is **1.06×** on the
COOL image. Since the pipeline calls `GaussianBlur` six times and `medianBlur`
once, that is the one change that should have moved throughput.

### On the pipeline it is 3.8× slower, and one call is the whole story

| | per field | slides/hour |
|---|---|---|
| OpenCV 5, Arm HAL off | 0.214 s | 566.9 |
| OpenCV 5 wheel (shipped) | 0.222 s | 547.4 |
| COOL image | **0.852 s** | **142.7** |

Six of the seven stages are faster or level on the COOL image — `prepare` 0.62×,
`white cells` 0.66×, `patches` 0.88×. One stage, `parasite candidates`, is
**9.5× slower** and swallows all of it.

The ten tile cases could not say which call did that, because they time a 5×5
kernel over `uint8` at one megapixel and `parasites.tophat` runs a 31×31 ellipse
over `float32` at twelve. So `cool_bench.py --group pipeline` times the real
calls at the real shape and dtype, and `--threads` sweeps the thread count,
because a function that is fine at 1 MP and catastrophic at 12 MP is what a
filter that stopped parallelising looks like. It was:

| call, 4032×3024 | stock @32 | COOL @32 | stock @1 | COOL @1 | stock gain | COOL gain |
|---|---|---|---|---|---|---|
| `cvtColor` BGR2Lab | 1.66 ms | 1.68 ms | 32.85 ms | 33.02 ms | 19.8× | **19.6×** |
| `GaussianBlur` f32 | 1.63 ms | 25.11 ms | 24.56 ms | 25.20 ms | 15.1× | **1.0×** |
| `morphologyEx` TOPHAT f32, ellipse 31 | 53.69 ms | 1137.37 ms | 1155.56 ms | 1139.70 ms | 21.5× | **1.0×** |
| `dilate` f32, ellipse 15 | 5.09 ms | 133.84 ms | 125.58 ms | 132.88 ms | 24.7× | **1.0×** |
| `morphologyEx` TOPHAT u8, ellipse 31 | 19.69 ms | 339.74 ms | 344.77 ms | 344.19 ms | 17.5× | **1.0×** |

Read the last two columns. **Forced to one thread the two images agree to within
6% on every call** — 0.99×, 1.03×, 0.99×, 1.06×, 1.00×. The kernels are the same
speed. The stock wheel then gets 15–25× back by spreading the work over the 32
cores; on the COOL image those four calls get **nothing**, at any thread count.

`cvtColor` is the control, and it matters: it parallelises on the COOL image
exactly as well as on the stock wheel, 19.6× against 19.8×. So TBB is working
and the threads exist. What does not happen is that the HAL-intercepted filter
and morphology paths use them. On a 1 MP tile that costs almost nothing, which
is why the microbenchmark reports 1.59× and the pipeline reports 0.26×.

### What this is worth to us, and to them

To us it decides the question: **we do not run on the COOL image as it stands.**
The shipped wheel on the same Graviton4 is 3.8× faster end to end, and 1.59× on
a function table is not a reason to take a 3.8× loss on throughput.

To the COOL team it is a bug report with a one-line reproduction, and it is the
more useful half of this work. The arm cost four failed launches to stand up
(`dnf` on an Ubuntu image, OpenCV on no default path, no AWS CLI, a `.` of a
missing file under `set -e`) and about $4 of instance time, and it is reproduced
by:

    GROUP=pipeline THREADS=0,1 COOL_AMI=<ami> bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool 10
    GROUP=pipeline THREADS=0,1                 bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock 10

One limit, stated plainly: our pipeline leans unusually hard on large-kernel
morphology, and a pipeline that did not would see the 1.59×, not the 0.26×. This
is a finding about an interaction between one image and one workload. It is not
a claim that the COOL image is slow.
