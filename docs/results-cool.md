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

## What a reader should take from this

The defensible claim is narrow and it is this: *on Graviton4, the
KleidiCV-accelerated build is 1.13× faster than the same OpenCV 5.0.0 with the
acceleration compiled out, on a ten-function microbenchmark — and that figure
is carried almost entirely by `medianBlur`, which this pipeline calls once.*

Anything broader is not supported yet.

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
