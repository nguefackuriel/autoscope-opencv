# Draft: issue for the COOL / KleidiCV team

Not posted. Posting it is a public action in the team's name and is Uriel's call.
Where to post: the KleidiCV tracker (Arm) and the AWS Marketplace listing's
support contact for "Cloud Optimized OpenCV for AWS Graviton4". Every number
below is in `bench/results/` with the `cv.getBuildInformation()` lines of the
binary that produced it.

---

**Title:** Filter and morphology calls run on one core on the Graviton4 COOL
image at 12 MP; a band-parallel workaround recovers 16x

**Image:** Cloud Optimized OpenCV for AWS Graviton4, `ami-01db31139bc5615d8`,
`us-east-1`, 16 October 2026.
OpenCV `4.13.0-153-g9a6e0d1bff-dirty` (reports 4.14.0-pre), Custom HAL
`carotene 0.0.1 + KleidiCV 0.7.0`, Parallel framework `TBB 2022.1`, Baseline
`SVE NEON FP16 NEON_DOTPROD NEON_FP16 NEON_BF16`, Ubuntu 24.04.3.
Python 3.12 from `/opt/cool/venvs/python_3.12`, bindings from
`/opt/cool/python_3.12/site-packages`, `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib`.

**Instance:** `c8g.8xlarge`, 32 vCPU. `cv.getNumThreads()` reports 32.

**Control:** the same instance type with `opencv-python 5.0.0.93` from PyPI
(`5.0.0-dirty`, carotene 0.0.1 + KleidiCV 26.03, pthreads, Baseline NEON FP16)
on Amazon Linux 2023.

## What we see

Large-kernel filter and morphology calls on a 4032x3024 image take the same
time at 32 threads as at 1 on the COOL image. The pip wheel gains 15 to 25x
from its threads on the same calls. `cvtColor` parallelises on both, so TBB is
working. Medians of 10 runs, milliseconds:

| call, 4032x3024 | wheel @32 | COOL @32 | wheel @1 | COOL @1 | wheel gain | COOL gain |
|---|---|---|---|---|---|---|
| `cvtColor` BGR2Lab | 1.66 | 1.68 | 32.85 | 33.02 | 19.8x | 19.6x |
| `GaussianBlur` float32, sigma 1.4 | 1.63 | 25.11 | 24.56 | 25.20 | 15.1x | 1.0x |
| `morphologyEx` TOPHAT float32, ellipse 31x31 | 53.69 | 1137.37 | 1155.56 | 1139.70 | 21.5x | 1.0x |
| `dilate` float32, ellipse 15x15 | 5.09 | 133.84 | 125.58 | 132.88 | 24.7x | 1.0x |
| `morphologyEx` TOPHAT uint8, ellipse 31x31 | 19.69 | 339.74 | 344.77 | 344.19 | 17.5x | 1.0x |

At one thread the two builds agree within 6% on every call, so the kernels run
at the same speed. Only the threading differs. On a 1 MP tile the same calls
are at parity or faster on the COOL image, which is why a function benchmark
does not show this.

## Effect on a real pipeline

Our malaria thick-film pipeline runs a 31x31 top-hat on a float32 stain map at
full resolution. On twelve real fields:

| | wheel | COOL image |
|---|---|---|
| parasite candidates stage | 0.075 s | 0.718 s |
| per field | 0.222 s | 0.852 s |
| slides per hour | 547 | 143 |

Six of seven stages are faster or level on the COOL image. This one stage is
9.5x slower and dominates.

## Workaround

Cut the image into one horizontal band per core, overlapping by the kernel's
full support (the radius times the number of chained passes: 30 rows for a
31x31 top-hat, 7 for a 15x15 dilate), run the same call per band from a thread
pool, and stitch. OpenCV releases the GIL inside the call, so the bands run on
separate cores. The result is identical to the whole-image call.

| call | COOL plain | COOL tiled | gain |
|---|---|---|---|
| TOPHAT float32, ellipse 31 | 1125.8 ms | 68.5 ms | 16.4x |
| `dilate` float32, ellipse 15 | 130.6 ms | 9.7 ms | 13.4x |

With the workaround in the pipeline: 0.229 s per field, 532 slides per hour,
0.97x the wheel. The remaining gap is thread-pool overhead against the wheel's
native `parallel_for_`. On the wheel itself the workaround is slower (77.5 ms
against 54.7), as expected.

## The two builds give the same answers

A speed comparison across libraries is only about our pipeline if the libraries
agree. We ran the whole pipeline twice on the same twelve fields, once on each
build, keeping the parasite and white cell count of every field:

| | pip wheel | COOL image, tiled |
|---|---|---|
| OpenCV | 5.0.0 | 4.14.0-pre |
| seconds per field | 0.218 | 0.225 |
| slides per hour | 559 | 540 |
| fields counted differently | — | 0 of 12 |

605 parasites and every white cell, identical. So the numbers above are a
measurement of the library, not of a difference in what the two builds detect.

## Reproduce

```
GROUP=pipeline THREADS=0,1 COOL_AMI=ami-01db31139bc5615d8 \
    bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool 10
GROUP=pipeline THREADS=0,1 \
    bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock 10

PROFILE=1 bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock-counts
PROFILE=1 TILED_MORPH=1 COOL_AMI=ami-01db31139bc5615d8 \
    bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool-counts
python bench/compare_counts.py profile-c8g-stock-counts profile-c8g-cool-counts
```

Repository: https://github.com/nguefackuriel/autoscope-opencv (private during
the OpenCV AI Competition 2026 judging; access on request). The benchmark is
`bench/cool_bench.py --group pipeline --threads 0,1`; the tiled helper is
`parasites._morph`.

## Question

Is the HAL path for large-kernel morphology and separable filters expected to
be single-threaded on this image, or is this a build configuration? The
function-level speedups the image delivers elsewhere (medianBlur 23x,
calcHist 12x, adaptiveThreshold 8x against the wheel) are real and we would
like to use them.
