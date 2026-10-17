### The behaviour

On the Cloud Optimized OpenCV image for AWS Graviton4 from the AWS
Marketplace, large-kernel filter and morphology calls on a 12 megapixel
image take the same time on 32 threads as on 1. The same calls on the
`opencv-python` wheel from PyPI, on the same instance type, gain roughly
15 to 26 times from the threads. `cvtColor` gains about 20 times on both,
so the threading itself is working.

At one thread the two builds agree to within a few percent on every call,
so the kernels run at the same speed. The difference is entirely whether
the work is spread across the cores.

### Builds

| | Marketplace COOL image | `opencv-python` wheel |
|---|---|---|
| Version control | `4.13.0-153-g9a6e0d1bff-dirty` | `5.0.0-dirty` |
| Custom HAL | `YES (carotene (ver 0.0.1) KleidiCV (ver 0.7.0))` | `YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))` |
| Parallel framework | `TBB (ver 2022.1 interface 12150)` | `pthreads` |
| Baseline | `SVE NEON FP16 NEON_DOTPROD NEON_FP16 NEON_BF16` | `NEON FP16` |
| `cv.__version__` | 4.14.0-pre | 5.0.0 |
| OS | Ubuntu 24.04 | Amazon Linux 2023 |

Both on `c8g.8xlarge`, 32 vCPU, `us-east-1`. `cv.getNumThreads()` reports
32 on both. The COOL image is
`ami-01db31139bc5615d8`, the current Graviton4 listing. Python 3.12 from
`/opt/cool/venvs/python_3.12` with bindings from
`/opt/cool/python_3.12/site-packages` and
`LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib`.

Four things differ between these builds at once: the OpenCV version, the
KleidiCV version, the parallel backend and the operating system. We
cannot isolate which one causes this, because we cannot rebuild the
Marketplace image. What the one-thread column does rule out is a
difference in kernel speed.

### Measurements

Medians of 10 runs on one 4032x3024 image, `bench/cool_bench.py --group
pipeline --threads 0,1`. Milliseconds.

| call | wheel @32 | COOL @32 | wheel @1 | COOL @1 | wheel gain | COOL gain |
|---|---|---|---|---|---|---|
| `cvtColor` BGR2Lab | 1.67 | 1.50 | 32.38 | 32.33 | 19.4x | 21.6x |
| `GaussianBlur` float32, sigma 1.4 | 1.62 | 25.12 | 24.11 | 24.85 | 14.9x | **1.0x** |
| `morphologyEx` TOPHAT float32, ellipse 31x31 | 54.72 | 1125.82 | 1132.53 | 1133.34 | 20.7x | **1.0x** |
| `dilate` float32, ellipse 15x15 | 5.07 | 130.56 | 133.80 | 132.25 | 26.4x | **1.0x** |
| `morphologyEx` TOPHAT uint8, ellipse 31x31 | 20.56 | 339.52 | 345.51 | 343.04 | 16.8x | **1.0x** |

At one thread the two builds are within 3% of each other on
every call in that table.

### Effect on a real pipeline

Our malaria thick-film pipeline runs a 31x31 top-hat on a float32 stain
map at full resolution. The same twelve images on each build,
`bench/field_profile.py`:

| stage, seconds per field | wheel | COOL image | ratio |
|---|---|---|---|
| imread | 0.0573 | 0.0646 | 1.13x |
| assess_field | 0.0551 | 0.0587 | 1.07x |
| prepare (disc + stain) | 0.0238 | 0.0147 | 0.62x |
| white cells | 0.0701 | 0.0461 | 0.66x |
| parasite candidates | 0.0753 | 0.7177 | 9.53x |
| patches | 0.0024 | 0.0021 | 0.88x |
| patch network (cv.dnn) | 0.0500 | 0.0597 | 1.19x |
| **whole field** | **0.222** | **0.852** | **3.8x** |

3 of 7 stages are faster on the COOL image.
The rest are slower by 7% to 19%, except the candidate stage, which is
9.5x
slower and swallows every gain. That stage is the 31x31 top-hat.
547 slides an hour becomes 143.

### Workaround

Cut the image into one horizontal band per core, overlapping by the
kernel's full support (31 rows for the 31x31
top-hat, 7 for the dilate), run the same call per
band from a thread pool, and stitch. OpenCV releases the GIL inside the
call, so the bands run on separate cores. The result is checked against
the unsplit call before it is timed, and a test asserts they are exactly
equal.

| call, COOL image @32 threads | plain | banded | gain |
|---|---|---|---|
| TOPHAT float32, ellipse 31 | 1125.8 | 68.5 | 16.4x |
| `dilate` float32, ellipse 15 | 130.6 | 9.7 | 13.4x |

In the pipeline that is 532 slides an hour
against the wheel's 547. On the wheel the
same banding is slower than the plain call
(77.5 ms against 54.7), which is what you
would expect when the library is already using the cores.

### The two builds return the same answers

A speed comparison across two libraries only means something if they
agree. The whole pipeline was run twice on the same
12 images, once on each build, keeping the count of every
image: 605 detections and every white cell,
0 of 12 images differing.
559 slides an hour against 540,
so 0.97x with the banding on.

### Where the image is faster

On a 1024 px tile the image wins on several calls, which is why a
function-level benchmark does not show any of this:

| call, 1024 px | wheel | COOL | ratio |
|---|---|---|---|
| `calcHist` | 0.5135 | 0.0418 | 12.28x |
| `adaptiveThreshold` | 9.2060 | 1.1017 | 8.36x |
| `GaussianBlur` | 1.0275 | 0.6579 | 1.56x |
| `resize` | 0.1300 | 0.0909 | 1.43x |
| `findContours` | 29.5547 | 22.9916 | 1.29x |
| `medianBlur` | 0.2788 | 0.2280 | 1.22x |
| `warpPerspective` | 0.1775 | 0.5836 | 0.30x |

`warpPerspective` is the exception at 0.30x, which we have not looked
into. Note that the 1024 px group never runs a 31x31 top-hat or a 15x15
dilate: its morphology case is a 5x5 open on uint8, which is at parity
(1.01x).
That is the whole reason this took a pipeline to find.

### Reproduce

```
GROUP=pipeline THREADS=0,1 COOL_AMI=ami-01db31139bc5615d8 \
    bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool 10
GROUP=pipeline THREADS=0,1 \
    bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock 10
```
Each launches one instance, measures it, and terminates it. The banded
helper is `parasites._morph`; `bench/compare_counts.py` is the equality
check.

### The question

Is a HAL that implements `morphologyEx` and the separable filters
expected to do its own threading, with OpenCV calling it once for the
whole image? If so this is a KleidiCV or packaging question rather than
an OpenCV one, and we would like to know where to take it. If not, the
Marketplace image appears to be built in a way that loses the
parallelism on those paths. The same calls on the current PyPI wheel,
which also reports carotene and KleidiCV in its custom HAL, do use the
cores.

