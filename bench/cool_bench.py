"""Time the OpenCV calls AutoScope leans on.

Run the same script three ways and diff the JSON:

    c8g  + COOL AMI     the thing we are claiming
    c8g  stock OpenCV 5 the honest control
    c7i  stock OpenCV 5 the x86 comparison, so the claim can lose

We report the median and the interquartile range over N runs. Never the best
run: a best-of number is not reproducible and a judge is right to distrust it.
"""

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def _tile(size: int, seed: int = 0) -> np.ndarray:
    """A stand-in for a smear tile: noisy, textured, blob-ridden."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 210, (size, size, 3), dtype=np.uint8)
    for _ in range(size // 12):
        cx, cy = rng.integers(0, size, 2)
        cv.circle(base, (int(cx), int(cy)), int(rng.integers(4, 14)),
                  (int(rng.integers(20, 80)),) * 3, -1)
    return cv.GaussianBlur(base, (3, 3), 0)


def _cases(img: np.ndarray) -> dict:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    h, w = gray.shape
    homography = np.array(
        [[1.0, 0.04, 12.0], [0.02, 1.0, -8.0], [1e-5, 1e-5, 1.0]], dtype=np.float64
    )
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 8
    )
    return {
        "resize": lambda: cv.resize(img, (w // 2, h // 2), interpolation=cv.INTER_AREA),
        "medianBlur": lambda: cv.medianBlur(img, 5),
        "GaussianBlur": lambda: cv.GaussianBlur(img, (0, 0), 3.0),
        "adaptiveThreshold": lambda: cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 8
        ),
        "morphologyEx_open": lambda: cv.morphologyEx(binary, cv.MORPH_OPEN, kernel),
        "findContours": lambda: cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        ),
        "calcHist": lambda: cv.calcHist([gray], [0], None, [256], [0, 256]),
        "warpPerspective": lambda: cv.warpPerspective(img, homography, (w, h)),
        "Laplacian": lambda: cv.Laplacian(gray, cv.CV_64F),
        "cvtColor_BGR2Lab": lambda: cv.cvtColor(img, cv.COLOR_BGR2Lab),
    }


def _pipeline_cases(size: tuple[int, int] = (3024, 4032)) -> dict:
    """The calls the candidate stage actually makes, at its real shape and dtype.

    The ten cases above are a 1024x1024 uint8 tile, which is what a
    microbenchmark looks like and not what this pipeline does. The whole-pipeline
    profile on the COOL image came back with `parasite candidates` 9.5x slower
    than on the stock wheel while six of the other seven stages were faster, and
    the tile cases could not say which call did it: they time morphology on a
    5x5 kernel over uint8, and `parasites.tophat` runs a 31x31 ellipse over
    float32 at twelve megapixels.

    So these are the real ones, lifted from parasites.tophat and parasites.peaks
    with the shipped constants. A regression that shows up here is attributable
    to a named OpenCV call; one that shows up only in the stage timing is not.
    """
    rng = np.random.default_rng(0)
    h, w = size
    bgr = rng.integers(60, 220, (h, w, 3), dtype=np.uint8)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    stain = cv.GaussianBlur(lab[:, :, 1] - lab[:, :, 2], (0, 0), 1.4)
    k31 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))      # TOPHAT_PX
    k15 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))      # 2*PEAK_NMS_PX+1
    top = cv.morphologyEx(stain, cv.MORPH_TOPHAT, k31)
    # The workaround hypothesis. On the COOL image the HAL-intercepted filter
    # and morphology paths run on one core whatever cv.setNumThreads says, but
    # at one thread the COOL kernels are as fast as the wheel's. So split the
    # image into horizontal bands, overlapping by the kernel radius so no
    # band sees a cut edge, and run the same call per band from a thread
    # pool: OpenCV releases the GIL inside the C++ call, so the bands run on
    # separate cores. If this recovers the 15 to 25x the wheel gets from its
    # own threading, the COOL image wins the pipeline instead of losing it.
    # The halo is the kernel radius TIMES the number of chained passes. A
    # top-hat is src - dilate(erode(src)): the erode needs 15 rows of context
    # and the dilate then needs 15 rows of correctly ERODED context, so 30. A
    # halo of 15 passed the eye and failed the check below by 13.8 on a
    # float32 stain map. A plain dilate is one pass, so 7.
    tiled_tophat = lambda: _tiled(stain, lambda a: cv.morphologyEx(a, cv.MORPH_TOPHAT, k31), 30)
    tiled_dilate = lambda: _tiled(top, lambda a: cv.dilate(a, k15), 7)
    # Correctness, once, loudly: a tiled result that differs from the plain
    # one is a wrong answer delivered faster, which is worse than slow.
    for label, tiled, plain in (
        ("tophat", tiled_tophat, lambda: cv.morphologyEx(stain, cv.MORPH_TOPHAT, k31)),
        ("dilate", tiled_dilate, lambda: cv.dilate(top, k15)),
    ):
        err = float(np.abs(tiled() - plain()).max())
        print(f"tiled {label} vs plain: max abs diff {err:.3g}"
              + ("   OK" if err < 1e-3 else "   MISMATCH, the halo is wrong"))
        if err >= 1e-3:
            raise SystemExit("tiled workaround is incorrect; not timing a wrong answer")
    return {
        "12MP_cvtColor_Lab": lambda: cv.cvtColor(bgr, cv.COLOR_BGR2Lab),
        "12MP_GaussianBlur_f32": lambda: cv.GaussianBlur(stain, (0, 0), 1.4),
        "12MP_tophat_f32_e31": lambda: cv.morphologyEx(stain, cv.MORPH_TOPHAT, k31),
        "12MP_tophat_f32_e31_tiled": tiled_tophat,
        "12MP_dilate_f32_e15": lambda: cv.dilate(top, k15),
        "12MP_dilate_f32_e15_tiled": tiled_dilate,
        "12MP_tophat_u8_e31": lambda: cv.morphologyEx(
            cv.cvtColor(bgr, cv.COLOR_BGR2GRAY), cv.MORPH_TOPHAT, k31
        ),
    }


def _tiled(img: np.ndarray, fn, halo: int, bands: int | None = None) -> np.ndarray:
    """Run `fn` on horizontal bands of `img` concurrently and stitch the result.

    `halo` is the kernel radius: each band is cut `halo` rows wider on both
    sides so the morphology inside the kept rows sees the same neighbourhood
    it would in the whole image, then the halo is trimmed. One band per core
    by default. A thread pool, not processes: OpenCV drops the GIL in the
    call, and the bands are views into one array, so nothing is copied in.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = bands or max(os.cpu_count() or 1, 1)
    h = img.shape[0]
    edges = [round(i * h / n) for i in range(n + 1)]
    out = np.empty_like(img)

    def one(i: int) -> None:
        y0, y1 = edges[i], edges[i + 1]
        a, b = max(y0 - halo, 0), min(y1 + halo, h)
        res = fn(img[a:b])
        out[y0:y1] = res[y0 - a:y0 - a + (y1 - y0)]

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(one, range(n)))
    return out


def time_case(fn, runs: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    q1 = statistics.median(samples[: len(samples) // 2])
    q3 = statistics.median(samples[(len(samples) + 1) // 2 :])
    return {
        "median_ms": round(statistics.median(samples), 4),
        "iqr_ms": round(q3 - q1, 4),
        "min_ms": round(samples[0], 4),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--label", default="local", help="e.g. c8g-cool, c8g-stock, c7i-stock")
    ap.add_argument("--threads", default="0",
                    help="comma-separated thread counts to time each case under; "
                         "0 means leave OpenCV's default. The 12 MP cases came "
                         "back 14-25x slower on the COOL image while the 1 MP "
                         "tile cases were at parity, which is what a filter that "
                         "stopped parallelising looks like -- so the thread count "
                         "is a measured variable here, not a fixed condition.")
    ap.add_argument("--group", default="tile", choices=("tile", "pipeline"),
                    help="tile: the 1024px synthetic cases; pipeline: the real "
                         "candidate-stage calls at 12 MP (fewer runs, much slower)")
    args = ap.parse_args()

    img = _tile(args.tile)
    build = cv.getBuildInformation()
    report = {
        "label": args.label,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opencv_version": cv.__version__,
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "tile_px": args.tile,
        # Proof of which build actually ran. The COOL claim is only as good as
        # this line, so it goes in every result file.
        "kleidicv_in_build": "kleidicv" in build.lower(),
        "cpu_baseline": next(
            (l.strip() for l in build.splitlines() if "CPU/HW features" in l), "unknown"
        ),
        # The COOL image turned out to differ from the pip wheel in more than
        # KleidiCV: it is OpenCV 4.14 with carotene beside KleidiCV 0.7.0, TBB
        # instead of pthreads, Arm Performance Libraries for BLAS and SVE in
        # the baseline. A single "kleidicv: yes" would let the report imply the
        # speedup came from one library, so every line that differs is recorded
        # and the report quotes them.
        "build_lines": {
            key: next(
                (l.split(":", 1)[1].strip() for l in build.splitlines()
                 if l.strip().startswith(key)), "unknown"
            )
            for key in ("Version control", "Baseline", "Custom HAL",
                        "Parallel framework", "Timestamp")
        },
        "results": {},
    }

    cases = _cases(img) if args.group == "tile" else _pipeline_cases()
    report["group"] = args.group
    report["cv_threads_default"] = cv.getNumThreads()
    report["cpu_count"] = os.cpu_count()
    wanted = [int(t) for t in args.threads.split(",")]
    for n_threads in wanted:
        if n_threads:
            cv.setNumThreads(n_threads)
        else:
            cv.setNumThreads(report["cv_threads_default"])
        bucket = report["results"] if len(wanted) == 1 else \
            report.setdefault("by_threads", {}).setdefault(str(n_threads), {})
        label = "" if len(wanted) == 1 else f"  [{cv.getNumThreads()} threads]"
        for name, fn in cases.items():
            bucket[name] = time_case(fn, args.runs)
            r = bucket[name]
            print(f"{name:22} {r['median_ms']:8.3f} ms   IQR {r['iqr_ms']:6.3f}{label}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{args.label}.json" if args.group == "tile"
                     else f"{args.label}-pipeline.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\narch {report['machine']} | opencv {report['opencv_version']} | "
          f"kleidicv in build: {report['kleidicv_in_build']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
