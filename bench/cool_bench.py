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
        "results": {},
    }

    for name, fn in _cases(img).items():
        report["results"][name] = time_case(fn, args.runs)
        r = report["results"][name]
        print(f"{name:22} {r['median_ms']:8.3f} ms   IQR {r['iqr_ms']:6.3f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\narch {report['machine']} | opencv {report['opencv_version']} | "
          f"kleidicv in build: {report['kleidicv_in_build']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
