"""Standalone reproducer: does this build use its cores on a big morphology?

    python bench/repro_hal_threads.py

No repository, no data, no AWS. One file, numpy and OpenCV only. Run it on the
build you are testing and on a stock wheel on the same machine.

It times the same calls at all threads and at one. A build that parallelises
them reports a speedup near the core count. A build that does not reports 1.0x
on the morphology and filter rows while cvtColor still speeds up, which is the
shape of the defect in opencv/opencv#30373.
"""

from __future__ import annotations

import time

import cv2 as cv
import numpy as np

MP12 = (3024, 4032)          # the size a phone camera gives
RUNS = 7


def median_ms(fn) -> float:
    fn()                     # warm up: first-call allocations
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 255, (*MP12, 3), dtype=np.uint8)
    f32 = rng.random(MP12, dtype=np.float32)
    u8 = rng.integers(0, 255, MP12, dtype=np.uint8)
    e31 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    e15 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    cases = {
        "cvtColor BGR2Lab        ": lambda: cv.cvtColor(bgr, cv.COLOR_BGR2Lab),
        "GaussianBlur f32 s=1.4  ": lambda: cv.GaussianBlur(f32, (0, 0), 1.4),
        "morphologyEx TOPHAT f32 ": lambda: cv.morphologyEx(f32, cv.MORPH_TOPHAT, e31),
        "dilate f32 ellipse 15   ": lambda: cv.dilate(f32, e15),
        "morphologyEx TOPHAT u8  ": lambda: cv.morphologyEx(u8, cv.MORPH_TOPHAT, e31),
    }

    info = cv.getBuildInformation()

    def line(label: str) -> str:
        for row in info.splitlines():
            head, _, value = row.partition(":")
            if head.strip().lower() == label:
                return value.strip()
        return "?"

    print(f"OpenCV {cv.__version__}")
    print(f"  Custom HAL         {line('custom hal')}")
    print(f"  Parallel framework {line('parallel framework')}")
    print(f"  CPU baseline       {line('cpu/hw features')}")
    default_threads = cv.getNumThreads()
    print(f"  cv.getNumThreads() {default_threads}, {MP12[1]}x{MP12[0]} image, "
          f"median of {RUNS}\n")

    print(f"{'call':<26}{'all threads':>13}{'1 thread':>11}{'speedup':>10}")
    gains = {}
    for name, fn in cases.items():
        cv.setNumThreads(0)
        many = median_ms(fn)
        cv.setNumThreads(1)
        one = median_ms(fn)
        gains[name.strip()] = one / many if many else 0.0
        print(f"{name:<26}{many:>11.1f}ms{one:>9.1f}ms{gains[name.strip()]:>9.1f}x")
    cv.setNumThreads(0)

    # cvtColor is the control, and it has to be read first. If IT does not speed
    # up, setNumThreads is not reaching this build and nothing below means
    # anything: that is the case on macOS with the GCD backend, where the
    # thread count is not ours to set. Without this check the script cries
    # wolf on any such machine.
    control = gains["cvtColor BGR2Lab"]
    rest = {k: v for k, v in gains.items() if k != "cvtColor BGR2Lab"}
    worst = min(rest.values())

    print()
    if default_threads < 4:
        print(f"Only {default_threads} threads here. Run this on a machine with "
              f"more cores; the effect is invisible on a few.")
    elif control < 2:
        print(f"cvtColor gains only {control:.1f}x, so setNumThreads is not "
              f"changing anything on this build")
        print(f"({line('parallel framework')}). Nothing above is a measurement "
              f"of the HAL. Try a Linux build.")
    elif worst < 2:
        slow = [k for k, v in rest.items() if v < 2]
        print(f"cvtColor gains {control:.1f}x, so the threads are working, and "
              f"these do not:")
        for k in slow:
            print(f"  {k}  {gains[k]:.1f}x")
        print("That is the behaviour reported in opencv/opencv#30373.")
    else:
        print(f"Every call gains from the threads ({worst:.1f}x at worst). "
              f"This build is not affected.")


if __name__ == "__main__":
    main()
