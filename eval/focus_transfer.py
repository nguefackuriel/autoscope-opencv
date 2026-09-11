"""Does the sharpness score mean the same thing on two sets from the same rig?

Written on 2 September 2026 after the uninfected set scored a median focus of
9.2 against 17.0 on the infected set, with 45 of 46 fields failing the gate.

    ../.venv/bin/python focus_transfer.py

Result, 80 random images from each set, statistics inside the eyepiece disc:

    measure           infected jpg (p5/med/p95)   uninfected tiff        gap
    lapvar            99.9   149   244            69.4   110   163      1.35x
    lap_s1.0          11.8  21.9  37.7            11.6  20.2  33.0      1.08x
    lap_s1.5          4.93  10.1  18.2            5.14  9.48  15.9      1.06x
    lap_minus_noise   29.8  58.3  96.0            30.7  52.3  89.6      1.11x
    noise             0.89  1.04  1.39            0.67  0.83  1.03      1.25x

    rank correlation with lapvar inside the infected set:
      noise 0.973   lap_s1.0 0.861   lap_minus_noise 0.853

The current score (variance of a 3x3 Laplacian) correlates at 0.97 with the
residual noise of the image: it measures grain, not focus. The TIFFs are
smoother than the phone JPEGs, so they score lower without being softer, and
re-encoding a JPEG at quality 95 raises its score without changing a cell.
A Laplacian after a sigma 1.0 Gaussian brings the two sets within 8% of each
other and still orders fields much as before. That is the replacement for
quality.py, with FOCUS_MIN re-surveyed on the new scale.
"""

import glob
import random

import cv2 as cv
import numpy as np
from scipy.stats import spearmanr

from autoscope import fov

random.seed(0)
INFECTED = sorted(glob.glob("../data/raw/TF*/*.jpg"))
UNINFECTED = sorted(glob.glob("../data/raw/NIH-NLM-ThickBloodSmearsU/Uninfected Patients/*/tiled/*.tiff"))
KEYS = ["lapvar", "lap_s1.0", "lap_s1.5", "lap_s2.0", "lap_minus_noise", "ratio_e2_e4", "noise"]


def measures(img: np.ndarray) -> dict:
    c, m = fov.crop(img)
    g = cv.cvtColor(c, cv.COLOR_BGR2GRAY)
    m = m > 0
    gf = g.astype(np.float32)
    out = {}
    lap = cv.Laplacian(gf, cv.CV_32F, ksize=3)
    out["lapvar"] = float(lap[m].var())
    for s in (1.0, 1.5, 2.0):
        l = cv.Laplacian(cv.GaussianBlur(gf, (0, 0), s), cv.CV_32F, ksize=3)
        out[f"lap_s{s}"] = float(l[m].var())
    res = gf - cv.medianBlur(g, 3).astype(np.float32)
    out["lap_minus_noise"] = out["lapvar"] - float(cv.Laplacian(res, cv.CV_32F, ksize=3)[m].var())

    def grad_energy(s: float) -> float:
        gs = cv.GaussianBlur(gf, (0, 0), s)
        gx, gy = cv.Sobel(gs, cv.CV_32F, 1, 0), cv.Sobel(gs, cv.CV_32F, 0, 1)
        return float((gx * gx + gy * gy)[m].mean())

    out["ratio_e2_e4"] = grad_energy(2.0) / max(grad_energy(4.0), 1e-6)
    out["noise"] = float(res[m].std())
    return out


def main(n: int = 80) -> None:
    sets = {"infected jpg": random.sample(INFECTED, n), "uninfected tiff": random.sample(UNINFECTED, n)}
    rows = {k: [measures(cv.imread(p)) for p in v] for k, v in sets.items()}
    print(f"{'measure':<16}" + "".join(f"{s:>28}" for s in rows) + "   gap of medians")
    for k in KEYS:
        med, line = {}, f"{k:<16}"
        for s, rs in rows.items():
            v = np.array([r[k] for r in rs])
            med[s] = np.median(v)
            line += f"{np.percentile(v, 5):9.3g} {np.median(v):8.3g} {np.percentile(v, 95):8.3g} "
        a, b = med.values()
        print(line + f"   {a / b:6.2f}x")
    print("\nrank correlation with lapvar inside the infected set:")
    lv = [r["lapvar"] for r in rows["infected jpg"]]
    for k in KEYS[1:]:
        print(f"  {k:<16} {spearmanr(lv, [r[k] for r in rows['infected jpg']]).correlation:6.3f}")


if __name__ == "__main__":
    main()
