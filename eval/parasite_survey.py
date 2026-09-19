"""Measure what a parasite looks like before writing a detector for it.

Runs on the fit patients only (the same 100 the white cell sweep used), so the
50 held-out patients stay untouched.

For every annotated parasite we sample the image at the marked centre, and for
comparison we sample (a) the field far from any annotation and (b) random
"look-alike" spots: the darkest small purple blobs that are NOT within a
parasite or white cell annotation. The look-alikes are what a detector has to
reject: platelets, stain precipitate, debris.
"""

import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

# --- repo paths: run from anywhere, installed or not -------------------------
# A remote runner executes the code snapshot without `pip install -e .`, so the
# editable install that normally puts src/ on sys.path is absent and every
# `from autoscope import ...` fails. True of AWS Batch and SageMaker as much as
# of the GPU box the detectors are trained on. This depends on __file__ alone,
# so it survives any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

import splits

from autoscope import annotations as ann, fov, photometry
from autoscope.segmentation import stain_map

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def lab_at(lab, x, y, r):
    h, w = lab.shape[:2]
    x0, x1 = max(int(x - r), 0), min(int(x + r) + 1, w)
    y0, y1 = max(int(y - r), 0), min(int(y + r) + 1, h)
    patch = lab[y0:y1, x0:x1].reshape(-1, 3)
    return patch


def main():
    rng = random.Random(1)
    sweep = json.loads((OUT / "wbc_sweep.json").read_text())
    fit = sorted(sweep["fit_patients"])
    rng.shuffle(fit)

    para_centre, para_ring, field, lookalike = [], [], [], []
    radii = []
    n_images = 0
    for name in fit[:40]:
        patient = RAW / name
        paths = splits.annotated_images(patient)
        if not paths:
            continue
        path = rng.choice(paths)
        img = cv.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        truth = ann.for_image(path, w, h)
        if truth is None:
            continue
        n_images += 1
        disc = fov.detect(img)          # on the original: normalising can lift the surround
        if disc is None:
            continue
        img = photometry.normalise(img)
        lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)

        for p in truth.parasites:
            radii.append(p.radius)
            para_centre.append(lab_at(lab, p.x, p.y, 2).mean(axis=0))
            # a ring 2 radii out: what surrounds the parasite
            ring = []
            for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                rx, ry = p.x + 2.5 * max(p.radius, 8) * np.cos(ang), p.y + 2.5 * max(p.radius, 8) * np.sin(ang)
                if 0 <= rx < w and 0 <= ry < h:
                    ring.append(lab[int(ry), int(rx)])
            if ring:
                para_ring.append(np.mean(ring, axis=0))

        # field samples: random points inside the disc, far from every annotation
        pts = np.array([[a.x, a.y] for a in truth.items]) if truth.items else np.zeros((0, 2))
        tries = 0
        while len(field) < 40 * n_images and tries < 500:
            tries += 1
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.9 * disc.radius)
            x, y = disc.cx + rad * np.cos(ang), disc.cy + rad * np.sin(ang)
            if pts.size and np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)) < 60:
                continue
            field.append(lab[int(y), int(x)])

        # look-alikes: small purple blobs not covered by any annotation
        stain = stain_map(img)
        _, mask = fov.crop(img, disc)
        # mask is cropped-size; rebuild a full-frame one
        full_mask = np.zeros((h, w), np.uint8)
        cv.circle(full_mask, (disc.cx, disc.cy), int(disc.radius * 0.9), 255, -1)
        binary = ((stain > 40) & (full_mask > 0)).astype(np.uint8)
        n, labels, stats, cents = cv.connectedComponentsWithStats(binary)
        cand = [(i, cents[i]) for i in range(1, n) if 20 <= stats[i, cv.CC_STAT_AREA] <= 2000]
        rng.shuffle(cand)
        taken = 0
        for i, (cx, cy) in cand:
            if pts.size and np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) < 40:
                continue
            lookalike.append(np.r_[lab_at(lab, cx, cy, 2).mean(axis=0), stats[i, cv.CC_STAT_AREA]])
            taken += 1
            if taken >= 40:
                break

    def summ(arr, name):
        a = np.array(arr)
        q = np.percentile(a, [10, 50, 90], axis=0)
        print(f"{name:14s} n={len(a):5d}  L {q[0,0]:6.1f} {q[1,0]:6.1f} {q[2,0]:6.1f} | "
              f"a {q[0,1]:6.1f} {q[1,1]:6.1f} {q[2,1]:6.1f} | b {q[0,2]:6.1f} {q[1,2]:6.1f} {q[2,2]:6.1f}"
              + (f" | area {q[0,3]:6.0f} {q[1,3]:6.0f} {q[2,3]:6.0f}" if a.shape[1] > 3 else ""))
        return {"n": len(a), "p10": q[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist()}

    print(f"{n_images} images from fit patients\n                        L (p10 p50 p90)      |  a                   |  b")
    r = {}
    r["parasite_centre"] = summ(para_centre, "parasite ctr")
    r["parasite_ring"] = summ(para_ring, "parasite ring")
    r["field"] = summ(field, "field")
    r["lookalike"] = summ(lookalike, "lookalike")
    rq = np.percentile(radii, [10, 50, 90])
    print(f"\nparasite radius px  p10 {rq[0]:.1f}  median {rq[1]:.1f}  p90 {rq[2]:.1f}   (n={len(radii)})")
    r["radius_px"] = {"p10": rq[0], "p50": rq[1], "p90": rq[2], "n": len(radii)}

    # the discriminating quantities
    pc, la, fi = np.array(para_centre), np.array(lookalike), np.array(field)
    for nm, arr in [("parasite", pc), ("lookalike", la[:, :3]), ("field", fi)]:
        ab = arr[:, 1] - arr[:, 2]
        print(f"{nm:10s} a-b  p10 {np.percentile(ab,10):6.1f}  p50 {np.percentile(ab,50):6.1f}  p90 {np.percentile(ab,90):6.1f}"
              f"   L-b... b-only p10 {np.percentile(arr[:,2],10):6.1f} p50 {np.percentile(arr[:,2],50):6.1f} p90 {np.percentile(arr[:,2],90):6.1f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "parasite_survey.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
