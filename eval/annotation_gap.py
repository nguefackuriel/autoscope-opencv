"""How far is the annotation from the object it marks?

Every number this project publishes is scored against a human click. If that
click is systematically off, or if its stated size bears no relation to the
object, then some of what looks like detector error is annotation error, and no
amount of model work removes it. This measures the gap before we try.

Two shapes, two questions.

    Parasite, Circle   centre plus a point on the rim. Is the centre on the
                       chromatin dot, and does the radius describe the dot?

    White cell, Point  one click, no size. Is the click inside the nucleus,
                       and where is the nucleus actually centred?

The reference is not another annotation: it is the pixels. A parasite is
refined against the top-hat stain map (the same map the candidate generator
reaches 88% recall with), a white cell against the nucleus mask that
segmentation.py already builds. Both are OpenCV contour operations, so what
this reports is exactly what "apply findContours and friends" would buy.

A systematic offset would show up as a mean displacement vector much larger
than zero; random jitter averages out. The two cases need different answers:
a systematic shift is a transform bug worth fixing, jitter is a noise floor to
be respected in the matching tolerance and otherwise left alone.

Fit patients only. The held-out 50 are not read.
"""

import argparse
import json
import random

# --- repo paths: run from anywhere, installed or not -------------------------
# Azure ML runs the code snapshot without `pip install -e .`, so the editable
# install that normally puts src/ on sys.path is absent and every `from
# autoscope import ...` fails. This depends on __file__ alone, so it survives
# any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann, parasites as par, segmentation as seg
from autoscope.boxes import LEVEL, PEAK_SEARCH_PX, WINDOW_PX
import splits

OUT = Path(__file__).resolve().parent / "results"

# A white cell nucleus is ~100 px across, so a click may be well off centre and
# still be inside the cell. Search wide enough to find the nucleus it belongs to.
WBC_SEARCH_PX = 150


def refine_parasite(top, cx, cy):
    """Contour the stain around an annotated parasite centre.

    Returns (peak_x, peak_y, centroid_x, centroid_y, equivalent_radius) or None.
    Equivalent radius is that of a circle with the same area as the contour,
    which is the fairest single number to set against the annotator's radius.
    """
    h, w = top.shape
    cxi, cyi = int(round(cx)), int(round(cy))
    if not (0 <= cxi < w and 0 <= cyi < h):
        return None

    x0, x1 = max(cxi - PEAK_SEARCH_PX, 0), min(cxi + PEAK_SEARCH_PX + 1, w)
    y0, y1 = max(cyi - PEAK_SEARCH_PX, 0), min(cyi + PEAK_SEARCH_PX + 1, h)
    win = top[y0:y1, x0:x1]
    if win.size == 0:
        return None
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    px, py = x0 + int(ix), y0 + int(iy)
    peak = float(win[iy, ix])
    if peak <= par.PEAK_MIN:
        return None

    gx0, gx1 = max(px - WINDOW_PX, 0), min(px + WINDOW_PX + 1, w)
    gy0, gy1 = max(py - WINDOW_PX, 0), min(py + WINDOW_PX + 1, h)
    patch = top[gy0:gy1, gx0:gx1]
    binary = (patch >= LEVEL * peak).astype(np.uint8)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        if cv.pointPolygonTest(c, (float(px - gx0), float(py - gy0)), False) >= 0:
            best = c
            break
    if best is None or cv.contourArea(best) <= 0:
        return None

    m = cv.moments(best)
    if m["m00"] == 0:
        return None
    mx = gx0 + m["m10"] / m["m00"]
    my = gy0 + m["m01"] / m["m00"]
    equiv_r = float(np.sqrt(cv.contourArea(best) / np.pi))
    return px, py, mx, my, equiv_r


def nucleus_cells(image_bgr, mask):
    """Per-cell nucleus centroids, WITH the watershed split.

    The split is not optional here. Contouring the raw nucleus mask merges two
    touching cells into one blob whose centroid sits in the isthmus between
    them, far from either click. Measured that way the click-to-centroid p90
    came out at 89 px and 22.9% of white cells looked unreachable inside the
    55 px tolerance -- which cannot be true, because the shipped detector
    scores recall 0.803 on these patients. The pipeline splits before counting,
    so anything measuring against it has to split too.

    Returns [(x, y, equivalent_radius)] at full resolution.
    """
    small = cv.resize(image_bgr, None, fx=seg.SCALE, fy=seg.SCALE,
                      interpolation=cv.INTER_AREA)
    small_mask = cv.resize(mask, (small.shape[1], small.shape[0]),
                           interpolation=cv.INTER_NEAREST)
    close_px = max(int(seg.CLOSE_KERNEL_FULL * seg.SCALE) | 1, 3)
    nuc = seg._nucleus_mask(small, small_mask, close_px)
    labels = seg._split_touching(nuc)

    inv = 1.0 / seg.SCALE
    out = []
    for label in range(2, int(labels.max()) + 1):
        blob = (labels == label).astype(np.uint8)
        area = int(blob.sum())
        if area < seg.AREA_MIN_FULL * seg.SCALE ** 2:
            continue
        m = cv.moments(blob)
        if m["m00"] == 0:
            continue
        out.append((m["m10"] / m["m00"] * inv, m["m01"] / m["m00"] * inv,
                    float(np.sqrt(area * inv * inv / np.pi))))
    return out


def refine_white_cell(cells, cx, cy):
    """The split nucleus a click belongs to. Returns (centre, equiv_r, inside).

    `inside` means the click is within the cell's equivalent radius, which is
    the honest test once the blob has been split: watershed labels are regions,
    not outlines, so a point-in-polygon test no longer applies.
    """
    if not cells:
        return None
    arr = np.array([[c[0], c[1]] for c in cells], dtype=float)
    d = np.hypot(arr[:, 0] - cx, arr[:, 1] - cy)
    i = int(d.argmin())
    if d[i] > WBC_SEARCH_PX:
        return None
    x, y, equiv_r = cells[i]
    return (x, y), equiv_r, bool(d[i] <= equiv_r)


def collect(patients, per_patient, seed):
    root = splits.patient_root()
    rng = random.Random(seed)
    par_rows, wbc_rows = [], []

    for n, patient in enumerate(sorted(patients), 1):
        candidates = [p for p in sorted((root / patient).glob("*.jpg"))
                      if ann.gt_path(p).exists()]
        if not candidates:
            continue
        for path in rng.sample(candidates, min(per_patient, len(candidates))):
            image = cv.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue

            prepared, mask, ox, oy = par._prepare(image, True, True)
            top, _, _ = par.tophat(prepared)

            for p in truth.parasites:
                r = refine_parasite(top, p.x - ox, p.y - oy)
                if r is None:
                    par_rows.append({"refined": False})
                    continue
                px, py, mx, my, equiv = r
                ax, ay = p.x - ox, p.y - oy
                par_rows.append({
                    "refined": True,
                    "dx_peak": px - ax, "dy_peak": py - ay,
                    "dx_centroid": mx - ax, "dy_centroid": my - ay,
                    "off_peak": float(np.hypot(px - ax, py - ay)),
                    "off_centroid": float(np.hypot(mx - ax, my - ay)),
                    "annot_r": p.radius, "equiv_r": equiv,
                })

            cells = nucleus_cells(prepared, mask)
            for a in truth.white_cells:
                res = refine_white_cell(cells, a.x - ox, a.y - oy)
                if res is None:
                    wbc_rows.append({"refined": False})
                    continue
                (mx, my), equiv, inside = res
                ax, ay = a.x - ox, a.y - oy
                wbc_rows.append({
                    "refined": True, "inside": bool(inside),
                    "dx": mx - ax, "dy": my - ay,
                    "off": float(np.hypot(mx - ax, my - ay)),
                    "equiv_r": equiv,
                })
        if n % 5 == 0:
            print(f"  {n} patients, {len(par_rows)} parasites, {len(wbc_rows)} white cells")
    return par_rows, wbc_rows


def summarise(par_rows, wbc_rows):
    p = [r for r in par_rows if r["refined"]]
    wb = [r for r in wbc_rows if r["refined"]]

    def stats(v):
        v = np.asarray(v, dtype=float)
        return {"n": int(v.size), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "mean": float(v.mean())}

    off_peak = stats([r["off_peak"] for r in p])
    off_cent = stats([r["off_centroid"] for r in p])
    dxy = np.array([[r["dx_peak"], r["dy_peak"]] for r in p], dtype=float)
    mean_vec = dxy.mean(axis=0)
    spread = float(np.median(np.hypot(dxy[:, 0], dxy[:, 1])))

    annot_r = np.array([r["annot_r"] for r in p if r["annot_r"]], dtype=float)
    equiv_r = np.array([r["equiv_r"] for r in p if r["annot_r"]], dtype=float)

    wof = stats([r["off"] for r in wb])
    wdxy = np.array([[r["dx"], r["dy"]] for r in wb], dtype=float)
    inside = float(np.mean([r["inside"] for r in wb])) if wb else 0.0
    wequiv = np.array([r["equiv_r"] for r in wb], dtype=float)

    return {
        "parasites": {
            "annotations": len(par_rows),
            "refined": len(p),
            "refine_rate": round(len(p) / max(len(par_rows), 1), 4),
            "offset_to_stain_peak_px": off_peak,
            "offset_to_contour_centroid_px": off_cent,
            "mean_offset_vector_px": [float(mean_vec[0]), float(mean_vec[1])],
            "mean_vector_length_px": float(np.hypot(*mean_vec)),
            "median_offset_length_px": spread,
            "systematic_fraction": round(float(np.hypot(*mean_vec)) / max(spread, 1e-6), 3),
            "annotated_radius_px": {"median": float(np.median(annot_r))},
            "contour_equivalent_radius_px": {"median": float(np.median(equiv_r))},
            "radius_ratio_contour_over_annotated": {
                "median": float(np.median(equiv_r / annot_r)),
                "p90": float(np.percentile(equiv_r / annot_r, 90)),
            },
        },
        "white_cells": {
            "annotations": len(wbc_rows),
            "refined": len(wb),
            "click_within_nucleus_radius_fraction": round(inside, 4),
            "offset_to_nucleus_centroid_px": wof,
            "mean_offset_vector_px": [float(wdxy[:, 0].mean()), float(wdxy[:, 1].mean())],
            "mean_vector_length_px": float(np.hypot(wdxy[:, 0].mean(), wdxy[:, 1].mean())),
            "nucleus_equivalent_radius_px": {"median": float(np.median(wequiv))},
        },
    }


def figures(par_rows, wbc_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = [r for r in par_rows if r["refined"]]
    wb = [r for r in wbc_rows if r["refined"]]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    off = np.array([r["off_peak"] for r in p])
    ax[0, 0].hist(off, bins=40, color="#e76f51")
    ax[0, 0].axvline(np.median(off), color="k", ls="--",
                     label=f"median {np.median(off):.1f} px")
    ax[0, 0].axvline(15, color="b", ls=":", label="15 px match tolerance")
    ax[0, 0].set_title("Parasite: click to stain peak"); ax[0, 0].set_xlabel("px")
    ax[0, 0].legend(fontsize=8)

    d = np.array([[r["dx_peak"], r["dy_peak"]] for r in p])
    ax[0, 1].scatter(d[:, 0], d[:, 1], s=4, alpha=.25, color="#e76f51")
    ax[0, 1].axhline(0, color="k", lw=.5); ax[0, 1].axvline(0, color="k", lw=.5)
    mv = d.mean(axis=0)
    ax[0, 1].plot(*mv, "k+", ms=18, mew=3,
                  label=f"mean ({mv[0]:+.1f}, {mv[1]:+.1f})")
    ax[0, 1].set_title("Parasite offset vectors\n(a cloud on zero = jitter, not bias)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_aspect("equal")

    ar = np.array([r["annot_r"] for r in p if r["annot_r"]])
    er = np.array([r["equiv_r"] for r in p if r["annot_r"]])
    ax[0, 2].scatter(ar, er, s=4, alpha=.25, color="#2a9d8f")
    lim = max(ar.max(), er.max())
    ax[0, 2].plot([0, lim], [0, lim], "k--", lw=1, label="equal")
    ax[0, 2].set_xlabel("annotated radius px"); ax[0, 2].set_ylabel("contour radius px")
    ax[0, 2].set_title(f"Parasite size: annotation vs pixels\nratio median "
                       f"{np.median(er/ar):.2f}")
    ax[0, 2].legend(fontsize=8)

    woff = np.array([r["off"] for r in wb])
    ax[1, 0].hist(woff, bins=40, color="#264653")
    ax[1, 0].axvline(np.median(woff), color="w", ls="--",
                     label=f"median {np.median(woff):.1f} px")
    ax[1, 0].axvline(55, color="r", ls=":", label="55 px match tolerance")
    ax[1, 0].set_title("White cell: click to nucleus centroid"); ax[1, 0].set_xlabel("px")
    ax[1, 0].legend(fontsize=8)

    wd = np.array([[r["dx"], r["dy"]] for r in wb])
    ax[1, 1].scatter(wd[:, 0], wd[:, 1], s=4, alpha=.25, color="#264653")
    ax[1, 1].axhline(0, color="k", lw=.5); ax[1, 1].axvline(0, color="k", lw=.5)
    wmv = wd.mean(axis=0)
    ax[1, 1].plot(*wmv, "r+", ms=18, mew=3,
                  label=f"mean ({wmv[0]:+.1f}, {wmv[1]:+.1f})")
    ax[1, 1].set_title("White cell offset vectors"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_aspect("equal")

    inside = np.mean([r["inside"] for r in wb]) * 100
    wer = np.array([r["equiv_r"] for r in wb])
    ax[1, 2].hist(wer, bins=30, color="#457b9d")
    ax[1, 2].set_title(f"Nucleus radius from contours\nmedian {np.median(wer):.0f} px, "
                       f"{inside:.0f}% of clicks within radius")
    ax[1, 2].set_xlabel("px")

    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=20)
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_ids, val_ids = splits.train_val(seed=args.seed)
    fit = sorted(set(train_ids) | set(val_ids))[:args.patients]
    print(f"{len(fit)} fit patients, {args.per_patient} images each "
          f"(held-out patients are not read)")

    par_rows, wbc_rows = collect(fit, args.per_patient, args.seed)
    report = summarise(par_rows, wbc_rows)
    report["patients"] = len(fit)
    report["per_patient"] = args.per_patient

    OUT.mkdir(exist_ok=True)
    (OUT / "annotation_gap.json").write_text(json.dumps(report, indent=2))
    figures(par_rows, wbc_rows, OUT / "annotation_gap.png")

    p, w = report["parasites"], report["white_cells"]
    print(f"\nPARASITES  {p['refined']} of {p['annotations']} refined "
          f"({p['refine_rate']*100:.1f}%)")
    print(f"  click to stain peak      median {p['offset_to_stain_peak_px']['median']:.1f} px, "
          f"p90 {p['offset_to_stain_peak_px']['p90']:.1f}")
    print(f"  mean offset vector       ({p['mean_offset_vector_px'][0]:+.2f}, "
          f"{p['mean_offset_vector_px'][1]:+.2f}) px, length "
          f"{p['mean_vector_length_px']:.2f}")
    print(f"  systematic / random      {p['systematic_fraction']:.3f}   "
          f"(near 0 = jitter, near 1 = a fixable shift)")
    print(f"  annotated radius         median {p['annotated_radius_px']['median']:.1f} px")
    print(f"  contour radius           median {p['contour_equivalent_radius_px']['median']:.1f} px")
    print(f"  ratio contour/annotated  median "
          f"{p['radius_ratio_contour_over_annotated']['median']:.2f}")

    print(f"\nWHITE CELLS  {w['refined']} of {w['annotations']} matched to a nucleus")
    print(f"  click within nucleus radius {w['click_within_nucleus_radius_fraction']*100:.1f}%")
    print(f"  click to centroid        median {w['offset_to_nucleus_centroid_px']['median']:.1f} px, "
          f"p90 {w['offset_to_nucleus_centroid_px']['p90']:.1f}")
    print(f"  mean offset vector       length {w['mean_vector_length_px']:.2f} px")
    print(f"  nucleus radius           median {w['nucleus_equivalent_radius_px']['median']:.0f} px")
    print(f"\nwritten to {OUT/'annotation_gap.json'} and annotation_gap.png")


if __name__ == "__main__":
    main()
