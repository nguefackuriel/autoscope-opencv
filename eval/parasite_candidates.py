"""Build the candidate table: every candidate blob on the fit patients, with
its features and whether it sits on an annotated parasite.

Also reports candidate recall, which is the ceiling for the whole detector: a
parasite that never becomes a candidate cannot be found by any classifier.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann, parasites as par
from eval_helpers import RAW, load_pairs

OUT = Path(__file__).resolve().parent / "results"
MATCH_PX = 15.0   # the annotated centre is itself ~8 px imprecise (median), so 12 was too tight


def label_candidates(cands, truth_pts, tol=MATCH_PX):
    """Greedy nearest matching; returns per-candidate label and number of GT hit."""
    y = np.zeros(len(cands), np.int32)
    if not cands or len(truth_pts) == 0:
        return y, 0
    c = np.array([[k.x, k.y] for k in cands])
    d = np.linalg.norm(c[:, None, :] - truth_pts[None, :, :], axis=2)
    used_t = set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    hits = 0
    for i, j in order:
        if d[i, j] > tol:
            break
        if y[i] or j in used_t:
            continue
        y[i] = 1
        used_t.add(int(j))
        hits += 1
    return y, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--per-patient", type=int, default=1)
    args = ap.parse_args()

    sweep = json.loads((OUT / "wbc_sweep.json").read_text())
    fit = set(sweep["fit_patients"])

    rows_X, rows_y, meta = [], [], []
    total_truth = total_hit = total_cand = n_img = 0
    t_all = 0.0
    for img, truth in load_pairs(args.images * 2, per_patient=args.per_patient):
        if truth.path.parent.name not in fit:
            continue
        if n_img >= args.images:
            break
        n_img += 1
        t0 = time.perf_counter()
        prepared, mask, ox, oy = par._prepare(img, True, True)
        cands, _ = par.candidates(prepared, mask)
        t_all += time.perf_counter() - t0
        pts = np.array([[p.x - ox, p.y - oy] for p in truth.parasites]) if truth.parasites else np.zeros((0, 2))
        y, hits = label_candidates(cands, pts)
        total_truth += len(pts); total_hit += hits; total_cand += len(cands)
        for c, lab in zip(cands, y):
            rows_X.append(c.features); rows_y.append(lab)
            meta.append(truth.path.parent.name)
        if n_img % 20 == 0:
            print(f"  {n_img} images, candidate recall so far {total_hit / max(total_truth,1):.3f}, "
                  f"{total_cand / n_img:.0f} candidates/image")

    X = np.stack(rows_X); y = np.array(rows_y)
    np.savez_compressed(OUT / "parasite_candidates.npz", X=X, y=y, patient=np.array(meta))
    report = {
        "images": n_img, "truth": total_truth, "candidates": total_cand,
        "candidate_recall": round(total_hit / max(total_truth, 1), 4),
        "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "seconds_per_image": round(t_all / max(n_img, 1), 3),
        "feature_names": par.FEATURE_NAMES,
    }
    (OUT / "parasite_candidates.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
