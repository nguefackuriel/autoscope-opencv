"""Choose the classifier threshold on the calibration patients.

The rule, from results-parasites.md section 11: the threshold at which the
count over the calibration images matches every annotated parasite, not just
the ones that became candidates. Scored against candidates only, the 11% of
parasites the top-hat never proposes are missing from both sides, and 0.80
looked unbiased in August while under-counting by 12%.

Reads parasite_scores_calibration.npz (parasite_scores.py --group
calibration: the 20 calibration patients, every image, every candidate) and
writes threshold_fixedstats.json, which the chain reads for every step after
this one. Chains 1 and 2 did this inline; now it is a file.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def main():
    d = np.load(OUT / "parasite_scores_calibration.npz")
    scores = d["scores"].astype(np.float32)
    truth = int(d["truth_parasites"].sum())
    matched = int(d["labels"].sum())
    recall = matched / max(truth, 1)
    print(f"{len(d['offsets']) - 1} calibration images, {len(scores)} candidates, "
          f"{truth} annotated parasites, candidate recall {recall:.3f}")

    grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    bias = np.array([(float((scores >= t).sum()) - truth) / truth for t in grid])
    best = int(np.argmin(np.abs(bias)))
    print("\n  thr    count   bias vs annotated")
    for i in range(max(0, best - 4), min(len(grid), best + 5)):
        mark = "  <- chosen" if i == best else ""
        print(f"  {grid[i]:.2f}  {int((scores >= grid[i]).sum()):6d}   {bias[i]:+.1%}{mark}")

    (OUT / "threshold_fixedstats.json").write_text(json.dumps({
        "threshold": float(grid[best]),
        "bias_vs_annotated": float(bias[best]),
        "candidate_recall": recall,
        "rule": "closest to zero count bias against every annotated parasite, 20 calibration patients",
    }, indent=2))
    print(f"\nthreshold {grid[best]:.2f} ({bias[best]:+.1%}) -> threshold_fixedstats.json")


if __name__ == "__main__":
    main()
