"""The soft-kept count, and the nine slides: the measured roadmap for the detector.

    .venv/bin/python eval/soft_count.py

Two things the report's future-work section quotes, computed from the cached
candidate scores so the figures come from a result file and not from a note.

SOFT-KEPT COUNT. The shipped counter counts candidates whose score clears the
threshold. The slides it over-counts are the ones whose kept detections are
uncertain: a false alarm that clears 0.53 usually clears it by little. So count
the kept detections by their score instead of by one each, and rescale by a
single factor fitted on the 20 calibration patients so the total matches the
annotated total there. Nothing is fitted on held-out.

THE NINE SLIDES. Per-slide variance of the found/true ratio is what the density
error tracks. Most of it sits in a handful of slides. This names them and says
what share of the variance they carry, and what every other slide looks like
without them.

Writes eval/results/soft_count.json. Reads nothing it could fit on.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = Path(__file__).resolve().parent / "results"
WBC_PER_UL = 8000.0
THRESHOLD = 0.53


def _load(tag: str):
    z = np.load(RESULTS / f"parasite_scores_{tag}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    rows = defaultdict(list)
    for i, name in enumerate(d["patient"].tolist()):
        rows[name].append(i)
    return d, rows


def _per_patient(d, rows, soft: bool, k: float = 1.0):
    """(ours per uL, expert per uL) per patient, hard or soft count."""
    out = {}
    for patient, members in rows.items():
        found = found_w = true_p = true_w = 0.0
        for i in members:
            s = d["scores"][d["offsets"][i]:d["offsets"][i + 1]]
            kept = s[s >= THRESHOLD]
            found += (k * kept.sum()) if soft else len(kept)
            found_w += d["found_wbc"][i]
            true_p += d["truth_parasites"][i]
            true_w += d["truth_wbc"][i]
        if found_w == 0 or true_w == 0 or true_p == 0:
            continue
        out[patient] = (found / found_w * WBC_PER_UL, true_p / true_w * WBC_PER_UL)
    return out


def _stats(pp: dict) -> dict:
    ours = np.array([o for o, _ in pp.values()])
    exp = np.array([e for _, e in pp.values()])
    rel = (ours - exp) / exp
    ratio = ours / exp
    return {
        "slides": len(pp),
        "mean_abs_error_pct": round(float(np.abs(rel).mean() * 100), 1),
        "bias_pct": round(float(rel.mean() * 100), 1),
        "ratio_cv_pct": round(float(ratio.std(ddof=1) / ratio.mean() * 100), 1),
        "within_25_pct": round(float((np.abs(rel) <= 0.25).mean() * 100), 1),
    }


def main() -> None:
    cal, cal_rows = _load("calibration")
    held, held_rows = _load("holdout")

    # The one scalar, from calibration only: soft-kept sum -> annotated count.
    soft_sum = truth_sum = 0.0
    for members in cal_rows.values():
        for i in members:
            s = cal["scores"][cal["offsets"][i]:cal["offsets"][i + 1]]
            soft_sum += s[s >= THRESHOLD].sum()
            truth_sum += cal["truth_parasites"][i]
    k = float(truth_sum / soft_sum)

    hard = _per_patient(held, held_rows, soft=False)
    soft = _per_patient(held, held_rows, soft=True, k=k)
    report = {
        "threshold": THRESHOLD,
        "soft_scale_k": round(k, 3),
        "k_fitted_on": "20 calibration patients, total annotated / total soft-kept sum",
        "shipped": _stats(hard),
        "soft_kept": _stats(soft),
    }

    # The nine slides: rank by |log ratio| under the shipped count.
    ratio = {p: o / e for p, (o, e) in hard.items()}
    logr = {p: float(np.log(r)) for p, r in ratio.items()}
    mean_lr = float(np.mean(list(logr.values())))
    contrib = {p: (v - mean_lr) ** 2 for p, v in logr.items()}
    total = sum(contrib.values())
    worst = sorted(contrib, key=contrib.get, reverse=True)[:9]
    rest = {p: hard[p] for p in hard if p not in worst}
    report["nine_slides"] = {
        "patients": [{"patient": p, "ratio": round(ratio[p], 2),
                      "expert_per_ul": round(hard[p][1])} for p in worst],
        "share_of_log_ratio_variance_pct": round(sum(contrib[p] for p in worst) / total * 100, 1),
        "others": _stats(rest),
    }
    # Where the variance is: recall or false alarms. ratio = recall + FA/truth.
    rec, fa = [], []
    for patient, members in held_rows.items():
        hits = fas = truth = 0
        for i in members:
            sl = slice(held["offsets"][i], held["offsets"][i + 1])
            s, l = held["scores"][sl], held["labels"][sl]
            kept = s >= THRESHOLD
            hits += int((kept & (l == 1)).sum()); fas += int((kept & (l != 1)).sum())
            truth += int(held["truth_parasites"][i])
        if truth:
            rec.append(hits / truth); fa.append(fas / truth)
    rec, fa = np.array(rec), np.array(fa)
    report["ratio_variance_split"] = {
        "recall_var": round(float(rec.var(ddof=1)), 4),
        "false_alarm_var": round(float(fa.var(ddof=1)), 4),
        "recall_cv_pct": round(float(rec.std(ddof=1) / rec.mean() * 100), 1),
        "false_alarm_cv_pct": round(float(fa.std(ddof=1) / fa.mean() * 100), 1),
    }

    (RESULTS / "soft_count.json").write_text(json.dumps(report, indent=2))
    s, h = report["soft_kept"], report["shipped"]
    print(f"k = {k:.3f} from calibration")
    print(f"{'held-out 50':<14}{'|err|':>8}{'bias':>8}{'CV':>8}{'within25':>10}")
    for name, st in (("shipped", h), ("soft-kept", s)):
        print(f"{name:<14}{st['mean_abs_error_pct']:>7.1f}%{st['bias_pct']:>+7.1f}%{st['ratio_cv_pct']:>7.1f}%{st['within_25_pct']:>9.0f}%")
    n = report["nine_slides"]
    print(f"\nnine slides carry {n['share_of_log_ratio_variance_pct']}% of the ratio variance: "
          + ", ".join(f"{x['patient']} x{x['ratio']}" for x in n["patients"]))
    o = n["others"]
    print(f"the other {o['slides']}: |err| {o['mean_abs_error_pct']}%, CV {o['ratio_cv_pct']}%")
    v = report["ratio_variance_split"]
    print(f"variance split: recall {v['recall_var']} (CV {v['recall_cv_pct']}%), "
          f"false alarms {v['false_alarm_var']} (CV {v['false_alarm_cv_pct']}%)")


if __name__ == "__main__":
    main()
