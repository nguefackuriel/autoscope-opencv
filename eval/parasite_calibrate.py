"""A count that does not depend on picking one threshold for every slide.

THE PROBLEM

The shipped classifier reports a probability per candidate and we turn that
into a count by keeping everything above 0.80. That single number has to serve
slides that are nothing like each other, and it does not: on held-out fields
with fewer than ten parasites the precision is 0.575, so we over-call; on
fields with eighty or more it is 0.894 with recall 0.73, so we under-call. The
bias also crossed zero at 0.80 on the validation patients and near 0.72 on the
held-out ones. One threshold cannot sit in the right place for every slide.

THREE WAYS OUT, MEASURED AGAINST EACH OTHER

  fixed        keep candidates above t. What ships today.
  expected     do not threshold at all. If the score were a probability, the
               expected number of parasites is the sum of the scores, and a
               sum needs no cut-off. It adapts to the slide for free, because a
               dense slide simply has more high scores.
  per-slide    count once at a provisional threshold to see what kind of slide
               this is, then look up the threshold that suited slides like it.
  corrected    the expected count, divided by the share of parasites that
               become candidates at all. See below: this is where the bias
               actually lives.

WHERE THE BIAS ACTUALLY LIVES

Measured on the calibration patients, the -12% count bias splits as:

    expected count vs candidates that really are parasites   +1.6%
    those candidates vs every annotated parasite            -11.4%

So the scoring is close to right and the candidate step is what loses
parasites. No threshold, per slide or otherwise, can find an object that never
became a candidate. That reframes the work: a threshold buys a point or two,
and the remaining ten points are in `parasites.candidates()`.

WHY THE SCORES NEED CALIBRATING FIRST

The network was trained with a positive class weight, so its scores are not
probabilities: they are pushed towards 1 to buy recall. Summing them straight
would over-count badly. So we fit a one-dimensional map from score to
probability, and only then sum.

WHERE THE CALIBRATION IS FITTED, AND WHY IT MATTERS

On the twenty fit patients the shipped model was NOT trained on (it kept them
back for its own threshold table). Fitting on patients the network memorised
would give scores that are too confident and a calibration that flatters
itself. The fifty held-out patients are still never touched by any fitting
here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def load(group: str):
    d = np.load(OUT / f"parasite_scores_{group}.npz", allow_pickle=False)
    off = d["offsets"]
    scores = [d["scores"][off[i]:off[i + 1]].astype(np.float32) for i in range(len(off) - 1)]
    out = {
        "scores": scores,
        "patient": d["patient"], "image": d["image"],
        "truth_parasites": d["truth_parasites"].astype(float),
        "truth_wbc": d["truth_wbc"].astype(float),
        "found_wbc": d["found_wbc"].astype(float),
    }
    if "labels" in d:
        out["labels"] = [d["labels"][off[i]:off[i + 1]].astype(np.int32)
                         for i in range(len(off) - 1)]
    return out


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool adjacent violators. Returns the breakpoints of a monotone step map.

    Written out rather than imported because the runtime pins opencv, numpy and
    scipy only, and this is twenty lines.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    values = y.astype(float).copy()
    weights = np.ones_like(values)
    idx = list(range(len(values)))
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_w
        values[i] = pooled
        weights[i] = total_w
        values = np.delete(values, i + 1)
        weights = np.delete(weights, i + 1)
        x = np.delete(x, i + 1)
        if i > 0:
            i -= 1
    return x, values


class Calibration:
    """score -> probability, as a monotone lookup fitted on binned frequencies."""

    def __init__(self, edges: np.ndarray, probs: np.ndarray):
        self.edges, self.probs = edges, probs

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, bins: int = 40) -> "Calibration":
        edges = np.quantile(scores, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        centres, freqs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (scores >= lo) & (scores < hi if hi < edges[-1] else scores <= hi)
            if m.sum() < 20:
                continue
            centres.append(float(scores[m].mean()))
            freqs.append(float(labels[m].mean()))
        cx, cy = isotonic(np.array(centres), np.array(freqs))
        return cls(cx, cy)

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self.edges, self.probs)

    def as_dict(self) -> dict:
        return {"score": self.edges.tolist(), "probability": self.probs.tolist()}


def counts_fixed(scores: list[np.ndarray], t: float) -> np.ndarray:
    return np.array([float((s >= t).sum()) for s in scores])


def counts_expected(scores: list[np.ndarray], cal: Calibration, floor: float = 0.0) -> np.ndarray:
    """Sum of calibrated probabilities. `floor` drops the long tail of near-zero
    candidates, whose individual probabilities are small but whose number is
    large enough to add up to a real count if left in."""
    return np.array([float(cal(s[s >= floor]).sum()) for s in scores])


def candidate_recall(data) -> float:
    """Share of annotated parasites that became candidates at all."""
    matched = sum(float(l.sum()) for l in data["labels"])
    return matched / max(data["truth_parasites"].sum(), 1.0)


def per_slide_threshold(provisional: float, table: list[tuple[float, float]]) -> float:
    """Look up the threshold that suited slides with this provisional density."""
    for upper, t in table:
        if provisional <= upper:
            return t
    return table[-1][1]


def slide_report(name: str, ours: np.ndarray, truth: np.ndarray,
                 wbc_ours: np.ndarray, wbc_truth: np.ndarray, patients: np.ndarray) -> dict:
    """Per-image count error, and the per-slide ratio that becomes the density."""
    err = ours - truth
    bias = err.sum() / max(truth.sum(), 1) * 100
    mae = float(np.abs(err).mean())

    rows = []
    for p in sorted(set(patients.tolist())):
        m = patients == p
        if wbc_ours[m].sum() > 0 and wbc_truth[m].sum() > 0:
            rows.append((ours[m].sum() / wbc_ours[m].sum(),
                         truth[m].sum() / wbc_truth[m].sum()))
    a = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
    keep = (a > 0) & (b > 0)
    ld = np.log(a[keep]) - np.log(b[keep])
    return {
        "method": name,
        "count_bias_pct": round(float(bias), 2),
        "count_mae": round(mae, 2),
        "slide_density_bias_pct": round(float((np.exp(ld.mean()) - 1) * 100), 2),
        "slide_density_abs_error_pct": round(float(np.abs(np.exp(np.abs(ld)) - 1).mean() * 100), 1),
        "slide_loa_pct": [round(float((np.exp(ld.mean() - 1.96 * ld.std()) - 1) * 100), 1),
                          round(float((np.exp(ld.mean() + 1.96 * ld.std()) - 1) * 100), 1)],
        "slides": int(keep.sum()),
    }


def fit_slide_table(cal, cal_counts_at, buckets=(2.0, 10.0, 30.0, 1e9)) -> list[tuple[float, float]]:
    """The threshold that zeroed each calibration patient's bias, by how many
    parasites per field a provisional pass at 0.80 suggested.

    Two passes, not one: the first says what kind of slide this is, the second
    counts it with a threshold suited to slides like it. The first pass costs
    nothing extra, because the scores are already computed.
    """
    provisional = counts_fixed(cal["scores"], 0.80)
    grid = np.arange(0.40, 0.96, 0.01)
    per_patient = {}
    for p in sorted(set(cal["patient"].tolist())):
        m = cal["patient"] == p
        truth = cal["truth_parasites"][m].sum()
        if truth < 5:
            continue
        best, best_err = 0.80, 1e9
        for t in grid:
            got = counts_fixed([s for s, k in zip(cal["scores"], m) if k], t).sum()
            err = abs(got - truth) / truth
            if err < best_err:
                best, best_err = float(t), err
        per_patient[p] = (float(provisional[m].mean()), best)

    table = []
    lower = 0.0
    for upper in buckets:
        ts = [t for dens, t in per_patient.values() if lower <= dens < upper]
        table.append((float(upper), float(np.median(ts)) if ts else 0.80))
        lower = upper
    return table, per_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", type=float, default=0.80)
    ap.add_argument("--floor", type=float, default=0.05,
                    help="drop candidates below this before summing probabilities")
    args = ap.parse_args()

    cal = load("calibration")
    hold = load("holdout")
    print(f"calibration: {len(cal['scores'])} images, "
          f"{len(set(cal['patient'].tolist()))} patients the model never trained on")
    print(f"held-out:    {len(hold['scores'])} images, "
          f"{len(set(hold['patient'].tolist()))} patients\n")

    scores = np.concatenate(cal["scores"])
    labels = np.concatenate(cal["labels"])
    calib = Calibration.fit(scores, labels)

    print("what the score means, before and after calibrating:")
    print(f"  {'score band':<14s} {'candidates':>11s} {'really parasites':>17s} {'calibrated':>11s}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        print(f"  {lo:.2f} to {hi:<6.2f} {m.sum():>11d} {labels[m].mean():>16.1%} "
              f"{calib(scores[m]).mean():>10.1%}")

    table, per_patient = fit_slide_table(cal, counts_fixed)
    dens = np.array([v[0] for v in per_patient.values()])
    best = np.array([v[1] for v in per_patient.values()])
    print(f"\nbest threshold per calibration slide: p10 {np.percentile(best, 10):.2f}, "
          f"median {np.median(best):.2f}, p90 {np.percentile(best, 90):.2f}")
    print(f"correlation with the provisional density: {np.corrcoef(dens, best)[0, 1]:+.3f}"
          "   <- if this is near zero the lookup cannot work")
    print("\nthreshold that suited each kind of calibration slide:")
    lower = 0.0
    for upper, t in table:
        label = f"under {upper:g}" if upper < 1e8 else "30 and over"
        n = sum(1 for d, _ in per_patient.values() if lower <= d < upper)
        print(f"  provisional parasites per field {label:<14s} n={n:<3d} threshold {t:.2f}")
        lower = upper

    recall = candidate_recall(cal)
    print(f"\ncandidate recall on the calibration patients: {recall:.3f}")
    exp_cal = counts_expected(cal["scores"], calib, args.floor)
    matched_cal = sum(float(l.sum()) for l in cal["labels"])
    print(f"  expected count vs candidates that are parasites: "
          f"{(exp_cal.sum() / matched_cal - 1) * 100:+.1f}%   <- scoring")
    print(f"  those candidates vs every annotated parasite:    "
          f"{(matched_cal / cal['truth_parasites'].sum() - 1) * 100:+.1f}%   <- candidates")

    methods = {}
    methods[f"fixed {args.fixed:.2f} (shipped)"] = counts_fixed(hold["scores"], args.fixed)
    methods["expected count"] = counts_expected(hold["scores"], calib, args.floor)
    methods["expected / recall"] = methods["expected count"] / recall

    provisional = counts_fixed(hold["scores"], 0.80)
    by_patient_prov = {p: provisional[hold["patient"] == p].mean()
                       for p in set(hold["patient"].tolist())}
    per_slide = np.array([
        (hold["scores"][i] >= per_slide_threshold(by_patient_prov[hold["patient"][i]], table)).sum()
        for i in range(len(hold["scores"]))], float)
    methods["per-slide threshold"] = per_slide

    rows = [slide_report(name, c, hold["truth_parasites"], hold["found_wbc"],
                         hold["truth_wbc"], hold["patient"])
            for name, c in methods.items()]

    print(f"\nheld-out, {len(hold['scores'])} images, "
          f"{len(set(hold['patient'].tolist()))} slides\n")
    print(f"{'method':<24s} {'count bias':>11s} {'count MAE':>10s} {'density bias':>13s} "
          f"{'|density err|':>14s} {'95% limits':>22s}")
    for r in rows:
        loa = f"{r['slide_loa_pct'][0]:+.0f}% to {r['slide_loa_pct'][1]:+.0f}%"
        print(f"{r['method']:<24s} {r['count_bias_pct']:>+10.1f}% {r['count_mae']:>10.2f} "
              f"{r['slide_density_bias_pct']:>+12.1f}% {r['slide_density_abs_error_pct']:>13.1f}% "
              f"{loa:>22s}")

    (OUT / "parasite_calibration.json").write_text(json.dumps({
        "candidate_recall": round(recall, 4),
        "calibration_patients": sorted(set(cal["patient"].tolist())),
        "calibration": calib.as_dict(),
        "slide_table": table,
        "per_patient_best_threshold": per_patient,
        "floor": args.floor,
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
