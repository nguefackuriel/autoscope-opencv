"""What the white cell detector's false alarms look like, on the FIT patients.

6 September 2026. With the chain-3 parasite network the white cell count is
the larger of the two errors: +6% per density band on the held-out patients
and +26 to +30% on three slides (results-wbc.md). Before writing a gate,
look: every detection on the fit images is matched to the annotation, and
the hits and the false alarms are described by area, solidity, brightness,
purpleness and how much of the blob is dark. The held-out patients are not
touched here; whatever gate comes out of this is fitted by wbc_sweep.py on
the fit set and scored once on the held-out set by apply_sweep.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from autoscope import segmentation
from eval_helpers import load_pairs
from wbc_eval import TOLERANCE_PX

OUT = Path(__file__).resolve().parent / "results"
FEATURES = ("area", "solidity", "l_mean", "l_min", "stain_mean", "dark_fraction")


def label_found(found, truth):
    """Greedy nearest matching, returns a hit flag per detection."""
    hit = np.zeros(len(found), bool)
    if not found or not truth:
        return hit
    f = np.array([[c.x, c.y] for c in found]); t = np.array([[a.x, a.y] for a in truth])
    d = np.linalg.norm(f[:, None, :] - t[None, :, :], axis=2)
    used_t = set()
    for i, j in np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]:
        if d[i, j] > TOLERANCE_PX:
            break
        if hit[i] or j in used_t:
            continue
        hit[i] = True; used_t.add(int(j))
    return hit


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-patient", type=int, default=2)
    args = ap.parse_args()
    fit = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    pairs = [(img, t) for img, t in load_pairs(100 * args.per_patient, per_patient=args.per_patient, normalise=True)
             if t.path.parent.name in fit]
    print(f"{len(pairs)} fit images from {len({t.path.parent.name for _, t in pairs})} patients")

    rows = []
    per_patient = {}
    for img, truth in pairs:
        found = segmentation.find_white_cells(img, normalise_stain=False)
        hit = label_found(found, truth.white_cells)
        pat = truth.path.parent.name
        per_patient.setdefault(pat, [0, 0]); per_patient[pat][0] += len(found); per_patient[pat][1] += len(truth.white_cells)
        for c, h in zip(found, hit):
            rows.append({"patient": pat, "image": truth.path.name, "hit": bool(h),
                         **{k: getattr(c, k) for k in FEATURES}})
    hits = [r for r in rows if r["hit"]]; fas = [r for r in rows if not r["hit"]]
    print(f"{len(rows)} detections: {len(hits)} hits, {len(fas)} false alarms ({100 * len(fas) / len(rows):.1f}%)\n")

    print(f"{'feature':<14s} {'hits p5':>9s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p95':>8s}   {'FA p5':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p95':>8s}")
    stats = {}
    for k in FEATURES:
        h = np.array([r[k] for r in hits]); f = np.array([r[k] for r in fas])
        stats[k] = {"hits": [pct(h, q) for q in (5, 25, 50, 75, 95)], "false_alarms": [pct(f, q) for q in (5, 25, 50, 75, 95)]}
        print(f"{k:<14s} " + " ".join(f"{v:8.2f}" for v in stats[k]["hits"]) + "   " + " ".join(f"{v:8.2f}" for v in stats[k]["false_alarms"]))

    # What a single gate on each feature would cost and buy, on the fit set:
    # for each candidate cut, hits lost and false alarms removed.
    print("\nsingle gates, fit set (hits lost / false alarms removed):")
    gates = {"area >=": (2000, 2500, 3000, 3500, 4000), "l_mean <=": (100, 110, 120, 130, 140),
             "stain_mean >=": (40, 50, 60, 70), "dark_fraction >=": (0.3, 0.5, 0.7, 0.9), "solidity >=": (0.6, 0.7, 0.8)}
    gate_table = []
    for name, cuts in gates.items():
        k, op = name.split(" ")
        for cut in cuts:
            keep = (lambda v: v >= cut) if op == ">=" else (lambda v: v <= cut)
            lost = sum(not keep(r[k]) for r in hits); removed = sum(not keep(r[k]) for r in fas)
            gate_table.append({"gate": name, "cut": cut, "hits_lost": lost, "false_alarms_removed": removed})
            print(f"  {name} {cut:>6}: lose {lost:4d} of {len(hits)} hits, remove {removed:4d} of {len(fas)} false alarms")

    worst = sorted(((f / max(t, 1), p, f, t) for p, (f, t) in per_patient.items()), reverse=True)[:8]
    print("\nfit patients over-counting most (found / annotated):")
    for r in worst:
        print(f"  {r[1]:<14s} {r[0]:.2f}  {r[2]}/{r[3]}")
    (OUT / "wbc_fp_look.json").write_text(json.dumps({
        "images": len(pairs), "detections": len(rows), "hits": len(hits), "false_alarms": len(fas),
        "percentiles": stats, "gates": gate_table,
        "worst_fit_patients": [{"patient": p, "ratio": round(r, 3), "found": f, "truth": t} for r, p, f, t in worst],
        "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
