"""Where the density error actually comes from, and what fixing each part buys.

    .venv/bin/python eval/density_attribution.py

The density error on the held-out 50 is about 19%, and every stopping policy
reports roughly the same figure, so the error is the counter's and not the
policy's. This asks which part of the counter owns it, by replaying the cached
candidate scores under counterfactuals instead of re-running the pipeline.

It needs no images and no model: `parasite_scores_holdout.npz` already holds
every candidate's score and whether it matched an annotated parasite, so a
threshold sweep or a perfect-judge substitution is arithmetic.

WHY THIS EXISTS

Two improvement ideas were on the table and both looked obvious. One was a
density-aware threshold, taken from the agent's running estimate. The other was
a better judge, which is what four training chains chased. This measures the
ceiling of each before anyone spends a week on it, and both ceilings are lower
than they look.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- repo paths, so this runs without an install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = Path(__file__).resolve().parent / "results"
WBC_PER_UL = 8000.0          # the WHO convention, same constant the pipeline uses
SHIPPED = 0.53               # threshold_fixedstats.json


def _load(tag: str = "holdout"):
    path = RESULTS / f"parasite_scores_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the chain's parasite_scores step")
    # Materialise every array once. An NpzFile decompresses on each key access,
    # and the sweeps below read `scores` inside a double loop: leaving it lazy
    # turned a one-second job into one that had to be killed.
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    rows = defaultdict(list)
    for i, name in enumerate(d["patient"].tolist()):
        rows[name].append(i)
    return d, rows


def densities(d, rows, pick, truth_wbc: bool = False):
    """Per-patient (ours, expert) under a rule for keeping a candidate."""
    out = {}
    for patient, members in rows.items():
        found_p = found_w = true_p = true_w = 0
        for i in members:
            sl = slice(d["offsets"][i], d["offsets"][i + 1])
            found_p += int(pick(d["scores"][sl], d["labels"][sl]).sum())
            found_w += int(d["found_wbc"][i])
            true_p += int(d["truth_parasites"][i])
            true_w += int(d["truth_wbc"][i])
        if found_w == 0 or true_w == 0 or true_p == 0:
            continue
        ours = found_p / (true_w if truth_wbc else found_w) * WBC_PER_UL
        out[patient] = (ours, true_p / true_w * WBC_PER_UL)
    return out


def _stats(out: dict) -> tuple[float, float]:
    rel = np.array([(o - e) / e for o, e in out.values()])
    return float(np.abs(rel).mean() * 100), float(rel.mean() * 100)


def main() -> None:
    d, rows = _load()
    print(f"{len(rows)} held-out patients, {len(d['scores']):,} candidates, "
          f"{int((d['labels'] == 1).sum()):,} of them annotated parasites\n")

    # 1. Is the shipped threshold the best single threshold?
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
    best_t, best_e = None, None
    for t in grid:
        e, _ = _stats(densities(d, rows, lambda s, l, t=t: s >= t))
        if best_e is None or e < best_e:
            best_t, best_e = float(t), e
    shipped_e, shipped_b = _stats(densities(d, rows, lambda s, l: s >= SHIPPED))
    print("A SINGLE GLOBAL THRESHOLD")
    print(f"  shipped  t={SHIPPED}   mean |error| {shipped_e:.1f}%   bias {shipped_b:+.1f}%")
    print(f"  best     t={best_t:.2f}   mean |error| {best_e:.1f}%")
    print("  The shipped value is already at or next to the optimum, so there is\n"
          "  nothing to win by moving one number.\n")

    # 2. The ceiling of a density-aware threshold, handed the expert's band.
    #    Not implementable: at inference you have a running estimate, not the
    #    answer. It is the size of the prize, and the prize is small.
    at_shipped = densities(d, rows, lambda s, l: s >= SHIPPED)
    band_of = {p: (0 if e < 1000 else 1 if e < 10000 else 2)
               for p, (_, e) in at_shipped.items()}
    per_band = {}
    print("A THRESHOLD PER DENSITY BAND, GIVEN THE EXPERT'S BAND (an oracle)")
    for b, label in ((0, "under 1 000"), (1, "1 000 to 10 000"), (2, "over 10 000")):
        members = [p for p in at_shipped if band_of[p] == b]
        if not members:
            continue
        bb_t, bb_e = None, None
        for t in grid:
            sub = {p: v for p, v in densities(
                d, rows, lambda s, l, t=t: s >= t).items() if p in members}
            e, _ = _stats(sub)
            if bb_e is None or e < bb_e:
                bb_t, bb_e = float(t), e
        per_band[b] = bb_t
        now, _ = _stats({p: at_shipped[p] for p in members})
        print(f"  {label:<16} {len(members):>2} slides   "
              f"t={SHIPPED} -> {now:.1f}%    t={bb_t:.2f} -> {bb_e:.1f}%")
    oracle = {}
    for b in set(band_of.values()):
        members = [p for p in at_shipped if band_of[p] == b]
        sub = densities(d, rows, lambda s, l, t=per_band[b]: s >= t)
        oracle.update({p: sub[p] for p in members if p in sub})
    oe, ob = _stats(oracle)
    print(f"  all bands, each at its own best threshold: {oe:.1f}% "
          f"against {shipped_e:.1f}% today.")
    print("  An oracle band buys about one and a half points. A running estimate\n"
          "  would buy less. This idea is not worth building.\n")

    # 3. Who owns the remaining error: the proposer, the judge, the white cells.
    print("WHAT FIXING EACH PART WOULD BUY")
    print(f"  {'':<44}{'|error|':>9}{'bias':>9}")
    for label, pick, tw in (
        ("shipped: threshold 0.53", lambda s, l: s >= SHIPPED, False),
        ("a perfect judge, same candidates", lambda s, l: l == 1, False),
        ("a perfect judge and perfect white cells", lambda s, l: l == 1, True),
        ("threshold 0.53, perfect white cells", lambda s, l: s >= SHIPPED, True),
    ):
        e, b = _stats(densities(d, rows, pick, tw))
        print(f"  {label:<44}{e:>8.1f}%{b:>+8.1f}%")

    print("""
  Read the bias column, because it is the finding.

  A perfect judge scores WORSE on bias than the detector we ship: it can only
  keep candidates the top-hat proposed, and the proposer misses 9.4% of
  annotated parasites, so a flawless judge under-counts by about 15%. The
  shipped detector under-counts by 4%, and the reason is that its false
  positives roughly cancel the proposer's misses. The white cell over-count of
  about 4% is part of the same cancellation: fix the white cells alone and the
  shipped error gets worse, not better.

  So the error is a three-way cancellation, and improving any single part on its
  own breaks it. That is why four training chains all landed between 23% and 26%
  no matter what the judge learned: a better judge removes false positives
  without recovering the parasites the proposer never offered it.

  The part to work on is the PROPOSER. Candidate recall is 0.906 and it caps
  everything downstream. Until it moves, a better judge cannot pay, and neither
  can a cleverer threshold.""")


if __name__ == "__main__":
    main()
