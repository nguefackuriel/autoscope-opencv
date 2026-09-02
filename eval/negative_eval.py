"""Specificity: what the pipeline says about slides with no parasites.

Reads negative_fields.json (uninfected patients, from negative_fields.py) and
agent_fields.json (the 50 held-out infected patients) and answers three
questions the proposal promised and nothing had measured:

  1. How many false parasites does a clean field produce, and how many clean
     fields produce none at all.
  2. What density does a clean slide read at, through the real protocol
     (WHO 200 white cells, and the rule-based agent). This is the noise floor:
     an infected patient below it cannot be told from a clean one.
  3. Per-patient sensitivity and specificity at a density cut-off, and the
     rank AUC between infected and uninfected patients.

Replays the real `read_slide` loop through agent_eval's CachedStage, so the
policies here are the ones that ship.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_eval import CachedStage, CachedTools, reference
from autoscope.agent.loop import read_slide
from autoscope.agent.policy import RuleBased, WhoProtocol
from autoscope.density import WBC_PER_UL_DEFAULT, density

OUT = Path(__file__).resolve().parent / "results"


def by_patient(table: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in table["rows"]:
        out.setdefault(row["patient"], []).append(row)
    return out


def replay(fields: list[dict], policy, source: str = "found") -> dict:
    stage = CachedStage(fields)
    tools = CachedTools(stage, source)
    trace = read_slide(tools, policy, slide=fields[0]["patient"], max_fields=len(fields))
    d = density(tools.parasites, max(tools.wbc, 1))
    return {"patient": fields[0]["patient"], "policy": policy.name,
            "fields_read": tools.fields_read, "fields_available": len(fields),
            "wbc": tools.wbc, "parasites": tools.parasites, "per_ul": round(d.per_ul, 1),
            "stopped_because": trace.stopped_because, "escalated": tools.escalation is not None}


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random infected patient reads higher than a random clean one."""
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", default="negative_fields.json")
    ap.add_argument("--positives", default="agent_fields.json")
    ap.add_argument("--target-ci", type=float, default=0.20)
    ap.add_argument("--min-fields", type=int, default=5)
    ap.add_argument("--out", default="negative_eval.json")
    args = ap.parse_args()

    neg_table = json.loads((OUT / args.negatives).read_text())
    pos_table = json.loads((OUT / args.positives).read_text())
    neg, pos = by_patient(neg_table), by_patient(pos_table)
    neg_rows = neg_table["rows"]
    print(f"{len(neg)} uninfected patients, {len(neg_rows)} fields; "
          f"{len(pos)} infected held-out patients, {len(pos_table['rows'])} fields; "
          f"threshold {neg_table['threshold']} / {pos_table['threshold']}\n")

    # 1. Per field.
    fp = np.array([r["found_parasites"] for r in neg_rows], float)
    wbc = np.array([r["found_wbc"] for r in neg_rows], float)
    per_field = {
        "fields": len(neg_rows),
        "clean_fields_pct": round(float((fp == 0).mean() * 100), 1),
        "false_parasites_mean": round(float(fp.mean()), 2),
        "false_parasites_median": float(np.median(fp)),
        "false_parasites_p90": float(np.percentile(fp, 90)),
        "false_parasites_max": int(fp.max()),
        "false_per_200_wbc_pooled": round(float(fp.sum() / max(wbc.sum(), 1) * 200), 2),
        "wbc_per_field_mean": round(float(wbc.mean()), 1),
    }
    print("per clean field:")
    print(f"  {per_field['clean_fields_pct']}% of fields report zero parasites; "
          f"mean {per_field['false_parasites_mean']}, median {per_field['false_parasites_median']:.0f}, "
          f"p90 {per_field['false_parasites_p90']:.0f}, max {per_field['false_parasites_max']}")
    print(f"  pooled: {per_field['false_per_200_wbc_pooled']} false parasites per 200 white cells "
          f"= {per_field['false_per_200_wbc_pooled'] / 200 * WBC_PER_UL_DEFAULT:,.0f}/uL noise floor\n")

    # 2. Per patient, through the real protocols.
    arms = {}
    for make in (lambda: WhoProtocol(200), lambda: RuleBased(args.target_ci, min_fields=args.min_fields)):
        neg_runs = [replay(f, make()) for _, f in sorted(neg.items())]
        pos_runs = [replay(f, make()) for _, f in sorted(pos.items())]
        name = neg_runs[0]["policy"]
        n_ul = np.array([r["per_ul"] for r in neg_runs]); p_ul = np.array([r["per_ul"] for r in pos_runs])
        called_negative = sum("meets the WHO threshold for a negative" in r["stopped_because"] for r in neg_runs)
        cutoffs = [0, 100, 200, 500, 1000, 2000]
        table = []
        for c in cutoffs:
            spec = float((n_ul <= c).mean()); sens = float((p_ul > c).mean())
            table.append({"cutoff_per_ul": c, "specificity": round(spec, 3), "sensitivity": round(sens, 3)})
        arms[name] = {
            "uninfected_per_ul_median": float(np.median(n_ul)),
            "uninfected_per_ul_p90": float(np.percentile(n_ul, 90)),
            "uninfected_per_ul_max": float(n_ul.max()),
            "uninfected_called_negative_by_rule": int(called_negative),
            "uninfected_escalated": int(sum(r["escalated"] for r in neg_runs)),
            "uninfected_fields_read": int(sum(r["fields_read"] for r in neg_runs)),
            "uninfected_fields_available": int(sum(r["fields_available"] for r in neg_runs)),
            "auc": round(rank_auc(p_ul, n_ul), 4),
            "cutoffs": table,
            "runs_uninfected": neg_runs,
        }
        print(f"{name}: clean slides read at median {np.median(n_ul):,.0f}/uL, "
              f"p90 {np.percentile(n_ul, 90):,.0f}, max {n_ul.max():,.0f}; "
              f"{called_negative}/{len(neg_runs)} called negative by the stopping rule, "
              f"{arms[name]['uninfected_escalated']} escalated; "
              f"fields read {arms[name]['uninfected_fields_read']}/{arms[name]['uninfected_fields_available']}")
        print(f"  rank AUC infected vs uninfected: {arms[name]['auc']:.3f}")
        print(f"  {'cut-off /uL':>12s} {'specificity':>12s} {'sensitivity':>12s}")
        for t in table:
            print(f"  {t['cutoff_per_ul']:>12d} {t['specificity']:>12.2f} {t['sensitivity']:>12.2f}")
        print()

    # Where the infected patients sit, by expert density, so the reader can see
    # what a given cut-off would cost at the light end.
    refs = np.array(sorted(reference(f) for f in pos.values()))
    print("infected held-out patients by expert density: "
          f"min {refs[0]:,.0f}, p10 {np.percentile(refs, 10):,.0f}, median {np.median(refs):,.0f}/uL")

    (OUT / args.out).write_text(json.dumps({
        "threshold": neg_table["threshold"], "uninfected_patients": len(neg),
        "infected_patients": len(pos), "per_field": per_field, "arms": arms,
        "infected_expert_density_percentiles": {k: float(np.percentile(refs, k)) for k in (0, 10, 25, 50)},
    }, indent=2))


if __name__ == "__main__":
    main()
