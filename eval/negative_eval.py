"""Specificity: what the pipeline says about slides with no parasites.

Reads negative_fields.json (uninfected patients, from negative_fields.py) and
agent_fields.json (the 50 held-out infected patients) and answers four
questions the proposal promised and nothing had measured:

  1. How many false parasites does a clean field produce, and how many clean
     fields produce none at all.
  2. What density does a clean slide read at, through the real protocol
     (WHO 200 white cells, and the rule-based agent). This is the noise floor:
     an infected patient below it cannot be told from a clean one.
  3. Per-patient sensitivity and specificity at a density cut-off, and the
     rank AUC between infected and uninfected patients.
  4. Since 5 September 2026: what the decision rule CALLS. Negative on the
     noise floor after enough white cells, positive when the interval clears
     it, or handed to a person. Specificity is then "clean slides called
     negative", sensitivity "infected slides called positive", and the third
     column, "sent to a person", is the honest cost of both.

THE FLOOR IS DESIGNED ON THE FIT HALF AND TESTED ON THE FROZEN HALF

The uninfected set is split 25/25 (uninfected_split.json). The classifier
trains on the fit half, and so does the rule: the floor is the p95 of what
the fit half's slides read at through the pipeline (--floor-quantile), or a
value given with --floor. The frozen half is then scored once with that
floor and is the only specificity number that counts. Mind that the fit
half's fields have been seen by the classifier since chain 3, so its floor
is optimistic; the frozen half says by how much.

Replays the real `read_slide` loop through agent_eval's CachedStage, so the
policies here are the ones that ship.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_eval import CachedStage, CachedTools, reference
from autoscope.agent.loop import read_slide
from autoscope.agent.policy import DecisionRule, RuleBased, WhoProtocol
from autoscope.density import NEGATIVE_WBC_MIN, WBC_PER_UL_DEFAULT, band, density

OUT = Path(__file__).resolve().parent / "results"


def by_patient(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
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
            "stopped_because": trace.stopped_because, "escalated": tools.escalation is not None,
            "call": tools.decision.call if tools.decision else None,
            "band": tools.decision.band_name if tools.decision else None}


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random infected patient reads higher than a random clean one."""
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def slide_per_ul(fields: list[dict]) -> float:
    """What a slide reads at with every field counted."""
    p = sum(f["found_parasites"] for f in fields)
    w = sum(f["found_wbc"] for f in fields)
    return p / w * WBC_PER_UL_DEFAULT if w else 0.0


def decision_arm(neg: dict, pos: dict, floor: float, negative_wbc: int, refs: dict) -> dict:
    """Replay the decision rule on clean and infected slides and count the calls."""
    make = lambda: DecisionRule(floor, negative_wbc)  # noqa: E731
    neg_runs = [replay(f, make()) for _, f in sorted(neg.items())]
    pos_runs = [replay(f, make()) for _, f in sorted(pos.items())]
    calls = lambda runs, what: sum(r["call"] == what for r in runs)  # noqa: E731
    by_band = {}
    for r in pos_runs:
        b = band(refs[r["patient"]])
        by_band.setdefault(b, [0, 0])
        by_band[b][0] += 1
        by_band[b][1] += r["call"] == "positive"
    return {
        "floor_per_ul": floor, "negative_wbc_min": negative_wbc,
        "clean_slides": len(neg_runs),
        "specificity": round(calls(neg_runs, "negative") / len(neg_runs), 3),
        "clean_called_positive": calls(neg_runs, "positive"),
        "clean_to_a_person": sum(r["call"] not in ("negative", "positive") for r in neg_runs),
        "clean_fields_read": sum(r["fields_read"] for r in neg_runs),
        "infected_slides": len(pos_runs),
        "sensitivity": round(calls(pos_runs, "positive") / len(pos_runs), 3),
        "infected_called_negative": calls(pos_runs, "negative"),
        "infected_to_a_person": sum(r["call"] not in ("negative", "positive") for r in pos_runs),
        "infected_band_settled": sum(r["band"] is not None for r in pos_runs),
        "infected_fields_read": sum(r["fields_read"] for r in pos_runs),
        "sensitivity_by_expert_band": {int(k): {"slides": v[0], "called_positive": v[1]} for k, v in sorted(by_band.items())},
        "runs_uninfected": neg_runs, "runs_infected": pos_runs,
    }


def print_decision(tag: str, a: dict) -> None:
    print(f"  {tag:<28s} floor {a['floor_per_ul']:>6,.0f}/uL  N {a['negative_wbc_min']:>5d}   "
          f"spec {a['specificity']:.2f} ({a['clean_called_positive']} FP, {a['clean_to_a_person']} to a person)   "
          f"sens {a['sensitivity']:.2f} ({a['infected_called_negative']} FN, {a['infected_to_a_person']} to a person, "
          f"{a['infected_band_settled']} banded)   fields {a['clean_fields_read']} + {a['infected_fields_read']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", default="negative_fields.json")
    ap.add_argument("--positives", default="agent_fields.json")
    ap.add_argument("--target-ci", type=float, default=0.20)
    ap.add_argument("--min-fields", type=int, default=5)
    ap.add_argument("--out", default="negative_eval.json")
    ap.add_argument("--all", action="store_true",
                    help="score all 50 uninfected patients, not just the frozen test half")
    ap.add_argument("--floor", type=float, default=None,
                    help="noise floor /uL for the decision rule; default: --floor-quantile of the fit half's slides")
    ap.add_argument("--floor-quantile", type=float, default=95)
    ap.add_argument("--negative-wbc", type=int, default=NEGATIVE_WBC_MIN)
    ap.add_argument("--sweep", action="store_true", help="sweep the floor and the white cell minimum")
    args = ap.parse_args()

    neg_table = json.loads((OUT / args.negatives).read_text())
    all_rows = neg_table["rows"]
    split = json.loads((OUT / "uninfected_split.json").read_text())
    fit_half = by_patient([r for r in all_rows if r["patient"] in set(split["fit"])])
    # From 3 September 2026 the fit half of the uninfected set supplies
    # negative training patches, so it is no longer test material. Only the
    # frozen half is reported unless --all is asked for, and the output says which.
    if not args.all:
        test = set(split["test"])
        neg_table["rows"] = [r for r in all_rows if r["patient"] in test]
        neg_table["which"] = "frozen test half"
    else:
        neg_table["which"] = "all 50 patients"
    neg_table["patients"] = len({r["patient"] for r in neg_table["rows"]})
    print(f"uninfected patients scored: {neg_table['which']}")
    pos_table = json.loads((OUT / args.positives).read_text())
    neg, pos = by_patient(neg_table["rows"]), by_patient(pos_table["rows"])
    neg_rows = neg_table["rows"]
    refs = {p: reference(f) for p, f in pos.items()}
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
    ref_sorted = np.array(sorted(refs.values()))
    print("infected held-out patients by expert density: "
          f"min {ref_sorted[0]:,.0f}, p10 {np.percentile(ref_sorted, 10):,.0f}, median {np.median(ref_sorted):,.0f}/uL")

    # 3. The decision rule. Floor designed on the fit half, tested on the frozen half.
    fit_ul = np.array(sorted(slide_per_ul(f) for f in fit_half.values()))
    floor_stats = {"fit_half_slides": len(fit_ul),
                   "fit_half_per_ul": {"median": float(np.median(fit_ul)), "p90": float(np.percentile(fit_ul, 90)),
                                       "p95": float(np.percentile(fit_ul, 95)), "max": float(fit_ul.max())}}
    floor = args.floor if args.floor is not None else float(np.percentile(fit_ul, args.floor_quantile))
    print(f"\nnoise floor, fit half ({len(fit_ul)} slides, every field): median {np.median(fit_ul):,.0f}, "
          f"p90 {np.percentile(fit_ul, 90):,.0f}, p95 {np.percentile(fit_ul, 95):,.0f}, max {fit_ul.max():,.0f}/uL"
          f"  ->  floor {floor:,.0f}/uL" + (" (given)" if args.floor is not None else f" (p{args.floor_quantile:g})"))
    decision = decision_arm(neg, pos, floor, args.negative_wbc, refs)
    print(f"\ndecision rule on the {neg_table['which']} and the {len(pos)} infected held-out patients:")
    print_decision("shipped", decision)
    print("  sensitivity by expert band: " + ", ".join(
        f"band {k} {v['called_positive']}/{v['slides']}" for k, v in decision["sensitivity_by_expert_band"].items()))

    sweep = []
    if args.sweep:
        print("\nsweep (design numbers: pick on the fit half, not here):")
        floors = sorted({round(float(np.percentile(fit_ul, q))) for q in (90, 95, 100)} | {300, 500, 800, 1000, 1500})
        for n_wbc in (200, 500, 1000):
            for fl in floors:
                a = decision_arm(neg, pos, fl, n_wbc, refs)
                a.pop("runs_uninfected"); a.pop("runs_infected")
                sweep.append(a)
                print_decision("", a)

    (OUT / "decision_rule.json").write_text(json.dumps({
        "floor_per_ul": floor, "negative_wbc_min": args.negative_wbc,
        "floor_from": "given" if args.floor is not None else f"p{args.floor_quantile:g} of the fit half's slides, every field",
        "threshold": neg_table["threshold"], **floor_stats,
    }, indent=2))
    (OUT / args.out).write_text(json.dumps({
        "threshold": neg_table["threshold"], "uninfected_patients": len(neg), "which": neg_table["which"],
        "infected_patients": len(pos), "per_field": per_field, "arms": arms,
        "decision": decision, "decision_sweep": sweep, "floor": floor_stats,
        "infected_expert_density_percentiles": {k: float(np.percentile(ref_sorted, k)) for k in (0, 10, 25, 50)},
    }, indent=2))


if __name__ == "__main__":
    main()
