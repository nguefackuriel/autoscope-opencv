"""Limit of detection by dilution: how light an infection the decision rule
still calls positive.

Six of the 150 NLM patients are under 1 000/uL, so the sensitivity at the
densities a screening tool is for cannot be read off the patients. A
laboratory validating a new method makes a dilution series instead, and the
cached field tables let us do the same without touching a pixel: a
pseudo-slide is a shuffled mix of clean fields from the FROZEN half of the
uninfected set and a few sparse infected fields from the HELD-OUT patients,
with a known expert count, mixed to a target density. Both pools are test
material the classifier never trained on and the rule was never fitted on.

For each level (0, 50, 100, 200, 500, 1 000, 2 000/uL) it builds --repeats
pseudo-slides, replays the decision rule over each, and reports the share
called positive, negative, or sent to a person, and the fields read. Level 0
is pure clean pseudo-slides of the same length, which is the specificity of
the rule at that read length.

What it does not model: a real light slide's parasites are spread thinly
over every field, where here they sit in a few fields among clean ones. The
counts per white cell are the same; the per-field clustering is not, and the
rule reads the running total, so the difference is second order. It is
stated in the report.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from agent_eval import CachedStage, CachedTools
from autoscope.agent.loop import read_slide
from autoscope.agent.policy import DecisionRule
from autoscope.density import FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, SETTLE_CI_WIDTH, WBC_PER_UL_DEFAULT, density

OUT = Path(__file__).resolve().parent / "results"
LEVELS = (0, 50, 100, 200, 500, 1000, 2000)


def pseudo_slide(rng: random.Random, clean: list[dict], sparse: list[dict], target_per_ul: float,
                 min_parasites: int = 3, min_wbc: int = 600) -> list[dict]:
    """Infected fields until at least `min_parasites` expert parasites, then
    clean fields until the expert density falls to the target; at least
    `min_wbc` white cells so a negative can be called at all."""
    fields: list[dict] = []
    par = wbc = 0
    if target_per_ul > 0:
        # enough parasites that the target is reached at min_wbc, never fewer than min_parasites
        want = max(min_parasites, int(np.ceil(target_per_ul * min_wbc / WBC_PER_UL_DEFAULT)))
        pool = sparse[:]
        rng.shuffle(pool)
        for f in pool:
            fields.append(f); par += f["truth_parasites"]; wbc += f["truth_wbc"]
            if par >= want:
                break
        need_wbc = par / target_per_ul * WBC_PER_UL_DEFAULT
    else:
        need_wbc = 0
    pool = clean[:]
    rng.shuffle(pool)
    for f in pool:
        if wbc >= max(need_wbc, min_wbc):
            break
        fields.append(f); wbc += f["truth_wbc"]
    rng.shuffle(fields)
    return [dict(f, patient=f"pseudo-{target_per_ul:g}") for f in fields]


def expert_per_ul(fields: list[dict]) -> float:
    p = sum(f["truth_parasites"] for f in fields); w = sum(f["truth_wbc"] for f in fields)
    return p / w * WBC_PER_UL_DEFAULT if w else 0.0


def replay(fields: list[dict], policy) -> dict:
    stage = CachedStage(fields)
    tools = CachedTools(stage, "found")
    trace = read_slide(tools, policy, slide=fields[0]["patient"], max_fields=len(fields))
    d = density(tools.parasites, max(tools.wbc, 1))
    return {"fields": len(fields), "fields_read": tools.fields_read, "expert_per_ul": round(expert_per_ul(fields)),
            "found_per_ul": round(d.per_ul), "wbc": tools.wbc, "parasites": tools.parasites,
            "call": tools.decision.call if tools.decision else None,
            "escalated": tools.escalation is not None, "stopped_because": trace.stopped_because}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", default="negative_fields.json")
    ap.add_argument("--positives", default="agent_fields.json")
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--sparse-max", type=int, default=12, help="an infected field qualifies with at most this many expert parasites")
    ap.add_argument("--floor", type=float, default=None, help="default: decision_rule.json")
    ap.add_argument("--negative-wbc", type=int, default=NEGATIVE_WBC_MIN)
    ap.add_argument("--settle-ci", type=float, default=SETTLE_CI_WIDTH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dilution_eval.json")
    args = ap.parse_args()
    if args.floor is None:
        rule = OUT / "decision_rule.json"
        args.floor = json.loads(rule.read_text())["floor_per_ul"] if rule.exists() else FLOOR_PER_UL_DEFAULT

    split = json.loads((OUT / "uninfected_split.json").read_text())
    frozen = set(split["test"])
    neg = json.loads((OUT / args.negatives).read_text())
    clean = [r for r in neg["rows"] if r["patient"] in frozen and r["truth_wbc"] > 0]
    pos = json.loads((OUT / args.positives).read_text())
    sparse = [r for r in pos["rows"] if 1 <= r["truth_parasites"] <= args.sparse_max and r["truth_wbc"] > 0]
    print(f"clean pool: {len(clean)} frozen-half fields from {len({r['patient'] for r in clean})} patients; "
          f"sparse infected pool: {len(sparse)} held-out fields (1 to {args.sparse_max} expert parasites) "
          f"from {len({r['patient'] for r in sparse})} patients")
    print(f"decision rule: floor {args.floor:,.0f}/uL, negative after {args.negative_wbc} white cells, "
          f"settle {args.settle_ci:.0%}; {args.repeats} pseudo-slides per level\n")

    rng = random.Random(args.seed)
    levels = []
    print(f"{'target /uL':>10s} {'realised':>9s} {'fields':>7s} {'read':>5s} {'positive':>9s} {'negative':>9s} {'person':>7s} {'found /uL':>10s}")
    for level in LEVELS:
        runs = []
        for _ in range(args.repeats):
            fields = pseudo_slide(rng, clean, sparse, level, min_wbc=max(args.negative_wbc + 100, 600))
            runs.append(replay(fields, DecisionRule(args.floor, args.negative_wbc, settle_ci_width=args.settle_ci)))
        calls = lambda what: sum(r["call"] == what for r in runs) / len(runs)  # noqa: E731
        row = {"target_per_ul": level,
               "realised_per_ul_median": float(np.median([r["expert_per_ul"] for r in runs])),
               "fields_median": float(np.median([r["fields"] for r in runs])),
               "fields_read_median": float(np.median([r["fields_read"] for r in runs])),
               "positive": calls("positive"), "negative": calls("negative"),
               "to_a_person": sum(r["call"] not in ("positive", "negative") for r in runs) / len(runs),
               "found_per_ul_median": float(np.median([r["found_per_ul"] for r in runs])),
               "runs": runs}
        levels.append(row)
        print(f"{level:>10d} {row['realised_per_ul_median']:>9.0f} {row['fields_median']:>7.0f} {row['fields_read_median']:>5.0f} "
              f"{row['positive']:>9.0%} {row['negative']:>9.0%} {row['to_a_person']:>7.0%} {row['found_per_ul_median']:>10.0f}")

    lod = next((r["target_per_ul"] for r in levels if r["target_per_ul"] > 0 and r["positive"] >= 0.95), None)
    print(f"\nlimit of detection at 95% sensitivity: {lod if lod else 'above 2 000'}/uL; "
          f"specificity at level 0: {levels[0]['negative']:.0%} called negative")
    (OUT / args.out).write_text(json.dumps({
        "floor_per_ul": args.floor, "negative_wbc_min": args.negative_wbc, "settle_ci": args.settle_ci,
        "repeats": args.repeats, "sparse_max": args.sparse_max, "seed": args.seed,
        "clean_pool_fields": len(clean), "sparse_pool_fields": len(sparse),
        "lod_95_per_ul": lod, "levels": levels,
    }, indent=2))


if __name__ == "__main__":
    main()
