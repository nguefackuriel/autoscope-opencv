"""Every figure the technical report quotes, read from the result files.

    python docs/report/report_numbers.py              # print them, to check a run
    from report_numbers import figures; figures()     # what build_report.py calls

NOT named numbers.py: build_report.py puts this directory on sys.path, and a
module called `numbers` there shadows the standard library's, which breaks
`decimal` and so `statistics` with a baffling AttributeError.

WHY THIS EXISTS AND IS NOT A LIST OF CONSTANTS IN THE REPORT

The report is the one document a judge reads instead of the repository, so a
number that is right in `eval/results/` and stale in the report is the worst
failure mode available to us: it is invisible, it survives review, and it makes
every other number suspect. Nothing in the report is typed by hand. Each figure
below names the file and the key it came from, and `build_report.py` substitutes
them, so re-running the evaluation and rebuilding the PDF cannot disagree.

A missing key RAISES. An earlier draft returned "??" for anything absent, which
produced a report that looked finished and quoted nothing; a crash with the list
of what is missing is the more useful outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "results"
BENCH = ROOT / "bench" / "results"


class Missing(Exception):
    """A figure the report quotes is not in the result files."""


def _load(path: Path) -> dict:
    if not path.exists():
        raise Missing(f"{path.relative_to(ROOT)} does not exist — run the chain")
    return json.loads(path.read_text())


def _dig(d: dict, *keys, src: str):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise Missing(f"{src}: no key {' -> '.join(map(str, keys))}")
        cur = cur[k]
    return cur


def figures() -> dict:
    """Flat dict of every figure, keyed by the name the template uses."""
    held = _load(EVAL / "parasite_eval_holdout_cnn.json")
    neg = _load(EVAL / "negative_eval.json")
    bands = _load(EVAL / "density_bands.json")
    rule = _load(EVAL / "decision_rule.json")
    dil = _load(EVAL / "dilution_eval.json")
    thr = _load(EVAL / "threshold_fixedstats.json")
    cal = _load(EVAL / "parasite_calibration.json")
    agent = _load(EVAL / "agent_eval.json")
    tz = _load(EVAL / "tanzania_eval.json")
    pv = _load(EVAL / "vivax_eval.json")

    dec = _dig(neg, "decision", src="negative_eval.json")
    per_field = _dig(neg, "per_field", src="negative_eval.json")

    f = {
        # --- the detector, held-out patients the fit never saw
        "held_patients": held["patients"],
        "held_images": held["images"],
        "held_truth": held["parasites_in_truth"],
        "held_precision": held["precision"],
        "held_recall": held["recall"],
        "held_f1": held["f1"],
        "held_count_bias_pct": held["count_bias_pct"],
        "held_ratio_bias_pct": held["ratio_bias_pct"],
        "held_ratio_spearman": held["ratio_spearman"],
        "threshold": thr["threshold"],
        "candidate_recall": cal["candidate_recall"],

        # --- density agreement with the expert count
        "density_slides": bands["slides"],
        "density_mae_pct": bands["mean_abs_density_error_pct"],
        "density_within_25": bands["within_25_pct"],
        "density_within_50": bands["within_50_pct"],

        # --- specificity, on the frozen uninfected half
        "neg_which": neg["which"],
        "neg_patients": neg["uninfected_patients"],
        "neg_fields": per_field["fields"],
        "neg_clean_fields_pct": per_field["clean_fields_pct"],
        "neg_false_per_200": per_field["false_per_200_wbc_pooled"],
        "neg_false_max": per_field["false_parasites_max"],

        # --- the decision rule, which is what a user actually gets
        "floor_per_ul": rule["floor_per_ul"],
        "negative_wbc_min": rule["negative_wbc_min"],
        "dec_clean_slides": dec["clean_slides"],
        "dec_clean_called_positive": dec["clean_called_positive"],
        "dec_clean_to_person": dec["clean_to_a_person"],
        "dec_specificity": dec["specificity"],
        "dec_infected_slides": dec["infected_slides"],
        "dec_sensitivity": dec["sensitivity"],
        "dec_infected_called_negative": dec["infected_called_negative"],
        "dec_infected_to_person": dec["infected_to_a_person"],
        "dec_band_settled": dec["infected_band_settled"],
        "lod_per_ul": dil["lod_95_per_ul"],
        "lod_repeats": dil["repeats"],

        # --- generalisation, each run once
        "tz_infected_images": tz["infected_images"],
        "tz_uninfected_images": tz["uninfected_images"],
        "tz_auc": tz["auc_parasites_per_image"],
        "tz_auc_density": tz["auc_density"],
        "tz_uninfected_at_zero": tz["uninfected_at_zero_pct"],
        "tz_commit": tz["commit"],
        "pv_patients": pv["patients_scored"],
        "pv_images": pv["images_scored"],
        "pv_f1": pv["f1_overall"],
        "pv_recall": pv["recall_overall"],
        "pv_precision": pv["precision_overall"],
        "pv_score_matched": pv["score_median_matched"],
        "pv_score_unmatched": pv["score_median_unmatched"],
        "pv_under10": pv["patients_under_10pc_recall"],

        # --- the agent
        "agent_patients": agent["patients"],
        "agent_fields_total": agent["fields_total"],
        "agent_model": _dig(agent, "bedrock", "model", src="agent_eval.json"),
        "agent_calls_cached": _dig(agent, "bedrock", "served_from_cache",
                                   src="agent_eval.json"),
        "agent_calls_live": _dig(agent, "bedrock", "live_calls", src="agent_eval.json"),
    }

    # The durable figure is the size of the replay cache, not one run's
    # cache-hit split: the split changes every time a prompt edit invalidates a
    # key, while the cache is what actually ships and what a judge replays.
    # Counted from the file rather than copied out of a summary, because the
    # file is the artefact being described.
    cache = EVAL / "agent_bedrock_calls.jsonl"
    if not cache.exists():
        raise Missing("eval/results/agent_bedrock_calls.jsonl is missing — "
                      "the agent comparison cannot be replayed without it")
    models = set()
    n = 0
    for line in cache.read_text().splitlines():
        if not line.strip():
            continue
        n += 1
        models.add(json.loads(line).get("model", "?"))
    f["cache_calls"] = n
    f["cache_models"] = len(models)

    # The agent table. `found` is the shipped counter; `truth` is the oracle arm
    # and is reported separately because it is the ceiling, not the product.
    for arm in ("who-200", "rule-based", "decision", "bedrock",
                "gpt-oss-120b-1", "llama4-maverick-17b-instruct-v1"):
        for counter in ("found", "truth"):
            key = f"{arm}/{counter}"
            if key not in agent["arms"]:
                raise Missing(f"agent_eval.json: no arm {key}")
            a = agent["arms"][key]
            slug = arm.replace("-", "_").replace(".", "_")
            f[f"arm_{slug}_{counter}_fields"] = a["fields_read_total"]
            f[f"arm_{slug}_{counter}_mape"] = a["density_mape_pct"]
            f[f"arm_{slug}_{counter}_escalated"] = a["escalated"]
            f[f"arm_{slug}_{counter}_settled"] = a["derived_band_settled"]
            f[f"arm_{slug}_{counter}_positive"] = a["derived_positive"]

    # --- COOL, four arms. The build lines are the proof of which binary ran.
    for label in ("c8g-stock", "c8g-no-kleidicv", "c8g-cool", "c7i-stock"):
        tile = _load(BENCH / f"{label}.json")
        slug = label.replace("-", "_")
        f[f"cool_{slug}_opencv"] = tile["opencv_version"]
        f[f"cool_{slug}_kleidicv"] = tile["kleidicv_in_build"]
        f[f"cool_{slug}_tile"] = {k: v["median_ms"] for k, v in tile["results"].items()}
        prof = BENCH / f"profile-{label}.json"
        if prof.exists():
            p = _load(prof)
            f[f"cool_{slug}_per_field"] = p["seconds_per_field_median"]
            f[f"cool_{slug}_slides_hour"] = p["slides_per_hour"]
            f[f"cool_{slug}_stages"] = p["stages"]

    cool_tile = _load(BENCH / "c8g-cool.json")
    f["cool_build_lines"] = cool_tile["build_lines"]
    f["cool_stock_build_lines"] = _dig(
        _load(BENCH / "c8g-stock-pipeline.json"), "build_lines",
        src="c8g-stock-pipeline.json")

    # the 12 MP thread sweep: the measurement that explains the pipeline gap
    for label in ("c8g-cool", "c8g-stock"):
        d = _load(BENCH / f"{label}-pipeline.json")
        slug = label.replace("-", "_")
        for n in ("0", "1"):
            block = _dig(d, "by_threads", n, src=f"{label}-pipeline.json")
            f[f"{slug}_12mp_t{n}"] = {k: v["median_ms"] for k, v in block.items()}
    return f


def main() -> None:
    f = figures()
    for k, v in f.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for a, b in v.items():
                print(f"    {a:<26}{b}")
        else:
            print(f"{k:<40}{v}")
    print(f"\n{len(f)} figures, none typed by hand")


if __name__ == "__main__":
    main()
