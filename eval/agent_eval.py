"""Does the agent help?

The question the proposal promised to answer: at the same quality of answer,
does the stopping rule read fewer fields than the WHO protocol, and does it
keep reading when the slide is light and the fixed method would stop too early?

HOW THIS RUNS

The real `read_slide` loop, the real policies, the real stopping rule. The only
thing replaced is where the numbers come from: `agent_fields.py` counted every
field of every held-out patient once, and this replays sessions over that
table. A field returns the same counts whoever asked for it, so caching changes
no decision, and it turns a two-hour experiment into a one-second one. There is
no second copy of the policy logic to drift.

Two counters, same sessions:

  truth   what the experts annotated. Isolates the agent.
  found   what the pipeline found. What a user would get.

Reporting only one would hide which half is at fault.

The reference density for a patient is its own expert annotation over ALL its
fields. That is what "the right answer" means here.

WHAT THIS CANNOT MEASURE

Refocus. The NLM set has one capture per field, so a policy can decide to
refocus and we can log it, but no count changes as a result. Any claim about
recovering a soft field is out of reach on this data and is not made.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from autoscope.agent.loop import read_slide
from autoscope.agent.bedrock import DEFAULT_MODEL, BedrockPolicy
from autoscope.agent.policy import DecisionRule, RuleBased, WhoProtocol
from autoscope.agent.replay import RecordingClient, ReplayClient, cost_of
from autoscope.agent.tools import Tools
from autoscope.density import (FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, SETTLE_CI_WIDTH,
                               WBC_PER_UL_DEFAULT, decide, density)
from autoscope.quality import FieldQuality
from autoscope.virtual_stage import Frame

OUT = Path(__file__).resolve().parent / "results"
_DUMMY = np.zeros((1, 1, 3), np.uint8)


class CachedStage:
    """One patient's fields in order. No pixels, just their identity.

    The NLM fields do not overlap (timestamps a minute apart), so a patient is
    a row of fields, not a grid to wander in.
    """

    def __init__(self, fields: list[dict]) -> None:
        self.fields = fields
        self._x = self._y = self._z = 0
        self.visited: list[tuple[int, int]] = []
        self.refocus_calls = 0

    @property
    def grid(self) -> tuple[int, int]:
        return (len(self.fields), 1)

    def current(self) -> Frame:
        return Frame(_DUMMY, self._x, self._y, self._z, synthetic=False)

    def move_stage(self, dx: int, dy: int) -> Frame:  # noqa: ARG002
        self._x = int(np.clip(self._x + dx, 0, len(self.fields) - 1))
        self.visited.append((self._x, self._y))
        return self.current()

    def refocus(self, dz: int) -> Frame:
        self._z += dz
        self.refocus_calls += 1
        return self.current()

    def at_edge(self) -> bool:
        return self._x >= len(self.fields) - 1


class CachedTools(Tools):
    """The real Tools, with the two measuring calls served from the table."""

    def __init__(self, stage: CachedStage, source: str) -> None:
        super().__init__(stage, counter=lambda image: (0, 0))
        self.source = source          # "found" or "truth"

    def _row(self) -> dict:
        return self.stage.fields[self.stage._x]  # noqa: SLF001

    def assess_field(self) -> FieldQuality:
        row = self._row()
        return FieldQuality(
            focus_score=row["focus"],
            illumination_uniformity=row["even"],
            cell_density=0.04,        # measured median; never the binding gate
            debris_ratio=row["debris"],
        )

    def count_field(self) -> tuple[int, int]:
        row = self._row()
        parasites = row[f"{self.source}_parasites"]
        wbc = row[f"{self.source}_wbc"]
        self.parasites += parasites
        self.wbc += wbc
        self.fields_read += 1
        return parasites, wbc


def reference(fields: list[dict]) -> float:
    p = sum(f["truth_parasites"] for f in fields)
    w = sum(f["truth_wbc"] for f in fields)
    return p / w * WBC_PER_UL_DEFAULT if w else 0.0


def run_one(fields: list[dict], policy, source: str,
            floor: float = 0.0, negative_wbc: int = NEGATIVE_WBC_MIN) -> dict:
    stage = CachedStage(fields)
    tools = CachedTools(stage, source)
    trace = read_slide(tools, policy, slide=fields[0]["patient"], max_fields=len(fields))
    d = density(tools.parasites, max(tools.wbc, 1))

    # THE CALL IS DERIVED, NOT TAKEN FROM THE POLICY.
    #
    # `decide` reads the counts and nothing else, so every arm gets its call
    # from the same function and the arms are comparable. A policy chooses when
    # to stop; what the numbers then say is not its to decide. For the agent
    # that is also the safety property we claim in writing — it picks where to
    # look and when to stop, never the diagnosis.
    #
    # `call`/`band` below stay exactly as they were: policy-set, None for every
    # arm but the decision rule. They are what results-agent.md already
    # publishes (49 positives, 40 settled bands), and redefining a published
    # column in place would move those numbers with nothing to show for it.
    derived = decide(d, floor, negative_wbc)

    return {
        "patient": fields[0]["patient"], "policy": policy.name, "source": source,
        "fields_read": tools.fields_read, "fields_available": len(fields),
        "wbc": tools.wbc, "parasites": tools.parasites,
        "per_ul": round(d.per_ul, 1), "ci_width": round(d.ci_relative_width, 4),
        "refocus_calls": stage.refocus_calls,
        "stopped_because": trace.stopped_because,
        "escalated": tools.escalation is not None,
        "call": tools.decision.call if tools.decision else None,
        "band": tools.decision.band_name if tools.decision else None,
        "derived_call": derived.call,
        "derived_band": derived.band_name,
        # What the model said its own verdict was, when it stopped. None for
        # every policy that is not a model. Kept so the two can be compared.
        "agent_call": getattr(policy, "last_call", None),
        # Token counts come off the recorded calls, so a run dict is identical
        # whether it was recorded or replayed. That is what makes "replay is
        # byte-identical to the recording" a testable claim.
        **_model_cost(policy),
    }


def _model_cost(policy) -> dict:
    """Empty for a policy that is not a model, so the rule arms' run dicts keep
    exactly the shape already committed in eval/results/agent_eval.json."""
    calls = getattr(policy, "calls", None)
    if not calls:
        return {}
    return {
        "model_calls": len(calls),
        "input_tokens": sum((c["usage"].get("input_tokens") or 0) for c in calls),
        "output_tokens": sum((c["usage"].get("output_tokens") or 0) for c in calls),
    }


def summarise(runs: list[dict], refs: dict[str, float], target: float) -> dict:
    f = np.array([r["fields_read"] for r in runs], float)
    ci = np.array([r["ci_width"] for r in runs], float)
    ours = np.array([r["per_ul"] for r in runs], float)
    ref = np.array([refs[r["patient"]] for r in runs], float)
    keep = (ours > 0) & (ref > 0)
    ld = np.log(ours[keep]) - np.log(ref[keep])
    return {
        "patients": len(runs),
        "fields_read_total": int(f.sum()),
        "fields_read_median": float(np.median(f)),
        "fields_available_total": int(sum(r["fields_available"] for r in runs)),
        "ci_width_median": round(float(np.median(ci)), 4),
        "reached_target": int((ci <= target).sum()),
        "density_bias_pct": round(float((np.exp(ld.mean()) - 1) * 100), 2) if keep.any() else None,
        "density_loa_pct": [round(float((np.exp(ld.mean() - 1.96 * ld.std()) - 1) * 100), 1),
                            round(float((np.exp(ld.mean() + 1.96 * ld.std()) - 1) * 100), 1)] if keep.any() else None,
        "density_mape_pct": round(float(np.abs(np.exp(np.abs(ld)) - 1).mean() * 100), 1) if keep.any() else None,
        "escalated": int(sum(r["escalated"] for r in runs)),
        "compared_on": int(keep.sum()),
        "called_positive": int(sum(r.get("call") == "positive" for r in runs)),
        "called_negative": int(sum(r.get("call") == "negative" for r in runs)),
        "band_settled": int(sum(r.get("band") is not None for r in runs)),
        # Derived from the counts where each arm stopped, so every arm has one
        # and they can be compared with each other.
        "derived_positive": int(sum(r.get("derived_call") == "positive" for r in runs)),
        "derived_negative": int(sum(r.get("derived_call") == "negative" for r in runs)),
        "derived_undecided": int(sum(r.get("derived_call") == "undecided" for r in runs)),
        "derived_band_settled": int(sum(r.get("derived_band") is not None for r in runs)),
        # Only meaningful for a model arm: how often its own verdict matches
        # what its own counts say. A gap is a finding, not a bug to hide.
        "agent_calls_given": int(sum(r.get("agent_call") is not None for r in runs)),
        "agent_call_agreed": int(sum(
            r.get("agent_call") is not None
            and _same_call(r["agent_call"], r.get("derived_call"))
            for r in runs)),
        "model_calls": int(sum(r.get("model_calls", 0) for r in runs)),
        "input_tokens": int(sum(r.get("input_tokens", 0) for r in runs)),
        "output_tokens": int(sum(r.get("output_tokens", 0) for r in runs)),
    }


def _same_call(agent: str, derived: str | None) -> bool:
    """The model answers positive / negative / settled; `decide` answers
    positive / negative / undecided. "settled" is the model saying the interval
    sits in one band, which is a positive that has converged, so it counts as
    agreement with a positive and never with a negative."""
    if agent == "settled":
        return derived == "positive"
    return agent == derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ci", type=float, default=0.20)
    ap.add_argument("--min-fields", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="sweep min_fields and target error bar")
    ap.add_argument("--fields", default="agent_fields.json")
    ap.add_argument("--out", default="agent_eval.json")
    ap.add_argument("--floor", type=float, default=None,
                    help="noise floor /uL for the decision rule with the real counter; default: decision_rule.json if present")
    ap.add_argument("--negative-wbc", type=int, default=NEGATIVE_WBC_MIN)
    ap.add_argument("--settle-ci", type=float, default=SETTLE_CI_WIDTH,
                    help="a positive with a settled band stops once its error bar is within this")
    ap.add_argument("--sweep-settle", action="store_true", help="sweep the settle width for the decision arms")
    ap.add_argument("--policies", default="who,rule,decision",
                    help="comma-separated: who, rule, decision, bedrock. The default is the "
                         "three that cost nothing; bedrock is opt-in because it calls a model.")
    ap.add_argument("--bedrock-cache", default="agent_bedrock_calls.jsonl",
                    help="the recorded model calls, under eval/results/")
    ap.add_argument("--record", action="store_true",
                    help="CALLS BEDROCK AND SPENDS MONEY. Records every call to the cache. "
                         "Without this the bedrock arm replays from the cache and costs nothing.")
    ap.add_argument("--allow-live", action="store_true",
                    help="with --record, let a cache miss make a live call; on its own, extend "
                         "an existing recording instead of failing on a miss")
    ap.add_argument("--bedrock-model", default=DEFAULT_MODEL)
    ap.add_argument("--bedrock-effort", default="low")
    args = ap.parse_args()

    args.policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    known = {"who", "rule", "decision", "bedrock"}
    unknown = set(args.policies) - known
    if unknown:
        ap.error(f"unknown policies {sorted(unknown)}; pick from {sorted(known)}")

    # Two guards against buying something nobody asked for.
    if args.record and "bedrock" not in args.policies:
        ap.error("--record only means anything with bedrock in --policies")
    if args.record and (args.sweep or args.sweep_settle):
        ap.error("--record with a sweep would pay for the same experiment nine times over, "
                 "and the sweep varies two knobs the agent never reads. Record once, then sweep "
                 "the rule arms.")
    if args.floor is None:
        rule_path = OUT / "decision_rule.json"
        args.floor = json.loads(rule_path.read_text())["floor_per_ul"] if rule_path.exists() else FLOOR_PER_UL_DEFAULT

    cache = None
    if "bedrock" in args.policies:
        cache_path = OUT / args.bedrock_cache
        if args.record:
            from anthropic import AnthropicBedrockMantle
            live = AnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION_NAME", "us-east-1"))
            cache = RecordingClient(live, cache_path)
            print(f"RECORDING to {cache_path.name}: this calls Bedrock and spends money.\n"
                  f"{len(cache._seen)} calls already on disk will be reused.\n")  # noqa: SLF001
        else:
            live = None
            if args.allow_live:
                from anthropic import AnthropicBedrockMantle
                live = AnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION_NAME", "us-east-1"))
            try:
                cache = ReplayClient(cache_path, allow_live=live)
            except FileNotFoundError as missing:
                # The first thing a judge with no recording will see. A
                # traceback here would read as a broken repository rather than
                # as a missing artefact.
                raise SystemExit(str(missing)) from None
            print(f"replaying {len(cache.rows)} recorded model calls from {cache_path.name}, "
                  f"no network and no cost\n")

    table = json.loads((OUT / args.fields).read_text())
    by_patient: dict[str, list[dict]] = {}
    for row in table["rows"]:
        by_patient.setdefault(row["patient"], []).append(row)
    refs = {p: reference(f) for p, f in by_patient.items()}

    def arms(target: float, min_fields: int, settle: float | None = None,
             names: tuple[str, ...] | None = None) -> dict:
        settle = args.settle_ci if settle is None else settle
        names = tuple(args.policies) if names is None else names
        out = {}
        for source in ("truth", "found"):
            # With expert counts there are no false alarms to clear, so the
            # decision rule's floor is 0 there; with the real counter it is
            # the measured noise floor (negative_eval.py, fit half).
            floor = 0.0 if source == "truth" else args.floor
            builders = {
                "who": lambda: WhoProtocol(200),
                "rule": lambda: RuleBased(target, min_fields=min_fields),
                "decision": lambda: DecisionRule(floor, args.negative_wbc, target,
                                                 min_fields=min_fields, settle_ci_width=settle),
                "bedrock": lambda: BedrockPolicy(client=cache, model=args.bedrock_model,
                                                 effort=args.bedrock_effort,
                                                 floor_per_ul=floor,
                                                 negative_wbc_min=args.negative_wbc),
            }
            for name in names:
                runs = [run_one(f, builders[name](), source, floor, args.negative_wbc)
                        for _, f in sorted(by_patient.items())]
                out[f"{runs[0]['policy']}/{source}"] = {"summary": summarise(runs, refs, target),
                                                        "runs": runs}
        return out

    print(f"{len(by_patient)} held-out patients, {len(table['rows'])} annotated fields, "
          f"{sum(len(f) for f in by_patient.values())} fields available\n")

    if args.sweep:
        print(f"{'target':>7s} {'min':>4s}  {'arm':<16s} {'fields':>7s} {'err bar':>8s} "
              f"{'bias':>8s} {'|err|':>7s} {'at target':>10s}")
        sweep = []
        # Bedrock is excluded here, not merely skipped when printing. arms()
        # RUNS every policy it is handed at each of the nine grid points, so
        # leaving it in would mean nine agent runs — and the two knobs being
        # swept, target error bar and minimum fields, are read by the rule
        # policies and not by the model.
        swept = tuple(n for n in args.policies if n != "bedrock")
        if not swept:
            ap.error("nothing to sweep: the sweep varies knobs only the rule policies read")
        for target in (0.15, 0.20, 0.30):
            for min_fields in (3, 5, 8):
                a = arms(target, min_fields, names=swept)
                for name, arm in a.items():
                    s = arm["summary"]
                    if name.startswith("who-200") and (target, min_fields) != (0.15, 3):
                        continue    # the baseline does not depend on these
                    sweep.append({"target_ci": target, "min_fields": min_fields,
                                  "arm": name, **s})
                    print(f"{target:>7.2f} {min_fields:>4d}  {name:<16s} {s['fields_read_total']:>7d} "
                          f"{s['ci_width_median']:>7.0%} {s['density_bias_pct']:>+7.1f}% "
                          f"{s['density_mape_pct']:>6.1f}% {s['reached_target']:>7d}/{s['patients']}")
        (OUT / "agent_sweep.json").write_text(json.dumps(sweep, indent=2))
        return

    if args.sweep_settle:
        print(f"{'settle':>7s}  {'arm':<16s} {'fields':>7s} {'err bar':>8s} {'bias':>8s} {'|err|':>7s} {'flagged':>8s} {'banded':>7s}")
        rows = []
        for settle in (1.0, 0.5, 0.3, 0.2):
            for name, arm in arms(args.target_ci, args.min_fields, settle).items():
                if not name.startswith("decision"):
                    continue
                s = arm["summary"]
                rows.append({"settle_ci": settle, "arm": name, **s})
                print(f"{settle:>7.2f}  {name:<16s} {s['fields_read_total']:>7d} {s['ci_width_median']:>7.0%} "
                      f"{s['density_bias_pct']:>+7.1f}% {s['density_mape_pct']:>6.1f}% {s['escalated']:>8d} {s['band_settled']:>7d}")
        (OUT / "agent_settle_sweep.json").write_text(json.dumps(rows, indent=2))
        return

    a = arms(args.target_ci, args.min_fields)
    report = {"target_ci": args.target_ci, "min_fields": args.min_fields,
              "patients": len(by_patient), "fields_total": len(table["rows"]),
              "threshold": table.get("threshold"),
              "decision_floor_per_ul": args.floor, "decision_negative_wbc": args.negative_wbc,
              "decision_settle_ci": args.settle_ci,
              "arms": {k: v["summary"] for k, v in a.items()},
              "runs": {k: v["runs"] for k, v in a.items()}}

    if cache is not None:
        report["bedrock"] = {
            "cache": args.bedrock_cache,
            "mode": "record" if args.record else "replay",
            "model": args.bedrock_model,
            "effort": args.bedrock_effort,
            "recorded_this_run": getattr(cache, "recorded", 0),
            "served_from_cache": getattr(cache, "served_from_cache", 0) + getattr(cache, "hits", 0),
            "live_calls": getattr(cache, "live", 0) + getattr(cache, "recorded", 0),
        }
    (OUT / args.out).write_text(json.dumps(report, indent=2))

    print(f"target error bar {args.target_ci:.0%}, minimum {args.min_fields} fields; "
          f"decision rule: floor {args.floor:,.0f}/uL (0 with expert counts), negative after {args.negative_wbc} white cells\n")
    print(f"{'arm':<18s} {'fields':>7s} {'median':>7s} {'err bar':>8s} {'bias':>8s} "
          f"{'|err|':>7s} {'at target':>10s} {'flagged':>8s} {'positive':>9s} {'negative':>9s} {'band':>5s}")
    for name, arm in a.items():
        s = arm["summary"]
        calls = (f"{s['called_positive']:>9d} {s['called_negative']:>9d} {s['band_settled']:>5d}"
                 if name.startswith(("decision", "bedrock")) else f"{'':>9s} {'':>9s} {'':>5s}")
        print(f"{name:<18s} {s['fields_read_total']:>7d} {s['fields_read_median']:>7.0f} "
              f"{s['ci_width_median']:>7.0%} {s['density_bias_pct']:>+7.1f}% "
              f"{s['density_mape_pct']:>6.1f}% {s['reached_target']:>7d}/{s['patients']} "
              f"{s['escalated']:>8d} {calls}")


if __name__ == "__main__":
    main()
