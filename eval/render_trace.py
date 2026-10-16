"""Print the real reading trace of one slide, from the replay cache.

The report quotes a trace. A trace that was assembled by hand to read well is
not evidence of anything, so this prints the one the agent actually produced,
step by step, from the recorded model calls. No AWS account, no cost.

    .venv/bin/python eval/render_trace.py TF89_CM5
    .venv/bin/python eval/render_trace.py TF89_CM5 --out docs/figures/agent-trace-example.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_eval import CachedStage, CachedTools
from autoscope.agent.bedrock import BedrockPolicy
from autoscope.agent.converse import ConversePolicy
from autoscope.agent.loop import read_slide
from autoscope.agent.replay import ReplayClient

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def compact(trace) -> str:
    """The same trace, narrow enough for a printed column.

    Trace.render pads its columns for a terminal. Wrapped lines in a PDF are
    harder to read than the padding is worth, and the stop reason appears twice
    there, once on the step and once in the tail.
    """
    lines = [f"slide: {trace.slide}"]
    for step in trace.steps:
        nums = "  ".join(f"{k}={v}" for k, v in step.measured.items())
        lines.append(f"{step.n:>2}  {step.where:<10} {step.action:<14} {nums}")
        lines.append(f"    {step.reason}")
    lines.append(f"read {trace.fields_read} fields, {trace.parasites} parasites "
                 f"in {trace.wbc} white cells")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patient")
    ap.add_argument("--source", default="found", choices=("found", "truth"))
    ap.add_argument("--model", default="",
                    help="a Converse model id; the default is the shipped "
                         "BedrockPolicy arm, the one the report quotes")
    ap.add_argument("--negative-wbc", type=int, default=500)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--cache", default="agent_bedrock_calls.jsonl")
    ap.add_argument("--fields", default="agent_fields.json")
    ap.add_argument("--floor", type=float, default=148.05007075743845)
    ap.add_argument("--out", default="")
    ap.add_argument("--compact", action="store_true",
                    help="narrow columns and drop the repeated stop reason, so "
                         "the trace fits a report column without wrapping")
    args = ap.parse_args()

    table = json.loads((RESULTS / args.fields).read_text())
    fields: dict[str, list[dict]] = {}
    for row in table["rows"]:
        fields.setdefault(row["patient"], []).append(row)
    if args.patient not in fields:
        raise SystemExit(f"no patient {args.patient}; have {len(fields)}")
    rows = fields[args.patient]

    # Exactly the arm agent_eval builds, or the key will not be in the cache:
    # the key hashes the prompt, the tool definitions and the model.
    cache = ReplayClient(RESULTS / args.cache)
    floor = 0.0 if args.source == "truth" else args.floor
    policy = (ConversePolicy(model=args.model, client=cache, floor_per_ul=floor,
                             negative_wbc_min=args.negative_wbc)
              if args.model else
              BedrockPolicy(client=cache, effort=args.effort, floor_per_ul=floor,
                            negative_wbc_min=args.negative_wbc))
    stage = CachedStage(rows)
    tools = CachedTools(stage, args.source)
    trace = read_slide(tools, policy, slide=args.patient, max_fields=len(rows))

    text = compact(trace) if args.compact else trace.render()
    print(text)
    print(f"\n{len(rows)} fields were available. "
          f"The WHO protocol reads to 200 white cells.")
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
