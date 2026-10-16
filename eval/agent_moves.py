"""What each model actually chose, counted from the recorded calls.

    .venv/bin/python eval/agent_moves.py

The arm table says how many fields a policy read and how many slides it sent to
a person. It does not say WHY. This reads every recorded call in
agent_bedrock_calls.jsonl and counts the move each model picked, so a claim like
"Nova Pro never chose stop" is a count from the cache and not an inference from
an escalation column.

Two runs per slide are in the cache (the shipped counter and the oracle), so a
model that stops on every slide shows about 100 stops over 50 patients.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
MOVES = ("count_then_move", "stop", "flag_for_human", "skip", "refocus")


def counts(cache: Path = RESULTS / "agent_bedrock_calls.jsonl") -> dict[str, collections.Counter]:
    by_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in cache.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        move = d.get("move") or {}
        by_model[d.get("model", "?")][move.get("name", "?")] += 1
    return by_model


def main() -> None:
    by_model = counts()
    w = max(len(m) for m in by_model)
    print(f"{'model':<{w}}" + "".join(f"{m:>17}" for m in MOVES) + f"{'calls':>8}")
    for model, c in sorted(by_model.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{model:<{w}}" + "".join(f"{c.get(m, 0):>17}" for m in MOVES)
              + f"{sum(c.values()):>8}")
    print()
    for model, c in by_model.items():
        if c.get("stop", 0) == 0:
            print(f"{model}: honours forced tool choice and never once chose stop "
                  f"in {sum(c.values())} calls. Forced tool choice is necessary and not sufficient.")


if __name__ == "__main__":
    main()
