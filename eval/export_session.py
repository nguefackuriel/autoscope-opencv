"""Export one real reading session in the shape the API serves.

The demo page has to work with nothing deployed, so it falls back to a recorded
session sitting beside it. That recording is not a fixture somebody typed: it
is the decision rule replayed over the cached field table, which holds what
OpenCV actually measured on 609 held-out fields across 50 patients
(`eval/results/agent_fields.json`). Every focus score, every count and every
decision in the exported file was produced by the pipeline.

    python eval/export_session.py --patient TF101_233 --out infra/site/sample-session.json
    python eval/export_session.py --list

Replaying from the cache rather than re-reading the images is the same thing
`agent_eval.py` does, and for the same reason: the vision is expensive and
deterministic, so the policy can be swapped and compared in a second.
"""

import argparse
import json
from pathlib import Path

from autoscope.agent.bedrock import observation
from autoscope.agent.policy import DecisionRule
from autoscope.density import (FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, density,
                               should_stop, should_stop_decision)
from autoscope.quality import FieldQuality

RESULTS = Path(__file__).resolve().parent / "results"


class ReplayTools:
    """The slice of Tools a policy touches, fed from the cached rows."""

    def __init__(self, rows, wbc_per_ul: float = 8000.0):
        self.rows = rows
        self.wbc_per_ul = wbc_per_ul
        self.parasites = 0
        self.wbc = 0
        self.fields_read = 0
        self.escalation = None
        self.decision = None
        self._i = 0

    @property
    def row(self):
        return self.rows[min(self._i, len(self.rows) - 1)]

    def quality(self) -> FieldQuality:
        r = self.row
        return FieldQuality(focus_score=r["focus"], illumination_uniformity=r["even"],
                            cell_density=0.0, debris_ratio=r["debris"])

    def estimate_parasitaemia(self):
        return density(self.parasites, max(self.wbc, 1), self.wbc_per_ul)

    def count(self):
        r = self.row
        self.parasites += r["found_parasites"]
        self.wbc += r["found_wbc"]
        self.fields_read += 1
        return r["found_parasites"], r["found_wbc"]

    def advance(self) -> bool:
        self._i += 1
        return self._i < len(self.rows)

    def flag_for_human(self, reason):
        self.escalation = reason

    # What DecisionRule calls. Same arguments and same order as
    # agent.tools.Tools, so the replay exercises the shipped rule rather than a
    # paraphrase of it.
    def decide_call(self, floor_per_ul, negative_wbc_min, target_ci_width,
                    min_fields, settle_ci_width):
        stop, why, decision = should_stop_decision(
            self.estimate_parasitaemia(), self.fields_read, floor_per_ul,
            negative_wbc_min, target_ci_width=target_ci_width,
            min_fields=min_fields, settle_ci_width=settle_ci_width)
        self.decision = decision
        return stop, why

    def decide_stop(self, target_ci_width, min_fields):
        return should_stop(self.estimate_parasitaemia(), self.fields_read,
                           target_ci_width, min_fields)


def replay(rows, floor: float) -> list[dict]:
    """One session: a step per field, each carrying what was measured, what was
    counted, and the move the rule made."""
    tools = ReplayTools(rows)
    policy = DecisionRule(floor_per_ul=floor)
    steps = []

    for step_index in range(len(rows) * 2):
        quality = tools.quality()
        obs = observation(quality, tools, floor, NEGATIVE_WBC_MIN)
        obs["field"]["index"] = step_index
        action, args = policy.next_move(quality, tools)

        step = {"step": step_index, "observation": obs,
                "move": {"action": action, **args}, "model": "decision-rule (replayed)"}

        if action == "count_then_move":
            parasites, wbc = tools.count()
            step["counted"] = {"parasites": parasites, "wbc": wbc}
            # The totals in the observation were taken before this field was
            # counted; the page shows the running total after it, so restate it.
            step["observation"]["totals"] = {
                "fields_counted": tools.fields_read,
                "parasites": tools.parasites,
                "white_cells": tools.wbc,
            }
            steps.append(step)
            if not tools.advance():
                steps.append({"step": step_index + 1, "observation": obs,
                              "move": {"action": "flag_for_human",
                                       "reason": "ran out of annotated slide"},
                              "model": "decision-rule (replayed)"})
                break
            continue

        steps.append(step)

        if action in ("stop", "flag_for_human"):
            break
        if action == "skip" and not tools.advance():
            break
        if action == "refocus":
            # The NLM set has one capture per field, so a refocus cannot change
            # anything. The rule notices and skips; this keeps the replay honest
            # rather than pretending the focus improved.
            continue

    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", default=str(RESULTS / "agent_fields.json"))
    ap.add_argument("--patient", default=None)
    ap.add_argument("--list", action="store_true", help="patients available, with their fields")
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor", type=float, default=FLOOR_PER_UL_DEFAULT)
    args = ap.parse_args()

    table = json.loads(Path(args.fields).read_text())
    rows = table["rows"]
    by_patient: dict[str, list] = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)

    if args.list or not args.patient:
        print(f"{len(by_patient)} patients in {Path(args.fields).name}\n")
        for patient, patient_rows in sorted(by_patient.items()):
            found = sum(r["found_parasites"] for r in patient_rows)
            truth = sum(r["truth_parasites"] for r in patient_rows)
            print(f"  {patient:<18}{len(patient_rows):>3} fields  "
                  f"found {found:>4}  annotated {truth:>4}")
        if not args.patient:
            return

    if args.patient not in by_patient:
        raise SystemExit(f"{args.patient} is not in the table. --list shows what is.")

    steps = replay(by_patient[args.patient], args.floor)
    session = {
        "session_id": args.patient,
        "source": "replayed from eval/results/agent_fields.json, the cached field table",
        "policy": "decision-rule",
        "steps": steps,
        "disclaimer": "Screening aid, not a diagnostic device. "
                      "Every result is reviewed by a person.",
    }

    counted = [s for s in steps if "counted" in s]
    parasites = sum(s["counted"]["parasites"] for s in counted)
    wbc = sum(s["counted"]["wbc"] for s in counted)
    estimate = density(parasites, max(wbc, 1))
    print(f"\n{args.patient}: {len(counted)} fields counted, {parasites} parasites "
          f"in {wbc} white cells")
    print(f"  {estimate.per_ul:.0f} /uL  [{estimate.low_per_ul:.0f}, {estimate.high_per_ul:.0f}]")
    print(f"  stopped: {steps[-1]['move'].get('reason', '')}")

    if args.out:
        Path(args.out).write_text(json.dumps(session, indent=2))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
