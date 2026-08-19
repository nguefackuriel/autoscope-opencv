"""How the next move gets chosen.

Three policies, same tools, so we can measure what the agent is worth:

  FixedProtocol  walk the slide and read every field. The WHO baseline.
  RuleBased      thresholds and a search. No model, no cost. The honest control.
  Bedrock        the real agent.                                    (to write)

If the model policy cannot beat RuleBased, the agent is decoration, and the
report should say so rather than bury it. Keeping the control in the repo is
what makes that claim checkable.
"""

from typing import Protocol

from autoscope.agent.tools import Tools
from autoscope.quality import FieldQuality

# What a policy asks for next.
Move = tuple[str, dict]

# How many times we will hunt for focus on one field before giving up on it.
# Without a cap, a policy that always answers "refocus" never terminates, and
# the stage limits alone will not save it.
MAX_REFOCUS_TRIES = 4


class Policy(Protocol):
    name: str

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move: ...


class FixedProtocol:
    """The WHO method as written: walk the slide, read every field, then stop.

    No quality gate, because a human at a microscope refocuses without thinking
    about it. This is the cost baseline, not a strawman.
    """

    name = "fixed"

    def __init__(self, fields: int = 100) -> None:
        self.fields = fields

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:  # noqa: ARG002
        if tools.fields_read >= self.fields:
            return ("stop", {"reason": f"read the fixed {self.fields} fields"})
        return ("count_then_move", {})


class RuleBased:
    """Thresholds plus a small focus search. No model involved.

    When a field is soft, hunt for the focal plane by stepping up, then down,
    then further, giving up after MAX_REFOCUS_TRIES and skipping the field.
    Alternating direction matters: the objective can be on either side of the
    plane where this field is sharp, and we do not know which.
    """

    name = "rule-based"

    def __init__(self, target_ci_width: float = 0.20, step: int = 2) -> None:
        self.target_ci_width = target_ci_width
        self.step = step
        self._tries = 0
        self._best_seen = 0.0

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:
        stop, why = tools.decide_stop(self.target_ci_width)
        if stop:
            return ("stop", {"reason": why})

        problem = quality.why_not()

        if problem is None:
            self._reset()
            return ("count_then_move", {})

        if "focus" in problem and self._tries < MAX_REFOCUS_TRIES:
            dz = self._search_step()
            self._tries += 1
            self._best_seen = max(self._best_seen, quality.focus_score)
            return ("refocus", {"dz": dz, "reason": problem})

        self._reset()
        return ("skip", {"reason": problem})

    def _search_step(self) -> int:
        """+s, -2s, +3s, -4s: widen the hunt and alternate side each time."""
        n = self._tries + 1
        magnitude = self.step * n
        return magnitude if n % 2 else -magnitude

    def _reset(self) -> None:
        self._tries = 0
        self._best_seen = 0.0
