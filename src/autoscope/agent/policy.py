"""How the next move gets chosen.

Three policies, same tools, so we can measure what the agent is worth:

  FixedProtocol  walk the slide and read a set number of fields.
  WhoProtocol    read until 200 white cells have gone by. The real baseline:
                 this is what the WHO manual actually asks a technician to do.
  RuleBased      thresholds and a stopping rule. No model, no cost. The honest
                 control, and what the agent has to beat to be worth anything.
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

# Below this, two sharpness readings are the same measurement. Laplacian
# variance on these tiles sits around 17 and moves by whole units when the
# focus really changes, so 0.5 is well inside the noise floor.
FOCUS_NOISE = 0.5


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


class WhoProtocol:
    """The WHO thick film method: count parasites against 200 white cells.

    This, not a fixed number of fields, is the baseline the proposal argues
    with. How many fields it takes depends on the slide, so it is not a
    constant we can quote: on this data the median field carries 19 white
    cells, so about ten fields, but a thin patch of the smear takes many more.

    No quality gate. A technician refocuses without thinking about it, and
    pretending otherwise would make the baseline a strawman.
    """

    name = "who-200"

    def __init__(self, wbc_target: int = 200) -> None:
        self.wbc_target = wbc_target

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:  # noqa: ARG002
        if tools.wbc >= self.wbc_target:
            return ("stop", {"reason": f"counted {tools.wbc} white cells, "
                                       f"target was {self.wbc_target}"})
        return ("count_then_move", {})


class RuleBased:
    """Thresholds plus a small focus search. No model involved.

    When a field is soft, hunt for the focal plane by stepping up, then down,
    then further, giving up after MAX_REFOCUS_TRIES and skipping the field.
    Alternating direction matters: the objective can be on either side of the
    plane where this field is sharp, and we do not know which.

    GIVING UP EARLY WHEN NOTHING MOVES

    If a refocus does not change the sharpness at all, the objective is not
    responding: a stuck focus motor on real hardware, or a dataset with one
    capture per field, which is what the NLM set is. Hunting four times then
    skipping wastes four tool calls and fills the log with a search that could
    never have worked. So a refocus that leaves the score unchanged ends the
    search immediately. Measured over the 50 held-out slides: refocus calls
    fall from 100 to 25 and the log from 662 steps to 587, with all 469
    counted fields identical.
    """

    name = "rule-based"

    def __init__(self, target_ci_width: float = 0.20, step: int = 2,
                 min_fields: int = 5) -> None:
        self.target_ci_width = target_ci_width
        self.step = step
        self.min_fields = min_fields
        self._tries = 0
        self._best_seen = 0.0
        self._last_focus: float | None = None

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:
        stop, why = tools.decide_stop(self.target_ci_width, self.min_fields)
        if stop:
            return ("stop", {"reason": why})

        problem = quality.why_not()

        if problem is None:
            self._reset()
            return ("count_then_move", {})

        if "focus" in problem and self._tries < MAX_REFOCUS_TRIES:
            unresponsive = (
                self._last_focus is not None
                and abs(quality.focus_score - self._last_focus) < FOCUS_NOISE
            )
            if unresponsive:
                self._reset()
                return ("skip", {"reason": f"{problem}, and refocusing did not change it"})
            dz = self._search_step()
            self._tries += 1
            self._best_seen = max(self._best_seen, quality.focus_score)
            self._last_focus = quality.focus_score
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
        self._last_focus = None
