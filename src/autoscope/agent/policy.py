"""How the next move gets chosen.

Three policies, same tools, so we can measure what the agent is worth:

  FixedProtocol  walk the slide and read a set number of fields.
  WhoProtocol    read until 200 white cells have gone by. The real baseline:
                 this is what the WHO manual actually asks a technician to do.
  RuleBased      thresholds and a stopping rule. No model, no cost. The honest
                 control, and what the agent has to beat to be worth anything.
  DecisionRule   the same, stopping on a decision instead of an error bar:
                 negative on the measured noise floor, positive when the
                 interval clears it, settled when it sits in one density band.
                 What a technician actually does (5 September 2026).
  BedrockPolicy  the real agent: Claude on Bedrock, one call per field, the
                 five moves given as tools. In agent/bedrock.py, behind this
                 same protocol, so agent_eval.py replays it against the three
                 above on the same cached field table (16 September 2026).

If the model policy cannot beat RuleBased, the agent is decoration, and the
report should say so rather than bury it. Keeping the control in the repo is
what makes that claim checkable.
"""

from typing import Protocol

from autoscope.agent.tools import Tools
from autoscope.density import FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, SETTLE_CI_WIDTH
from autoscope.quality import FieldQuality

# What a policy asks for next.
Move = tuple[str, dict]

# How many times we will hunt for focus on one field before giving up on it.
# Without a cap, a policy that always answers "refocus" never terminates, and
# the stage limits alone will not save it.
MAX_REFOCUS_TRIES = 4

# Below this, two sharpness readings are the same measurement. The smoothed
# Laplacian variance sits around 3.1 on a real field (p5 2.4, p95 4.4) and
# moves by tenths when the focus really changes, so 0.1, about 3% of a
# typical reading, is inside the noise floor. It was 0.5 on the raw scale,
# where the median was 17: the same fraction.
FOCUS_NOISE = 0.1


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
        # Two fixed arms in one table have to be told apart, and the number is
        # the whole difference between them.
        self.name = f"fixed-{fields}"

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
        stop, why = self._stop(tools)
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

    def _stop(self, tools: Tools) -> tuple[bool, str]:
        return tools.decide_stop(self.target_ci_width, self.min_fields)

    def _search_step(self) -> int:
        """+s, -2s, +3s, -4s: widen the hunt and alternate side each time."""
        n = self._tries + 1
        magnitude = self.step * n
        return magnitude if n % 2 else -magnitude

    def _reset(self) -> None:
        self._tries = 0
        self._best_seen = 0.0
        self._last_focus = None


class DecisionRule(RuleBased):
    """RuleBased's focus handling with the decision-driven stopping rule.

    `floor_per_ul` is the noise floor: what a clean slide reads at through
    this pipeline, measured on the fit half of the uninfected set. It belongs
    to the vision, not to the policy, and is re-measured whenever the
    classifier changes (eval/negative_eval.py). `negative_wbc_min` is how
    many white cells must go by before a negative can be called.
    """

    name = "decision"

    def __init__(self, floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
                 negative_wbc_min: int = NEGATIVE_WBC_MIN,
                 target_ci_width: float = 0.20, step: int = 2, min_fields: int = 5,
                 settle_ci_width: float = SETTLE_CI_WIDTH) -> None:
        super().__init__(target_ci_width, step, min_fields)
        self.floor_per_ul = floor_per_ul
        self.negative_wbc_min = negative_wbc_min
        self.settle_ci_width = settle_ci_width

    def _stop(self, tools: Tools) -> tuple[bool, str]:
        return tools.decide_call(self.floor_per_ul, self.negative_wbc_min,
                                 self.target_ci_width, self.min_fields, self.settle_ci_width)
