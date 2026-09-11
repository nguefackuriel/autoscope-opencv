"""The seven tools the agent can call.

Six of them read numbers out of OpenCV 5. The other two act on the stage.

The counter is injected rather than imported, so the loop can be built and
tested before segmentation exists, and so the evaluation can swap a perfect
oracle counter in to separate agent mistakes from vision mistakes. That
separation matters: without it, a bad result tells you nothing about which half
is at fault.
"""

from typing import Callable, Protocol

import numpy as np

from autoscope.density import (FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, SETTLE_CI_WIDTH, Decision, Density,
                               density, should_stop, should_stop_decision)
from autoscope.quality import FieldQuality, assess_field
from autoscope.virtual_stage import Frame, VirtualStage


class Counter(Protocol):
    """Counts parasites and white cells in one field."""

    def __call__(self, image: np.ndarray) -> tuple[int, int]: ...


class Tools:
    def __init__(
        self,
        stage: VirtualStage,
        counter: Counter,
        wbc_per_ul: float = 8000.0,
    ) -> None:
        self.stage = stage
        self.counter = counter
        self.wbc_per_ul = wbc_per_ul
        self.parasites = 0
        self.wbc = 0
        self.fields_read = 0
        self.escalation: str | None = None
        self.decision: Decision | None = None   # set by the decision rule when it stops
        self._frame: Frame = stage.current()

    # --- the six that read OpenCV output ---

    def assess_field(self) -> FieldQuality:
        return assess_field(self._frame.image)

    def count_field(self) -> tuple[int, int]:
        parasites, wbc = self.counter(self._frame.image)
        self.parasites += parasites
        self.wbc += wbc
        self.fields_read += 1
        return parasites, wbc

    def estimate_parasitaemia(self) -> Density:
        return density(self.parasites, max(self.wbc, 1), self.wbc_per_ul)

    def decide_stop(self, target_ci_width: float = 0.20,
                    min_fields: int = 10) -> tuple[bool, str]:
        """`min_fields` is a policy choice, not a constant of nature.

        It guards against slide heterogeneity: parasites are not spread evenly,
        so a tight error bar after two fields is still a bad estimate. It is
        not a proxy for the WHO count, and setting it near ten on this data
        would make the agent stop exactly where the WHO protocol does and the
        comparison vacuous. Swept in eval/agent_eval.py.
        """
        return should_stop(
            self.estimate_parasitaemia(),
            self.fields_read,
            target_ci_width=target_ci_width,
            min_fields=min_fields,
        )

    def decide_call(self, floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
                    negative_wbc_min: int = NEGATIVE_WBC_MIN,
                    target_ci_width: float = 0.20, min_fields: int = 5,
                    settle_ci_width: float = SETTLE_CI_WIDTH) -> tuple[bool, str]:
        """The decision-driven rule: negative on the noise floor, positive
        when the interval clears it, settled when it sits in one band.
        The call is kept on `self.decision` so the harness can score it."""
        stop, why, decision = should_stop_decision(
            self.estimate_parasitaemia(), self.fields_read, floor_per_ul, negative_wbc_min,
            target_ci_width=target_ci_width, min_fields=min_fields, settle_ci_width=settle_ci_width)
        self.decision = decision
        return stop, why

    # --- the two that act ---

    def refocus(self, dz: int) -> Frame:
        self._frame = self.stage.refocus(dz)
        return self._frame

    def move_stage(self, dx: int, dy: int) -> Frame:
        self._frame = self.stage.move_stage(dx, dy)
        return self._frame

    # --- the escape hatch ---

    def flag_for_human(self, reason: str) -> None:
        self.escalation = reason

    @property
    def frame(self) -> Frame:
        return self._frame


class OracleCounter:
    """A counter that already knows the answer for each tile.

    For testing the loop, and for the evaluation: running the same slide with a
    perfect counter and with the real one separates agent mistakes from vision
    mistakes. Without that split, a bad end-to-end result tells you nothing
    about which half to fix. It never appears in a published end-to-end number.
    """

    def __init__(self, stage: VirtualStage, truth: dict[tuple[int, int], tuple[int, int]]):
        self.stage = stage
        self.truth = truth

    def __call__(self, image: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        frame = self.stage.current()
        return self.truth.get((frame.x, frame.y), (0, 0))
