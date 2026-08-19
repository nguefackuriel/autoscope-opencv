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

from autoscope.density import Density, density, should_stop
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

    def decide_stop(self, target_ci_width: float = 0.20) -> tuple[bool, str]:
        return should_stop(
            self.estimate_parasitaemia(),
            self.fields_read,
            target_ci_width=target_ci_width,
        )

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
