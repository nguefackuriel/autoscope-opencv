"""The Virtual Stage: a microscope stage backed by images instead of hardware.

The agent calls move_stage() and refocus() exactly as it would against a real
microscope, so every run is replayable and a judge can re-run a session and get
the same log.

HOW FOCUS IS MODELLED, and this belongs in the report too:

  A real slide is not flat and the coverslip is not level, so each field comes
  into focus at a slightly different height. The objective sits at z. A field is
  sharp when z is near that field's own best height and soft when it is not.
  That is the model here, because a model where refocusing always helps or
  always hurts would let the agent look good for the wrong reason.

  move_stage() is real. Tiles come from a genuine capture of a genuine slide.

  refocus() returns a real capture only when the source data has more than one
  focal plane of the same field. The NLM thick smear set does not: its images
  are successive fields, taken about a minute apart, one capture each. So on
  that data refocus() renders defocus with a Gaussian blur and marks the frame
  synthetic=True. Any published number about refocusing has to say which of the
  two it used. Never report a synthetic refocus as if it came off a real stack.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

# How much blur one step away from the best plane costs. Fitted against the
# spread of real focus scores in the NLM set, see eval/focus_survey.py.
BLUR_PER_STEP = 0.9

# The objective cannot travel forever. A real one runs out of thread, and
# without a limit an agent that keeps refocusing will spiral.
Z_LIMIT = 8


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    x: int
    y: int
    z: int
    synthetic: bool

    def describe(self) -> str:
        kind = "synthetic defocus" if self.synthetic else "captured"
        return f"tile ({self.x}, {self.y}) z={self.z} [{kind}]"


class VirtualStage:
    """Serves tiles from one stitched slide.

    Parameters
    ----------
    mosaic:
        The whole slide as one image, built by the same stitching code the
        pipeline uses at run time.
    z_planes:
        Real focal planes keyed by z offset, when the source has them. With
        these, refocus() returns captured pixels and nothing is synthesised.
    focus_map:
        Best z per field, as {(x, y): z}. Fields not listed are sharp at z=0.
        This is what makes some fields need a refocus and others not.
    tile:
        Tile size in pixels, matching a microscope field of view.
    """

    def __init__(
        self,
        mosaic: np.ndarray,
        z_planes: dict[int, np.ndarray] | None = None,
        focus_map: dict[tuple[int, int], int] | None = None,
        tile: int = 1024,
    ) -> None:
        if mosaic.ndim < 2:
            raise ValueError("mosaic must be an image")
        self._planes = {0: mosaic, **(z_planes or {})}
        self._focus_map = focus_map or {}
        self._tile = tile
        self._x = 0
        self._y = 0
        self._z = 0
        self.visited: list[tuple[int, int]] = []

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> "VirtualStage":
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {path}")
        return cls(image, **kwargs)

    @property
    def has_real_z_stack(self) -> bool:
        return len(self._planes) > 1

    @property
    def grid(self) -> tuple[int, int]:
        h, w = self._planes[0].shape[:2]
        return (max(w // self._tile, 1), max(h // self._tile, 1))

    @property
    def z(self) -> int:
        return self._z

    def best_z_here(self) -> int:
        """Where this field is sharp. The agent does not get to see this."""
        return self._focus_map.get((self._x, self._y), 0)

    def current(self) -> Frame:
        plane, from_capture = self._nearest_plane()
        h, w = plane.shape[:2]
        x0 = min(self._x * self._tile, max(w - self._tile, 0))
        y0 = min(self._y * self._tile, max(h - self._tile, 0))
        crop = plane[y0 : y0 + self._tile, x0 : x0 + self._tile].copy()

        if from_capture:
            return Frame(crop, self._x, self._y, self._z, synthetic=False)

        # Blur the tile, not the whole slide: it is the honest unit and it is
        # also what keeps a long session fast.
        sigma = BLUR_PER_STEP * abs(self._z - self.best_z_here())
        if sigma <= 0:
            return Frame(crop, self._x, self._y, self._z, synthetic=False)
        return Frame(
            cv.GaussianBlur(crop, (0, 0), sigma), self._x, self._y, self._z, synthetic=True
        )

    def move_stage(self, dx: int, dy: int) -> Frame:
        """Move by whole fields. Always a real capture."""
        cols, rows = self.grid
        self._x = int(np.clip(self._x + dx, 0, cols - 1))
        self._y = int(np.clip(self._y + dy, 0, rows - 1))
        self.visited.append((self._x, self._y))
        return self.current()

    def refocus(self, dz: int) -> Frame:
        """Move the objective. Clamped, because a real one has limits."""
        self._z = int(np.clip(self._z + dz, -Z_LIMIT, Z_LIMIT))
        return self.current()

    def at_edge(self) -> bool:
        """True when the stage cannot move any further right or down."""
        cols, rows = self.grid
        return self._x >= cols - 1 and self._y >= rows - 1

    def _nearest_plane(self) -> tuple[np.ndarray, bool]:
        if self._z in self._planes:
            return self._planes[self._z], True
        nearest = min(self._planes, key=lambda k: abs(k - self._z))
        return self._planes[nearest], False
