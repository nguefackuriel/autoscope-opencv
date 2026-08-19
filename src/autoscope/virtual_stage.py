"""The Virtual Stage: a microscope stage backed by images instead of hardware.

The agent calls move_stage() and refocus() exactly as it would against a real
microscope. Here those calls serve tiles out of a stitched slide, which makes
every run replayable. A judge can re-run a session and get the same log.

HONESTY NOTE, and it belongs in the report as well as here:

  move_stage() is real. Tiles come from a genuine capture of a genuine slide.

  refocus() is real only when the dataset gives us more than one capture of the
  same field. When it does not, we fall back to synthesising defocus with a
  Gaussian blur, and the returned frame is marked synthetic=True. Any number we
  publish about how well refocusing works has to say which of the two it used.
  Never report a synthetic refocus as if it came off a real z-stack.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


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
        Optional real focal planes of the same slide, keyed by z offset. Supply
        these whenever the source data has them; without them refocus() is
        simulated.
    tile:
        Tile size in pixels, matching a microscope field of view.
    """

    def __init__(
        self,
        mosaic: np.ndarray,
        z_planes: dict[int, np.ndarray] | None = None,
        tile: int = 1024,
    ) -> None:
        if mosaic.ndim < 2:
            raise ValueError("mosaic must be an image")
        self._planes = {0: mosaic, **(z_planes or {})}
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
        return (w // self._tile, h // self._tile)

    def current(self) -> Frame:
        plane, synthetic = self._plane(self._z)
        h, w = plane.shape[:2]
        x0 = min(self._x * self._tile, max(w - self._tile, 0))
        y0 = min(self._y * self._tile, max(h - self._tile, 0))
        crop = plane[y0 : y0 + self._tile, x0 : x0 + self._tile].copy()
        return Frame(crop, self._x, self._y, self._z, synthetic)

    def move_stage(self, dx: int, dy: int) -> Frame:
        """Move by whole fields. Always a real capture."""
        cols, rows = self.grid
        self._x = int(np.clip(self._x + dx, 0, max(cols - 1, 0)))
        self._y = int(np.clip(self._y + dy, 0, max(rows - 1, 0)))
        self.visited.append((self._x, self._y))
        return self.current()

    def refocus(self, dz: int) -> Frame:
        """Change focal plane. Real when we have a z-stack, simulated otherwise."""
        self._z += dz
        return self.current()

    def _plane(self, z: int) -> tuple[np.ndarray, bool]:
        if z in self._planes:
            return self._planes[z], False
        # No captured plane at this offset: blur the in-focus one instead, and
        # say so. Sigma grows with distance from the nearest real plane.
        nearest = min(self._planes, key=lambda k: abs(k - z))
        sigma = 0.9 * abs(z - nearest)
        if sigma <= 0:
            return self._planes[nearest], False
        return cv.GaussianBlur(self._planes[nearest], (0, 0), sigma), True
