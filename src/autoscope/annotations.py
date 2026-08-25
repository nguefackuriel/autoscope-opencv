"""Read the NLM ground truth, in the image's own coordinate frame.

THE COORDINATE TRANSFORM, AND WHY IT IS NOT OBVIOUS

A ground truth file starts with `count,4032,3024` and the images are 3024 wide
by 4032 tall. So the header is landscape while the pixels are portrait: the
annotations were made on a rotated view. The stored numbers are

    coord1, coord2   in a 4032 x 3024 landscape frame

and the mapping into the portrait image is a 90 degree clockwise rotation:

    x = width - coord2
    y = coord1

We did not guess this. Seven candidate transforms were scored by how much
darker the pixels under the points were than the lit field, since a white cell
nucleus is far darker than the background. The winner was ahead by a wide
margin, 131 L units against 47 for the next best, and it puts 100% of points
inside the frame. Overlaying it on real images then put a nucleus under 11 of 12
markers, against 1 of 12 for the plain reading.

Get this wrong and every count, every precision figure and every Bland-Altman
plot is measured against noise, while looking perfectly reasonable.

TWO SHAPES

    White_Blood_Cell  Point   one click, no size
    Parasite          Circle  two points, a drag across the parasite

So white cells can be scored for counting and for position, never for area. A
detector that finds the right number of cells in the right places is right, even
if its masks are the wrong size, because the count is what the density needs.
"""

from dataclasses import dataclass
from pathlib import Path

PARASITE = "Parasite"
WHITE_CELL = "White_Blood_Cell"


@dataclass(frozen=True)
class Annotation:
    label: str
    x: float
    y: float
    radius: float | None = None  # only parasites carry one


@dataclass(frozen=True)
class GroundTruth:
    path: Path
    width: int
    height: int
    items: list[Annotation]

    @property
    def white_cells(self) -> list[Annotation]:
        return [a for a in self.items if a.label == WHITE_CELL]

    @property
    def parasites(self) -> list[Annotation]:
        return [a for a in self.items if a.label == PARASITE]

    def counts(self) -> tuple[int, int]:
        """Parasites, white cells. The two numbers the density needs."""
        return len(self.parasites), len(self.white_cells)


def to_image_frame(coord1: float, coord2: float, width: int) -> tuple[float, float]:
    """Landscape annotation coordinates into portrait image pixels."""
    return width - coord2, coord1


def read(path: str | Path, width: int, height: int) -> GroundTruth:
    """Parse one .txt, mapping every point into the image's frame.

    `width` and `height` are the real image dimensions, not the header's. The
    header is in the annotation frame and disagrees with the pixels on purpose.
    """
    path = Path(path)
    items: list[Annotation] = []

    for line in path.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        label, shape = parts[1], parts[3]

        if label == WHITE_CELL and shape == "Point":
            x, y = to_image_frame(float(parts[5]), float(parts[6]), width)
            items.append(Annotation(label, x, y))

        elif label == PARASITE and shape == "Circle" and len(parts) >= 9:
            a1, b1, a2, b2 = (float(v) for v in parts[5:9])
            x1, y1 = to_image_frame(a1, b1, width)
            x2, y2 = to_image_frame(a2, b2, width)
            # The two points are a drag across the parasite, so the distance
            # between them is a diameter.
            radius = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2
            items.append(Annotation(label, (x1 + x2) / 2, (y1 + y2) / 2, radius))

    return GroundTruth(path=path, width=width, height=height, items=items)


def for_image(image_path: str | Path, width: int, height: int) -> GroundTruth | None:
    """Find and read the ground truth that goes with an image, or None."""
    image_path = Path(image_path)
    gt = image_path.parents[1] / "GT_updated" / image_path.parent.name / f"{image_path.stem}.txt"
    return read(gt, width, height) if gt.exists() else None
