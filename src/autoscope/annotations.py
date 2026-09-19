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
    Parasite          Circle  two points: the CENTRE, then a point on the rim

The Circle is not a drag across the parasite. We first read it that way and
took the midpoint as the centre, which put every parasite about 13 px off, to
one side, on an object 10 px across. Measured on 834 parasites in 25 images:
the most purple pixel near the annotation sits a median 0.8 px from point 1
and 18 px from point 2. Point 1 is the centre. The distance to point 2 is the
annotator's radius, median 22 px, which is loose: the chromatin dot itself is
about 10 px across. Treat `radius` as an upper bound on the object, not its
size.

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
            cx, cy = to_image_frame(a1, b1, width)
            rx, ry = to_image_frame(a2, b2, width)
            # First point is the centre, second is on the rim. Not a diameter:
            # see the module docstring for the measurement behind this.
            radius = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
            items.append(Annotation(label, cx, cy, radius))

    return GroundTruth(path=path, width=width, height=height, items=items)


def gt_path(image_path: str | Path) -> Path:
    """Where the ground truth for an image lives, whether or not it exists.

    53 of the 1883 images (2.8%) have no file here, and that is meaningful
    rather than missing data: the dataset ReadMe says annotation stops once
    200 white cells have been counted, because that is enough to compute a
    parasitaemia. An image with no .txt was captured after the count stopped.
    Callers that sample images should skip those before sampling, not after,
    or they quietly get fewer images than they asked for.
    """
    image_path = Path(image_path)
    return (image_path.parents[1] / "GT_updated" / image_path.parent.name
            / f"{image_path.stem}.txt")


def for_image(image_path: str | Path, width: int, height: int) -> GroundTruth | None:
    """Find and read the ground truth that goes with an image, or None."""
    gt = gt_path(image_path)
    return read(gt, width, height) if gt.exists() else None
