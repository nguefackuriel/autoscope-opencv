"""The real counter for the agent loop: one field in, (parasites, white cells)
out, both from OpenCV.

`agent.tools.Tools` takes any callable with that signature. The oracle counter
in the evaluation returns the ground truth; this one returns what the pipeline
sees. Running the loop with both, on the same slide, is how agent mistakes are
told apart from vision mistakes.

The field is prepared once (find the eyepiece disc, crop to it, normalise the
stain) and both detectors run on the prepared image. Calling
`find_white_cells` and `find_parasites` separately would do that work twice,
and it is the expensive part of the pipeline.
"""

from dataclasses import dataclass, replace

import numpy as np

from autoscope import parasites as par
from autoscope.segmentation import Cell, find_white_cells


@dataclass(frozen=True)
class FieldCount:
    parasites: int
    white_cells: int
    parasite_points: list[par.Parasite]
    white_cell_points: list[Cell]

    def as_tuple(self) -> tuple[int, int]:
        return self.parasites, self.white_cells


def count_field(
    image_bgr: np.ndarray,
    model: str = "cnn",
    threshold: float | None = None,
    scorer=None,
) -> FieldCount:
    """Count one field. Coordinates are reported in the input image's frame.
    `threshold` defaults to the model's own."""
    threshold = par.threshold_for(model, threshold)
    prepared, mask, ox, oy = par._prepare(image_bgr, crop_to_fov=True, normalise_stain=True)

    cells = find_white_cells(prepared, crop_to_fov=False, normalise_stain=False)
    cells = [replace(c, x=c.x + ox, y=c.y + oy) for c in cells]

    found: list[par.Parasite] = []
    if model == "heatmap":
        scorer = scorer or par._heatmap_scorer()
        pts, scores = scorer.peaks(prepared, mask)
        found = [par.Parasite(float(x) + ox, float(y) + oy, 0.0, float(s))
                 for (x, y), s in zip(pts, scores) if s >= threshold]
        return FieldCount(len(found), len(cells), found, cells)

    cands, _ = par.candidates(prepared, mask, features=(model != "cnn"))
    if cands:
        if model == "cnn":
            scorer = scorer or par._patch_scorer()
            scores = scorer.score(par.patches_at(prepared, cands))
        elif model == "hybrid":
            scorer = scorer or par._hybrid_scorer()
            scores = scorer.score(par.patches_at(prepared, cands), np.stack([c.features for c in cands]))
        elif model == "mlp":
            scorer = scorer or par._scorer()
            scores = scorer.score(np.stack([c.features for c in cands]))
        else:
            raise ValueError(f"unknown model {model!r}")
        found = [par.Parasite(c.x + ox, c.y + oy, c.area, float(s))
                 for c, s in zip(cands, scores) if s >= threshold]

    return FieldCount(len(found), len(cells), found, cells)


def vision_counter(model: str = "cnn", threshold: float | None = None):
    """A `Counter` for `agent.tools.Tools`, with the model loaded once."""
    scorer = par.load_scorer(model)

    def count(image_bgr: np.ndarray) -> tuple[int, int]:
        return count_field(image_bgr, model=model, threshold=threshold, scorer=scorer).as_tuple()

    return count
