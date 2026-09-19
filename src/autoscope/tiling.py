"""How a field is cut into tiles, shared by the export and by inference.

WHY THIS IS ONE MODULE AND NOT TWO CONSTANTS

The training set and the inference pass have to tile identically. When they
drifted apart the failure was silent and expensive: both defaulted to
MIN_FOV = 0.9, which keeps only the tiles lying almost entirely inside the
eyepiece disc. Measured on validation fields that is 10.2 tiles per image out
of 29.6, covering 65.8% of annotated parasites.

The consequences compounded:

  * training saw about a third of the available tiles,
  * at inference the detector was never shown 34.8% of the parasites it was
    then scored against, so recall was capped at 0.652 before the model did
    anything at all,
  * and the export reported the shortfall as "lost to the disc edge and tile
    boundaries", which sounded like geometry rather than a threshold nobody
    had measured.

MIN_FOV = 0.2 covers 99.9% of parasites at 27.8 tiles per image. The disc is
inscribed in the cropped frame, so a 640 px tile near the rim is legitimately
part black; the detector should still look at it, because parasites sit right
up to the edge of the field.

Raising this again would silently shrink both the training set and the share of
the slide anyone is actually looking at, so it lives here, once.
"""

TILE_PX = 640
STRIDE_PX = 512          # 128 px of overlap
MIN_FOV = 0.2            # fraction of a tile that must lie inside the disc


def tile_origins(size: int, tile: int = TILE_PX, stride: int = STRIDE_PX) -> list[int]:
    """Tile starts covering `size`, with the last one flush against the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def covered_tiles(mask, tile: int = TILE_PX, stride: int = STRIDE_PX,
                  min_fov: float = MIN_FOV) -> list[tuple[int, int]]:
    """Every (x, y) origin whose tile is at least `min_fov` inside the disc.

    Takes the field-of-view mask rather than the image, so the caller cannot
    accidentally test coverage against a different array from the one the
    tiles are cut from.
    """
    height, width = mask.shape[:2]
    out = []
    for oy in tile_origins(height, tile, stride):
        for ox in tile_origins(width, tile, stride):
            if float((mask[oy:oy + tile, ox:ox + tile] > 0).mean()) >= min_fov:
                out.append((ox, oy))
    return out
