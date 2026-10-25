"""Tiles for the learned detector: half-resolution crops of the prepared
field with every annotated parasite inside them, from the fit patients.

6 September 2026. The candidate
stage caps recall at 0.91 (peaks closer than 7 px merge, the white cell
exclusion zone, the rim) and the patch network judges each candidate
without its neighbours. A fully convolutional network on the whole field
reads context and proposes its own peaks. It is trained from the point
annotations rendered as small Gaussians (parasite_heatmap_train.py).

Tiles are TILE x TILE at half resolution (a chromatin dot is about 5 px
there, the annotator's circle about 11 px), cut from the cropped,
stain-normalised field the pipeline itself sees. Per infected image,
--per-image tiles: half centred near a random parasite, half anywhere in
the disc, so the network sees empty field as often as parasites. Clean
fields from the fit half of the uninfected set add tiles with no parasites.
The calibration patients' tiles are kept (marked) so the trainer can hold
them out; the held-out patients are never read here.

Stored as uint8 tiles plus a flat (tile, x, y) point table; the Gaussian
targets are rendered at training time, which is cheap and keeps the file
small.
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann, parasites as par

OUT = Path(__file__).resolve().parent / "results"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SCALE = 0.5
TILE = 288          # stored; training crops 256 at a random offset


def tiles_of(half: np.ndarray, mask_half: np.ndarray, pts: np.ndarray, n: int, rng: random.Random, tile: int):
    """n tiles: half near parasites (when there are any), half anywhere in the disc."""
    h, w = half.shape[:2]
    if h < tile or w < tile:
        return []
    ys, xs = np.nonzero(mask_half[tile // 2:h - tile // 2, tile // 2:w - tile // 2])
    if len(xs) == 0:
        return []
    out = []
    for k in range(n):
        if len(pts) and k < n // 2:
            px, py = pts[rng.randrange(len(pts))]
            cx = int(np.clip(px + rng.randint(-tile // 3, tile // 3), tile // 2, w - tile // 2))
            cy = int(np.clip(py + rng.randint(-tile // 3, tile // 3), tile // 2, h - tile // 2))
        else:
            i = rng.randrange(len(xs))
            cx, cy = int(xs[i]) + tile // 2, int(ys[i]) + tile // 2
        x0, y0 = cx - tile // 2, cy - tile // 2
        crop = half[y0:y0 + tile, x0:x0 + tile]
        inside = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x0 + tile) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + tile)] - [x0, y0] if len(pts) else np.zeros((0, 2))
        out.append((crop, inside))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=6)
    ap.add_argument("--per-clean", type=int, default=2)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--uninfected-images", type=int, default=-1, help="clean fields from the fit half; -1 = all")
    ap.add_argument("--max-images", type=int, default=0, help="smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="parasite_tiles.npz")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fit = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    calibration = set(json.loads((OUT / "calibration_patients.json").read_text()))

    tiles, points, meta = [], [], []
    stats = {"fit": [0, 0, 0], "calibration": [0, 0, 0], "clean": [0, 0, 0]}   # images, tiles, parasites
    t0 = time.perf_counter()

    def add(crops, patient, src):
        for crop, inside in crops:
            idx = len(tiles)
            tiles.append(crop)
            for x, y in inside:
                points.append((idx, float(x), float(y)))
            meta.append((patient, src))
            stats[src][1] += 1; stats[src][2] += len(inside)
        stats[src][0] += 1

    n = 0
    for patient in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name in fit):
        src = "calibration" if patient.name in calibration else "fit"
        for path in sorted(patient.glob("*.jpg")):
            if args.max_images and n >= args.max_images:
                break
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue
            n += 1
            prepared, mask, ox, oy = par._prepare(img, True, True)
            half = cv.resize(prepared, None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_AREA)
            mask_half = cv.resize(mask, (half.shape[1], half.shape[0]), interpolation=cv.INTER_NEAREST)
            pts = (np.array([[(p.x - ox) * SCALE, (p.y - oy) * SCALE] for p in truth.parasites])
                   if truth.parasites else np.zeros((0, 2)))
            add(tiles_of(half, mask_half, pts, args.per_image, rng, args.tile), patient.name, src)
            if n % 100 == 0:
                print(f"  {n} images, {len(tiles)} tiles, {len(points)} parasites, {(time.perf_counter() - t0) / 60:.0f} min", flush=True)

    if args.uninfected_images:
        split = json.loads((OUT / "uninfected_split.json").read_text())
        root = RAW / "NIH-NLM-ThickBloodSmearsU" / "Uninfected Patients"
        n_clean = 0
        for pat in split["fit"]:
            for path in sorted((root / pat / "tiled").glob("*.tiff")):
                if (args.uninfected_images > 0 and n_clean >= args.uninfected_images) or (args.max_images and n_clean >= args.max_images):
                    break
                img = cv.imread(str(path), cv.IMREAD_COLOR)
                if img is None:
                    continue
                n_clean += 1
                prepared, mask, _, _ = par._prepare(img, True, True)
                half = cv.resize(prepared, None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_AREA)
                mask_half = cv.resize(mask, (half.shape[1], half.shape[0]), interpolation=cv.INTER_NEAREST)
                add(tiles_of(half, mask_half, np.zeros((0, 2)), args.per_clean, rng, args.tile), pat, "clean")
        print(f"  + {n_clean} clean fields, {len(tiles)} tiles now")

    T = np.stack(tiles).astype(np.uint8)
    P = np.array(points, np.float32).reshape(-1, 3)
    np.savez_compressed(OUT / args.out, tiles=T, points=P,
                        patient=np.array([m[0] for m in meta]), source=np.array([m[1] for m in meta]))
    summary = {"out": args.out, "scale": SCALE, "tile": args.tile, "per_image": args.per_image, "per_clean": args.per_clean,
               "tiles": int(len(T)), "parasites": int(len(P)),
               "by_source": {k: {"images": v[0], "tiles": v[1], "parasites": v[2]} for k, v in stats.items()},
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (OUT / (Path(args.out).stem + ".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
