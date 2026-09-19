"""Turn the NLM fit patients into a tiled detection dataset, YOLO and COCO.

WHY TILES

A parasite is about 10 px across in a field roughly 3000 px wide. Feeding the
whole field to a detector at 640 px would shrink it to 2 px, below what any box
head can regress. So tiles are cut at NATIVE resolution and the detector sees
the parasite at its true size. At inference the same tiling is applied and the
detections are merged back (see detector_eval.py).

WHY BOTH FORMATS OVER ONE SET OF TILES

YOLO26 wants a directory of images and .txt labels; torchvision's Faster R-CNN
wants COCO json. Both are written against the SAME jpg tiles, so any difference
between the two models is the model and not the data.

WHAT IS AND IS NOT EXPORTED

Fit patients only, split into train and val by patient. The 50 held-out
patients are never read here and never tiled: they are scored once, on whole
images, by detector_eval.py. Cutting them into tiles at all would invite
somebody to point a training run at the directory.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np


# --- repo paths: run from anywhere, installed or not -------------------------
# Azure ML runs the code snapshot without `pip install -e .`, so the editable
# install that normally puts src/ on sys.path is absent and every `from
# autoscope import ...` fails. This depends on __file__ alone, so it survives
# any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from autoscope import annotations as ann, boxes as bx, parasites as par
import splits

# Patient images: resolved via splits.patient_root(), not assumed to sit
# directly under data/raw -- download.py may nest them one level down.
CATEGORY = {"id": 1, "name": "parasite", "supercategory": "cell"}


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Tile starts covering `size`, with the last one flush against the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def boxes_in_tile(all_boxes, ox, oy, tile, edge=1):
    """Boxes lying wholly inside the tile, as (x, y, w, h) in tile coordinates.

    A box straddling the tile boundary is a bad target: the detector would be
    taught that a parasite looks like half a parasite. Dropping it is safe
    because the overlap between tiles means a neighbour contains it whole.
    """
    kept = []
    for b in all_boxes:
        x0, y0 = b.x - ox, b.y - oy
        x1, y1 = x0 + b.w, y0 + b.h
        if x0 < edge or y0 < edge or x1 > tile - edge or y1 > tile - edge:
            continue
        kept.append((x0, y0, b.w, b.h))
    return kept


def export(images_per_patient, tile, stride, out, seed, negative_ratio, min_fov,
           max_patients=0):
    root = splits.patient_root()
    train_ids, val_ids = splits.train_val(seed=seed)
    wanted = {p: "train" for p in train_ids}
    wanted.update({p: "val" for p in val_ids})
    holdout = set(splits.holdout_patients())

    if max_patients:
        # Smoke test only. Keeps both subsets represented so the run still
        # exercises the train/val paths rather than silently testing one.
        keep = set(train_ids[:max_patients]) | set(val_ids[:max(1, max_patients // 4)])
        wanted = {p: s for p, s in wanted.items() if p in keep}
        print(f"SMOKE TEST: {len(wanted)} patients, not the full fit set")

    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    coco = {s: {"images": [], "annotations": [], "categories": [CATEGORY]}
            for s in ("train", "val")}
    img_id = 0
    ann_id = 0
    stats = Counter()
    negatives_kept = Counter()
    sides = []

    for patient in sorted(wanted):
        # Belt and braces. The split says these are fit patients, but a bug
        # that leaked a held-out patient into training would silently
        # invalidate every number, so it is checked rather than trusted.
        assert patient not in holdout, f"{patient} is held out, never export it"
        subset = wanted[patient]
        # Only images that actually carry ground truth are candidates. 53 of
        # the 1883 images have none, because the dataset ReadMe says annotation
        # stops once 200 white cells are counted. Sampling first and skipping
        # afterwards would quietly return fewer images than asked for, and the
        # shortfall would land on whichever patients were annotated least.
        candidates = splits.annotated_images(root / patient)
        rng = splits.image_rng(seed, patient)
        stats["images_reserved_for_inference"] += len(
            splits.unannotated_images(root / patient))
        if not candidates:
            continue

        for path in rng.sample(candidates, min(images_per_patient, len(candidates))):
            image = cv.imread(str(path))
            if image is None:
                stats["unreadable"] += 1
                continue
            h_full, w_full = image.shape[:2]
            truth = ann.for_image(path, w_full, h_full)
            if truth is None:
                stats["no_annotation"] += 1
                continue

            prepared, mask, ox_fov, oy_fov = par._prepare(image, True, True)
            top, _, _ = par.tophat(prepared)
            centres = [(p.x - ox_fov, p.y - oy_fov) for p in truth.parasites]
            radii = [p.radius for p in truth.parasites]
            # fallback=False: never invent a box. Fabricating one where the
            # stain shows nothing would be altering the ground truth, and a
            # dataset altered that way would need per-image verification we are
            # not in a position to do. Instead the tiles containing an
            # unboxable annotation are dropped below, so nothing is invented
            # and no real parasite is left labelled as background.
            found, kept, provenance = bx.boxes_for(top, centres, radii, fallback=False)
            unboxable = [centres[i] for i in range(len(centres)) if i not in set(kept)]

            stats["images"] += 1
            stats["annotated_parasites"] += len(centres)
            stats["boxes_derived"] += len(found)
            stats["boxes_from_stain"] += sum(1 for s in provenance if s == "stain")
            stats["annotations_unboxable"] += len(unboxable)
            for b in found:
                sides.append(b.w)
                sides.append(b.h)

            height, width = prepared.shape[:2]
            positives = []
            empties = []
            for oy in tile_origins(height, tile, stride):
                for ox in tile_origins(width, tile, stride):
                    sub_mask = mask[oy:oy + tile, ox:ox + tile]
                    if float((sub_mask > 0).mean()) < min_fov:
                        stats["tiles_outside_fov"] += 1
                        continue
                    # A tile holding an annotation we could not box is not
                    # usable either way: keep it and a real parasite is labelled
                    # background, invent a box and we have altered the ground
                    # truth. Declining the tile does neither.
                    if any(ox <= ux < ox + tile and oy <= uy < oy + tile
                           for ux, uy in unboxable):
                        stats["tiles_with_unboxable"] += 1
                        continue
                    inside = boxes_in_tile(found, ox, oy, tile)
                    if inside:
                        positives.append((ox, oy, inside))
                    else:
                        empties.append((ox, oy, inside))

            # Background tiles teach the detector what is not a parasite, but a
            # dataset that is mostly background trains a detector that says no.
            n_neg = int(round(len(positives) * negative_ratio))
            chosen = positives + rng.sample(empties, min(n_neg, len(empties)))
            negatives_kept[subset] += len(chosen) - len(positives)

            for ox, oy, inside in chosen:
                crop = prepared[oy:oy + tile, ox:ox + tile]
                name = f"{patient}_{path.stem}_{ox}_{oy}.jpg"
                cv.imwrite(str(out / "images" / subset / name), crop,
                           [cv.IMWRITE_JPEG_QUALITY, 95])

                lines = []
                for x, y, w, h in inside:
                    lines.append(
                        f"0 {(x + w / 2) / tile:.6f} {(y + h / 2) / tile:.6f} "
                        f"{w / tile:.6f} {h / tile:.6f}")
                (out / "labels" / subset / (name[:-4] + ".txt")).write_text(
                    "\n".join(lines))

                coco[subset]["images"].append(
                    {"id": img_id, "file_name": name, "width": tile, "height": tile})
                for x, y, w, h in inside:
                    coco[subset]["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": 1,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(w * h), "iscrowd": 0,
                    })
                    ann_id += 1
                img_id += 1
                stats["tiles_" + subset] += 1
                stats["boxes_" + subset] += len(inside)

        done = stats["tiles_train"] + stats["tiles_val"]
        print(f"  {patient} ({subset}): {done} tiles so far")

    (out / "annotations").mkdir(exist_ok=True)
    for sub in ("train", "val"):
        (out / "annotations" / f"instances_{sub}.json").write_text(
            json.dumps(coco[sub]))

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: parasite\n")

    derived = stats["boxes_derived"]
    annotated = stats["annotated_parasites"]
    manifest = {
        "split_source": "eval/results/wbc_sweep.json:fit_patients",
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "holdout_patients_untouched": len(holdout),
        "images_per_patient": images_per_patient,
        "tile_px": tile,
        "stride_px": stride,
        "seed": seed,
        "annotated_parasites": annotated,
        "boxes_derived": derived,
        # One box per annotation is the contract, so this is 0.0 unless an
        # annotation sits off the edge of the cropped field entirely.
        "box_drop_rate": round(1 - derived / max(annotated, 1), 4),
        "boxes_from_stain": stats["boxes_from_stain"],
        "boxes_invented": 0,
        "annotations_unboxable": stats["annotations_unboxable"],
        "tiles_dropped_unboxable": stats["tiles_with_unboxable"],
        # Derived boxes that reached a tile. The rest fell in tiles skipped for
        # sitting outside the eyepiece disc, or straddled a tile edge in every
        # tile covering them. Rim parasites are excluded from counting
        # elsewhere in the pipeline too, so this is consistent -- but it is a
        # second, larger shrinkage than the drop rate and must not be invisible.
        "boxes_in_tiles": stats["boxes_train"] + stats["boxes_val"],
        "box_tile_loss_rate": round(
            1 - (stats["boxes_train"] + stats["boxes_val"]) / max(derived, 1), 4),
        "box_side_median_px": float(np.median(sides)) if sides else None,
        "box_side_p10_px": float(np.percentile(sides, 10)) if sides else None,
        "box_side_p90_px": float(np.percentile(sides, 90)) if sides else None,
        "negative_tiles_kept": dict(negatives_kept),
        "counts": dict(stats),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-per-patient", type=int, default=3,
                    help="3 is what the patch network's best held-out run used")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=512,
                    help="640 minus 512 gives 128 px of overlap")
    ap.add_argument("--negative-ratio", type=float, default=0.2)
    ap.add_argument("--min-fov", type=float, default=0.9,
                    help="skip tiles less than this fraction inside the disc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-patients", type=int, default=0,
                    help="smoke test on a few patients instead of the fit set")
    ap.add_argument("--out", default="data/derived/detect")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    m = export(args.images_per_patient, args.tile, args.stride, out,
               args.seed, args.negative_ratio, args.min_fov, args.max_patients)

    print(f"\ntrain {m['counts'].get('tiles_train', 0)} tiles, "
          f"{m['counts'].get('boxes_train', 0)} boxes")
    print(f"val   {m['counts'].get('tiles_val', 0)} tiles, "
          f"{m['counts'].get('boxes_val', 0)} boxes")
    print(f"\n{m['boxes_derived']} boxes from {m['annotated_parasites']} "
          f"annotations ({m['box_drop_rate'] * 100:.2f}% unboxed)")
    print(f"  every box contoured from the stain; none invented")
    print(f"  {m['annotations_unboxable']} annotations had no stain peak; the "
          f"{m['tiles_dropped_unboxable']} tiles containing them were dropped")
    print(f"box side px: p10 {m['box_side_p10_px']}, "
          f"median {m['box_side_median_px']}, p90 {m['box_side_p90_px']}")
    print(f"{m['boxes_in_tiles']} of those reached a tile "
          f"({m['box_tile_loss_rate'] * 100:.1f}% lost to the disc edge and "
          f"tile boundaries)")
    print(f"\nheld-out patients untouched: {m['holdout_patients_untouched']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
