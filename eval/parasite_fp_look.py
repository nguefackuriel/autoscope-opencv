"""Look at the false alarms before building for them: montages of what the
detector calls a parasite and the experts did not, next to what both agree on.

    ../.venv/bin/python parasite_fp_look.py --patients TF70_Scr440 TF37_Sc563 --per-patient 2
    ../.venv/bin/python parasite_fp_look.py --uninfected TF208_15_38 TF210_15_40 --per-patient 2

Writes docs/figures/fp-look-<tag>.png: 48 px crops of the prepared (cropped,
normalised) field around each detection, magnified 3x, score in the corner.
Diagnosis only; nothing here fits anything.
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann, parasites as par
from parasite_candidates import label_candidates

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
CROP, ZOOM, COLS = 48, 3, 12


def crops(prepared, cands, scores, idx):
    h, w = prepared.shape[:2]; half = CROP // 2
    out = []
    for i in idx:
        x, y = int(cands[i].x), int(cands[i].y)
        x0, y0 = min(max(x - half, 0), w - CROP), min(max(y - half, 0), h - CROP)
        tile = cv.resize(prepared[y0:y0 + CROP, x0:x0 + CROP], None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
        cv.putText(tile, f"{float(scores[i]):.2f}", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv.putText(tile, f"{float(scores[i]):.2f}", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        out.append(tile)
    return out


def montage(tiles, title):
    if not tiles:
        return None
    size = CROP * ZOOM
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = np.full((rows * size + 24, COLS * size, 3), 255, np.uint8)
    cv.putText(sheet, title, (4, 17), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for k, t in enumerate(tiles):
        r, c = divmod(k, COLS)
        sheet[24 + r * size:24 + (r + 1) * size, c * size:(c + 1) * size] = t
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=[])
    ap.add_argument("--uninfected", nargs="*", default=[])
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=par.DEFAULT_THRESHOLD)
    ap.add_argument("--max", type=int, default=60, help="tiles per group")
    ap.add_argument("--tag", default="worst")
    args = ap.parse_args()
    scorer = par.PatchScorer.load()
    fp_tiles, tp_tiles, fn_note = [], [], []
    jobs = [(RAW / p, False) for p in args.patients] + \
           [(RAW / "NIH-NLM-ThickBloodSmearsU" / "Uninfected Patients" / p / "tiled", True) for p in args.uninfected]
    for folder, clean in jobs:
        images = sorted(folder.glob("*.tiff" if clean else "*.jpg"))[:args.per_patient]
        for path in images:
            img = cv.imread(str(path)); h, w = img.shape[:2]
            prepared, mask, ox, oy = par._prepare(img, crop_to_fov=True, normalise_stain=True)
            cands, _ = par.candidates(prepared, mask, features=False)
            if not cands:
                continue
            scores = scorer.score(par.patches_at(prepared, cands))
            if clean:
                labels = np.zeros(len(cands), np.int32)
            else:
                truth = ann.for_image(path, w, h)
                pts = np.array([[a.x - ox, a.y - oy] for a in truth.parasites]) if truth and truth.parasites else np.zeros((0, 2))
                labels, _ = label_candidates(cands, pts)
            keep = scores >= args.threshold
            fp = [i for i in np.argsort(-scores) if keep[i] and labels[i] == 0]
            tp = [i for i in np.argsort(-scores) if keep[i] and labels[i] == 1]
            name = f"{folder.parent.name if clean else folder.name}/{path.stem[-6:]}"
            print(f"{name}: {len(cands)} candidates, {int(keep.sum())} detections, {len(fp)} false, {len(tp)} true", flush=True)
            fp_tiles += crops(prepared, cands, scores, fp[: max(1, args.max // (len(jobs) * args.per_patient))])
            tp_tiles += crops(prepared, cands, scores, tp[: max(1, args.max // (len(jobs) * args.per_patient))])
    FIG.mkdir(exist_ok=True)
    for tiles, kind in ((fp_tiles, "false"), (tp_tiles, "true")):
        sheet = montage(tiles[: args.max], f"{kind} positives at {args.threshold:.2f}, {args.tag}: {len(tiles[:args.max])} shown")
        if sheet is not None:
            out = FIG / f"fp-look-{args.tag}-{kind}.png"; cv.imwrite(str(out), sheet); print("wrote", out.name, sheet.shape)


if __name__ == "__main__":
    main()
