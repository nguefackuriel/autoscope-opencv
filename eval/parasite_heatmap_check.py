"""Runtime-venv check that cv.dnn reproduces the torch heat map of the
exported detector, on the tiles the trainer saved, and that it runs on a
field-sized input. Run after parasite_heatmap_train.py."""

import json
from pathlib import Path

import numpy as np

from autoscope import parasites as par

OUT = Path(__file__).resolve().parent / "results"

train = json.loads((OUT / "parasite_heatmap_train.json").read_text())
d = np.load(OUT / train.get("tiles_file", "parasite_tiles.npz"))
ref = np.load(OUT / "parasite_heatmap_check.npy")
crop = int(train["crop"])
scorer = par.HeatmapScorer.load()
T = d["tiles"][train["check_tiles"]]
o = (T.shape[1] - crop) // 2
ours = np.stack([scorer.heat(t[o:o + crop, o:o + crop]) for t in T])
err = float(np.abs(ours - ref).max())
print(f"cv.dnn vs torch on {len(ref)} tiles: max abs diff {err:.2e}")
assert err < 1e-3, "cv.dnn does not reproduce the exported detector"
h = scorer.heat(np.zeros((1335, 1335, 3), np.uint8) + 200)
print(f"field-sized input 1335 x 1335 -> heat {h.shape}, padded to a multiple of 8 and cropped back")
assert h.shape == (1335, 1335)
