"""Runtime-venv check that cv.dnn reproduces the torch predictions of the
exported patch network. Run after parasite_cnn_train.py."""

import json
import sys
from pathlib import Path

import numpy as np

from autoscope import parasites as par

OUT = Path(__file__).resolve().parent / "results"

hybrid = "--hybrid" in sys.argv
args = [a for a in sys.argv[1:] if a != "--hybrid"]
# Compare against the patch file the trainer actually used, recorded in its
# JSON. Defaulting to parasite_patches.npz failed the check on 2 September
# 2026 for no reason: the model was trained on the x3 file.
train_json = OUT / ("parasite_hybrid_train.json" if hybrid else "parasite_cnn_train.json")
train = json.loads(train_json.read_text()) if train_json.exists() else {}
default = train.get("patches_file", "parasite_patches.npz")
size = int(train.get("crop", par.PATCH_PX))
# The runtime cuts PATCH_PX patches; a model trained at another size would
# run without error and score nonsense. The constant and the model land together.
assert size == par.PATCH_PX, f"model trained at {size} px but parasites.PATCH_PX is {par.PATCH_PX}"
d = np.load(OUT / (args[0] if args else default))
ref = np.load(OUT / ("parasite_hybrid_check.npy" if hybrid else "parasite_cnn_check.npy"))
P = d["patches"][:len(ref)]
o = (P.shape[1] - size) // 2
crop = P[:, o:o + size, o:o + size]
if hybrid:
    ours = par.PatchScorer.load(par.HYBRID_PATH).score(crop, d["X"][:len(ref)].astype(np.float32))
else:
    ours = par.PatchScorer.load().score(crop)
err = float(np.abs(ours - ref).max())
print(f"cv.dnn vs torch on {len(ref)} patches: max abs diff {err:.2e}")
assert err < 1e-3, "cv.dnn does not reproduce the exported network"
