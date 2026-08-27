"""Runtime-venv check that cv.dnn reproduces the torch predictions of the
exported patch network. Run after parasite_cnn_train.py."""

from pathlib import Path

import numpy as np

from autoscope import parasites as par

OUT = Path(__file__).resolve().parent / "results"

d = np.load(OUT / "parasite_patches.npz")
ref = np.load(OUT / "parasite_cnn_check.npy")
P = d["patches"][:len(ref)]
o = (P.shape[1] - par.PATCH_PX) // 2
ours = par.PatchScorer.load().score(P[:, o:o + par.PATCH_PX, o:o + par.PATCH_PX])
err = float(np.abs(ours - ref).max())
print(f"cv.dnn vs torch on {len(ref)} patches: max abs diff {err:.2e}")
assert err < 1e-3, "cv.dnn does not reproduce the exported network"
