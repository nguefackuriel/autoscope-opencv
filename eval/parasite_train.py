"""Train the parasite classifier on the candidate table, honestly, and write it
as an ONNX graph for cv.dnn.

Two numbers come out:

  cross-validated   5 folds split BY PATIENT, so the network never sees a
                    candidate from a patient it is scored on. Fit-set estimate.
  final model       trained on all fit candidates, exported to
                    models/parasite_mlp.onnx, and scored end to end on the
                    held-out patients by parasite_eval.py. That is the number
                    for the report.

Splitting by candidate instead of by patient would leak: two candidates from
the same image share stain, focus and background, and the network would learn
the image rather than the parasite.

The network is small enough (19 -> 64 -> 32 -> 1) that training is plain
numpy: forward, backward, Adam. No torch. That is deliberate: the only wheel
of torch that runs on this x86 Mac was built against NumPy 1 and will not load
next to the NumPy 2 the pipeline pins. The `onnx` package is the one extra
dependency (requirements-train.txt), used to write the graph. Standardisation
is folded into the first layer so the runtime passes raw features, and the
export is checked against cv.dnn before anything is saved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# --- repo paths: run from anywhere, installed or not -------------------------
# Azure ML runs the code snapshot without `pip install -e .`, so the editable
# install that normally puts src/ on sys.path is absent and every `from
# autoscope import ...` fails. This depends on __file__ alone, so it survives
# any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from autoscope import parasites as par

OUT = Path(__file__).resolve().parent / "results"
WIDTH = (64, 32)


def prf(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    bias = (tp + fp - (tp + fn)) / max(tp + fn, 1) * 100
    return p, r, f, bias


class MLP:
    """Sigmoid(W3 relu(W2 relu(W1 std(x)))), trained with Adam on weighted BCE."""

    def __init__(self, n_in, mean, std, seed=0):
        rng = np.random.RandomState(seed)
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        dims = (n_in,) + WIDTH + (1,)
        # np.float32 on the scale: under NumPy 2 a float64 scalar promotes the whole
        # array to float64, which then leaks into the ONNX file (see to_onnx).
        self.W = [rng.randn(dims[i], dims[i + 1]).astype(np.float32) * np.float32(np.sqrt(2.0 / dims[i])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1], np.float32) for i in range(len(dims) - 1)]

    def _forward(self, X):
        h = (X - self.mean) / self.std
        acts = [h]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return acts

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-self._forward(X.astype(np.float32))[-1][:, 0]))

    def fit(self, X, y, epochs=60, lr=2e-3, wd=1e-4, bs=512, pos_weight=None, seed=0):
        rng = np.random.RandomState(seed)
        X = X.astype(np.float32); y = y.astype(np.float32)
        if pos_weight is None:
            pos = y.mean(); pos_weight = (1 - pos) / pos * 0.5
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        t = 0; n = len(y); steps = epochs * ((n + bs - 1) // bs)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb, yb = X[idx], y[idx]
                acts = self._forward(xb)
                logit = acts[-1][:, 0]
                p = 1.0 / (1.0 + np.exp(-logit))
                w = np.where(yb == 1, pos_weight, 1.0).astype(np.float32)
                # d(weighted BCE)/dlogit
                g = (w * (p - yb) / len(yb)).astype(np.float32)[:, None]
                grads = []
                for k in range(len(self.W) - 1, -1, -1):
                    gW = acts[k].T @ g + wd * self.W[k]
                    gb = g.sum(axis=0)
                    grads.append((gW, gb))
                    if k > 0:
                        g = (g @ self.W[k].T) * (acts[k] > 0)
                grads = grads[::-1]
                t += 1
                cur_lr = lr * 0.5 * (1 + np.cos(np.pi * t / steps))   # cosine decay
                for j, (gW, gb) in enumerate(grads):
                    for pi, gr in ((j, gW), (len(self.W) + j, gb)):
                        m[pi] = 0.9 * m[pi] + 0.1 * gr
                        v[pi] = 0.999 * v[pi] + 0.001 * gr * gr
                        mh = m[pi] / (1 - 0.9 ** t); vh = v[pi] / (1 - 0.999 ** t)
                        params[pi] -= cur_lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def to_onnx(self, path: Path):
        """features (n, 19) -> p (n, 1). Standardisation is layer 0."""
        n_in = len(self.mean)
        # Every tensor is cast to float32 explicitly. cv.dnn's Gemm reads a
        # float64 initializer as if it were float32 and returns NaN without
        # complaint; only its MatMul path bothers to raise. Do not remove.
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        inits = [
            numpy_helper.from_array(f32(np.diag(1.0 / self.std)), "W0"),
            numpy_helper.from_array(f32(-self.mean / self.std), "b0"),
        ]
        nodes = [helper.make_node("Gemm", ["features", "W0", "b0"], ["h0"])]
        prev = "h0"
        for i, (W, b) in enumerate(zip(self.W, self.b), start=1):
            inits += [numpy_helper.from_array(f32(W), f"W{i}"), numpy_helper.from_array(f32(b), f"b{i}")]
            nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"b{i}"], [f"z{i}"]))
            if i < len(self.W):
                nodes.append(helper.make_node("Relu", [f"z{i}"], [f"h{i}"])); prev = f"h{i}"
            else:
                nodes.append(helper.make_node("Sigmoid", [f"z{i}"], ["p"]))
        graph = helper.make_graph(
            nodes, "parasite_mlp",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", n_in])],
            [helper.make_tensor_value_info("p", TensorProto.FLOAT, ["n", 1])],
            initializer=inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="autoscope")
        model.ir_version = 8
        onnx.checker.check_model(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = np.load(OUT / "parasite_candidates.npz")
    X, y, patient = d["X"].astype(np.float32), d["y"], d["patient"]
    print(f"{len(y)} candidates, {int(y.sum())} parasites, {len(set(patient))} patients")

    pats = sorted(set(patient))
    rng = np.random.RandomState(0); rng.shuffle(pats)
    fold_of = {p: i % args.folds for i, p in enumerate(pats)}
    folds = np.array([fold_of[p] for p in patient])

    scores = np.zeros(len(y), np.float32)
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        net = MLP(X.shape[1], X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-6, seed=k).fit(X[tr], y[tr], args.epochs, seed=k)
        scores[te] = net.predict(X[te])
        _, _, f, _ = prf(y[te], scores[te], 0.5)
        print(f"  fold {k}: {int(te.sum())} candidates, F1 {f:.3f}")

    print("\ncross-validated by patient (candidate level):")
    print("  thr   precision  recall   F1    count bias")
    table = []
    for thr in np.arange(0.2, 0.81, 0.05):
        p, r, f, b = prf(y, scores, thr)
        table.append({"threshold": round(float(thr), 2), "precision": round(p, 4),
                      "recall": round(r, 4), "f1": round(f, 4), "count_bias_pct": round(b, 2)})
        print(f"  {thr:.2f}   {p:.3f}      {r:.3f}   {f:.3f}   {b:+.1f}%")

    cand = json.loads((OUT / "parasite_candidates.json").read_text())
    print(f"\ncandidate recall ceiling: {cand['candidate_recall']:.3f}")

    final = MLP(X.shape[1], X.mean(axis=0), X.std(axis=0) + 1e-6, seed=0).fit(X, y, args.epochs, seed=0)
    final.to_onnx(par.MODEL_PATH)
    check = X[:2000]
    err = float(np.abs(final.predict(check) - par.Scorer.load().score(check)).max())
    print(f"\nexported {par.MODEL_PATH.name}; numpy vs cv.dnn max abs diff {err:.2e}")

    (OUT / "parasite_train.json").write_text(json.dumps({
        "candidates": int(len(y)), "positives": int(y.sum()), "patients": len(pats),
        "epochs": args.epochs, "folds": args.folds, "width": WIDTH, "features": par.FEATURE_NAMES,
        "cv_by_patient": table, "candidate_recall": cand["candidate_recall"],
        "export_max_abs_diff": err,
    }, indent=2))


if __name__ == "__main__":
    main()
