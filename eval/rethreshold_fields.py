"""Rebuild agent_fields.json at a different classifier threshold, without
re-running the pipeline.

The per-field quality metrics (sharpness, lighting, debris) do not depend on
the threshold, so they are carried over from the existing table. The counts do,
and they are recomputed from every candidate's score in
parasite_scores_holdout.npz. That is the same arithmetic the pipeline would do,
on the same candidates, so the result is identical to a rerun and takes a
second instead of half an hour.

The white cell count is threshold-free and comes from the score table too.
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--source", default="agent_fields.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old = json.loads((OUT / args.source).read_text())
    quality = {(r["patient"], r["image"]): r for r in old["rows"]}

    d = np.load(OUT / "parasite_scores_holdout.npz")
    off = d["offsets"]
    rows, missing = [], 0
    for i in range(len(off) - 1):
        key = (str(d["patient"][i]), str(d["image"][i]))
        if key not in quality:
            missing += 1
            continue
        scores = d["scores"][off[i]:off[i + 1]].astype(np.float32)
        q = quality[key]
        rows.append({**q,
                     "found_parasites": int((scores >= args.threshold).sum()),
                     "found_wbc": int(d["found_wbc"][i])})
    if missing:
        raise SystemExit(f"{missing} scored images are not in {args.source}")

    (OUT / args.out).write_text(json.dumps({
        "threshold": args.threshold, "patients": old["patients"], "fields": len(rows),
        "rebuilt_from": "parasite_scores_holdout.npz",
        "quality_from": args.source,
        "rows": rows,
    }, indent=2))
    print(f"{len(rows)} fields rebuilt at threshold {args.threshold}")
    a = sum(r["found_parasites"] for r in rows)
    b = sum(r["truth_parasites"] for r in rows)
    print(f"parasites: {a} found against {b} annotated ({(a / b - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
