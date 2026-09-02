"""Recompute the quality columns of a cached field table.

agent_fields.json and negative_fields.json store, per field, what the counter
found and what the quality tool measured. The counts do not depend on the
quality gate, so when the sharpness measure changes only the quality columns
need recomputing, and the counts, which cost 3 s a field, stay as they are.

    ../.venv/bin/python refresh_quality.py agent_fields.json
    ../.venv/bin/python refresh_quality.py negative_fields.json

Rewrites the file in place with focus / even / debris / usable refreshed, and
records which sharpness measure produced them.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv

from autoscope import quality
from autoscope.quality import assess_field

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def image_path(table: dict, row: dict) -> Path:
    if table.get("label") == "uninfected":
        return RAW / table["source_set"] / "Uninfected Patients" / row["patient"] / "tiled" / row["image"]
    return RAW / row["patient"] / row["image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    args = ap.parse_args()
    path = OUT / args.table
    table = json.loads(path.read_text())
    rows = table["rows"]
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        img = cv.imread(str(image_path(table, row)), cv.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {image_path(table, row)}")
        q = assess_field(img)
        row["focus"] = round(q.focus_score, 2)
        row["even"] = round(q.illumination_uniformity, 4)
        row["debris"] = round(q.debris_ratio, 4)
        row["usable"] = q.why_not() is None
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} fields, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    table["quality_measure"] = f"laplacian variance after gaussian sigma {quality.FOCUS_SMOOTH_SIGMA}, gate {quality.FOCUS_MIN}"
    table.pop("quality_from", None)
    path.write_text(json.dumps(table, indent=2))
    usable = sum(r["usable"] for r in rows)
    print(f"{len(rows)} fields refreshed, {usable} usable by the gate, written to {path.name}")


if __name__ == "__main__":
    main()
