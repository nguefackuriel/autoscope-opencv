"""Did two builds read the same fields the same way?

A speed number from a different library is only about AutoScope if the library
gives back the same answer. Every profile run records the parasite and white
cell count of each field it timed; this compares two of those runs field by
field.

    python bench/compare_counts.py profile-c8g-stock profile-c8g-cool-tiled

Exit code 1 if any field differs, so it can gate a claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def counts(label: str) -> dict[str, tuple[int, int]]:
    path = RESULTS / f"{label}.json"
    d = json.loads(path.read_text())
    if "counts" not in d:
        raise SystemExit(f"{path.name} was recorded before counts were kept. Re-run it.")
    return {c["field"]: (c["parasites"], c["white_cells"]) for c in d["counts"]}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    ca, cb = counts(a), counts(b)
    shared = sorted(set(ca) & set(cb))
    if not shared:
        raise SystemExit(f"{a} and {b} have no field in common")
    bad = [f for f in shared if ca[f] != cb[f]]
    print(f"{len(shared)} fields in common, {len(bad)} differ\n")
    print(f"{'field':<34}{a:>22}{b:>22}")
    for f in shared:
        mark = "  <-- differs" if ca[f] != cb[f] else ""
        print(f"{f:<34}{str(ca[f]):>22}{str(cb[f]):>22}{mark}")
    if bad:
        raise SystemExit(1)
    print("\nidentical counts on every field")


if __name__ == "__main__":
    main()
