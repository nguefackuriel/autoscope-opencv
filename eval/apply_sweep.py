"""Take the sweep's least-biased parameters, write them into segmentation.py,
and re-score on the held-out patients.

Selection rule, stated once so it does not drift: choose the least absolute
count bias, break ties on F1. The count is the denominator of the density, so a
systematic surplus or deficit poisons every parasitaemia in the same direction,
while localisation errors average out across a session.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEG = HERE.parent / "src" / "autoscope" / "segmentation.py"


def pick(grid: list[dict]) -> dict:
    return min(grid, key=lambda r: (abs(r["count_bias_pct"]), -r["f1"]))


def main() -> None:
    sweep = json.loads((HERE / "results" / "wbc_sweep.json").read_text())
    chosen = pick(sweep["grid"])
    best_f1 = max(sweep["grid"], key=lambda r: r["f1"])

    print(f"chosen: seed_fraction={chosen['seed_fraction']} close_px={chosen['close_px']}")
    print(f"  fit-set F1 {chosen['f1']:.3f}, bias {chosen['count_bias_pct']:+.1f}%")
    if best_f1 is not chosen:
        print(f"passed over: F1 {best_f1['f1']:.3f} at bias {best_f1['count_bias_pct']:+.1f}% "
              f"(seed={best_f1['seed_fraction']}, close={best_f1['close_px']})")

    src = SEG.read_text()
    src = re.sub(r"SEED_FRACTION = [0-9.]+", f"SEED_FRACTION = {chosen['seed_fraction']}", src)
    src = re.sub(r"CLOSE_KERNEL_FULL = \d+", f"CLOSE_KERNEL_FULL = {chosen['close_px']}", src)
    SEG.write_text(src)
    print(f"\nwrote parameters into {SEG.name}")

    print("\nscoring on held-out patients, normalisation on:")
    subprocess.run(
        [sys.executable, str(HERE / "wbc_eval.py"),
         "--per-patient", "3", "--tag", "holdout_refit", "--holdout", "--normalise"],
        check=True,
    )


if __name__ == "__main__":
    main()
