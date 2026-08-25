"""Take the sweep's least-biased parameters, write them into segmentation.py,
and re-score on the held-out patients.

Selection rule, stated once so it does not drift: among settings whose absolute
count bias is under BIAS_TOLERANCE, take the best F1; if none qualify, take the
least biased. The count is the denominator of the density, so a systematic
surplus poisons every parasitaemia in the same direction, which is why bias
gates the choice. But bias differences well inside measurement noise do not
justify giving up real F1: on a 100-image fit set, 0.1pp of bias is noise and
5pp of F1 is not.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BIAS_TOLERANCE = 1.0  # percentage points

HERE = Path(__file__).resolve().parent
SEG = HERE.parent / "src" / "autoscope" / "segmentation.py"


def pick(grid: list[dict]) -> dict:
    unbiased = [r for r in grid if abs(r["count_bias_pct"]) <= BIAS_TOLERANCE]
    if unbiased:
        return max(unbiased, key=lambda r: r["f1"])
    return min(grid, key=lambda r: abs(r["count_bias_pct"]))


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
