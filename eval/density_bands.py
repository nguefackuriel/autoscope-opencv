"""Where the density error lives: numerator or denominator, sparse or dense
slides, and what a clean field costs, from the cached field tables.

Reads agent_fields.json (every held-out field, real counter and expert
counts) and negative_fields.json (every clean field of the uninfected set).
Takes a second. Written on 5 September 2026 because the answer decided the
plan: the four worst slides were all parasite over-counts with the white
cells within 6%, and the dense slides under-count parasites with the white
cells right too, so both failures are in the numerator.
"""

import collections
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
WBC_PER_UL = 8000.0


def band(per_ul: float) -> str:
    if per_ul < 1000:
        return "under 1 000"
    if per_ul < 10000:
        return "1 000 to 10 000"
    if per_ul < 100000:
        return "10 000 to 100 000"
    return "over 100 000"


def main():
    af = json.loads((OUT / "agent_fields.json").read_text())
    by = collections.defaultdict(lambda: np.zeros(4))
    for r in af["rows"]:
        by[r["patient"]] += [r["found_parasites"], r["found_wbc"], r["truth_parasites"], r["truth_wbc"]]

    slides = []
    for p, (fp, fw, tp, tw) in by.items():
        truth = tp / tw * WBC_PER_UL
        found = fp / max(fw, 1) * WBC_PER_UL
        slides.append({"patient": p, "expert_per_ul": round(truth), "found_per_ul": round(found),
                       "density_error_pct": round((found - truth) / truth * 100, 1),
                       "parasite_error_pct": round((fp - tp) / max(tp, 1) * 100, 1),
                       "wbc_error_pct": round((fw - tw) / max(tw, 1) * 100, 1)})
    err = np.array([s["density_error_pct"] for s in slides])
    print(f"held-out, {len(slides)} slides, every field, threshold {af['threshold']}:")
    print(f"  mean |density error| {np.abs(err).mean():.1f}%, within 25%: {100 * (np.abs(err) <= 25).mean():.0f}%, "
          f"within 50%: {100 * (np.abs(err) <= 50).mean():.0f}%, bias {err.mean():+.1f}%")

    bands = collections.OrderedDict()
    print(f"\n  {'expert density':<18s} {'n':>3s} {'|err|':>7s} {'in 25%':>7s} {'parasites':>12s} {'white cells':>12s}")
    for name in ["under 1 000", "1 000 to 10 000", "10 000 to 100 000", "over 100 000"]:
        m = [s for s in slides if band(s["expert_per_ul"]) == name]
        if not m:
            continue
        tot = sum((by[s["patient"]] for s in m), np.zeros(4))
        e = np.array([s["density_error_pct"] for s in m])
        row = {"slides": len(m), "mean_abs_error_pct": round(float(np.abs(e).mean()), 1),
               "within_25_pct": round(float(100 * (np.abs(e) <= 25).mean()), 0),
               "parasite_bias_pct": round(float((tot[0] - tot[2]) / tot[2] * 100), 1),
               "wbc_bias_pct": round(float((tot[1] - tot[3]) / tot[3] * 100), 1)}
        bands[name] = row
        print(f"  {name:<18s} {len(m):3d} {row['mean_abs_error_pct']:6.1f}% {row['within_25_pct']:6.0f}% "
              f"{row['parasite_bias_pct']:+11.1f}% {row['wbc_bias_pct']:+11.1f}%")

    worst = sorted(slides, key=lambda s: -abs(s["density_error_pct"]))[:8]
    print("\n  worst slides       expert/uL   density   parasites   white cells")
    for s in worst:
        print(f"  {s['patient']:<16s} {s['expert_per_ul']:9d} {s['density_error_pct']:+8.0f}% {s['parasite_error_pct']:+10.0f}% "
              f"{s['wbc_error_pct']:+12.0f}%")

    clean = {}
    nf_path = OUT / "negative_fields.json"
    if nf_path.exists():
        nf = json.loads(nf_path.read_text())
        split = json.loads((OUT / "uninfected_split.json").read_text())
        print(f"\nclean fields, threshold {nf['threshold']}: false parasites per field")
        print(f"  {'half':<12s} {'fields':>6s} {'mean':>6s} {'p90':>5s} {'max':>4s} {'zero':>5s} {'per 200 WBC':>12s} {'floor /uL':>10s}")
        for name in ("fit", "test"):
            pats = set(split[name])
            fp = np.array([r["found_parasites"] for r in nf["rows"] if r["patient"] in pats])
            wbc = np.array([r["found_wbc"] for r in nf["rows"] if r["patient"] in pats])
            per200 = 200 * fp.sum() / max(wbc.sum(), 1)
            clean[name] = {"fields": int(len(fp)), "false_per_field_mean": round(float(fp.mean()), 2),
                           "p90": float(np.percentile(fp, 90)), "max": int(fp.max()),
                           "zero_pct": round(float(100 * (fp == 0).mean()), 1),
                           "false_per_200_wbc": round(float(per200), 1),
                           "noise_floor_per_ul": round(float(per200 / 200 * WBC_PER_UL))}
            c = clean[name]
            label = "fit (trained)" if name == "fit" else "frozen"
            print(f"  {label:<12s} {c['fields']:6d} {c['false_per_field_mean']:6.2f} {c['p90']:5.0f} {c['max']:4d} "
                  f"{c['zero_pct']:4.0f}% {c['false_per_200_wbc']:12.1f} {c['noise_floor_per_ul']:10d}")

    (OUT / "density_bands.json").write_text(json.dumps({
        "threshold": af["threshold"], "slides": len(slides),
        "mean_abs_density_error_pct": round(float(np.abs(err).mean()), 1),
        "within_25_pct": round(float(100 * (np.abs(err) <= 25).mean()), 0),
        "within_50_pct": round(float(100 * (np.abs(err) <= 50).mean()), 0),
        "bands": bands, "worst": worst, "clean_fields": clean, "per_slide": sorted(slides, key=lambda s: s["patient"]),
    }, indent=2))


if __name__ == "__main__":
    main()
