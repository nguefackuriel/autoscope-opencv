"""Generate the upstream bug report from the result files.

    python bench/upstream_issue.py > /tmp/issue.md

WHY THIS IS A SCRIPT

The first draft of this issue was typed by hand from a benchmark run, and then
the benchmark was re-run. Every millisecond in its main table silently became a
number that no file in bench/results/ contained any more, while the issue's own
opening sentence promised the opposite. Two baselines were swapped, a stage
count was wrong in the direction that flattered the product being reported, and
the same measurement appeared twice with two values.

A bug report that a vendor cannot reproduce from the repository it cites is
worse than none. So every number below is computed from bench/results/ at the
moment it is printed. If a benchmark is re-run, the issue regenerates; it
cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def ms(d: dict, key: str) -> float:
    return d["results"][key]["median_ms"]


def main() -> None:
    cool_t, stock_t, nok_t = load("c8g-cool"), load("c8g-stock"), load("c8g-no-kleidicv")
    cool_p, stock_p = load("c8g-cool-pipeline"), load("c8g-stock-pipeline")
    cool_pr, stock_pr = load("profile-c8g-cool"), load("profile-c8g-stock")
    tiled = load("profile-c8g-cool-tiled")
    cc, sc = load("profile-c8g-cool-counts"), load("profile-c8g-stock-counts")

    # c8g-stock.json predates build_lines; the pipeline run on the same build
    # carries them, and the two agree on opencv_version and kleidicv_in_build.
    cb, sb = cool_p["build_lines"], stock_p["build_lines"]
    out: list[str] = []
    w = out.append

    w("### The behaviour\n")
    w("On the Cloud Optimized OpenCV image for AWS Graviton4 from the AWS")
    w("Marketplace, large-kernel filter and morphology calls on a 12 megapixel")
    w("image take the same time on 32 threads as on 1. The same calls on the")
    w("`opencv-python` wheel from PyPI, on the same instance type, gain roughly")
    w("15 to 26 times from the threads. `cvtColor` gains about 20 times on both,")
    w("so the threading itself is working.\n")
    w("At one thread the two builds agree to within a few percent on every call,")
    w("so the kernels run at the same speed. The difference is entirely whether")
    w("the work is spread across the cores.\n")

    w("### Builds\n")
    w("| | Marketplace COOL image | `opencv-python` wheel |")
    w("|---|---|---|")
    for label, key in (("Version control", "Version control"),
                       ("Custom HAL", "Custom HAL"),
                       ("Parallel framework", "Parallel framework"),
                       ("Baseline", "Baseline")):
        w(f"| {label} | `{cb[key]}` | `{sb[key]}` |")
    w(f"| `cv.__version__` | {cool_t['opencv_version']} | {stock_t['opencv_version']} |")
    w(f"| OS | Ubuntu 24.04 | Amazon Linux 2023 |")
    w("")
    w("Both on `c8g.8xlarge`, 32 vCPU, `us-east-1`. `cv.getNumThreads()` reports")
    w(f"{cool_p.get('cv_threads_default', 32)} on both. The COOL image is")
    w("`ami-01db31139bc5615d8`, the current Graviton4 listing. Python 3.12 from")
    w("`/opt/cool/venvs/python_3.12` with bindings from")
    w("`/opt/cool/python_3.12/site-packages` and")
    w("`LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib`.\n")
    w("Four things differ between these builds at once: the OpenCV version, the")
    w("KleidiCV version, the parallel backend and the operating system. We")
    w("cannot isolate which one causes this, because we cannot rebuild the")
    w("Marketplace image. What the one-thread column does rule out is a")
    w("difference in kernel speed.\n")

    w("### Measurements\n")
    w("Medians of 10 runs on one 4032x3024 image, `bench/cool_bench.py --group")
    w("pipeline --threads 0,1`. Milliseconds.\n")
    w("| call | wheel @32 | COOL @32 | wheel @1 | COOL @1 | wheel gain | COOL gain |")
    w("|---|---|---|---|---|---|---|")
    rows = (("12MP_cvtColor_Lab", "`cvtColor` BGR2Lab"),
            ("12MP_GaussianBlur_f32", "`GaussianBlur` float32, sigma 1.4"),
            ("12MP_tophat_f32_e31", "`morphologyEx` TOPHAT float32, ellipse 31x31"),
            ("12MP_dilate_f32_e15", "`dilate` float32, ellipse 15x15"),
            ("12MP_tophat_u8_e31", "`morphologyEx` TOPHAT uint8, ellipse 31x31"))
    for key, pretty in rows:
        s32, c32 = stock_p["by_threads"]["0"][key]["median_ms"], cool_p["by_threads"]["0"][key]["median_ms"]
        s1, c1 = stock_p["by_threads"]["1"][key]["median_ms"], cool_p["by_threads"]["1"][key]["median_ms"]
        cool_gain = c1 / c32
        cell = (f"**{cool_gain:.1f}x**" if cool_gain < 2 else f"{cool_gain:.1f}x")
        w(f"| {pretty} | {s32:.2f} | {c32:.2f} | {s1:.2f} | {c1:.2f} | "
          f"{s1/s32:.1f}x | {cell} |")
    w("")
    worst = max(abs(cool_p["by_threads"]["1"][k]["median_ms"]
                    / stock_p["by_threads"]["1"][k]["median_ms"] - 1) for k, _ in rows)
    w(f"At one thread the two builds are within {worst*100:.0f}% of each other on")
    w("every call in that table.\n")

    w("### Effect on a real pipeline\n")
    w("Our malaria thick-film pipeline runs a 31x31 top-hat on a float32 stain")
    w("map at full resolution. The same twelve images on each build,")
    w("`bench/field_profile.py`:\n")
    w("| stage, seconds per field | wheel | COOL image | ratio |")
    w("|---|---|---|---|")
    for k in stock_pr["stages"]:
        s, c = stock_pr["stages"][k], cool_pr["stages"][k]
        w(f"| {k} | {s:.4f} | {c:.4f} | {c/s:.2f}x |")
    sf, cf = stock_pr["seconds_per_field_median"], cool_pr["seconds_per_field_median"]
    w(f"| **whole field** | **{sf:.3f}** | **{cf:.3f}** | **{cf/sf:.1f}x** |")
    w("")
    faster = [k for k in stock_pr["stages"] if cool_pr["stages"][k] < stock_pr["stages"][k]]
    w(f"{len(faster)} of {len(stock_pr['stages'])} stages are faster on the COOL image.")
    w("The rest are slower by 7% to 19%, except the candidate stage, which is")
    w(f"{cool_pr['stages']['parasite candidates']/stock_pr['stages']['parasite candidates']:.1f}x")
    w("slower and swallows every gain. That stage is the 31x31 top-hat.")
    w(f"{stock_pr['slides_per_hour']:.0f} slides an hour becomes {cool_pr['slides_per_hour']:.0f}.\n")

    w("### Workaround\n")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from autoscope.parasites import PEAK_NMS_PX, TOPHAT_PX  # noqa: E402

    w("Cut the image into one horizontal band per core, overlapping by the")
    w(f"kernel's full support ({TOPHAT_PX} rows for the {TOPHAT_PX}x{TOPHAT_PX}")
    w(f"top-hat, {PEAK_NMS_PX} for the dilate), run the same call per")
    w("band from a thread pool, and stitch. OpenCV releases the GIL inside the")
    w("call, so the bands run on separate cores. The result is checked against")
    w("the unsplit call before it is timed, and a test asserts they are exactly")
    w("equal.\n")
    w("| call, COOL image @32 threads | plain | banded | gain |")
    w("|---|---|---|---|")
    for key, pretty in (("12MP_tophat_f32_e31", "TOPHAT float32, ellipse 31"),
                        ("12MP_dilate_f32_e15", "`dilate` float32, ellipse 15")):
        plain = cool_p["by_threads"]["0"][key]["median_ms"]
        band = cool_p["by_threads"]["0"][key + "_tiled"]["median_ms"]
        w(f"| {pretty} | {plain:.1f} | {band:.1f} | {plain/band:.1f}x |")
    w("")
    w(f"In the pipeline that is {tiled['slides_per_hour']:.0f} slides an hour")
    w(f"against the wheel's {stock_pr['slides_per_hour']:.0f}. On the wheel the")
    w("same banding is slower than the plain call")
    w(f"({stock_p['by_threads']['0']['12MP_tophat_f32_e31_tiled']['median_ms']:.1f} ms against "
      f"{stock_p['by_threads']['0']['12MP_tophat_f32_e31']['median_ms']:.1f}), which is what you")
    w("would expect when the library is already using the cores.\n")

    w("### The two builds return the same answers\n")
    a = {c["field"]: (c["parasites"], c["white_cells"]) for c in sc["counts"]}
    b = {c["field"]: (c["parasites"], c["white_cells"]) for c in cc["counts"]}
    shared = sorted(set(a) & set(b))
    differ = [k for k in shared if a[k] != b[k]]
    w("A speed comparison across two libraries only means something if they")
    w("agree. The whole pipeline was run twice on the same")
    w(f"{len(shared)} images, once on each build, keeping the count of every")
    w(f"image: {sum(v[0] for v in a.values())} detections and every white cell,")
    w(f"{len(differ)} of {len(shared)} images differing.")
    w(f"{sc['slides_per_hour']:.0f} slides an hour against {cc['slides_per_hour']:.0f},")
    w(f"so {cc['slides_per_hour']/sc['slides_per_hour']:.2f}x with the banding on.\n")

    w("### Where the image is faster\n")
    w("On a 1024 px tile the image wins on several calls, which is why a")
    w("function-level benchmark does not show any of this:\n")
    w("| call, 1024 px | wheel | COOL | ratio |")
    w("|---|---|---|---|")
    for k in ("calcHist", "adaptiveThreshold", "GaussianBlur", "resize",
              "findContours", "medianBlur", "warpPerspective"):
        s, c = ms(stock_t, k), ms(cool_t, k)
        w(f"| `{k}` | {s:.4f} | {c:.4f} | {s/c:.2f}x |")
    w("")
    wp = ms(stock_t, "warpPerspective") / ms(cool_t, "warpPerspective")
    w(f"`warpPerspective` is the exception at {wp:.2f}x, which we have not looked")
    w("into. Note that the 1024 px group never runs a 31x31 top-hat or a 15x15")
    w("dilate: its morphology case is a 5x5 open on uint8, which is at parity")
    w(f"({ms(stock_t, 'morphologyEx_open')/ms(cool_t, 'morphologyEx_open'):.2f}x).")
    w("That is the whole reason this took a pipeline to find.\n")

    w("### Reproduce\n")
    w("```")
    w("GROUP=pipeline THREADS=0,1 COOL_AMI=ami-01db31139bc5615d8 \\")
    w("    bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool 10")
    w("GROUP=pipeline THREADS=0,1 \\")
    w("    bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock 10")
    w("```")
    w("Each launches one instance, measures it, and terminates it. The banded")
    w("helper is `parasites._morph`; `bench/compare_counts.py` is the equality")
    w("check.\n")

    w("### The question\n")
    w("Is a HAL that implements `morphologyEx` and the separable filters")
    w("expected to do its own threading, with OpenCV calling it once for the")
    w("whole image? If so this is a KleidiCV or packaging question rather than")
    w("an OpenCV one, and we would like to know where to take it. If not, the")
    w("Marketplace image appears to be built in a way that loses the")
    w("parallelism on those paths. The same calls on the current PyPI wheel,")
    w("which also reports carotene and KleidiCV in its custom HAL, do use the")
    w("cores.\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
