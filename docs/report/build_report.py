"""Build the technical report PDF for the 26 October submission.

    .venv/bin/python docs/report/build_report.py
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf="AutoScope - Technical Report.pdf" \
      docs/report/report-print.html

The second command is the same one that produced the proposal PDF in August;
there is no pandoc, weasyprint or wkhtmltopdf on this machine, and adding one
two weeks before a deadline is not a good trade. `--build-pdf` runs it.

Every figure comes from `report_numbers.py`, which reads the result files. Nothing in
the prose below is a typed-in number, so re-running the evaluation and
rebuilding cannot produce a report that disagrees with the repository.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_numbers import AGENT_ARMS, arm_slug, figures  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: A4; margin: 19mm 17mm 16mm; }
:root {
  --ink: #1a1618; --muted: #5d555f; --rule: #d8d0da; --giemsa: #6d2a80;
  --wash: #f6f1f7; --good: #1f5c40; --miss: #8a3324;
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; background: #fff; color: var(--ink);
  font: 10.2pt/1.5 "Source Serif 4", Georgia, "Times New Roman", serif;
}
h1, h2, h3, .eyebrow, th, .chip, figcaption, .fignum {
  font-family: "IBM Plex Sans", -apple-system, "Helvetica Neue", sans-serif;
}
code, .num, td.num, pre { font-family: "IBM Plex Mono", Menlo, Consolas, monospace; }

/* ---- cover ---- */
.cover { height: 247mm; display: flex; flex-direction: column; justify-content: space-between; }
.cover .top { padding-top: 26mm; }
.eyebrow { font-size: 8.2pt; letter-spacing: .15em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
h1 { font-size: 40pt; line-height: 1; letter-spacing: -.025em; margin: 7mm 0 4mm; font-weight: 600; }
.cover .sub { font-size: 14pt; line-height: 1.38; color: var(--muted); max-width: 118mm; margin: 0; }
.cover .who { font-size: 9.6pt; color: var(--muted); border-top: 1.5pt solid var(--ink); padding-top: 4mm; }
.cover .who b { color: var(--ink); font-weight: 600; }
.cover .headline { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0; border: 1pt solid var(--rule); margin: 9mm 0; }
.cover .headline div { padding: 5mm 5mm; border-right: 1pt solid var(--rule); }
.cover .headline div:nth-child(2n) { border-right: none; }
.cover .headline div:nth-child(-n+2) { border-bottom: 1pt solid var(--rule); }
.cover .headline .n { font-family: "IBM Plex Mono", monospace; font-size: 19pt; font-variant-numeric: tabular-nums; display: block; }
.cover .headline .k { font-size: 8.6pt; color: var(--muted); display: block; margin-top: 1.5mm; line-height: 1.35; }

/* ---- running text ---- */
/* Sections flow rather than each starting a page. With a forced break per
   section, the ones shorter than a page left half-empty sheets and the
   break-inside:avoid on tables and callouts pushed those onto pages of their
   own: a 20-page document carrying 13 pages of text. */
section { margin-top: 9mm; }
section:first-of-type { margin-top: 0; }
h2 {
  font-size: 15pt; font-weight: 600; letter-spacing: -.012em; margin: 0 0 1mm;
  padding-bottom: 2mm; border-bottom: 1.5pt solid var(--ink);
  break-before: auto; break-after: avoid;
}
h3 { break-after: avoid; }
.lede { break-after: avoid; }
h2 .fignum { color: var(--giemsa); margin-right: 3mm; font-weight: 600; }
h3 { font-size: 10.8pt; font-weight: 600; margin: 7mm 0 1.5mm; }
p { margin: 0 0 3.2mm; max-width: 162mm; orphans: 3; widows: 3; }
.lede { font-size: 11.4pt; color: var(--muted); margin: 3mm 0 5mm; }
ul, ol { margin: 0 0 3.2mm; padding-left: 5mm; }
li { margin-bottom: 1.4mm; }
strong { font-weight: 600; }
code { font-size: 8.8pt; background: var(--wash); padding: .3mm 1mm; border-radius: .6mm; }
a { color: var(--giemsa); text-decoration: none; border-bottom: .4pt solid var(--rule); }

/* ---- tables ---- */
table { width: 100%; border-collapse: collapse; font-size: 9.1pt; margin: 2mm 0 4mm; break-inside: avoid; }
caption { caption-side: top; text-align: left; font-size: 8.4pt; color: var(--muted); padding-bottom: 1.5mm; font-family: "IBM Plex Sans", sans-serif; }
th { text-align: left; font-size: 7.8pt; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; padding: 0 3mm 1.6mm 0; border-bottom: 1pt solid var(--ink); }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td { padding: 1.5mm 3mm 1.5mm 0; border-bottom: .4pt solid var(--rule); vertical-align: top; }
tbody tr:last-child td { border-bottom: none; }
tr.total td { border-top: 1pt solid var(--ink); border-bottom: none; font-weight: 600; }
tr.group td { border-top: 1pt solid var(--ink); }
td .muted { color: var(--muted); font-size: 8.2pt; }
.chip { font-size: 7.6pt; letter-spacing: .06em; text-transform: uppercase; font-weight: 600; padding: .4mm 1.6mm; border-radius: .6mm; white-space: nowrap; }
.chip.met { color: var(--good); border: .7pt solid var(--good); }
.chip.missed { color: var(--miss); border: .7pt dashed var(--miss); }

/* ---- callouts ---- */
.note { background: var(--wash); border-left: 2pt solid var(--giemsa); padding: 4mm 5mm; margin: 4mm 0; break-inside: avoid; }
.note p:last-child { margin-bottom: 0; }
.note .label { font-family: "IBM Plex Sans", sans-serif; font-size: 7.8pt; letter-spacing: .12em; text-transform: uppercase; color: var(--giemsa); font-weight: 600; display: block; margin-bottom: 1.5mm; }
pre { font-size: 7.4pt; line-height: 1.45; background: var(--wash); padding: 3.5mm 4mm; margin: 2mm 0 4mm; overflow: hidden; white-space: pre-wrap; break-inside: avoid; padding-left: 6mm; text-indent: -2.5mm; }
figure { margin: 3mm 0 5mm; break-inside: avoid; }
figure img { width: 100%; border: .4pt solid var(--rule); }
figcaption { font-size: 8.4pt; color: var(--muted); margin-top: 1.5mm; }
footer { margin-top: 9mm; border-top: 1.5pt solid var(--ink); padding-top: 4mm; font-size: 8.8pt; color: var(--muted); }
"""


# How many decimals each figure is worth, by key. Anything not listed has its
# trailing zeros trimmed.
#
# Two opposite mistakes are possible here and this table exists because the
# report made both. A blanket `:,.3f` printed "70.000% of slides" and "count
# bias -3.830%", three digits of precision we do not have. Trimming everything
# instead printed the held-out F1 of 0.7899 as "0.79", dropping a significant
# figure from the number the whole evaluation turns on. A score keeps its three
# decimals; a percentage keeps one; a count keeps none.
_DECIMALS = {
    "cool_c8g_stock_slides_hour": 0, "cool_c8g_cool_slides_hour": 0,
    "cool_c8g_no_kleidicv_slides_hour": 0, "cool_tiled_slides_hour": 0,
    "alloc_agent_r": 2, "alloc_fixed_r": 2,
    "cost_stock": 2, "cost_cool": 2, "cost_nokleidi": 2,
    "cost_stock_spot": 2, "cost_cool_spot": 2, "cost_nokleidi_spot": 2,
    "same_stock_slides_hour": 0, "same_cool_slides_hour": 0,
    "same_stock_per_field": 3, "same_cool_per_field": 3,
    "held_f1": 3, "held_precision": 3, "held_recall": 3, "held_ratio_spearman": 3,
    "candidate_recall": 3, "tz_auc": 3, "tz_auc_density": 3,
    "pv_f1": 3, "pv_recall": 3, "pv_precision": 3,
    "pv_score_matched": 3, "pv_score_unmatched": 3,
    "dec_sensitivity": 2, "dec_specificity": 2, "threshold": 2,
    "neg_false_per_200": 2,
    "sc_k": 3,
}


def _fmt(v, key: str = "") -> str:
    """One number, formatted the way a reader of this report expects it."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if key in _DECIMALS:
            return f"{v:,.{_DECIMALS[key]}f}"
        out = f"{v:,.3f}".rstrip("0").rstrip(".")
        return out if out not in ("", "-") else "0"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _pct(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}%"


def _row(label: str, *cells: str, cls: str = "") -> str:
    tds = "".join(f'<td class="num">{c}</td>' for c in cells)
    return f'<tr class="{cls}"><td>{label}</td>{tds}</tr>'


def build(f: dict) -> str:
    """The report. Prose here, numbers from `f`."""
    # --- derived comparisons, computed rather than asserted
    tile_names = list(f["cool_c8g_stock_tile"])
    geo = {}
    for arm in ("c8g_no_kleidicv", "c8g_stock", "c8g_cool", "c7i_stock"):
        r = [f["cool_c8g_stock_tile"][n] / f[f"cool_{arm}_tile"][n] for n in tile_names]
        geo[arm] = statistics.geometric_mean(r)
    cool_vs_stock_pipeline = f["cool_c8g_stock_slides_hour"] / f["cool_c8g_cool_slides_hour"]
    cand_ratio = (f["cool_c8g_cool_stages"]["parasite candidates"]
                  / f["cool_c8g_stock_stages"]["parasite candidates"])

    sweep_rows = []
    for n, pretty in (("12MP_cvtColor_Lab", "cvtColor BGR2Lab"),
                      ("12MP_GaussianBlur_f32", "GaussianBlur, float32"),
                      ("12MP_tophat_f32_e31", "morphologyEx TOPHAT, float32, ellipse 31"),
                      ("12MP_dilate_f32_e15", "dilate, float32, ellipse 15"),
                      ("12MP_tophat_u8_e31", "morphologyEx TOPHAT, uint8, ellipse 31")):
        s32, c32 = f["c8g_stock_12mp_t0"][n], f["c8g_cool_12mp_t0"][n]
        s1, c1 = f["c8g_stock_12mp_t1"][n], f["c8g_cool_12mp_t1"][n]
        sweep_rows.append(_row(
            pretty, f"{s32:,.2f}", f"{c32:,.2f}", f"{s1:,.2f}", f"{c1:,.2f}",
            f"{s1/s32:.1f}&times;", f"<strong>{c1/c32:.1f}&times;</strong>"))

    # The three fixed policies first, then every model by fields read. A model
    # that did not count on every slide says so in its label, because its error
    # is over fewer slides and cannot be read against the others.
    fixed = [(arm, label) for arm, model, label in AGENT_ARMS if model is None]
    models = sorted(((arm, label) for arm, model, label in AGENT_ARMS if model),
                    key=lambda al: f[f"arm_{arm_slug(al[0])}_found_fields"])
    agent_rows = []
    for i, (arm, label) in enumerate(fixed + models):
        slug = arm_slug(arm)
        compared = f[f"arm_{slug}_found_compared"]
        if compared < f["agent_patients"]:
            label += (f' <span class="muted">({compared} of {f["agent_patients"]} '
                      f'slides counted)</span>')
        stops = f"{f[f'arm_{slug}_stops']}" if f"arm_{slug}_stops" in f else ""
        agent_rows.append(_row(
            label,
            f"{f[f'arm_{slug}_found_fields']}",
            _pct(f[f"arm_{slug}_found_mape"]),
            f"{f[f'arm_{slug}_found_settled']}",
            f"{f[f'arm_{slug}_found_escalated']}",
            stops,
            cls="group" if i == len(fixed) else ""))

    # What the system does, measured. An earlier version of this table scored
    # the work against seven targets we set ourselves in September, which is
    # our business and not a reader's.
    results = [
        ("Parasites found, held-out patients",
         f"<strong>F1 {f['held_f1']:.3f}</strong>, precision {f['held_precision']:.3f}, "
         f"recall {f['held_recall']:.3f}, on {f['held_patients']} patients and "
         f"{f['held_images']} images", "below"),
        ("Infected slides called positive",
         f"<strong>{f['dec_infected_slides'] - f['dec_infected_called_negative'] - f['dec_infected_to_person']}"
         f" of {f['dec_infected_slides']}</strong>, none called negative, "
         f"{f['dec_infected_to_person']} sent to a person", "good"),
        ("Clean slides called positive",
         f"<strong>{f['dec_clean_called_positive']} of {f['dec_clean_slides']}</strong>; "
         f"{f['dec_clean_to_person']} sent to a person, giving specificity "
         f"{f['dec_specificity']:.2f}", "below"),
        ("Density against the expert count",
         f"within &plusmn;25% on <strong>{_pct(f['density_within_25'])}</strong> of "
         f"{f['density_slides']} slides, mean absolute error "
         f"{_pct(f['density_mae_pct'])}", "good"),
        ("Limit of detection, by dilution",
         f"<strong>{f['lod_per_ul']} parasites/&micro;L</strong> at 95% sensitivity, "
         f"over {f['lod_repeats']} repeats", "good"),
        ("Fields read for the same density error",
         f"<strong>{f['arm_bedrock_found_fields']}</strong> by the agent against "
         f"{f['arm_who_200_found_fields']} by the WHO protocol, over "
         f"{f['agent_patients']} slides", "good"),
        ("Another laboratory, scored once",
         f"image-level AUC <strong>{f['tz_auc']:.3f}</strong> on "
         f"{f['tz_infected_images'] + f['tz_uninfected_images']:,} Tanzanian images, "
         f"after a ten-slide calibration", "good"),
        ("Another species, scored once",
         f"<em>P. vivax</em> <strong>F1 {f['pv_f1']:.3f}</strong> against "
         f"{f['held_f1']:.3f} on falciparum", "below"),
        ("Throughput and cost on Graviton4",
         f"<strong>{f['cool_c8g_stock_slides_hour']:.0f} slides an hour</strong> on one "
         f"c8g.8xlarge, ${f['cost_stock']:.2f} per 1,000 slides on demand", "good"),
    ]
    crit_rows = "".join(
        f'<tr><td>{c}</td><td>{m}</td></tr>' for c, m, _ in results)

    t = Template(_TEMPLATE)
    return t.substitute(
        css=CSS,
        built=date.today().strftime("%-d %B %Y"),
        crit_rows=crit_rows,
        agent_rows="".join(agent_rows),
        sweep_rows="".join(sweep_rows),
        geo_cool=f"{geo['c8g_cool']:.2f}",
        geo_nokleidi=f"{geo['c8g_no_kleidicv']:.2f}",
        geo_c7i=f"{geo['c7i_stock']:.2f}",
        cool_vs_stock_pipeline=f"{cool_vs_stock_pipeline:.1f}",
        cand_ratio=f"{cand_ratio:.1f}",
        **{k: _fmt(v, k) for k, v in f.items() if not isinstance(v, dict)},
        # a few that want their own formatting
        f1=f"{f['held_f1']:.3f}", prec=f"{f['held_precision']:.3f}",
        rec=f"{f['held_recall']:.3f}", pvf1=f"{f['pv_f1']:.3f}",
        tzauc=f"{f['tz_auc']:.3f}", sens=f"{f['dec_sensitivity']:.2f}",
        spec=f"{f['dec_specificity']:.2f}",
        floor=f"{f['floor_per_ul']:.0f}",
        stock_sph=f"{f['cool_c8g_stock_slides_hour']:.0f}",
        cool_sph=f"{f['cool_c8g_cool_slides_hour']:.0f}",
        nokleidi_sph=f"{f['cool_c8g_no_kleidicv_slides_hour']:.0f}",
        stock_spf=f"{f['cool_c8g_stock_per_field']:.3f}",
        cool_spf=f"{f['cool_c8g_cool_per_field']:.3f}",
        tiled_sph=f"{f['cool_tiled_slides_hour']:.0f}",
        tiled_spf=f"{f['cool_tiled_per_field']:.3f}",
        cool_c8g_cool_stages_cand=f"{f['cool_c8g_cool_stages']['parasite candidates']:.3f}",
        cool_hal=f["cool_build_lines"]["Custom HAL"],
        cool_par=f["cool_build_lines"]["Parallel framework"],
        cool_base=f["cool_build_lines"]["Baseline"],
        cool_vc=f["cool_build_lines"]["Version control"],
        stock_hal=f["cool_stock_build_lines"]["Custom HAL"],
        stock_par=f["cool_stock_build_lines"]["Parallel framework"],
        stock_base=f["cool_stock_build_lines"]["Baseline"],
        stock_vc=f["cool_stock_build_lines"]["Version control"],
    )


_TEMPLATE = ""  # filled in by report_text.py, kept separate so prose edits are readable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "report-print.html"))
    ap.add_argument("--build-pdf", action="store_true",
                    help="also run headless Chrome, as the proposal did")
    args = ap.parse_args()

    global _TEMPLATE
    _TEMPLATE = (HERE / "report_text.html").read_text()

    html = build(figures())
    out = Path(args.out)
    out.write_text(html)
    # Template.substitute, not safe_substitute: it raises on a placeholder with
    # no value instead of printing it as "$held_f1" in the finished PDF. A
    # literal dollar in the prose has to be written $$ for the same reason.
    print(f"wrote {out.relative_to(ROOT)}  ({len(html):,} bytes)")

    if args.build_pdf:
        if not Path(CHROME).exists():
            raise SystemExit(f"no Chrome at {CHROME}; see the docstring")
        # Into the repository, because the report is the submission and a judge
        # who clones this should not have to install Chrome to read it. The
        # copy beside the repository is the one we attach to things.
        pdf = HERE / "AutoScope-Technical-Report.pdf"
        subprocess.run([CHROME, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", str(out)], check=True)
        print(f"pdf: {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1e6:.2f} MB)")
        beside = ROOT.parent / "AutoScope - Technical Report.pdf"
        beside.write_bytes(pdf.read_bytes())
        print(f"copied to {beside}")


if __name__ == "__main__":
    main()
