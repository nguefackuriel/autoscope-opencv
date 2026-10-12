"""Render a reading session as the frames a person would have seen.

The agent's work is invisible in a table. This draws it: for each field the
agent read, the actual NLM image, the parasites the pipeline found circled,
the white cells marked, and underneath the running estimate with its interval
narrowing and the move the agent chose with its own reason.

    python eval/render_session.py --session TF101_233 --out docs/figures/session
    python eval/render_session.py --session TF101_233 --gif

WHAT IS AND IS NOT DRAWN

Everything on a frame is real: the image is the NLM capture, the circles are
`count_field`'s detections at the shipped threshold, and the numbers are the
ones in `agent_eval.json`. Nothing is re-run and nothing is staged.

What is NOT drawn is a slide being traversed. NLM fields are separate captures
minutes apart, not tiles of a mosaic, so there is no spatial layout to pan
across and any animation of a stage gliding over a smear would be fiction.
The honest visual is a sequence of fields, which is what a technician actually
sees down an eyepiece: one view, then another.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope.counter import count_field
from autoscope.density import density
from autoscope.parasites import PatchScorer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "eval" / "results"

W, H = 1280, 860          # frame size
PANEL = 250               # height of the text panel under the image
INK = (235, 233, 228)
MUTED = (154, 149, 142)
ACCENT = (224, 154, 195)  # BGR of the page's violet
FOUND = (124, 199, 156)   # BGR green, same as the page's "negative" colour
WBC = (200, 160, 90)


def _font(img, text, org, scale=0.6, colour=INK, weight=1):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv.LINE_AA)


def _bar(img, x, y, w, h, per_ul, lo, hi, floor):
    """The interval on a log axis, the same one the demo page draws."""
    LO, HI = 10.0, 200_000.0
    def pos(v):
        v = min(max(v, LO), HI)
        return x + int((np.log10(v) - np.log10(LO)) / (np.log10(HI) - np.log10(LO)) * w)
    cv.rectangle(img, (x, y), (x + w, y + h), (58, 55, 62), -1)
    if per_ul > 0:
        cv.rectangle(img, (pos(lo), y + 5), (pos(hi), y + h - 5), (120, 90, 150), -1)
        cv.line(img, (pos(per_ul), y), (pos(per_ul), y + h), ACCENT, 2)
    cv.line(img, (pos(floor), y), (pos(floor), y + h), MUTED, 1)
    for v, label in ((10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k"), (100_000, "100k")):
        _font(img, label, (pos(v) - 8, y + h + 16), 0.42, MUTED)


def frames(session: str, steps: list[dict], scorer) -> list[np.ndarray]:
    """One frame per field: the image on the left, the numbers on the right.

    Markers are drawn AFTER the downscale, with the coordinates scaled and a
    fixed pixel thickness. Drawing them at 4032 px and then shrinking to 600
    turns a 3 px ring into a sub-pixel smudge, which is how the first version
    produced frames with invisible detections.
    """
    from autoscope import fov

    out = []
    counted = [s for s in steps if s.get("counted")]
    floor = steps[0]["observation"]["reference"]["noise_floor_per_ul"]
    IMG_W = 620

    for i, step in enumerate(counted, 1):
        path = RAW / session / step["image"]
        img = cv.imread(str(path))
        if img is None:
            continue

        result = count_field(img, scorer=scorer)

        # Crop to the lit disc: the black surround is most of the frame and
        # none of the slide.
        disc = fov.detect(img)
        if disc is not None:
            x0, y0, w, h = disc.bounds
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
        else:
            x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]

        crop = img[y0:y1, x0:x1]
        k = IMG_W / crop.shape[1]
        small = cv.resize(crop, None, fx=k, fy=k, interpolation=cv.INTER_AREA)

        for c in result.white_cell_points:
            cv.circle(small, (int((c.x - x0) * k), int((c.y - y0) * k)),
                      max(int(44 * k), 6), WBC, 2, cv.LINE_AA)
        for pt in result.parasite_points:
            cv.circle(small, (int((pt.x - x0) * k), int((pt.y - y0) * k)),
                      max(int(26 * k), 7), FOUND, 2, cv.LINE_AA)

        frame = np.full((H, W, 3), 26, np.uint8)
        top = (H - small.shape[0]) // 2
        frame[top:top + small.shape[0], 30:30 + small.shape[1]] = small

        totals = step["observation"]["totals"]
        d = density(totals["parasites"], max(totals["white_cells"], 1))
        move = step.get("move", {})

        x = 30 + IMG_W + 48
        # The patient goes on the frame, not only in the page caption. One GIF
        # runs four sessions back to back and the slide changing is the whole
        # point of it; without this the viewer sees the numbers jump and has no
        # way to know why.
        _font(frame, session, (x, 92), 0.62, INK, 2)
        _font(frame, f"FIELD {i} OF {len(counted)}", (x, 124), 0.5, MUTED)
        _font(frame, step["image"], (x, 150), 0.46, MUTED)

        cv.circle(frame, (x + 10, 196), 8, FOUND, 2, cv.LINE_AA)
        _font(frame, f"{result.parasites} parasites", (x + 32, 202), 0.66, FOUND)
        cv.circle(frame, (x + 10, 238), 8, WBC, 2, cv.LINE_AA)
        _font(frame, f"{result.white_cells} white cells", (x + 32, 244), 0.66, WBC)

        _font(frame, f"running total   {totals['parasites']} in {totals['white_cells']}",
              (x, 300), 0.56, INK)

        _font(frame, f"{d.per_ul:,.0f}", (x, 390), 1.5, INK, 2)
        _font(frame, "parasites / uL", (x + 10, 420), 0.5, MUTED)
        _font(frame, f"95%  {d.low_per_ul:,.0f} to {d.high_per_ul:,.0f}", (x, 452), 0.52, MUTED)

        _bar(frame, x, 474, W - x - 40, 22, d.per_ul, d.low_per_ul, d.high_per_ul, floor)
        _font(frame, "grey line: measured noise floor on clean slides",
              (x, 530), 0.42, MUTED)

        action = move.get("action", "")
        colour = ACCENT if action in ("stop", "flag_for_human") else FOUND
        _font(frame, "THE AGENT", (x, 600), 0.5, MUTED)
        _font(frame, action, (x, 636), 0.72, colour, 2)
        words = str(move.get("reason", "")).split()
        line, y = "", 672
        for word in words:
            if len(line) + len(word) > 44:
                _font(frame, line, (x, y), 0.48, MUTED); y += 24; line = ""
            line += word + " "
        if line:
            _font(frame, line, (x, y), 0.48, MUTED)

        _font(frame, "AutoScope  -  screening aid, not a diagnostic device",
              (30, H - 18), 0.44, MUTED)
        out.append(frame)
    return out


def _sorted_fields(session: str) -> list[str]:
    """The capture files in the order the uploader numbered them.

    `infra/site/add_patient.sh` uploads `sorted(*.jpg)` as field-0000 upward, and
    the live trace refers to a field only by that index. Sorting the same way
    here is what makes `observation.field.index` point at a real file.
    """
    return sorted(p.name for p in (RAW / session).glob("*.jpg"))


def _steps_from_api(api: str, session: str) -> list[dict]:
    """The reading AWS actually did, with the moves and reasons the model gave."""
    import urllib.request

    url = f"{api.rstrip('/')}/sessions/{session}/trace"
    with urllib.request.urlopen(url, timeout=30) as r:
        trace = json.loads(r.read())
    names = _sorted_fields(session)
    steps = []
    for raw in trace.get("steps", []):
        obs = raw.get("observation")
        obs = json.loads(obs) if isinstance(obs, str) else (obs or {})
        move = raw.get("move")
        move = json.loads(move) if isinstance(move, str) else (move or {})
        idx = (obs.get("field") or {}).get("index")
        if idx is None or idx >= len(names):
            continue
        steps.append({"image": names[idx], "counted": True,
                      "observation": obs, "move": move})
    if not steps:
        raise SystemExit(f"{url} returned no renderable steps")
    return steps


def _steps_from_arm(session: str, arm: str) -> list[dict]:
    """The offline reading, rebuilt from the cached field table."""
    runs = json.loads((RESULTS / "agent_eval.json").read_text())["runs"][arm]
    run = next((r for r in runs if r["patient"] == session), None)
    if run is None:
        raise SystemExit(f"{session} is not in arm {arm}")

    table = json.loads((RESULTS / "agent_fields.json").read_text())
    rows = [r for r in table["rows"] if r["patient"] == session][:run["fields_read"]]

    parasites = wbc = 0
    steps = []
    floor = json.loads((RESULTS / "decision_rule.json").read_text())["floor_per_ul"]
    for r in rows:
        parasites += r["found_parasites"]; wbc += r["found_wbc"]
        steps.append({
            "image": r["image"],
            "counted": {"parasites": r["found_parasites"], "wbc": r["found_wbc"]},
            "observation": {"totals": {"parasites": parasites, "white_cells": wbc},
                            "reference": {"noise_floor_per_ul": floor}},
            "move": {"action": "count_then_move", "reason": ""},
        })
    if steps:
        steps[-1]["move"] = {"action": "stop", "reason": run["stopped_because"]}
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="TF101_233",
                    help="one session, or several comma-separated. With several "
                         "and --gif you also get one gif running them back to "
                         "back, which is the asset the video wants: the agent "
                         "spending three fields on one slide and eleven on "
                         "another is only legible side by side.")
    ap.add_argument("--arm", default="bedrock/found",
                    help="which arm's run to draw, from agent_eval.json")
    ap.add_argument("--from-api", metavar="URL",
                    help="render the session AWS actually read, pulled from "
                         "<URL>/sessions/<id>/trace, instead of the offline arm. "
                         "This is what the demo page should show: the page prints "
                         "the live trace in its log, so frames built from the "
                         "offline arm can disagree with the table beside them. "
                         "They did: TF101_233 reads 12 fields offline and 8 on AWS.")
    ap.add_argument("--out", default=None,
                    help="default: docs/figures/session/<session>")
    ap.add_argument("--gif", action="store_true", help="also write an animated gif")
    ap.add_argument("--gif-only", action="store_true",
                    help="build the gifs from the frames already on disk, "
                         "without re-rendering them")
    ap.add_argument("--seconds", type=float, default=2.2, help="per frame, in the gif")
    ap.add_argument("--gif-width", type=int, default=0,
                    help="downscale the gif to this width; 0 keeps full size")
    ap.add_argument("--hold", type=int, default=2,
                    help="extra repeats of each session's last frame in the "
                         "combined gif, so the decision has time to be read")
    args = ap.parse_args()

    sessions = [s.strip() for s in args.session.split(",") if s.strip()]
    if len(sessions) > 1 and args.out:
        raise SystemExit("--out names one directory; drop it when rendering several")

    base = ROOT / "docs" / "figures" / "session"
    everything = []
    if args.gif_only:
        # Re-pace or re-size a gif without re-rendering. Drawing 27 frames means
        # running count_field on 27 twelve-megapixel captures, about five
        # minutes, and the thing you actually iterate on is the timing: the
        # combined gif has to fit the beat it plays under in the video.
        for session in sessions:
            out = base / session
            files = sorted(out.glob("field-*.png"))
            if not files:
                raise SystemExit(f"--gif-only but no frames in {out}")
            everything.append((session, [cv.imread(str(f)) for f in files], out))
            print(f"{len(files)} frames read from {out}")
    else:
        scorer = PatchScorer.load()
        for session in sessions:
            steps = (_steps_from_api(args.from_api, session) if args.from_api
                     else _steps_from_arm(session, args.arm))
            out = Path(args.out) if args.out else base / session
            out.mkdir(parents=True, exist_ok=True)
            made = frames(session, steps, scorer)
            for i, f in enumerate(made):
                cv.imwrite(str(out / f"field-{i:02d}.png"), f)
            print(f"{len(made)} frames in {out}")
            everything.append((session, made, out))

    if not args.gif:
        return
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("--gif needs pillow: uv pip install pillow")

    def write(path: Path, mats, seconds: float) -> None:
        pil = [Image.fromarray(cv.cvtColor(m, cv.COLOR_BGR2RGB)) for m in mats]
        if args.gif_width and args.gif_width < pil[0].width:
            k = args.gif_width / pil[0].width
            size = (args.gif_width, int(pil[0].height * k))
            pil = [im.resize(size, Image.LANCZOS) for im in pil]
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=int(seconds * 1000), loop=0, optimize=True)
        print(f"gif: {path} ({path.stat().st_size / 1e6:.1f} MB, {len(pil)} frames)")

    for session, made, out in everything:
        if made:
            write(base / f"{session}-agent.gif", made, args.seconds)

    if len(everything) > 1:
        combined = [m for _, made, _ in everything for m in made]
        # The last frame of each session is the one that carries the decision,
        # so it is held longer than the rest. Done by repeating it rather than
        # with per-frame durations, which several GIF players ignore.
        held = []
        for _, made, _ in everything:
            held.extend(made)
            if made:
                held.extend([made[-1]] * args.hold)
        write(base / "four-sessions.gif", held, args.seconds)
        print(f"    {len(everything)} sessions, "
              f"{sum(len(m) for _, m, _ in everything)} fields")


if __name__ == "__main__":
    main()
