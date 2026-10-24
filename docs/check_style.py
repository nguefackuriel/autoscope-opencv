"""Check prose against the house style in docs/style-rules.md.

    python docs/check_style.py                       # everything we publish
    python docs/check_style.py docs/video-script.md  # one file

Reads only prose. In HTML that means the text inside paragraphs, list items,
captions and table cells, with the contents of <code>, <pre>, <script> and
<style> removed first, because an identifier is not a sentence. In the video
script it means the lines beginning with "> ", which are the spoken ones.

Quoted material is skipped. The report prints a reading trace in the model's
own words and the agent's reasons appear on the demo page; editing either to
suit our own style would make them something other than quotes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["docs/video-script.md", "docs/report/report_text.html",
           "infra/site/index.html", "README.md"]

CONTRACTION = re.compile(
    r"\b[A-Za-z]+'(t|re|ll|ve|m)\b"
    r"|\b(it|that|there|here|what|who|he|she|let|where|how|one|this)'s\b"
    r"|\b(we|you|they|I|he|she|it|who|that)'d\b", re.IGNORECASE)
PARTICIPLE = re.compile(
    r",\s+(?!and |or |so |but )\w+ing\b"
    r"|\bby\s+\w+ing\b"
    r"|(?:^|(?<=[.!?]) )\s*[A-Z]\w+ing\b", re.MULTILINE)
NOT_PARTICIPLES = {"nothing", "something", "anything", "everything", "during",
                   "morning", "string", "thing", "things", "ring", "spring",
                   "ceiling", "sibling"}
NOT_BEFORE = re.compile(
    r"\brather than\b"
    r"|,\s+not\s+(a|an|the|one|just|only)\b"
    r"|\b(is|was|are|were)\s+not\s+(a|an|one|the)\b"
    r"|\bnot\s+because\b[^.]{0,60}?\bbut\b", re.IGNORECASE)
THREE_ITEMS = re.compile(
    r"[\w'-]+(?:\s+[\w'-]+){0,7},"
    r"\s+(?!which|who|that|so|but|because|and|or|if|where|when)"
    r"[\w'-]+(?:\s+[\w'-]+){0,7},\s+(and|or)\s+[\w'-]+", re.IGNORECASE)
# Words with a plainer everyday equivalent. The right-hand side is what to
# write instead. Uriel has asked three times for simple English; a list is
# more use than another promise.
SIMPLER = {
    "obtain": "get", "obtains": "gets", "obtained": "got",
    "utilise": "use", "utilize": "use", "utilised": "used",
    "demonstrate": "show", "demonstrates": "shows", "demonstrated": "showed",
    "indicate": "show", "indicates": "shows", "indicated": "showed",
    "require": "need", "requires": "needs", "required": "needed",
    "sufficient": "enough", "insufficient": "too little",
    "additional": "more", "numerous": "many", "approximately": "about",
    "subsequently": "then", "consequently": "so", "furthermore": "and",
    "moreover": "and", "however": "but", "nevertheless": "but",
    "therefore": "so", "thus": "so", "hence": "so",
    "prior to": "before", "in order to": "to", "with respect to": "for",
    "commence": "start", "terminate": "end", "ascertain": "find",
    "facilitate": "help", "implement": "build", "implemented": "built",
    "component": "part", "components": "parts", "methodology": "method",
    "initial": "first", "initially": "at first", "ultimately": "in the end",
    "attempt": "try", "attempts": "tries", "assist": "help",
    "possess": "have", "possesses": "has", "purchase": "buy",
    "sustain": "keep", "comprise": "have", "comprises": "has",
    "constitute": "make up", "exhibit": "show", "exhibits": "shows",
    "observe": "see", "observed": "seen", "yield": "give", "yields": "gives",
    "employ": "use", "employs": "uses", "employed": "used",
    "determine": "find", "determines": "finds", "determined": "found",
    "perform": "do", "performs": "does", "performed": "done",
    "accomplish": "do", "modify": "change", "modified": "changed",
    "verify": "check", "verifies": "checks", "verified": "checked",
    "optimal": "best", "minimal": "least", "maximal": "most",
    "substantial": "large", "significant": "large", "considerable": "large",
    "approximately": "about", "regarding": "about", "concerning": "about",
    "whilst": "while", "amongst": "among", "upon": "on",
}

INFLATED = ("enhance", "boost", "elevate", "ensure", "crucial", "vital",
            "seamless", "leverage", "robust", "unlock", "harness", "pivotal",
            "revolution", "testament", "delve", "fundamentally", "reshape",
            "extends beyond", "extend beyond", "game-chang", "cutting-edge",
            "state-of-the-art", "paradigm")
PHRASAL = ("panned across", "panning across",
           "figure out", "come up with", "carry out", "take on", "look into",
           "give up", "run into", "end up", "point out", "bring in",
           "sort out", "beat us", "took us a while", "comes down to",
           "boils down to", "pays off", "falls short", "stands out")
DRAMA = ("no one expected", "here is the part", "what happened next",
         "little did", "surprisingly", "remarkably", "strikingly",
         "it turns out that")

# "A screening aid, not a diagnostic device" states what the thing is not, and
# rule 9 would have it rewritten. It stays. The phrasing is the standard one
# for this kind of warning and the negation is what carries it, so weakening it
# to satisfy a style rule would be the wrong trade. Any other exception needs
# the same kind of reason written next to it.
ALLOW = ("screening aid, not a diagnostic device",)

TAG = re.compile(r"<[^>]+>")
DROP = re.compile(r"<(code|pre|script|style|title|footer)\b.*?</\1>", re.S | re.I)
# A numbered reference list is terse on purpose and is not written by us.
CITATION = re.compile(r"^\s*\d\.\s+\w|doi:")
PROSE = re.compile(r"<(p|li|figcaption|caption|td|h[1-6])\b[^>]*>(.*?)</\1>", re.S | re.I)


def prose(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix == ".md":
        if path.name == "video-script.md":
            lines = [re.sub(r"\*\*[A-Z][A-Z ]+\.\*\*", "", l[2:])
                     for l in text.splitlines() if l.startswith("> ")]
        else:
            # Everything between ``` fences is code, not prose. Excluding only
            # the fence lines left a directory tree being read as sentences.
            lines, fenced = [], False
            for line in text.splitlines():
                if line.startswith("```"):
                    fenced = not fenced
                    continue
                if fenced or not line:
                    continue
                if line.startswith(("#", "|", "    ", ">", "-", "*")):
                    continue
                lines.append(line)
        return [x.strip() for x in lines if x.strip()]

    # A page builds half its prose in JavaScript. Dropping <script> wholesale
    # hid the field-viewer caption, the empty-session message and the note on
    # the detection crops, all of which a reader sees.
    out = []
    # An apostrophe inside a comment opens a spurious literal and swallows the
    # code after it, so a literal must stay on one line and carry no JavaScript
    # punctuation to be read as prose.
    for block in re.findall(r"<script\b[^>]*>(.*?)</script>", text, re.S | re.I):
        block = re.sub(r"^\s*//.*$", "", block, flags=re.M)
        for lit in re.findall(r"\"([^\"\\\n]{24,})\"|'([^'\\\n]{24,})'", block):
            piece = (lit[0] or lit[1]).strip()
            if any(c in piece for c in "{}();=<>$|&"):
                continue
            if piece.count(" ") >= 4 and re.search(r"[a-z] [a-z]", piece):
                out.append(piece)

    text = DROP.sub(" ", text)
    for _, body in PROSE.findall(text):
        body = TAG.sub(" ", body)
        body = re.sub(r"&[a-z]+;|&#\d+;", " ", body)
        body = re.sub(r"\$[a-z_][a-z_0-9]*", "N", body)   # a figure, read as a word
        body = re.sub(r"\s+", " ", body).strip()
        if body and not CITATION.search(body):
            out.append(body)
    return out


def check(path: Path) -> list[str]:
    lines = prose(path)
    text = " ".join(lines)
    words = len(text.split())
    if not words:
        return [f"{path}: no prose found"]
    problems = []

    def flag(rule, hit, where):
        problems.append(f"  {rule}: {hit!r}\n      {where.strip()[:100]}")

    for line in lines:
        if any(a in line for a in ALLOW):
            line = line
            for a in ALLOW:
                line = line.replace(a, "a screening aid")
        for m in CONTRACTION.finditer(line):
            flag("contraction", m.group(0), line)
        for m in NOT_BEFORE.finditer(line):
            flag("says what it is not first", m.group(0)[:46], line)
        for m in THREE_ITEMS.finditer(line):
            flag("list of three", m.group(0)[:56], line)
        low = line.lower()
        for w in INFLATED:
            if w in low:
                flag("inflated word", w, line)
        for w in PHRASAL:
            if w in low:
                flag("phrasal verb", w, line)
        for w in DRAMA:
            if w in low:
                flag("suspense", w, line)
        for word, plain in SIMPLER.items():
            if re.search(rf"(?<![\w-])\b{re.escape(word)}\b(?![\w-])", low):
                flag(f"use {plain!r} instead of", word, line)

    parts = [m.group(0) for line in lines for m in PARTICIPLE.finditer(line)
             if m.group(0).strip().split()[-1].lower() not in NOT_PARTICIPLES]
    allowed = max(1, round(words / 500))
    if len(parts) > allowed:
        problems.append(f"  participle clauses: {len(parts)} in {words} words, "
                        f"at most {allowed}\n      {parts[:12]}")

    sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    lens = sorted(len(x.split()) for x in sents)
    passives = len(re.findall(r"\b(is|was|are|were|been|be)\s+\w+(ed|en)\b", text))
    print(f"{path.relative_to(ROOT)}: {words} words, {len(sents)} sentences, "
          f"length {lens[0]} to {lens[-1]}, {passives} passive")
    if lens[-1] - lens[0] < 20:
        problems.append(f"  sentence length spread only {lens[-1] - lens[0]}; vary it")

    # A wide spread across a whole document hides runs of short sentences
    # inside it. Four or more in a row is the pattern a reader notices, and it
    # is what made an early draft of this report read as machine-written.
    worst = []
    for line in lines:
        run = []
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if not sent.strip():
                continue
            if len(sent.split()) <= 9:
                run.append(sent)
                if len(run) > len(worst):
                    worst = list(run)
            else:
                run = []
    if len(worst) >= 4:
        joined = " ".join(worst)[:150]
        problems.append(f"  {len(worst)} short sentences in a row; join some\n"
                        f"      {joined}")

    # A run of nine-word sentences is the obvious case. The one that slipped
    # past was eight sentences averaging eleven words, every one a separate
    # clause that could have been joined. A paragraph of five or more
    # sentences whose mean is under sixteen words reads the same way.
    for line in lines:
        pieces = [x for x in re.split(r"(?<=[.!?])\s+", line) if x.strip()]
        if len(pieces) < 5:
            continue
        mean = sum(len(x.split()) for x in pieces) / len(pieces)
        if mean < 16:
            problems.append(
                f"  {len(pieces)} sentences averaging {mean:.0f} words in one "
                f"paragraph; connect them\n      {line[:140]}")
    return problems


def main() -> None:
    targets = sys.argv[1:] or DEFAULT
    bad = 0
    for name in targets:
        path = ROOT / name
        if not path.exists():
            print(f"{name}: missing")
            bad += 1
            continue
        problems = check(path)
        if problems:
            print(f"  {len(problems)} to fix")
            print("\n".join(problems))
            bad += 1
        print()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
