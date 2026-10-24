"""Check the spoken lines of the video script against the house style rules.

    python docs/check_script_style.py

The rules were set by Uriel on 24 October 2026. They are not suggestions about
taste; each one names a pattern that makes written English sound generated.
This reads only the lines beginning with "> " in video-script.md, because the
shot directions are notes to ourselves and are not spoken.

Reports every hit with the sentence it sits in. Exits non-zero if any rule is
broken, so the script cannot drift back.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# "wheel's 547" and "one person's attention" are genitives, not contractions.
# Only flag 's where the word before it cannot own anything.
CONTRACTION = re.compile(
    r"\b[A-Za-z]+'(t|re|ll|ve|m)\b"
    r"|\b(it|that|there|here|what|who|he|she|let|where|how|one|this)'s\b"
    r"|\b(we|you|they|I|he|she|it|who|that)'d\b",
    re.IGNORECASE)
# A participle clause can open with the participle ("reached by watching"),
# or sit after a comma, or follow "by". The first version of this check only
# looked for the comma form and reported zero on a text that had several.
PARTICIPLE = re.compile(
    r",\s+(?!and |or |so |but )\w+ing\b"
    r"|\bby\s+\w+ing\b"
    r"|^\s*\w+ing\b",
    re.MULTILINE)
NOT_PARTICIPLES = {"nothing", "something", "anything", "everything", "during",
                   "morning", "string", "thing", "things", "ring", "spring"}
NOT_BEFORE = re.compile(
    r"\b(is|was|are|were|it is|that is)\s+not\b.*?\b(but|rather|instead)\b"
    r"|\bnot\s+\w+,\s+(but|rather)\b"
    r"|\brather than\b",
    re.IGNORECASE)
# Items in a list can be long. Allowing only three words per item let
# "refocus, skip a field with debris, or continue reading" through.
THREE_ITEMS = re.compile(
    r"[\w'-]+(?:\s+[\w'-]+){0,7},\s+[\w'-]+(?:\s+[\w'-]+){0,7},\s+(and|or)\s+[\w'-]+")
INFLATED = ("enhance", "boost", "elevate", "ensure", "crucial", "vital",
            "seamless", "leverage", "robust", "unlock", "harness", "pivotal",
            "transform", "revolution", "landscape", "testament", "delve",
            "fundamentally", "reshape", "extends beyond", "game-chang")
PHRASAL = ("figure out", "come up with", "carry out", "set up", "take on",
           "look into", "give up", "run into", "back out", "turn out",
           "come back", "end up", "point out", "bring in", "sort out",
           "keep going", "pushes it", "beat us", "took us a while")
DRAMA = ("no one expected", "here is the part", "here's the part",
         "what happened next", "the truth is", "imagine", "little did",
         "took us a while to believe", "surprisingly", "remarkably")


def spoken(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("> "):
            out.append(re.sub(r"\*\*[A-Z][A-Z ]+\.\*\*", "", line[2:]).strip())
    return [x for x in out if x]


def main() -> None:
    path = Path(__file__).resolve().parent / "video-script.md"
    lines = spoken(path)
    text = " ".join(lines)
    words = len(text.split())
    problems: list[str] = []

    def flag(rule: str, hit: str, where: str) -> None:
        problems.append(f"  {rule}: {hit!r}\n      in: {where.strip()[:96]}")

    for line in lines:
        for m in CONTRACTION.finditer(line):
            flag("contraction", m.group(0), line)
        for m in NOT_BEFORE.finditer(line):
            flag("says what it was not", m.group(0), line)
        for m in THREE_ITEMS.finditer(line):
            flag("list of three", m.group(0)[:52], line)
        low = line.lower()
        for word in INFLATED:
            if word in low:
                flag("inflated word", word, line)
        for phrase in PHRASAL:
            if phrase in low:
                flag("phrasal verb", phrase, line)
        for phrase in DRAMA:
            if phrase in low:
                flag("suspense", phrase, line)

    participles = [m.group(0) for line in lines for m in PARTICIPLE.finditer(line)
                   if m.group(0).strip().split()[-1].lower() not in NOT_PARTICIPLES]
    allowed = max(1, round(words / 500))
    if len(participles) > allowed:
        problems.append(f"  participle clauses: {len(participles)} in {words} words, "
                        f"at most {allowed} allowed\n      {participles}")

    # burstiness: a uniform sentence length is the clearest tell of all
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    lengths = sorted(len(x.split()) for x in sentences)
    mid = lengths[len(lengths) // 2]
    spread = lengths[-1] - lengths[0]
    passives = len(re.findall(r"\b(is|was|are|were|been|be)\s+\w+(ed|en)\b", text))

    print(f"{words} spoken words, {len(sentences)} sentences")
    print(f"  length: shortest {lengths[0]}, median {mid}, longest {lengths[-1]}, "
          f"spread {spread}")
    print(f"  passive constructions: {passives}")
    print(f"  participle clauses: {len(participles)} (at most {allowed})\n")

    if spread < 20:
        problems.append(f"  sentence length spread is only {spread} words; vary it")
    if passives < 3:
        problems.append(f"  only {passives} passive constructions; mix the voice")

    if problems:
        print(f"{len(problems)} to fix:\n")
        print("\n".join(problems))
        sys.exit(1)
    print("No rule broken.")


if __name__ == "__main__":
    main()
