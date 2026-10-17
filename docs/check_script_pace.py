"""Does the video script fit in five minutes?

    python docs/check_script_pace.py

Five minutes is the competition limit. The script's beats carry their own time
budget in the heading; this reads the spoken lines under each one, the ones
starting with "> ", and says how long they take at a normal narration pace.

Written because the first draft of the script was 1,183 spoken words, which is
nearly eight minutes, in a document whose first line said three.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WPM = 156          # a brisk but comfortable narration pace
LIMIT_SECONDS = 300


def seconds(stamp: str) -> int:
    minutes, secs = stamp.split(":")
    return int(minutes) * 60 + int(secs)


def main() -> None:
    path = Path(__file__).resolve().parent / "video-script.md"
    beats: list[tuple[str, int]] = []
    current, words = None, 0
    for line in path.read_text().splitlines():
        head = re.match(r"^## (\d:\d\d-\d:\d\d), (.+)", line)
        if head:
            if current:
                beats.append((current, words))
            current, words = head.group(1), 0
        elif line.startswith("> ") and current:
            words += len(line.split()) - 1
    if current:
        beats.append((current, words))

    if not beats:
        raise SystemExit("no timed beats found in video-script.md")

    print(f"{'beat':<14}{'words':>7}{'budget':>9}{'needs':>8}")
    total_words = total_budget = 0
    over = []
    for beat, count in beats:
        start, end = beat.split("-")
        budget = seconds(end) - seconds(start)
        needs = count / WPM * 60
        total_words += count
        total_budget += budget
        flag = ""
        if needs > budget * 1.05:
            flag = f"   over by {needs - budget:.0f}s"
            over.append(beat)
        print(f"{beat:<14}{count:>7}{budget:>8}s{needs:>7.0f}s{flag}")

    total_needs = total_words / WPM * 60
    print(f"\n{total_words} spoken words, {total_needs:.0f}s at {WPM} words a minute, "
          f"in a {total_budget}s video.")
    if total_budget > LIMIT_SECONDS:
        print(f"The beats add up to {total_budget}s. The limit is {LIMIT_SECONDS}s.")
    if total_needs > LIMIT_SECONDS:
        print(f"Cut about {(total_needs - LIMIT_SECONDS) * WPM / 60:.0f} words, "
              f"or the video runs past five minutes.")
        sys.exit(1)
    print("Fits.")


if __name__ == "__main__":
    main()
