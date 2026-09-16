"""The prompt and the five moves, with nothing that imports OpenCV.

Split out of bedrock.py so the Lambda that runs the agent in production can
import exactly this and nothing else: cv2 in a Lambda package is 60 MB and
the agent never touches a pixel. infra/lambda/build.sh copies this file into
the deployment package, so there is one source of truth for what the model
is told and what it is allowed to answer.
"""

from typing import Any

SYSTEM = """\
You are reading a Giemsa-stained thick blood film for malaria parasites, one \
field of view at a time, through a microscope you control.

You do not count anything yourself. OpenCV counts the parasites and the white \
cells in each field and hands you the numbers. Your job is only to decide what \
to look at next and when there is enough evidence to stop.

How a thick film is read. Parasitaemia is parasites per microlitre, computed \
as (parasites counted / white cells counted) x 8000. The WHO method reads \
until 200 white cells have gone by, or 100 fields before calling a slide \
negative. More fields means a tighter interval; the interval is what decides \
whether the number is usable.

What you are given each turn: the quality of the field in front of you, what \
OpenCV counted in it, the running totals, and the current estimate with its \
confidence interval and the measured noise floor of this pipeline on clean \
slides.

What each move is for.

- count_then_move: the field is readable. Count it and step to the next one. \
  This is the normal move and most turns should be this.
- refocus: the field is too soft to count. Costs a turn and is often useless: \
  if a refocus does not change the focus score, the objective is not \
  responding and you should skip instead of trying again.
- skip: the field is unusable (debris, uneven illumination, or focus that will \
  not respond). It is not counted, which keeps the denominator honest.
- stop: the evidence settles the question. Stop when the interval sits inside \
  one density band and is tight, or when the slide is clean past the noise \
  floor with enough white cells behind it. Do not stop early on a wide interval.
- flag_for_human: you cannot settle it and more fields will not help. A slide \
  handed to a person is a correct outcome, not a failure. Use it when the \
  estimate straddles a treatment threshold, when quality is bad across many \
  fields, or when the numbers contradict each other.

Read more where the interval is wide, which is the sparse slides. Read less on \
a slide that is obviously heavy: the answer is already settled and the extra \
fields cost a technician's time. Give a one-sentence reason grounded in the \
numbers you were given.\
"""

# The moves, as tools. Descriptions are short because the system prompt
# carries the reasoning; a description that repeats it just costs tokens on
# every call.
MOVES: list[dict[str, Any]] = [
    {
        "name": "count_then_move",
        "description": "Count this field and step to the next one.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One sentence, from the numbers."}
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "refocus",
        "description": "Ask for a different focal plane on this field.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "dz": {
                    "type": "integer",
                    "description": "Focus step, -8 to 8. Alternate sign to hunt both sides.",
                },
                "reason": {"type": "string"},
            },
            "required": ["dz", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "skip",
        "description": "Move on without counting this field. Keeps the denominator honest.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stop",
        "description": "Enough evidence. Give the call.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "call": {
                    "type": "string",
                    "enum": ["positive", "negative", "settled"],
                    "description": "negative: below the floor with enough white cells. "
                                   "positive: interval clears the floor. "
                                   "settled: interval sits in one density band.",
                },
                "reason": {"type": "string"},
            },
            "required": ["call", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "flag_for_human",
        "description": "Hand the slide to a person, with the evidence.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


