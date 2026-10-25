"""The agent: Claude on Bedrock, picking the next move.

It implements the same `Policy` protocol as `RuleBased` and `DecisionRule`, so
`read_slide` cannot tell the difference and `agent_eval.py` can replay all
three over the same cached field table. That is the point: if the model cannot
beat the rule, the agent is decoration and the report says so. Keeping it
behind the same interface is what makes that claim checkable rather than
asserted. Section 7 of the report carries the answer, and it is not the
flattering one.

WHAT THE MODEL IS AND IS NOT ALLOWED TO DO

It picks where to look and when to stop. It never picks the diagnosis. The
count comes from OpenCV and is passed to the model as a number it cannot
change; the density and the confidence interval are computed by
`autoscope.density` from that count, not by the model. `flag_for_human` is
always available, and the loop's field ceiling and the Step Functions limits
sit outside the policy entirely, so a model that keeps asking for one more
field is stopped by something that is not a model.

The moves are given as tools with `tool_choice={"type": "any"}` and parallel
use disabled, so exactly one move comes back per call. A move is a decision,
not prose that has to be parsed.

COST

One call per field. At 30 to 60 fields a slide and a prompt of a few hundred
tokens, a slide is fractions of a cent. `effort="low"` is deliberate: this is
a small decision made many times, the numbers are already computed, and higher
effort buys nothing measurable here. Raise it only if the measurement says so.
"""

from __future__ import annotations

import json
import os
from typing import Any

from autoscope.agent.moves import MOVES, SYSTEM
from autoscope.agent.policy import MAX_REFOCUS_TRIES, Move
from autoscope.agent.tools import Tools
from autoscope.density import FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN
from autoscope.quality import FOCUS_MIN, FieldQuality

# Bedrock model ids carry an `anthropic.` prefix. Confirm what the account
# offers before deploying: aws bedrock list-foundation-models --by-provider anthropic
DEFAULT_MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
DEFAULT_REGION = os.environ.get("AWS_REGION_NAME", "us-east-1")


def observation(
    quality: FieldQuality,
    tools: Tools,
    floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
    negative_wbc_min: int = NEGATIVE_WBC_MIN,
) -> dict[str, Any]:
    """Everything the model is allowed to see, as plain numbers.

    Kept as a function so the Lambda can build the same payload from a
    DynamoDB row without importing OpenCV, and so a trace can be replayed
    exactly.

    `ci_relative_width` is infinite until something has been counted. JSON has
    no infinity, so it goes over as null and the prompt says what null means;
    serialising it as a float would put the literal `Infinity` in the body,
    which is not valid JSON and which some parsers turn into a very large
    number instead of "no estimate yet".
    """
    est = tools.estimate_parasitaemia()
    width = est.ci_relative_width
    return {
        "field": {
            "index": tools.fields_read,
            "focus_score": round(quality.focus_score, 2),
            "focus_needed": round(FOCUS_MIN, 2),
            "illumination_uniformity": round(quality.illumination_uniformity, 3),
            "debris_ratio": round(quality.debris_ratio, 3),
            "usable": quality.usable,
            "problem": quality.why_not(),
        },
        "totals": {
            "fields_counted": tools.fields_read,
            "parasites": tools.parasites,
            "white_cells": tools.wbc,
        },
        "estimate": {
            "per_ul": round(est.per_ul, 1),
            "ci_low_per_ul": round(est.low_per_ul, 1),
            "ci_high_per_ul": round(est.high_per_ul, 1),
            # null means nothing counted yet, so there is no estimate to judge.
            "ci_relative_width": None if width == float("inf") else round(width, 3),
        },
        "reference": {
            "noise_floor_per_ul": round(floor_per_ul, 1),
            "white_cells_for_a_negative": negative_wbc_min,
            "density_bands_per_ul": [1000, 10000, 100000],
            "who_protocol_white_cells": 200,
        },
    }


class BedrockPolicy:
    """Claude on Bedrock, behind the Policy protocol.

    `client` is injected so tests can hand in a fake and the evaluation can
    replay a recorded conversation without spending anything.
    """

    name = "bedrock"

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        region: str = DEFAULT_REGION,
        effort: str = "low",
        max_refocus: int = MAX_REFOCUS_TRIES,
        floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
        negative_wbc_min: int = NEGATIVE_WBC_MIN,
        on_call: Any = None,
    ) -> None:
        self._client = client
        self.model = model
        self.region = region
        self.effort = effort
        self.max_refocus = max_refocus
        # The noise floor belongs to the vision, not the policy: it is
        # re-measured whenever the classifier changes. Same argument as
        # DecisionRule, and the model is told it rather than asked to guess it.
        self.floor_per_ul = floor_per_ul
        self.negative_wbc_min = negative_wbc_min
        # Every call and everything it saw, for the audit trail and for replay.
        self.calls: list[dict[str, Any]] = []
        self.on_call = on_call
        # What the model said its own verdict was, when it stopped: "positive",
        # "negative" or "settled". It is NOT the call that gets reported — that
        # comes from density.decide() on the counts, because the model chooses
        # when to stop and never what the answer is. Kept so the two can be
        # compared, which is the only way to find out whether the model's
        # self-assessment is worth anything.
        self.last_call: str | None = None
        self._tries = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            # AnthropicBedrock, not AnthropicBedrockMantle. Mantle is the client
            # the SDK recommends for new code, and on this account it answers
            # every model id with 404 "does not exist" -- anthropic.claude-*,
            # us.anthropic.claude-*, all of them -- after first needing a separate
            # bedrock-mantle:CreateInference grant. The classic client goes through
            # bedrock-runtime, which is the API we verified by hand with
            # `aws bedrock-runtime converse`, and it takes the us. inference
            # profile ids. Revisit if Mantle is enabled on the account later.
            from anthropic import AnthropicBedrock

            self._client = AnthropicBedrock(aws_region=self.region)
        return self._client

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:
        obs = observation(quality, tools, self.floor_per_ul, self.negative_wbc_min)

        # The refocus cap is not the model's to argue with. Without it a model
        # that answers "refocus" forever never terminates, and on this dataset
        # there is only one capture per field so refocusing can never work.
        if self._tries >= self.max_refocus:
            self._tries = 0
            return ("skip", {"reason": f"{self.max_refocus} refocus attempts, field abandoned"})

        response = self.client.messages.create(
            model=self.model,
            # 8k, not 1k. Thinking is on by default on this model family, and
            # max_tokens is a hard ceiling the model does not see: a call that
            # thinks past it returns no tool_use block at all, _move_from
            # raises, and a paid recording run dies in the middle. You are
            # billed for tokens generated, never for the ceiling, so a high
            # ceiling costs nothing and removes the failure.
            max_tokens=8192,
            system=SYSTEM,
            tools=MOVES,
            tool_choice={"type": "any"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": json.dumps(obs, separators=(",", ":"))}],
        )

        move = self._move_from(response)
        self.calls.append(
            {
                "observation": obs,
                "move": move,
                "usage": {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                },
            }
        )
        if self.on_call is not None:
            self.on_call(self.calls[-1])

        action, args = move
        self._tries = self._tries + 1 if action == "refocus" else 0

        if action == "flag_for_human":
            tools.flag_for_human(args["reason"])
            return ("stop", {"reason": f"handed to a person: {args['reason']}"})

        return move

    def _move_from(self, response: Any) -> Move:
        """One tool call in, one move out.

        `tool_choice={"type": "any"}` guarantees a tool call, so a response
        without one is a contract break and not something to paper over with a
        default: a silent fallback here would show up in the results as the
        agent making a choice it never made.
        """
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            raise RuntimeError(
                f"model returned no move (stop_reason={response.stop_reason}); "
                "tool_choice=any should make this impossible"
            )

        # Tool inputs are parsed JSON already, but escaping differs between
        # models: never string-match on the raw input.
        args = dict(block.input)

        if block.name == "refocus":
            args["dz"] = max(-8, min(8, int(args.get("dz", 2))))

        if block.name == "stop":
            # The prose is for the log; `call` stays a field. Flattening it
            # into the sentence, which is what this did until 17 September,
            # meant the only machine-readable record of what the model
            # concluded was a string somebody would have to parse back out.
            # `last_call` is what the evaluation reads to ask how often the
            # model's own verdict agrees with what its counts actually say.
            self.last_call = args.get("call", "settled")
            args["reason"] = f"{self.last_call}: {args.get('reason', '')}"

        return (block.name, args)
