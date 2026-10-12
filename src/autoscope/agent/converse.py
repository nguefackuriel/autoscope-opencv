"""The same agent, through Bedrock's Converse API, so any model can drive it.

`BedrockPolicy` speaks the `anthropic` SDK, which ties the agent to one vendor.
Converse is Bedrock's model-agnostic surface: the same request shape reaches
Anthropic, OpenAI, Meta, Mistral and the rest, including tool use. So this
policy runs the identical loop with the identical five moves against whatever
the account can reach, and the question "does the agent need a frontier model"
becomes something we can answer instead of assume.

It implements the same `Policy` protocol as everything else in policy.py, and
it builds its observation with the same `observation()` function
`BedrockPolicy` uses, so the only thing that differs between two arms is the
model.

WHAT IS AND IS NOT THE MODEL'S

Unchanged from bedrock.py, and it is the whole safety argument: the model
picks where to look and when to stop, never the diagnosis. The count comes
from OpenCV, the density and interval from `autoscope.density`, and the call
in the results table from `density.decide()` on those counts. The model's own
verdict is recorded beside it and compared, never substituted for it.

WHY NOT JUST USE CONVERSE EVERYWHERE

Nothing stops us, and it may be the right end state. The Anthropic SDK path
stays because it is the one the first fifty-patient recording was made with,
and re-recording to change client libraries would spend money to produce the
same numbers.
"""

from __future__ import annotations

import json
import os
from typing import Any

from autoscope.agent.bedrock import observation
from autoscope.agent.moves import MOVES, SYSTEM
from autoscope.agent.policy import MAX_REFOCUS_TRIES, Move
from autoscope.agent.tools import Tools
from autoscope.density import FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN
from autoscope.quality import FieldQuality

DEFAULT_REGION = os.environ.get("AWS_REGION_NAME", "us-east-1")


def as_tool_config(moves: list[dict[str, Any]], force: bool = True) -> dict[str, Any]:
    """The Anthropic tool list in Converse's shape.

    Converse nests the schema one level deeper (`inputSchema.json`) and has no
    equivalent of `strict`, which it drops. Translating here rather than
    keeping two copies of the move definitions means `moves.py` stays the one
    place that says what the agent may answer.
    """
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": m["name"],
                    "description": m["description"],
                    "inputSchema": {"json": m["input_schema"]},
                }
            }
            for m in moves
        ],
        # Force a tool call, as tool_choice={"type":"any"} does on the
        # Anthropic path: a move is a decision, not prose to be parsed.
        #
        # Not every model takes it. Llama 4 Maverick answers "This model
        # doesn't support the toolConfig.toolChoice.any field", so that arm
        # runs unforced and a turn with no tool call is treated as the model
        # declining to commit -- which is an outcome, recorded as
        # flag_for_human, not a crash in the middle of a paid recording.
        **({"toolChoice": {"any": {}}} if force else {}),
    }


class ConversePolicy:
    """Any Bedrock model, behind the Policy protocol.

    `client` is injected so tests hand in a fake and the evaluation replays a
    recording without spending anything — the same seam BedrockPolicy uses.
    """

    name = "converse"

    def __init__(
        self,
        model: str,
        client: Any = None,
        region: str = DEFAULT_REGION,
        max_refocus: int = MAX_REFOCUS_TRIES,
        floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
        negative_wbc_min: int = NEGATIVE_WBC_MIN,
        max_tokens: int = 8192,
        force_tool: bool = True,
        name: str | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self.region = region
        self.max_refocus = max_refocus
        self.floor_per_ul = floor_per_ul
        self.negative_wbc_min = negative_wbc_min
        self.max_tokens = max_tokens
        self.force_tool = force_tool
        # The arm name in the results table. Defaults to the model's short
        # name so two model arms do not collide under one key.
        self.name = name or model.split(".")[-1].split(":")[0]
        self.calls: list[dict[str, Any]] = []
        self.last_call: str | None = None
        self.no_move = 0
        self._tries = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def next_move(self, quality: FieldQuality, tools: Tools) -> Move:
        obs = observation(quality, tools, self.floor_per_ul, self.negative_wbc_min)

        # Not the model's to argue with, same as every other policy: without a
        # cap, a model that answers "refocus" forever never terminates, and on
        # this dataset there is one capture per field so it never could work.
        if self._tries >= self.max_refocus:
            self._tries = 0
            return ("skip", {"reason": f"{self.max_refocus} refocus attempts, field abandoned"})

        response = self.client.converse(
            modelId=self.model,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user",
                       "content": [{"text": json.dumps(obs, separators=(",", ":"))}]}],
            toolConfig=as_tool_config(MOVES, force=self.force_tool),
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )

        move = self._move_from(response)
        usage = response.get("usage", {})
        self.calls.append({
            "observation": obs,
            "move": move,
            "usage": {"input_tokens": usage.get("inputTokens", 0),
                      "output_tokens": usage.get("outputTokens", 0)},
        })

        action, args = move
        self._tries = self._tries + 1 if action == "refocus" else 0

        if action == "flag_for_human":
            tools.flag_for_human(args["reason"])
            return ("stop", {"reason": f"handed to a person: {args['reason']}"})

        return move

    def _move_from(self, response: dict) -> Move:
        """One tool call in, one move out.

        `toolChoice: any` guarantees one, so a response without a tool block is
        a contract break rather than something to paper over with a default: a
        silent fallback would record a decision the agent never made.

        Unlike the Anthropic path, a Converse response can also carry text
        blocks alongside the tool use, and some models always emit one. They
        are ignored rather than treated as an error.
        """
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        use = next((b["toolUse"] for b in blocks if "toolUse" in b), None)
        if use is None:
            said = " ".join(b.get("text", "") for b in blocks).strip()[:120]
            if self.force_tool:
                raise RuntimeError(
                    f"{self.model} returned no move "
                    f"(stopReason={response.get('stopReason')}); "
                    "toolChoice=any should make this impossible"
                )
            # Unforced arm: a model that will not pick a move has not decided,
            # and the honest reading of that is a slide for a person. Counted
            # in the results rather than hidden.
            self.no_move += 1
            return ("flag_for_human",
                    {"reason": f"the model gave no move: {said or "(no text)"}"})

        args = dict(use.get("input") or {})
        name = use["name"]

        if name == "refocus":
            args["dz"] = max(-8, min(8, int(args.get("dz", 2))))

        if name == "stop":
            self.last_call = args.get("call", "settled")
            args["reason"] = f"{self.last_call}: {args.get('reason', '')}"

        # Every move's schema requires a reason, but a weaker model will
        # sometimes omit it. An empty string keeps the loop running and the
        # trace honest about what was missing, rather than raising on a field
        # that carries no decision.
        args.setdefault("reason", "(no reason given)")
        return (name, args)
