"""The agent, as a Lambda. One field in, one move out.

Step Functions calls this between Batch jobs. The Batch job has already run
stages 1 to 4 on Graviton and written what OpenCV measured; this reads those
numbers and decides what happens next. It never touches a pixel, which is why
it is 512 MB of pure Python rather than a container with OpenCV in it.

`moves.py` is not written here: infra/lambda/build.sh copies it from
src/autoscope/agent/moves.py, so the prompt and the five moves have one
definition and the deployed agent cannot drift from the one we measured.
"""

import json
import os
import time

import boto3

try:
    from moves import MOVES, SYSTEM
except ImportError as exc:  # pragma: no cover - a packaging mistake, not a runtime path
    raise RuntimeError(
        "moves.py is missing from the deployment package. "
        "Run infra/lambda/build.sh before terraform apply."
    ) from exc

MODEL = os.environ["BEDROCK_MODEL_ID"]
REGION = os.environ.get("AWS_REGION_NAME", os.environ.get("AWS_REGION", "us-east-1"))
TRACE_TABLE = os.environ["TRACE_TABLE"]

_dynamodb = boto3.resource("dynamodb", region_name=REGION)
_trace = _dynamodb.Table(TRACE_TABLE)

# Built once per container, reused across invocations.
_client = None


def client():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrockMantle

        _client = AnthropicBedrockMantle(aws_region=REGION)
    return _client


def handler(event, context):  # noqa: ARG001
    """event carries session_id and field. The numbers come from DynamoDB.

    Not from the event: Batch's `.sync` integration hands Step Functions the
    job's metadata, never its stdout, so there is no path by which the measured
    numbers could arrive here inside the event. The Batch job writes the
    observation to the trace row; this reads it back. That is also what makes a
    session replayable — the evidence is on disk, not in flight.

    Returns {"action": ..., "reason": ..., ...}: Step Functions branches on
    `action`, and the limits it enforces afterwards are not ours to relax.
    """
    session_id = event["session_id"]
    field = int(event.get("field", 0))

    row = _trace.get_item(Key={"session_id": session_id, "step": field}).get("Item")
    if row is None or "observation" not in row:
        # The job ran but wrote nothing we can decide on. A person, not a guess.
        return {"action": "flag_for_human",
                "reason": f"no measurement recorded for field {field}"}

    observation = json.loads(row["observation"])

    response = client().messages.create(
        model=MODEL,
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
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": json.dumps(observation, separators=(",", ":"))}],
    )

    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        # tool_choice=any should make this impossible. Escalating is the right
        # answer: a default here would record a decision the agent never made.
        move = {"action": "flag_for_human",
                "reason": f"model returned no move (stop_reason={response.stop_reason})"}
    else:
        move = {"action": block.name, **dict(block.input)}

    if move["action"] == "refocus":
        move["dz"] = max(-8, min(8, int(move.get("dz", 2))))

    # The audit trail. An update, not a put: the observation in this row was
    # written by the Batch job and overwriting it would destroy the evidence
    # the decision was made on, which is the one thing the row exists for.
    _trace.update_item(
        Key={"session_id": session_id, "step": field},
        UpdateExpression=(
            "SET move = :m, model = :model, input_tokens = :i, "
            "output_tokens = :o, decided_at = :t"
        ),
        ExpressionAttributeValues={
            ":m": json.dumps(move),
            ":model": MODEL,
            ":i": getattr(response.usage, "input_tokens", 0),
            ":o": getattr(response.usage, "output_tokens", 0),
            ":t": int(time.time()),
        },
    )

    return move
