"""The demo endpoint.

Five routes, one Lambda, no framework. A judge has to be able to pick a slide,
watch it being read, and see the number with its error bar and the log that
produced it.

  POST /sessions            start a reading, return its id
  GET  /sessions/{id}       the decision, the totals, the estimate
  GET  /sessions/{id}/trace the audit trail: every field, every decision
  POST /uploads             a presigned POST for the phone
  GET  /health              is anything alive
"""

import json
import os
import time
import uuid
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

REGION = os.environ.get("AWS_REGION", "us-east-1")
UPLOADS = os.environ["UPLOADS_BUCKET"]
SESSIONS_TABLE = os.environ["SESSIONS_TABLE"]
TRACE_TABLE = os.environ["TRACE_TABLE"]
STATE_MACHINE = os.environ["STATE_MACHINE"]

_s3 = boto3.client("s3", region_name=REGION)
_sfn = boto3.client("stepfunctions", region_name=REGION)
_ddb = boto3.resource("dynamodb", region_name=REGION)
_sessions = _ddb.Table(SESSIONS_TABLE)
_trace = _ddb.Table(TRACE_TABLE)

DISCLAIMER = "Screening aid, not a diagnostic device. Every result is reviewed by a person."


def _json(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json",
                    "access-control-allow-origin": "*"},
        "body": json.dumps(body, default=_plain),
    }


def _plain(value):
    """DynamoDB hands back Decimal; JSON does not know what that is."""
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def handler(event, context):  # noqa: ARG001
    method = event["requestContext"]["http"]["method"]
    path = event["requestContext"]["http"]["path"].rstrip("/")
    parts = [p for p in path.split("/") if p]

    if method == "OPTIONS":
        return _json(200, {})

    if parts == ["health"]:
        return _json(200, {"ok": True, "disclaimer": DISCLAIMER})

    if method == "POST" and parts == ["uploads"]:
        return _presign(json.loads(event.get("body") or "{}"))

    if method == "POST" and parts == ["sessions"]:
        return _start(json.loads(event.get("body") or "{}"))

    if method == "GET" and len(parts) == 2 and parts[0] == "sessions":
        return _session(parts[1])

    if method == "GET" and len(parts) == 3 and parts[0] == "sessions" and parts[2] == "trace":
        return _trace_for(parts[1])

    return _json(404, {"error": f"no route for {method} {path}"})


def _presign(body):
    """The phone uploads straight to S3. Nothing sits between it and the bucket,
    which is the only part of this that has to work on a bad connection."""
    key = f"slides/{uuid.uuid4().hex}/{body.get('filename', 'slide.mp4')}"
    post = _s3.generate_presigned_post(
        Bucket=UPLOADS,
        Key=key,
        # 200 MB: a two-minute 1080p pass over a slide. Larger is a mistake,
        # not a slide, and an unbounded presign is an open door.
        Conditions=[["content-length-range", 1, 200 * 1024 * 1024]],
        ExpiresIn=900,
    )
    return _json(200, {"key": key, "upload": post})


def _start(body):
    """Start a reading on a slide already in the bucket.

    The EventBridge rule starts one automatically on upload; this is the path a
    judge uses to re-run a slide that is already there, which is what makes the
    demo reproducible rather than a one-shot.
    """
    key = body.get("key")
    if not key:
        return _json(400, {"error": "key is required: which slide to read"})

    session_id = body.get("session_id") or uuid.uuid4().hex
    _sfn.start_execution(
        stateMachineArn=STATE_MACHINE,
        name=session_id,
        input=json.dumps({"session_id": session_id, "field": 0, "key": key}),
    )
    _sessions.put_item(
        Item={"session_id": session_id, "key": key, "state": "reading",
              "started_at": int(time.time())}
    )
    return _json(202, {"session_id": session_id, "state": "reading",
                       "trace": f"/sessions/{session_id}/trace"})


def _session(session_id):
    row = _sessions.get_item(Key={"session_id": session_id}).get("Item")
    if row is None:
        return _json(404, {"error": "no such session"})
    return _json(200, {**row, "disclaimer": DISCLAIMER})


def _trace_for(session_id):
    """Every field read and every decision, in order.

    A screening aid that cannot show its working is not one a lab can use, so
    this is a first-class route and not a debugging afterthought.
    """
    rows = _trace.query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        ScanIndexForward=True,
        Limit=500,
    ).get("Items", [])

    steps = []
    for row in rows:
        steps.append({
            "step": row.get("step"),
            "observation": json.loads(row["observation"]) if "observation" in row else None,
            "move": json.loads(row["move"]) if "move" in row else None,
            "model": row.get("model"),
        })
    return _json(200, {"session_id": session_id, "steps": steps,
                       "disclaimer": DISCLAIMER})
