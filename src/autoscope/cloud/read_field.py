"""One Batch job: read one field, write what OpenCV measured.

This is stages 1 to 4 and 6 on Graviton, and it is the whole of the main
workload. The agent never runs here and never sees a pixel; it reads the row
this writes.

    python -m autoscope.cloud.read_field --session <id> --field 7

The job is deliberately one field rather than one slide. A slide is not a unit
of work here because which field comes next is the agent's decision, and a job
that read the whole slide would be a job that had already made that decision.
It also means a lost Spot instance costs one field, not a slide.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
import cv2 as cv
import numpy as np

from autoscope.agent.bedrock import observation as build_observation
from autoscope.counter import count_field
from autoscope.density import FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN, density
from autoscope.parasites import PatchScorer
from autoscope.quality import assess_field

# Clients and bucket names are resolved on first use, not at import. A module
# that reaches for four environment variables at import time cannot be imported
# by a test, or by anything that wants to call one function in it, and a smoke
# test that the entrypoint still imports is worth more than the two lines this
# costs.
_clients: dict[str, Any] = {}


def build_line() -> dict[str, str]:
    """Which OpenCV read this field, and on what.

    The COOL claim is about a binary, and a result that does not say which
    binary produced it is not evidence of anything. One line per row is cheap
    and it is the only place the claim and the data meet. Parsed from
    getBuildInformation rather than hard-coded, so it cannot drift from the
    container that is actually running.
    """
    info = cv.getBuildInformation()
    out = {"opencv": cv.__version__, "machine": platform.machine(),
           "hal": "", "parallel": ""}
    for line in info.splitlines():
        label, _, value = line.partition(":")
        key = label.strip().lower()
        if key == "custom hal" and not out["hal"]:
            out["hal"] = value.strip()
        elif key == "parallel framework" and not out["parallel"]:
            out["parallel"] = value.strip()
    return out


def _region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


def _uploads() -> str:
    return os.environ["AUTOSCOPE_UPLOADS_BUCKET"]


def s3():
    if "s3" not in _clients:
        _clients["s3"] = boto3.client("s3", region_name=_region())
    return _clients["s3"]


def _table(env_var: str, cache_key: str):
    if cache_key not in _clients:
        ddb = boto3.resource("dynamodb", region_name=_region())
        _clients[cache_key] = ddb.Table(os.environ[env_var])
    return _clients[cache_key]


def trace_table():
    return _table("AUTOSCOPE_TRACE_TABLE", "trace")


def sessions_table():
    return _table("AUTOSCOPE_SESSIONS_TABLE", "sessions")

# The model is baked into the image, not fetched per job: a 2 MB download on
# every field would dominate the field time we are trying to measure.
_scorer: PatchScorer | None = None


def scorer() -> PatchScorer:
    global _scorer
    if _scorer is None:
        _scorer = PatchScorer.load()
    return _scorer


class Totals:
    """The running totals, rebuilt from the trace.

    Step Functions holds no state we could trust for this, and a job is a fresh
    container every time. Reading the rows back is cheap (one query, a few
    hundred items at most) and it means the totals are always what the audit
    trail says they are, not a number that drifted from it.
    """

    def __init__(self, session_id: str, skip_field: int | None = None) -> None:
        rows = trace_table().query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        ).get("Items", [])

        self.parasites = 0
        self.wbc = 0
        self.fields_read = 0
        for row in rows:
            if "counted" not in row:
                continue
            # The caller is about to add this field's counts itself. If an
            # earlier attempt at the same field already wrote a row, summing it
            # here too counts the field twice. A retry is the normal case on
            # Spot, so this is not a corner.
            if skip_field is not None and int(row.get("step", -1)) == skip_field:
                continue
            counted = json.loads(row["counted"])
            self.parasites += int(counted["parasites"])
            self.wbc += int(counted["wbc"])
            self.fields_read += 1


class _ToolsView:
    """Just enough of agent.tools.Tools for `observation()` to read.

    The real Tools owns a VirtualStage, which is a slide in memory. Here the
    stage is S3 and the totals come from DynamoDB, but the shape the agent
    sees has to be identical or the cloud agent is not the agent we measured.
    """

    def __init__(self, totals: Totals, wbc_per_ul: float = 8000.0) -> None:
        self.parasites = totals.parasites
        self.wbc = totals.wbc
        self.fields_read = totals.fields_read
        self.wbc_per_ul = wbc_per_ul

    def estimate_parasitaemia(self):
        return density(self.parasites, max(self.wbc, 1), self.wbc_per_ul)


def field_key(session_id: str, field: int) -> str:
    return f"{session_id}/field-{field:04d}.jpg"


def load_field(session_id: str, field: int) -> np.ndarray | None:
    """Fetch one field's image. None when the slide has no field that far along,
    which is how a session finds its own end."""
    key = field_key(session_id, field)
    try:
        body = s3().get_object(Bucket=_uploads(), Key=key)["Body"].read()
    except s3().exceptions.NoSuchKey:
        return None
    buffer = np.frombuffer(body, dtype=np.uint8)
    return cv.imdecode(buffer, cv.IMREAD_COLOR)


def site_calibration(session_id: str) -> float:
    """The site's scale, or 1.0 for an NLM-scale rig.

    Every threshold in the pipeline was fitted at NLM scale. A hospital's own
    microscope is calibrated once, from ten of its own slides, and the factor
    is stored on the session. Running without it is the single most likely way
    to get confidently wrong numbers on someone else's microscope.
    """
    row = sessions_table().get_item(Key={"session_id": session_id}).get("Item") or {}
    return float(row.get("scale", 1.0))


def read(session_id: str, field: int) -> dict[str, Any]:
    started = time.perf_counter()
    image = load_field(session_id, field)

    if image is None:
        return {"end_of_slide": True}

    scale = site_calibration(session_id)

    quality = assess_field(image)
    # count_field takes the site scale itself and maps coordinates back to the
    # caller's pixels. Rescaling here as well would apply it twice.
    counted = count_field(image, scorer=scorer(), scale=scale)
    parasites, wbc = counted.as_tuple()

    # Re-reading a field used to double-count it: this field's row from an
    # earlier attempt was already in the sum, and then the attempt added its
    # counts again. Observed on 10 October, field 0 of TF101_233 reporting 46
    # parasites in 32 white cells, which is 23 in 16 twice. A retry is the
    # normal case on Spot, so Totals skips the row for the field being read.
    totals = Totals(session_id, skip_field=field)
    totals.parasites += parasites
    totals.wbc += wbc
    totals.fields_read += 1

    obs = build_observation(quality, _ToolsView(totals), FLOOR_PER_UL_DEFAULT, NEGATIVE_WBC_MIN)
    obs["field"]["index"] = field

    trace_table().put_item(
        Item={
            "session_id": session_id,
            "step": field,
            "observation": json.dumps(obs),
            "counted": json.dumps({"parasites": parasites, "wbc": wbc}),
            "seconds": str(round(time.perf_counter() - started, 3)),
            "scale": str(scale),
            "build": json.dumps(build_line()),
            "recorded_at": int(time.time()),
            "expires_at": int(time.time()) + 90 * 24 * 3600,
        }
    )
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--field", type=int, required=True)
    args = ap.parse_args()

    result = read(args.session, args.field)
    # Stdout is the only thing a human reads in CloudWatch; the row in DynamoDB
    # is what the agent reads.
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
