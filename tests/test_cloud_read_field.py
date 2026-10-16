"""The Batch entrypoint, with S3 and DynamoDB faked.

What is worth testing here is not boto3. It is the two things that would be
wrong silently: that the observation this job writes is the same shape the
agent was measured against, and that the running totals come from the trace
rather than from anything the agent said. If those two ever drift, the cloud
pipeline stops being the pipeline we have numbers for.
"""

import json

import cv2 as cv
import numpy as np
import pytest

from autoscope.cloud import read_field as rf


class FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3's own spelling
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": _Body(self.objects[Key])}


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class FakeTable:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.written = []

    def query(self, **kwargs):  # noqa: ARG002
        return {"Items": self.items}

    def put_item(self, Item):  # noqa: N803
        self.written.append(Item)
        self.items.append(Item)

    def get_item(self, Key):  # noqa: N803
        for item in self.items:
            if item.get("session_id") == Key.get("session_id"):
                return {"Item": item}
        return {}


@pytest.fixture
def wired(monkeypatch):
    """A field image in S3, an empty trace, one session row."""
    import cv2 as cv

    rng = np.random.default_rng(3)
    image = rng.integers(140, 200, (256, 256, 3), dtype=np.uint8)
    ok, buf = cv.imencode(".jpg", image)
    assert ok

    s3 = FakeS3({"sess1/field-0000.jpg": buf.tobytes()})
    trace = FakeTable()
    sessions = FakeTable([{"session_id": "sess1", "scale": "1.0"}])

    monkeypatch.setattr(rf, "s3", lambda: s3)
    monkeypatch.setattr(rf, "trace_table", lambda: trace)
    monkeypatch.setattr(rf, "sessions_table", lambda: sessions)
    monkeypatch.setattr(rf, "_uploads", lambda: "bucket")
    return s3, trace, sessions


def test_a_field_past_the_end_of_the_slide_says_so(wired):
    assert rf.read("sess1", 99) == {"end_of_slide": True}


def test_the_row_carries_the_observation_the_agent_expects(wired):
    _, trace, _ = wired

    obs = rf.read("sess1", 0)

    assert set(obs) == {"field", "totals", "estimate", "reference"}
    assert obs["field"]["index"] == 0
    assert "parasites" in obs["totals"] and "white_cells" in obs["totals"]
    assert obs["reference"]["noise_floor_per_ul"] > 0

    row = trace.written[0]
    assert row["session_id"] == "sess1" and row["step"] == 0
    # The count and the observation both land, and the count is the evidence
    # the totals are rebuilt from.
    assert json.loads(row["observation"]) == obs
    assert set(json.loads(row["counted"])) == {"parasites", "wbc"}


def test_totals_are_rebuilt_from_the_trace_not_carried_in(wired):
    """Three fields already read; the fourth has to see their sum.

    Step Functions holds no state we could trust for this and the container is
    new every job, so a total that did not come from the trace could only have
    come from something that guessed.
    """
    _, trace, _ = wired
    for step in range(3):
        trace.items.append({
            "session_id": "sess1",
            "step": step,
            "counted": json.dumps({"parasites": 2, "wbc": 20}),
        })

    totals = rf.Totals("sess1")

    assert (totals.parasites, totals.wbc, totals.fields_read) == (6, 60, 3)


def test_the_row_says_which_opencv_read_the_field():
    """A speed claim about a build needs the build to travel with the data.

    Parsed from getBuildInformation rather than hard-coded, so a container
    swap cannot leave the report describing a binary that is not running.
    """
    line = rf.build_line()

    assert line["opencv"] == cv.__version__
    assert line["machine"] in {"x86_64", "aarch64", "arm64"}
    # Both fields are present whatever the build; on a build with no custom
    # HAL they are empty strings, not missing keys.
    assert set(line) == {"opencv", "machine", "hal", "parallel"}


def test_a_retried_field_is_not_counted_twice(wired):
    """A job that reads field 2 again must not see field 2's old row.

    Spot reclaims a container mid-job and Batch runs it again. The retry adds
    its own counts, so if the rebuild also summed the row the first attempt
    wrote, the field would land in the totals twice. It did, once: field 0 of
    TF101_233 came back as 46 parasites in 32 white cells, which is 23 in 16
    twice.
    """
    _, trace, _ = wired
    for step in range(3):
        trace.items.append({
            "session_id": "sess1",
            "step": step,
            "counted": json.dumps({"parasites": 2, "wbc": 20}),
        })

    totals = rf.Totals("sess1", skip_field=2)

    assert (totals.parasites, totals.wbc, totals.fields_read) == (4, 40, 2)


def test_skipping_a_field_that_was_never_read_changes_nothing(wired):
    """The first attempt at a field is the common case and must be untouched."""
    _, trace, _ = wired
    trace.items.append({
        "session_id": "sess1", "step": 0,
        "counted": json.dumps({"parasites": 2, "wbc": 20}),
    })

    totals = rf.Totals("sess1", skip_field=1)

    assert (totals.parasites, totals.wbc, totals.fields_read) == (2, 20, 1)


def test_a_row_with_no_count_is_not_counted_as_a_field(wired):
    """A skipped field has an observation but no count. Counting it would put a
    field with no white cells into the denominator."""
    _, trace, _ = wired
    trace.items.append({"session_id": "sess1", "step": 0,
                        "observation": json.dumps({"field": {}})})

    totals = rf.Totals("sess1")

    assert totals.fields_read == 0


def test_the_site_scale_comes_off_the_session(wired):
    _, _, sessions = wired
    sessions.items[0]["scale"] = "0.2082"

    assert rf.site_calibration("sess1") == pytest.approx(0.2082)


def test_an_uncalibrated_session_runs_at_nlm_scale(wired):
    _, _, sessions = wired
    sessions.items[0].pop("scale")

    assert rf.site_calibration("sess1") == 1.0
