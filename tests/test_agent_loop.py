"""The loop has to produce the evidence the rules ask for, so we test for it."""

import cv2 as cv
import numpy as np
import pytest

from autoscope.agent.loop import read_slide
from autoscope.agent.policy import FixedProtocol, RuleBased
from autoscope.agent.tools import OracleCounter, Tools
from autoscope.virtual_stage import VirtualStage

TILE = 256


def synthetic_slide(cols: int = 12, rows: int = 4, blur_first: bool = False) -> np.ndarray:
    """A slide-ish image: textured background with dark blobs standing in for cells."""
    rng = np.random.default_rng(7)
    h, w = rows * TILE, cols * TILE
    img = rng.integers(150, 200, (h, w, 3), dtype=np.uint8)
    for _ in range(w * h // 4000):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv.circle(img, (cx, cy), int(rng.integers(5, 12)), (60, 60, 70), -1)
    if blur_first:
        # Leave the first field soft, so the agent has something to refocus.
        img[:TILE, :TILE] = cv.GaussianBlur(img[:TILE, :TILE], (0, 0), 6)
    return img


def build(policy, truth, blur_first=False, tile=TILE):
    stage = VirtualStage(synthetic_slide(blur_first=blur_first), tile=tile)
    return Tools(stage, OracleCounter(stage, truth)), policy


def heavy_truth(cols=12, rows=4):
    return {(x, y): (25, 30) for x in range(cols) for y in range(rows)}


def light_truth(cols=12, rows=4):
    return {(x, y): (0, 30) for x in range(cols) for y in range(rows)}


def test_heavy_slide_stops_before_the_fixed_protocol_would():
    tools, policy = build(RuleBased(), heavy_truth())
    trace = read_slide(tools, policy, slide="heavy")
    assert trace.fields_read < 100
    assert "within" in trace.stopped_because


def test_fixed_protocol_always_pays_full_price():
    tools, policy = build(FixedProtocol(fields=40), heavy_truth())
    trace = read_slide(tools, policy, slide="heavy")
    assert trace.fields_read == 40


def test_the_agent_reads_fewer_fields_than_the_fixed_protocol():
    """This is the saving we claim in the proposal, measured rather than asserted."""
    agent_tools, agent_policy = build(RuleBased(), heavy_truth())
    fixed_tools, fixed_policy = build(FixedProtocol(fields=40), heavy_truth())
    agent = read_slide(agent_tools, agent_policy)
    fixed = read_slide(fixed_tools, fixed_policy)
    assert agent.fields_read < fixed.fields_read


def test_clean_slide_is_called_negative_on_the_who_threshold():
    tools, policy = build(RuleBased(), light_truth())
    trace = read_slide(tools, policy, slide="clean")
    assert trace.parasites == 0
    assert "negative" in trace.stopped_because


def test_a_soft_field_makes_the_agent_refocus():
    """The trace the competition asks for: an OpenCV number changes the next action."""
    tools, policy = build(RuleBased(), heavy_truth(), blur_first=True)
    trace = read_slide(tools, policy, slide="soft-first-field")

    refocused = trace.refocus_events()
    assert refocused, "a blurred first field should have triggered a refocus"

    step = refocused[0]
    assert "focus" in step.reason
    assert step.measured["focus"] < 120


def test_the_trace_renders_for_a_human():
    tools, policy = build(RuleBased(), heavy_truth())
    trace = read_slide(tools, policy, slide="render")
    text = trace.render()
    assert "slide: render" in text
    assert "stopped:" in text
    assert "count_field()" in text


def test_unknown_action_is_a_loud_failure():
    class Broken:
        name = "broken"

        def next_move(self, quality, tools):  # noqa: ARG002
            return ("teleport", {})

    tools, _ = build(RuleBased(), heavy_truth())
    with pytest.raises(ValueError, match="unknown action"):
        read_slide(tools, Broken())


def test_who_protocol_stops_at_two_hundred_white_cells():
    """The baseline is a white cell count, not a field count. It must stop as
    soon as the target is met, and not one field later."""
    from autoscope.agent.policy import WhoProtocol

    stage = VirtualStage(synthetic_slide(), tile=TILE)
    tools = Tools(stage, counter=lambda image: (2, 30))
    trace = read_slide(tools, WhoProtocol(200), max_fields=40)

    assert tools.wbc >= 200
    assert tools.wbc - 30 < 200, "read a field it did not need"
    assert tools.fields_read == 7               # ceil(200 / 30)
    assert "200" in trace.stopped_because
