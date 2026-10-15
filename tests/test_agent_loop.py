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


def test_decision_rule_calls_a_clean_slide_negative_and_a_heavy_one_positive():
    """The old rule needed zero parasites for a negative; this one needs the
    count to sit within what a clean slide produces, after enough cells."""
    from autoscope.agent.policy import DecisionRule

    def noisy_clean(cols=12, rows=4):
        return {(x, y): (1, 30) for x in range(cols) for y in range(rows)}   # one false parasite a field: 267/uL

    tools, policy = build(DecisionRule(floor_per_ul=425, negative_wbc_min=300), noisy_clean())
    trace = read_slide(tools, policy, slide="clean")
    assert tools.decision.call == "negative"
    assert "negative" in trace.stopped_because
    assert tools.wbc >= 300

    tools, policy = build(DecisionRule(floor_per_ul=425), heavy_truth())
    trace = read_slide(tools, policy, slide="heavy")
    assert tools.decision.call == "positive"
    assert tools.decision.band_name is not None
    assert trace.fields_read < 100


def test_a_clean_slide_that_crosses_the_threshold_on_its_last_field_is_called_negative():
    """The decision the rule would have made on the final totals must not be
    thrown away because the slide happened to end there.

    Four fields of 130 white cells: 390 after three, 520 after the fourth. The
    rule is asked at the top of each turn, before the count it then requests,
    so it last saw 390 and said "keep going". The fourth count makes it 520,
    the slide is out, and the loop used to escalate without asking again. On
    the frozen uninfected half that was 7 of the 10 clean slides sent to a
    person: all of them past 500 cells with nothing found.
    """
    from autoscope.agent.policy import DecisionRule

    stage = VirtualStage(synthetic_slide(cols=4, rows=1), tile=TILE)
    tools = Tools(stage, counter=lambda image: (0, 130))
    policy = DecisionRule(floor_per_ul=148, negative_wbc_min=500, min_fields=1)
    trace = read_slide(tools, policy, slide="short clean", max_fields=40)

    assert tools.fields_read == 4 and tools.wbc == 520, "the look must add no field"
    assert tools.escalation is None, "the rule had a negative; it must not be escalated"
    assert tools.decision.call == "negative"
    assert "negative" in trace.stopped_because


def test_a_policy_without_a_rule_still_escalates_when_the_slide_runs_out():
    """The last look is only for policies that can decide from totals alone. A
    model policy has no `_stop`, so the end of the slide is still an escalation
    and, just as important, costs it no extra call."""

    class CountEverything:
        name = "count-everything"

        def next_move(self, quality, tools):  # noqa: ARG002
            return "count_then_move", {"reason": "keep counting"}

    stage = VirtualStage(synthetic_slide(cols=4, rows=1), tile=TILE)
    tools = Tools(stage, counter=lambda image: (0, 130))
    trace = read_slide(tools, CountEverything(), slide="short clean", max_fields=40)

    assert tools.fields_read == 4
    assert tools.escalation is not None
    assert "ran out of slide" in trace.stopped_because
