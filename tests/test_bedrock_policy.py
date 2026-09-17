"""The agent, with a fake Bedrock client.

No network, no cost, no credentials. What these tests are for is the part that
is not the model: that a move comes back as a move the loop understands, that
the observation carries real numbers and no infinities, and above all that a
model which answers "refocus" forever is stopped by something that is not a
model. That last one is the whole of "appropriate autonomy": the limits have
to hold when the model misbehaves, and the only way to know they do is to
build a model that misbehaves.
"""

import json

import pytest

from autoscope.agent.bedrock import MOVES, BedrockPolicy, observation
from autoscope.agent.loop import read_slide
from autoscope.agent.policy import MAX_REFOCUS_TRIES
from autoscope.agent.tools import OracleCounter, Tools
from autoscope.quality import assess_field
from autoscope.virtual_stage import VirtualStage

from test_agent_loop import heavy_truth, light_truth, synthetic_slide


class FakeBlock:
    type = "tool_use"

    def __init__(self, name, input_):
        self.name = name
        self.input = input_


class FakeUsage:
    input_tokens = 400
    output_tokens = 30


class FakeResponse:
    stop_reason = "tool_use"

    def __init__(self, name, input_):
        self.content = [FakeBlock(name, input_)]
        self.usage = FakeUsage()


class FakeClient:
    """Replays a scripted list of moves and records what it was asked.

    The last move repeats once the script runs out, so a test only has to write
    the moves it cares about.
    """

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.messages = self

    def create(self, **kwargs):
        self.seen.append(kwargs)
        name, args = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        return FakeResponse(name, args)


def build(script, truth, blur_first=False):
    stage = VirtualStage(synthetic_slide(blur_first=blur_first), tile=256)
    tools = Tools(stage, OracleCounter(stage, truth))
    policy = BedrockPolicy(client=FakeClient(script))
    return tools, policy


def test_a_move_reaches_the_loop_and_the_slide_gets_read():
    script = [("count_then_move", {"reason": "field is clean and in focus"})] * 8
    script.append(("stop", {"call": "settled", "reason": "interval sits in one band"}))
    tools, policy = build(script, heavy_truth())

    trace = read_slide(tools, policy, slide="fake")

    assert trace.fields_read == 8
    assert "settled" in trace.stopped_because
    assert len(policy.calls) == 9


def test_flag_for_human_is_an_outcome_not_a_failure():
    tools, policy = build([("flag_for_human", {"reason": "quality bad across the slide"})],
                          light_truth())

    trace = read_slide(tools, policy, slide="fake")

    assert tools.escalation == "quality bad across the slide"
    assert "handed to a person" in trace.stopped_because


def test_a_model_that_only_ever_refocuses_is_stopped_by_the_cap():
    """The failure mode that has no natural end.

    The cap is per field, not per slide, which is what RuleBased does too: a
    field that will not come into focus is abandoned after MAX_REFOCUS_TRIES
    and the stage moves on, and the next field gets its own fair try. So the
    invariant worth asserting is not a total but that no single field is ever
    hunted more than the cap allows, and that the slide ends by running out of
    fields rather than by hitting the step ceiling.
    """
    tools, policy = build([("refocus", {"dz": 2, "reason": "looks soft"})],
                          heavy_truth(), blur_first=True)

    trace = read_slide(tools, policy, slide="fake", max_steps=400)

    per_field: dict[str, int] = {}
    for step in trace.steps:
        if "refocus" in step.action:
            # z changes as the hunt widens; the field is the (x,y) part.
            field = step.where.split(" z=")[0]
            per_field[field] = per_field.get(field, 0) + 1

    assert per_field, "the blurred field should have been hunted at least once"
    assert max(per_field.values()) <= MAX_REFOCUS_TRIES
    assert "step ceiling" not in trace.stopped_because


def test_refocus_dz_is_clamped_however_the_model_answers():
    tools, policy = build([("refocus", {"dz": 9999, "reason": "way off"})],
                          heavy_truth(), blur_first=True)
    quality = assess_field(tools.frame.image)

    action, args = policy.next_move(quality, tools)

    assert action == "refocus"
    assert args["dz"] == 8


def test_no_tool_call_is_an_error_rather_than_a_silent_default():
    """tool_choice=any makes this impossible; if it ever happens we want to see
    it, not to record a decision the agent never made."""

    class NoToolClient(FakeClient):
        def create(self, **kwargs):
            response = FakeResponse("stop", {})
            response.content = []
            response.stop_reason = "end_turn"
            return response

    stage = VirtualStage(synthetic_slide(), tile=256)
    tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    policy = BedrockPolicy(client=NoToolClient([]))

    with pytest.raises(RuntimeError, match="no move"):
        policy.next_move(assess_field(tools.frame.image), tools)


def test_the_observation_is_json_and_carries_no_infinity():
    """ci_relative_width is inf before anything is counted. Infinity is not
    valid JSON and a parser that accepts it turns 'no estimate' into a very
    large number, which reads as a very wide but real interval."""
    stage = VirtualStage(synthetic_slide(), tile=256)
    tools = Tools(stage, OracleCounter(stage, light_truth()))

    obs = observation(assess_field(tools.frame.image), tools)
    body = json.dumps(obs)

    assert obs["estimate"]["ci_relative_width"] is None
    assert "Infinity" not in body
    assert obs["reference"]["noise_floor_per_ul"] > 0
    assert obs["totals"]["fields_counted"] == 0


def test_the_request_asks_for_exactly_one_move():
    tools, policy = build([("count_then_move", {"reason": "fine"})], heavy_truth())

    policy.next_move(assess_field(tools.frame.image), tools)
    sent = policy.client.seen[0]

    assert sent["tool_choice"] == {"type": "any"}
    assert {t["name"] for t in sent["tools"]} == {m["name"] for m in MOVES}
    assert sent["output_config"]["effort"] == "low"
    # The count is given to the model, never asked of it.
    body = json.loads(sent["messages"][0]["content"])
    assert "parasites" in body["totals"]


def test_every_call_is_recorded_for_the_audit_trail():
    script = [("count_then_move", {"reason": "clean"})] * 3
    script.append(("stop", {"call": "positive", "reason": "interval clears the floor"}))
    tools, policy = build(script, heavy_truth())

    read_slide(tools, policy, slide="fake")

    assert all("observation" in c and "move" in c for c in policy.calls)
    assert policy.calls[-1]["move"][0] == "stop"
    assert policy.calls[0]["usage"]["input_tokens"] == 400
