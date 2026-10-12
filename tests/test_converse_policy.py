"""The model-agnostic agent, with a fake Converse client.

The point of ConversePolicy is that the agent is not tied to one vendor, so
what these tests pin is that it behaves exactly like the Anthropic path in
every way that matters: the same five moves, the same refusal to invent a move
when the model gives none, the same refocus cap, and the same cache.

The extra thing Converse brings is text blocks. Several models return a
sentence alongside the tool call; that must not be read as a missing move.
"""

import json

import pytest

from autoscope.agent.converse import ConversePolicy, as_tool_config
from autoscope.agent.loop import read_slide
from autoscope.agent.moves import MOVES
from autoscope.agent.policy import MAX_REFOCUS_TRIES
from autoscope.agent.replay import CacheMiss, RecordingClient, ReplayClient
from autoscope.agent.tools import OracleCounter, Tools
from autoscope.quality import assess_field
from autoscope.virtual_stage import VirtualStage

from test_agent_loop import heavy_truth, synthetic_slide


class FakeConverse:
    """Replays scripted moves in Bedrock Converse's response shape."""

    def __init__(self, script, text_too=False):
        self.script = list(script)
        self.seen = []
        self.text_too = text_too

    def converse(self, **kwargs):
        self.seen.append(kwargs)
        name, args = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        content = [{"toolUse": {"name": name, "input": args}}]
        if self.text_too:
            content.insert(0, {"text": "Let me look at this field."})
        return {
            "output": {"message": {"content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1700, "outputTokens": 90},
        }


COUNT = ("count_then_move", {"reason": "readable"})
STOP = ("stop", {"call": "positive", "reason": "the interval clears the floor"})


def build(script, client=None, **kw):
    stage = VirtualStage(synthetic_slide(), tile=256)
    tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    policy = ConversePolicy(model="openai.gpt-oss-120b-1:0",
                            client=client or FakeConverse(script), **kw)
    return tools, policy


def test_the_moves_translate_into_converse_shape():
    tc = as_tool_config(MOVES)

    assert {t["toolSpec"]["name"] for t in tc["tools"]} == {m["name"] for m in MOVES}
    assert tc["toolChoice"] == {"any": {}}
    # the schema is nested one level deeper than Anthropic's, and intact
    stop = next(t["toolSpec"] for t in tc["tools"] if t["toolSpec"]["name"] == "stop")
    assert stop["inputSchema"]["json"]["properties"]["call"]["enum"] == \
        ["positive", "negative", "settled"]


def test_a_slide_is_read_and_the_arm_is_named_after_the_model():
    tools, policy = build([COUNT] * 6 + [STOP])

    trace = read_slide(tools, policy, slide="x")

    assert policy.name == "gpt-oss-120b-1"       # not "converse": two model arms must not collide
    assert trace.fields_read == 6
    assert "positive" in trace.stopped_because


def test_text_alongside_the_tool_call_is_not_a_missing_move():
    """Several models always emit a sentence next to the tool use."""
    tools, policy = build([COUNT], client=FakeConverse([COUNT], text_too=True))

    action, _ = policy.next_move(assess_field(tools.frame.image), tools)

    assert action == "count_then_move"


def test_no_tool_call_is_an_error_rather_than_a_silent_default():
    class NoTool(FakeConverse):
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "I think we should stop."}]}},
                    "stopReason": "end_turn", "usage": {}}

    tools, policy = build([], client=NoTool([]))

    with pytest.raises(RuntimeError, match="no move"):
        policy.next_move(assess_field(tools.frame.image), tools)


def test_a_move_with_no_reason_still_runs():
    """A weaker model omits a required field. The loop must not die on a key
    that carries no decision."""
    tools, policy = build([("count_then_move", {})])

    action, args = policy.next_move(assess_field(tools.frame.image), tools)

    assert action == "count_then_move"
    assert args["reason"] == "(no reason given)"


def test_the_refocus_cap_holds_for_this_policy_too():
    stage = VirtualStage(synthetic_slide(blur_first=True), tile=256)
    tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    policy = ConversePolicy(model="m", client=FakeConverse([("refocus", {"dz": 2, "reason": "soft"})]))

    trace = read_slide(tools, policy, slide="x", max_steps=400)

    per_field = {}
    for step in trace.steps:
        if "refocus" in step.action:
            f = step.where.split(" z=")[0]
            per_field[f] = per_field.get(f, 0) + 1
    assert max(per_field.values()) <= MAX_REFOCUS_TRIES
    assert "step ceiling" not in trace.stopped_because


def test_the_stop_verdict_survives_as_a_field():
    tools, policy = build([STOP])

    policy.next_move(assess_field(tools.frame.image), tools)

    assert policy.last_call == "positive"


def test_the_cache_round_trips_a_converse_call(tmp_path):
    path = tmp_path / "calls.jsonl"
    inner = FakeConverse([COUNT] * 6 + [STOP])
    tools, policy = build([], client=RecordingClient(inner, path))
    recorded = read_slide(tools, policy, slide="x")

    stage = VirtualStage(synthetic_slide(), tile=256)
    replay_tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    replay_policy = ConversePolicy(model="openai.gpt-oss-120b-1:0",
                                   client=ReplayClient(path))
    replayed = read_slide(replay_tools, replay_policy, slide="x")

    assert [c["move"] for c in replay_policy.calls] == [c["move"] for c in policy.calls]
    assert replayed.stopped_because == recorded.stopped_because


def test_two_models_do_not_share_a_cache_entry(tmp_path):
    """The model id is in the key, so one arm cannot replay another's answers."""
    path = tmp_path / "calls.jsonl"
    tools, policy = build([], client=RecordingClient(FakeConverse([COUNT]), path))
    policy.next_move(assess_field(tools.frame.image), tools)

    other = ConversePolicy(model="meta.llama4-maverick", client=ReplayClient(path))
    stage = VirtualStage(synthetic_slide(), tile=256)
    other_tools = Tools(stage, OracleCounter(stage, heavy_truth()))

    with pytest.raises(CacheMiss):
        other.next_move(assess_field(other_tools.frame.image), other_tools)


def test_the_request_carries_the_prompt_and_forces_one_move():
    tools, policy = build([COUNT])
    policy.next_move(assess_field(tools.frame.image), tools)
    sent = policy.client.seen[0]

    assert sent["modelId"] == "openai.gpt-oss-120b-1:0"
    assert sent["toolConfig"]["toolChoice"] == {"any": {}}
    assert sent["inferenceConfig"]["temperature"] == 0.0
    body = json.loads(sent["messages"][0]["content"][0]["text"])
    assert set(body) == {"field", "totals", "estimate", "reference"}
