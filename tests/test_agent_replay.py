"""The record/replay cache, with no credentials and no network.

What is worth testing here is not JSON round-tripping. It is the two
properties the published comparison rests on:

  a replayed run makes the same moves as the recorded one, so "replayed" is a
  true claim rather than a hopeful one;

  and a cache that no longer matches the prompt fails loudly. That is the
  failure this design exists to prevent: if editing moves.py left the old keys
  valid, the evaluation would replay decisions the current prompt never
  produced, publish them, and leave no trace.
"""

import json

import pytest

from autoscope.agent.bedrock import BedrockPolicy
from autoscope.agent.loop import read_slide
from autoscope.agent.moves import MOVES, SYSTEM
from autoscope.agent.replay import (CacheMiss, RecordingClient, ReplayClient,
                                    cache_key, cost_of, key_for_request)
from autoscope.agent.tools import OracleCounter, Tools
from autoscope.virtual_stage import VirtualStage

from test_agent_loop import heavy_truth, synthetic_slide
from test_bedrock_policy import FakeClient


def request(observation='{"totals":{"parasites":1}}', model="anthropic.claude-opus-5",
            effort="low", system=SYSTEM, tools=MOVES):
    return {
        "model": model,
        "system": system,
        "tools": tools,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": observation}],
    }


def build(script, tmp_path, truth=None):
    stage = VirtualStage(synthetic_slide(), tile=256)
    tools = Tools(stage, OracleCounter(stage, truth or heavy_truth()))
    inner = FakeClient(script)
    client = RecordingClient(inner, tmp_path / "calls.jsonl")
    return tools, BedrockPolicy(client=client), client


SCRIPT = [("count_then_move", {"reason": "clean"})] * 5 + \
         [("stop", {"call": "positive", "reason": "interval clears the floor"})]


def test_a_replayed_run_makes_the_same_moves_as_the_recorded_one(tmp_path):
    tools, policy, recorder = build(SCRIPT, tmp_path)
    recorded = read_slide(tools, policy, slide="p1")
    assert recorder.recorded > 0

    stage = VirtualStage(synthetic_slide(), tile=256)
    replay_tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    replay_policy = BedrockPolicy(client=ReplayClient(tmp_path / "calls.jsonl"))
    replayed = read_slide(replay_tools, replay_policy, slide="p1")

    assert [c["move"] for c in replay_policy.calls] == [c["move"] for c in policy.calls]
    assert replayed.stopped_because == recorded.stopped_because
    assert (replayed.parasites, replayed.wbc) == (recorded.parasites, recorded.wbc)


def test_replaying_a_call_that_was_never_recorded_raises(tmp_path):
    tools, policy, _ = build(SCRIPT, tmp_path)
    read_slide(tools, policy, slide="p1")

    client = ReplayClient(tmp_path / "calls.jsonl")
    with pytest.raises(CacheMiss, match="no recorded call"):
        client.messages.create(**request(observation='{"never":"seen"}'))


def test_a_miss_can_fall_through_to_live_but_only_when_asked(tmp_path):
    tools, policy, _ = build(SCRIPT, tmp_path)
    read_slide(tools, policy, slide="p1")

    live = FakeClient([("skip", {"reason": "live answer"})])
    client = ReplayClient(tmp_path / "calls.jsonl", allow_live=live)
    response = client.messages.create(**request(observation='{"never":"seen"}'))

    assert client.live == 1
    assert response.content[0].name == "skip"


def test_changing_the_prompt_invalidates_every_key():
    """The whole reason the key is not just the observation."""
    base = key_for_request(request())

    assert key_for_request(request(system=SYSTEM + " One more instruction.")) != base
    assert key_for_request(request(tools=MOVES[:-1])) != base
    assert key_for_request(request(model="anthropic.claude-sonnet-5")) != base
    assert key_for_request(request(effort="high")) != base
    assert key_for_request(request(observation='{"different":true}')) != base
    # and the same question still hashes the same
    assert key_for_request(request()) == base


def test_the_key_does_not_move_with_dict_ordering():
    a = cache_key("m", "low", "s", MOVES, {"one": 1, "two": 2})
    b = cache_key("m", "low", "s", MOVES, {"two": 2, "one": 1})
    assert a == b


def test_recording_twice_does_not_buy_the_same_call_twice(tmp_path):
    tools, policy, recorder = build(SCRIPT, tmp_path)
    read_slide(tools, policy, slide="p1")
    bought = recorder.recorded

    stage = VirtualStage(synthetic_slide(), tile=256)
    again_tools = Tools(stage, OracleCounter(stage, heavy_truth()))
    again = RecordingClient(FakeClient(SCRIPT), tmp_path / "calls.jsonl")
    read_slide(again_tools, BedrockPolicy(client=again), slide="p1")

    assert bought > 0
    assert again.recorded == 0
    assert again.served_from_cache == bought


def test_a_partial_file_is_still_usable(tmp_path):
    """A run that dies mid-way leaves the calls it made, not a corrupt file."""
    path = tmp_path / "calls.jsonl"
    tools, policy, _ = build(SCRIPT, tmp_path)
    read_slide(tools, policy, slide="p1")

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n")

    client = ReplayClient(path)
    assert len(client.rows) == 2


def test_a_missing_cache_file_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="--record"):
        ReplayClient(tmp_path / "nothing.jsonl")


def test_cost_counts_cached_input_at_a_tenth():
    calls = [{"usage": {"input_tokens": 1_000_000, "output_tokens": 0,
                        "cache_read_input_tokens": 0}}]
    cached = [{"usage": {"input_tokens": 0, "output_tokens": 0,
                         "cache_read_input_tokens": 1_000_000}}]

    assert cost_of(calls)["usd_estimated"] == pytest.approx(5.0)
    assert cost_of(cached)["usd_estimated"] == pytest.approx(0.5)


def test_the_recorded_file_carries_the_evidence_not_just_the_answer(tmp_path):
    """A recording a judge cannot read back is not evidence."""
    tools, policy, _ = build(SCRIPT, tmp_path)
    read_slide(tools, policy, slide="p1")

    rows = [json.loads(line) for line in
            (tmp_path / "calls.jsonl").read_text().splitlines()]

    assert rows, "nothing was recorded"
    for row in rows:
        assert set(row) >= {"key", "observation", "move", "usage", "model", "recorded_utc"}
        # the observation is the real one the policy built, not a stub
        assert set(row["observation"]) == {"field", "totals", "estimate", "reference"}
    assert rows[-1]["move"]["name"] == "stop"
    assert rows[-1]["move"]["input"]["call"] == "positive"
