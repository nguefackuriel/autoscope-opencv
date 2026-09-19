"""The agent as an arm of the comparison harness, driven offline.

This runs the real `run_one`, the real `read_slide` and the real `CachedTools`
over a small synthetic field table, with a scripted client. No credentials, no
network, no `anthropic` package.

The test that carries the most weight is
`test_the_reported_call_is_the_rule_s_and_not_the_model_s`. It encodes the one
design decision in this work: the agent chooses when to stop, and
`density.decide` says what the counts mean. If that ever flips, a model could
declare a slide negative and the results table would print "negative", which is
the single worst thing this pipeline could do quietly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import agent_eval  # noqa: E402

from test_bedrock_policy import FakeClient  # noqa: E402

from autoscope.agent.bedrock import BedrockPolicy  # noqa: E402
from autoscope.agent.policy import DecisionRule  # noqa: E402

FLOOR = 150.0


def fields(n: int = 12, parasites: int = 25, wbc: int = 50, patient: str = "TFX") -> list[dict]:
    """A patient's worth of rows in the shape agent_fields.json uses."""
    return [{
        "patient": patient, "image": f"{i}.jpg",
        "found_parasites": parasites, "found_wbc": wbc,
        "truth_parasites": parasites, "truth_wbc": wbc,
        "focus": 3.4, "even": 0.95, "debris": 0.5, "usable": True, "seconds": 1.7,
    } for i in range(n)]


def agent(script):
    return BedrockPolicy(client=FakeClient(script), floor_per_ul=FLOOR)


STOP_POSITIVE = ("stop", {"call": "positive", "reason": "the interval clears the floor"})
COUNT = ("count_then_move", {"reason": "readable field"})


def test_the_agent_arm_fills_the_decision_columns():
    """Wiring the agent in naively left these blank, which is why they exist.

    Eight fields at 25 parasites in 50 white cells is 4 000/uL with an interval
    of roughly 3 500 to 4 600, which sits inside the 1 000 to 10 000 band, so
    the band settles and the column is non-empty.
    """
    run = agent_eval.run_one(fields(), agent([COUNT] * 8 + [STOP_POSITIVE]), "found", FLOOR, 500)

    assert run["policy"] == "bedrock"
    assert run["derived_call"] == "positive"
    assert run["derived_band"] is not None
    assert run["model_calls"] > 0


def test_a_band_that_straddles_a_boundary_does_not_settle():
    """Not a defect: an interval spanning 1 000 to 10 000 and 10 000 to 100 000
    has not answered the question the band is there to answer. The column is
    None and the agent should keep reading."""
    run = agent_eval.run_one(fields(parasites=20, wbc=18),
                             agent([COUNT] * 6 + [STOP_POSITIVE]), "found", FLOOR, 500)

    assert run["derived_call"] == "positive"
    assert run["derived_band"] is None


def test_the_reported_call_is_the_rule_s_and_not_the_model_s():
    """A model that says "negative" over plainly positive counts must not make
    the table say negative. It chooses when to stop; the counts say what that
    means."""
    lying = [COUNT] * 6 + [("stop", {"call": "negative", "reason": "I think it is clean"})]

    run = agent_eval.run_one(fields(), agent(lying), "found", FLOOR, 500)

    assert run["derived_call"] == "positive"     # what the counts plainly mean
    assert run["agent_call"] == "negative"       # what the model claimed
    summary = agent_eval.summarise([run], {"TFX": 8000.0}, 0.20)
    assert summary["agent_calls_given"] == 1
    assert summary["agent_call_agreed"] == 0


def test_a_model_that_agrees_with_the_counts_is_counted_as_agreeing():
    run = agent_eval.run_one(fields(), agent([COUNT] * 8 + [STOP_POSITIVE]), "found", FLOOR, 500)
    summary = agent_eval.summarise([run], {"TFX": 8000.0}, 0.20)

    assert summary["agent_call_agreed"] == 1


def test_settled_counts_as_agreement_with_a_positive_and_never_a_negative():
    """The model's vocabulary and the rule's do not line up exactly: "settled"
    is a positive that has converged."""
    assert agent_eval._same_call("settled", "positive") is True
    assert agent_eval._same_call("settled", "negative") is False
    assert agent_eval._same_call("negative", "negative") is True
    assert agent_eval._same_call("positive", "undecided") is False


def test_a_run_that_reads_to_the_end_of_the_slide_still_has_a_call():
    """The agent never says stop; the slide runs out. The columns must still be
    filled, or the arms are not comparable on exactly the runs where a policy
    failed to decide."""
    run = agent_eval.run_one(fields(n=5), agent([COUNT] * 40), "found", FLOOR, 500)

    assert run["derived_call"] is not None
    assert run["agent_call"] is None       # it never gave one
    assert run["fields_read"] == 5


def test_the_rule_arms_keep_the_shape_already_committed():
    """No model keys leak into a policy that is not a model, so the committed
    run dicts in eval/results/agent_eval.json keep their shape."""
    run = agent_eval.run_one(fields(), DecisionRule(FLOOR, 500, 0.20, min_fields=5),
                             "found", FLOOR, 500)

    assert "model_calls" not in run
    assert "input_tokens" not in run
    assert run["agent_call"] is None
    # and the derived columns are there for it too, so the arms compare
    assert run["derived_call"] == "positive"


def test_the_agent_and_the_rule_are_scored_by_the_same_function():
    """Same counts in, same derived call out, whoever stopped there."""
    same = fields()
    by_agent = agent_eval.run_one(same, agent([COUNT] * 6 + [STOP_POSITIVE]), "found", FLOOR, 500)
    by_rule = agent_eval.run_one(same, DecisionRule(FLOOR, 500, 0.20, min_fields=5),
                                 "found", FLOOR, 500)

    if by_agent["fields_read"] == by_rule["fields_read"]:
        assert by_agent["derived_call"] == by_rule["derived_call"]
        assert by_agent["derived_band"] == by_rule["derived_band"]


def test_replaying_the_agent_never_imports_anthropic(tmp_path):
    """The zero-cost guarantee, as an assertion: a judge with no AWS account and
    no SDK installed must be able to reproduce the comparison."""
    from autoscope.agent.replay import RecordingClient, ReplayClient

    path = tmp_path / "calls.jsonl"
    recorder = RecordingClient(FakeClient([COUNT] * 6 + [STOP_POSITIVE]), path)
    agent_eval.run_one(fields(), BedrockPolicy(client=recorder, floor_per_ul=FLOOR),
                       "found", FLOOR, 500)

    sys.modules.pop("anthropic", None)
    replayed = agent_eval.run_one(
        fields(), BedrockPolicy(client=ReplayClient(path), floor_per_ul=FLOOR),
        "found", FLOOR, 500)

    assert "anthropic" not in sys.modules
    assert replayed["derived_call"] == "positive"


def test_a_replayed_run_dict_is_identical_to_the_recorded_one(tmp_path):
    """Including the token counts, which come off the recorded calls rather than
    being re-measured. Without this, "replayed" and "recorded" would be two
    different experiments wearing the same name."""
    import json

    from autoscope.agent.replay import RecordingClient, ReplayClient

    path = tmp_path / "calls.jsonl"
    script = [COUNT] * 6 + [STOP_POSITIVE]
    recorded = agent_eval.run_one(
        fields(), BedrockPolicy(client=RecordingClient(FakeClient(script), path),
                                floor_per_ul=FLOOR), "found", FLOOR, 500)
    replayed = agent_eval.run_one(
        fields(), BedrockPolicy(client=ReplayClient(path), floor_per_ul=FLOOR),
        "found", FLOOR, 500)

    assert json.dumps(recorded, sort_keys=True) == json.dumps(replayed, sort_keys=True)
