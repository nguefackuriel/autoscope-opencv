"""The stopping rule is the agentic claim, so it gets tested properly."""

import math

import pytest

from autoscope.density import (
    WHO_WBC_TARGET,
    Density,
    density,
    poisson_interval,
    should_stop,
)


def test_zero_count_has_zero_lower_bound():
    low, high = poisson_interval(0)
    assert low == 0.0
    assert high == pytest.approx(3.689, abs=1e-3)


def test_interval_brackets_the_count():
    for k in (1, 5, 20, 200):
        low, high = poisson_interval(k)
        assert low < k < high


def test_interval_tightens_as_the_count_grows():
    """Relative precision improves with more parasites. This is why the agent
    can stop early on a heavy slide and must keep going on a light one."""
    widths = [density(k, 200).ci_relative_width for k in (5, 50, 500)]
    assert widths[0] > widths[1] > widths[2]


def test_density_conversion_matches_the_who_formula():
    # 40 parasites per 200 white cells at 8000 WBC/uL is 1600 /uL.
    assert density(40, 200).per_ul == pytest.approx(1600.0)


def test_empty_estimate_is_never_called_precise():
    assert math.isinf(density(0, 200).ci_relative_width)


def test_minimum_fields_is_enforced():
    stop, why = should_stop(density(500, 200), fields_read=3)
    assert not stop
    assert "minimum" in why


def test_heavy_slide_stops_early():
    """A heavy slide reaches a tight interval well before 100 fields, which is
    where the compute and the technician's time get saved."""
    stop, why = should_stop(density(500, 60), fields_read=20)
    assert stop
    assert "within" in why


def test_light_slide_keeps_reading():
    stop, why = should_stop(density(2, 60), fields_read=20)
    assert not stop
    assert "keep reading" in why


def test_clean_slide_stops_on_the_who_threshold():
    stop, why = should_stop(density(0, WHO_WBC_TARGET), fields_read=100)
    assert stop
    assert "negative" in why


def test_ceiling_escalates_instead_of_answering():
    stop, why = should_stop(density(3, 900), fields_read=400)
    assert stop
    assert "needs a person" in why
