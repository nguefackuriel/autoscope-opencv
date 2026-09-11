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


# --- the decision rule, 5 September 2026 ---

from autoscope.density import BAND_NAMES, band, decide, should_stop_decision  # noqa: E402


def test_bands_split_at_the_who_lines():
    assert [band(v) for v in (0, 999, 1000, 9999, 10000, 99999, 100000)] == [0, 0, 1, 1, 2, 2, 3]
    assert len(BAND_NAMES) == 4


def test_clean_slide_is_negative_once_enough_cells_have_gone_by():
    """Ten false parasites in 500 white cells is what a clean slide produces;
    the old rule could never call it, this one does."""
    d = density(10, 500)                       # 160/uL
    assert decide(d, floor_per_ul=425, negative_wbc_min=500).call == "negative"
    assert decide(d, floor_per_ul=425, negative_wbc_min=1000).call == "undecided"


def test_heavy_slide_is_positive_and_settles_a_band():
    dec = decide(density(300, 200), floor_per_ul=425)   # 12 000/uL, interval inside 10k to 100k
    assert dec.call == "positive"
    assert dec.band_name == "10 000 to 100 000/uL"


def test_borderline_slide_is_positive_but_unsettled():
    """9 000/uL with a wide interval straddles the 10 000 line: positive, no band yet."""
    dec = decide(density(9, 8), floor_per_ul=425)
    assert dec.call == "positive"
    assert dec.band is None


def test_light_slide_near_the_floor_is_undecided_not_negative():
    """A count at the floor with a lower bound below it is neither: keep reading."""
    dec = decide(density(40, 500), floor_per_ul=425, negative_wbc_min=500)   # 640/uL, low ~ 460
    assert dec.call in ("positive", "undecided")
    dec = decide(density(30, 400), floor_per_ul=425, negative_wbc_min=500)   # 600/uL, low ~ 405
    assert dec.call == "undecided"


def test_decision_rule_stops_on_negative_positive_and_ceiling():
    stop, why, dec = should_stop_decision(density(10, 600), fields_read=20, floor_per_ul=425, negative_wbc_min=500)
    assert stop and dec.call == "negative" and "negative" in why
    stop, why, dec = should_stop_decision(density(2000, 200), fields_read=20, floor_per_ul=425)
    assert stop and dec.call == "positive" and dec.band is not None
    stop, why, dec = should_stop_decision(density(2, 60), fields_read=3, floor_per_ul=425)
    assert not stop and "minimum" in why
    stop, why, dec = should_stop_decision(density(30, 300), fields_read=400, floor_per_ul=425)
    assert stop and "needs a person" in why
