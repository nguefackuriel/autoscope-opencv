"""The cost arithmetic, because it is the number a lab reads.

The subtle part is the 4-vCPU slot. A profile is taken on a whole machine with
all its threads; a Batch job gets 4 vCPU. Charging the whole-machine field time
at the whole-machine hourly rate counts the parallelism twice and reports a
cost several times too low. There is no way to see that in the output — it is a
plausible-looking number — so it gets a test.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import cost_model  # noqa: E402


PRICING = {
    "region": "us-east-1",
    "verified": True,
    "instances": {
        "c8g.8xlarge": {"vcpu": 32, "on_demand_usd_hour": 1.0,
                        "spot_usd_hour_median_7d": 0.40},
        "c7i.8xlarge": {"vcpu": 32, "on_demand_usd_hour": 2.0,
                        "spot_usd_hour_median_7d": 0.80},
    },
}


def profile(label, seconds_per_field, threads=32, fields_per_slide=30.0):
    return {
        "label": label,
        "threads": threads,
        "seconds_per_field_median": seconds_per_field,
        "fields_per_slide": fields_per_slide,
        "kleidicv_in_build": label == "c8g-cool",
    }


def test_the_slot_scaling_does_not_count_parallelism_twice():
    """1 s/field on 32 threads is 8 s/field in a 4-vCPU slot, and 8 such slots
    run at once. The two cancel, so the machine still does 32 vCPU of work —
    but the per-slot time is what a Batch job actually experiences."""
    row = cost_model.cost_per_1000(profile("c8g-cool", 1.0), PRICING, spot=False)

    assert row["seconds_per_field_on_4_vcpu"] == pytest.approx(8.0)
    # 8 s x 30 fields = 240 s per slide per slot; 8 slots => 30 s of instance
    # time per slide; 1000 slides = 30 000 instance-seconds = 8.333 h at $1/h.
    assert row["usd_per_1000_slides"] == pytest.approx(8.33, abs=0.02)
    assert row["slides_per_hour_per_instance"] == pytest.approx(120.0, abs=0.5)


def test_a_faster_build_costs_less_on_the_same_instance():
    cool = cost_model.cost_per_1000(profile("c8g-cool", 1.0), PRICING, spot=False)
    control = cost_model.cost_per_1000(profile("c8g-no-kleidicv", 1.5), PRICING, spot=False)

    assert cool["usd_per_1000_slides"] < control["usd_per_1000_slides"]
    assert control["usd_per_1000_slides"] / cool["usd_per_1000_slides"] == pytest.approx(1.5, abs=0.01)


def test_spot_is_priced_off_the_spot_column():
    on_demand = cost_model.cost_per_1000(profile("c8g-cool", 1.0), PRICING, spot=False)
    spot = cost_model.cost_per_1000(profile("c8g-cool", 1.0), PRICING, spot=True)

    assert spot["basis"] == "spot"
    assert spot["usd_per_1000_slides"] == pytest.approx(on_demand["usd_per_1000_slides"] * 0.4,
                                                        abs=0.02)


def test_a_missing_price_produces_no_row_rather_than_a_zero():
    """A zero in a cost table reads as free, not as unknown."""
    pricing = json.loads(json.dumps(PRICING))
    pricing["instances"]["c8g.8xlarge"]["on_demand_usd_hour"] = None

    assert cost_model.cost_per_1000(profile("c8g-cool", 1.0), pricing, spot=False) is None


def test_a_laptop_has_no_hourly_price_and_is_left_out():
    assert cost_model.cost_per_1000(profile("local-x86", 2.1, threads=8), PRICING,
                                    spot=False) is None


def test_every_benchmark_arm_maps_to_an_instance_or_is_deliberately_excluded():
    """A new arm with no entry would silently vanish from the cost table."""
    for arm, instance in cost_model.ARM_INSTANCE.items():
        assert instance is None or instance in PRICING["instances"], arm
