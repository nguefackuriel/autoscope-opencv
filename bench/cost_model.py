"""Dollars per 1,000 slides, from the measured field time and real prices.

Section 9 of the proposal promises this number for three builds on Graviton
plus x86, "all at the same accuracy". It is also the only number in the COOL
story that a laboratory actually cares about: a 1.4x speedup is interesting,
a third off the bill is a decision.

    python bench/cost_model.py                      # every profile in results/
    python bench/cost_model.py --fetch              # refresh prices from AWS
    python bench/cost_model.py --markdown           # the table for the report

PRICES ARE NOT GUESSED. They come from bench/pricing.json, which starts out
marked unverified and stays that way until somebody either runs --fetch against
a real account or edits the file and sets verified to true. An unverified run
still prints, with the word UNVERIFIED on every line, because a cost table with
invented prices in it is worse than no cost table: it looks like a measurement.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRICING = HERE / "pricing.json"

# Which instance each benchmark arm ran on. The label in a profile file is the
# arm; this is how an arm becomes a price.
ARM_INSTANCE = {
    "c8g-cool": "c8g.8xlarge",
    "c8g-stock": "c8g.8xlarge",
    "c8g-no-kleidicv": "c8g.8xlarge",
    "c7i-stock": "c7i.8xlarge",
    "local-x86": None,          # a laptop has no hourly price
}


def load_pricing() -> dict:
    if not PRICING.exists():
        raise SystemExit(
            f"{PRICING} is missing. Create it with --fetch (needs AWS credentials) "
            "or copy pricing.example.json and fill in the prices by hand."
        )
    return json.loads(PRICING.read_text())


def fetch_pricing(region: str = "us-east-1") -> dict:
    """On-demand from the Pricing API, Spot from recent history.

    Spot is a moving price, so what goes in the file is the median over the
    last week rather than this second's number: a cost table built on a
    thirty-second dip is not reproducible.
    """
    import statistics
    from datetime import datetime, timedelta, timezone

    import boto3

    # The Pricing API only lives in a few regions and only speaks us-east-1
    # names for locations.
    pricing = boto3.client("pricing", region_name="us-east-1")
    ec2 = boto3.client("ec2", region_name=region)

    out = {"region": region, "fetched": date.today().isoformat(), "verified": True,
           "instances": {}}

    for instance in sorted({i for i in ARM_INSTANCE.values() if i}):
        response = pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            ],
            MaxResults=1,
        )
        if not response["PriceList"]:
            print(f"  no on-demand price found for {instance}, skipping")
            continue

        product = json.loads(response["PriceList"][0])
        terms = product["terms"]["OnDemand"]
        dimension = next(iter(next(iter(terms.values()))["priceDimensions"].values()))
        on_demand = float(dimension["pricePerUnit"]["USD"])
        vcpu = int(product["product"]["attributes"]["vcpu"])

        history = ec2.describe_spot_price_history(
            InstanceTypes=[instance],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc) - timedelta(days=7),
        )["SpotPriceHistory"]
        spot = statistics.median(float(h["SpotPrice"]) for h in history) if history else None

        out["instances"][instance] = {
            "vcpu": vcpu,
            "on_demand_usd_hour": round(on_demand, 5),
            "spot_usd_hour_median_7d": round(spot, 5) if spot else None,
        }
        print(f"  {instance}: ${on_demand:.4f}/h on demand"
              + (f", ${spot:.4f}/h spot" if spot else ", no spot history"))

    PRICING.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {PRICING}")
    return out


def profiles() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("profile-*.json"))]


def cost_per_1000(profile: dict, pricing: dict, spot: bool) -> dict | None:
    instance = ARM_INSTANCE.get(profile["label"])
    if instance is None or instance not in pricing.get("instances", {}):
        return None

    price = pricing["instances"][instance]
    hourly = price["spot_usd_hour_median_7d"] if spot else price["on_demand_usd_hour"]
    if hourly is None:
        return None

    # One job gets 4 vCPU (the Batch job definition), so one instance runs
    # vcpu/4 fields at once. The profile was taken on however many threads the
    # box had, so scale the per-field time to a 4-vCPU slot before dividing:
    # otherwise a 32-vCPU measurement is charged as if it were one job.
    slot_vcpu = 4
    jobs_at_once = max(price["vcpu"] // slot_vcpu, 1)
    seconds_per_field = profile["seconds_per_field_median"] * (
        profile["threads"] / slot_vcpu
    )

    slide_seconds = seconds_per_field * profile["fields_per_slide"]
    instance_seconds_per_1000 = slide_seconds * 1000 / jobs_at_once

    return {
        "label": profile["label"],
        "instance": instance,
        "hourly": hourly,
        "basis": "spot" if spot else "on demand",
        "seconds_per_field_on_4_vcpu": round(seconds_per_field, 3),
        "slides_per_hour_per_instance": round(3600 * jobs_at_once / slide_seconds, 1),
        "usd_per_1000_slides": round(instance_seconds_per_1000 / 3600 * hourly, 2),
        "kleidicv_in_build": profile.get("kleidicv_in_build"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="refresh prices from AWS first")
    ap.add_argument("--spot", action="store_true", help="price on Spot instead of on demand")
    ap.add_argument("--markdown", action="store_true", help="print the table for the report")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    if args.fetch:
        print(f"fetching prices for {args.region}")
        pricing = fetch_pricing(args.region)
    else:
        pricing = load_pricing()

    found = profiles()
    if not found:
        raise SystemExit(
            "No profiles in bench/results/. Run bench/field_profile.py --label <arm> "
            "on each machine first."
        )

    rows = [r for r in (cost_per_1000(p, pricing, args.spot) for p in found) if r]
    rows.sort(key=lambda r: r["usd_per_1000_slides"])

    warning = "" if pricing.get("verified") else "  UNVERIFIED"

    if not pricing.get("verified"):
        print("\n*** The prices in bench/pricing.json are placeholders. Run --fetch, "
              "or edit the file and set verified to true. ***\n")

    if args.markdown:
        print(f"| arm | instance | $/hour ({rows[0]['basis']}) | slides/hour | "
              f"$/1000 slides | KleidiCV in build |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['label']} | {r['instance']} | {r['hourly']:.4f} | "
                  f"{r['slides_per_hour_per_instance']} | "
                  f"**{r['usd_per_1000_slides']:.2f}**{warning} | "
                  f"{r['kleidicv_in_build']} |")
    else:
        print(f"{'arm':<20}{'instance':<14}{'$/1000 slides':>15}{'slides/h':>11}  build")
        for r in rows:
            print(f"{r['label']:<20}{r['instance']:<14}"
                  f"{r['usd_per_1000_slides']:>15.2f}{warning}"
                  f"{r['slides_per_hour_per_instance']:>11.1f}"
                  f"  kleidicv={r['kleidicv_in_build']}")

    if len(rows) > 1:
        best, worst = rows[0], rows[-1]
        saving = 1 - best["usd_per_1000_slides"] / worst["usd_per_1000_slides"]
        print(f"\n{best['label']} costs {saving:.0%} less than {worst['label']} "
              f"per 1,000 slides.")


if __name__ == "__main__":
    main()
