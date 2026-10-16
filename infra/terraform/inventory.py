"""What is actually deployed, read out of the Terraform state.

The state file holds credentials and is gitignored, so the report cannot read
it: a judge cloning the repository would get a build error instead of a report.
This reads the state and writes `deployed.json`, which carries counts and a
handful of non-secret settings and is committed. Regenerate it after any
`terraform apply`:

    python infra/terraform/inventory.py

Nothing here reads a secret. The fields are resource type counts and the four
numbers the report quotes: log retention, the alarm threshold and the two
budgets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def inventory(state_path: Path | None = None) -> dict:
    state = json.loads((state_path or HERE / "terraform.tfstate").read_text())
    managed = [r for r in state.get("resources", []) if r.get("mode") == "managed"]
    if not managed:
        raise SystemExit("no managed resources in the state; has terraform applied?")

    kinds: Counter[str] = Counter()
    out: dict = {"region": None, "budgets": {}}
    for r in managed:
        kinds[r["type"]] += len(r.get("instances", []))
        for inst in r.get("instances", []):
            a = inst.get("attributes", {})
            if r["type"] == "aws_budgets_budget":
                name = "bedrock" if "bedrock" in a.get("name", "") else "monthly"
                out["budgets"][name] = int(float(a["limit_amount"]))
            elif r["type"] == "aws_cloudwatch_metric_alarm":
                out["alarm_sessions_per_5_min"] = int(float(a["threshold"]))
            elif r["type"] == "aws_cloudwatch_log_group":
                out["log_retention_days"] = int(a.get("retention_in_days") or 0)
            elif r["type"] == "aws_s3_bucket" and not out["region"]:
                out["region"] = a.get("region")

    out["resources"] = sum(kinds.values())
    out["by_type"] = dict(sorted(kinds.items()))
    return out


def main() -> None:
    out = inventory()
    path = HERE / "deployed.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{out['resources']} resources, {len(out['by_type'])} types -> {path}")


if __name__ == "__main__":
    main()
