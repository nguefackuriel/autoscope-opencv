"""The session log.

The rules ask for a trace showing an OpenCV 5 output changing a later decision.
This module is what produces it, so the format is the deliverable and not an
afterthought. Every step records what was measured, what was decided, and why.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    n: int
    where: str
    action: str
    measured: dict
    reason: str

    def as_line(self) -> str:
        """One readable line. This is what a judge reads on the demo page."""
        numbers = "  ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in self.measured.items())
        return f"{self.n:>3}  {self.where:<22} {self.action:<26} {numbers}\n     {self.reason}"


@dataclass
class Trace:
    slide: str
    steps: list[Step] = field(default_factory=list)
    fields_read: int = 0
    parasites: int = 0
    wbc: int = 0
    stopped_because: str = ""

    def record(self, where: str, action: str, measured: dict, reason: str) -> Step:
        step = Step(len(self.steps) + 1, where, action, measured, reason)
        self.steps.append(step)
        return step

    def render(self) -> str:
        head = f"slide: {self.slide}"
        body = "\n".join(s.as_line() for s in self.steps)
        tail = (
            f"\nread {self.fields_read} fields, "
            f"{self.parasites} parasites in {self.wbc} white cells\n"
            f"stopped: {self.stopped_because}"
        )
        return f"{head}\n{body}{tail}"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    def refocus_events(self) -> list[Step]:
        """The steps that prove the point: a measurement forced a new capture."""
        return [s for s in self.steps if s.action.startswith("refocus")]
