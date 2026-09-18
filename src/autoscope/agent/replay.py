"""Record the agent's Bedrock calls once, replay them for ever after.

A full agent-versus-rule comparison is one model call per loop step: about
1 200 calls over the 50 held-out patients, roughly $9. Paying that every time
somebody wants to check the comparison is bad for us and impossible for a
judge, who has no AWS account and should not need one to reproduce a number we
publish.

So the first run records every call to a `.jsonl` file that goes in the repo,
and every run after that serves the same responses from disk. The evaluation
then costs nothing and runs anywhere, and the recorded file is the evidence:
what the model saw and what it answered, call by call.

    # once, with credentials
    client = RecordingClient(AnthropicBedrockMantle(aws_region="us-east-1"), path)
    # for ever after, with none
    client = ReplayClient(path)

Both are drop-in for `BedrockPolicy(client=...)`: they expose `.messages.create`
and nothing else, which is the whole of the surface the policy uses and the
same seam the tests already fake.

TWO DECISIONS THAT LOOK LIKE DETAILS AND ARE NOT

**The key covers the prompt, not just the observation.** If it hashed only what
the model was asked about, then editing `moves.py` — the system prompt or the
tool definitions — would leave every cached answer looking valid, and the
evaluation would replay decisions the current prompt never produced. That is
the failure mode most likely to put a confident wrong number in the report, and
it would leave no trace. Changing the prompt now changes every key, the replay
misses, and the miss is loud.

**A miss during replay raises.** Quietly falling through to a live call would
mean a run advertised as replayed is partly live: it spends money nobody asked
to spend, and a judge re-running it would get a different answer from a cache
that looked complete. `allow_live` is the explicit opt-in, and a run that used
it says so.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cache_key(model: str, effort: str, system: str, tools: Any, observation: Any) -> str:
    """A stable hash of everything that could change the answer.

    `sort_keys` because a dict that serialises in a different order is the same
    question, and a key that moved with dict ordering would miss for no reason.
    """
    payload = json.dumps(
        {
            "model": model,
            "effort": effort,
            "system": system,
            "tools": tools,
            "observation": observation,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def key_for_request(kwargs: dict[str, Any]) -> str:
    """The key for one `messages.create(**kwargs)` call.

    The observation is the user message the policy built; it is already a JSON
    string, so it is hashed as sent rather than re-parsed and re-serialised.
    """
    messages = kwargs.get("messages") or [{}]
    return cache_key(
        model=kwargs.get("model", ""),
        effort=(kwargs.get("output_config") or {}).get("effort", ""),
        system=kwargs.get("system", ""),
        tools=kwargs.get("tools", []),
        observation=messages[0].get("content", ""),
    )


class CachedBlock:
    """A `tool_use` block rebuilt from disk, shaped like the SDK's."""

    type = "tool_use"

    def __init__(self, name: str, input_: dict) -> None:
        self.name = name
        self.input = input_


class CachedUsage:
    def __init__(self, row: dict) -> None:
        self.input_tokens = row.get("input_tokens", 0)
        self.output_tokens = row.get("output_tokens", 0)
        self.cache_read_input_tokens = row.get("cache_read_input_tokens", 0)


class CachedResponse:
    """What `messages.create` returns, rebuilt from a recorded row.

    Only the parts `BedrockPolicy._move_from` reads: the content blocks, the
    stop reason and the usage. Rebuilding the whole SDK object would be
    pretending to more fidelity than the file holds.
    """

    def __init__(self, row: dict) -> None:
        self.content = [CachedBlock(row["move"]["name"], row["move"]["input"])]
        self.stop_reason = row.get("stop_reason", "tool_use")
        self.usage = CachedUsage(row.get("usage", {}))


def _row_from(response: Any) -> dict:
    """The recordable part of a live response."""
    block = next((b for b in response.content if b.type == "tool_use"), None)
    usage = getattr(response, "usage", None)
    return {
        "move": {"name": block.name, "input": dict(block.input)} if block else None,
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        },
    }


class CacheMiss(RuntimeError):
    """Asked to replay a call that was never recorded."""


class _Messages:
    """The `.messages` attribute both clients expose."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)  # noqa: SLF001


class RecordingClient:
    """Calls the real client and appends every answer to `path`.

    One JSON object per line, flushed per call, so a run that dies after 900
    calls has 900 usable calls on disk and resuming costs only the rest.
    """

    def __init__(self, inner: Any, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.messages = _Messages(self)
        self.recorded = 0
        self.served_from_cache = 0
        # Anything already on disk is reused rather than bought twice.
        self._seen: dict[str, dict] = _load(self.path)

    def _create(self, **kwargs: Any) -> Any:
        key = key_for_request(kwargs)
        if key in self._seen:
            self.served_from_cache += 1
            return CachedResponse(self._seen[key])

        response = self.inner.messages.create(**kwargs)
        row = {
            "key": key,
            "observation": json.loads(kwargs["messages"][0]["content"]),
            "model": kwargs.get("model"),
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_row_from(response),
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._seen[key] = row
        self.recorded += 1
        return response


class ReplayClient:
    """Serves recorded answers. Raises on anything it has not seen.

    `allow_live` takes a real client and turns a miss into a live call, which
    is how a cache gets extended after the prompt changes. It is deliberately
    not the default: see the note at the top of this module.
    """

    def __init__(self, path: str | Path, allow_live: Any | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist. Record it first with --record "
                "(needs Bedrock access), or point --bedrock-cache at the file."
            )
        self.rows = _load(self.path)
        self.allow_live = allow_live
        self.messages = _Messages(self)
        self.hits = 0
        self.live = 0

    def _create(self, **kwargs: Any) -> Any:
        key = key_for_request(kwargs)
        row = self.rows.get(key)
        if row is not None:
            self.hits += 1
            return CachedResponse(row)

        if self.allow_live is None:
            raise CacheMiss(
                f"no recorded call for key {key} in {self.path.name} "
                f"({len(self.rows)} recorded). The prompt or the tool definitions "
                "have probably changed since the recording, which invalidates "
                "every key on purpose. Re-record with --record, or pass "
                "--allow-live to extend the cache."
            )

        self.live += 1
        return self.allow_live.messages.create(**kwargs)


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # A later recording of the same key wins: re-recording one call should
        # not mean editing the file by hand.
        rows[row["key"]] = row
    return rows


def cost_of(calls: list[dict], usd_per_mtok_in: float = 5.0,
            usd_per_mtok_out: float = 25.0) -> dict:
    """What a run actually cost, from the token counts the calls carry.

    Cached input is billed at a tenth, so it is counted separately rather than
    folded in; if prompt caching turns out not to fire, that column reads zero
    and the total is the honest one.
    """
    fresh = sum(c["usage"].get("input_tokens", 0) or 0 for c in calls)
    cached = sum(c["usage"].get("cache_read_input_tokens", 0) or 0 for c in calls)
    out = sum(c["usage"].get("output_tokens", 0) or 0 for c in calls)
    usd = (fresh / 1e6 * usd_per_mtok_in
           + cached / 1e6 * usd_per_mtok_in * 0.1
           + out / 1e6 * usd_per_mtok_out)
    return {
        "calls": len(calls),
        "input_tokens": fresh,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "usd_estimated": round(usd, 4),
    }
