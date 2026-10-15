"""The look, decide, act loop.

Step Functions runs this in production. Here it is a plain function so it can be
tested, replayed and stepped through without any cloud.

Limits live here, not in the policy, so no policy can talk its way past them.
"""

from autoscope.agent.policy import Policy
from autoscope.agent.tools import Tools
from autoscope.agent.trace import Trace


def _advance(tools) -> bool:
    """Move one field along, wrapping to the next row. False at the end.

    The stage clamps at its edges, so without this the session would keep
    re-reading the last field and call it progress.
    """
    stage = tools.stage
    cols, rows = stage.grid
    x, y = stage._x, stage._y  # noqa: SLF001
    if x + 1 < cols:
        tools.move_stage(1, 0)
        return True
    if y + 1 < rows:
        tools.move_stage(-x, 1)
        return True
    return False


def _last_look(tools, policy, trace, where: str, measured: dict, fallback: str) -> None:
    """The loop is about to leave the slide right after a count. Let a rule
    that can decide from the totals alone look at them once more first.

    The policy is consulted at the top of each turn, before the field it then
    asks to count. So when the count it asked for is the last one the slide has,
    or the one that hits the field ceiling, the totals it has never seen are the
    final totals. Escalating straight away threw that decision away: on the
    frozen uninfected half, 7 of the 10 clean slides sent to a person had crossed
    500 white cells on exactly that last field with the estimate under the noise
    floor. The rule would have called them negative. It was never asked.

    Only policies with `_stop` get the look (RuleBased, DecisionRule). It takes
    the totals and nothing else, so it adds no field, no count and no model call,
    and the Bedrock replay cache stays valid. A model policy has no `_stop` and
    escalates exactly as before.
    """
    stop_now = getattr(policy, "_stop", None)
    if stop_now is not None:
        stop, why = stop_now(tools)
        if stop:
            trace.record(where, "stop", measured, why)
            trace.stopped_because = why
            return
    tools.flag_for_human(fallback)
    trace.record(where, "flag_for_human()", measured, fallback)
    trace.stopped_because = fallback


def read_slide(
    tools: Tools,
    policy: Policy,
    slide: str = "unnamed",
    max_steps: int = 2000,
    max_fields: int = 400,
) -> Trace:
    trace = Trace(slide=slide)

    for _ in range(max_steps):
        frame = tools.frame
        quality = tools.assess_field()
        where = f"({frame.x},{frame.y}) z={frame.z}"
        measured = {
            "focus": round(quality.focus_score),
            "even": round(quality.illumination_uniformity, 2),
            "debris": round(quality.debris_ratio, 2),
        }

        action, args = policy.next_move(quality, tools)

        if action == "stop":
            trace.record(where, "stop", measured, args["reason"])
            trace.stopped_because = args["reason"]
            break

        if action == "refocus":
            tools.refocus(args["dz"])
            trace.record(where, f"refocus({args['dz']:+d})", measured, args["reason"])
            continue

        if action == "skip":
            moved = _advance(tools)
            trace.record(where, "move_stage()", measured,
                         f"{args['reason']}, field skipped and not counted")
            if not moved:
                reason = "ran out of slide while skipping unusable fields"
                tools.flag_for_human(reason)
                trace.stopped_because = reason
                break
            continue

        if action == "count_then_move":
            parasites, wbc = tools.count_field()
            trace.record(
                where, "count_field()",
                {**measured, "parasites": parasites, "wbc": wbc},
                f"counted, running total {tools.parasites} in {tools.wbc} white cells",
            )
            if tools.fields_read >= max_fields:
                _last_look(tools, policy, trace, where, measured,
                           f"hit the {max_fields} field ceiling, handing to a person")
                break
            if not _advance(tools):
                _last_look(tools, policy, trace, where, measured,
                           "ran out of slide before the error bar was tight enough")
                break
            continue

        raise ValueError(f"policy {policy.name} asked for an unknown action: {action}")
    else:
        trace.stopped_because = f"hit the {max_steps} step ceiling"

    trace.fields_read = tools.fields_read
    trace.parasites = tools.parasites
    trace.wbc = tools.wbc
    return trace
