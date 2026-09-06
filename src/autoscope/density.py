"""Stage 6: turn parasite and white-cell counts into a density, with an error bar.

This module is the heart of the agent's stopping rule, so it is deliberately
small, pure and fully tested. No OpenCV in here.

WHO method: count parasites against 200 white blood cells, then convert using an
assumed white cell concentration (8000/uL is the WHO default when a real white
cell count is unavailable).
"""

from dataclasses import dataclass

from scipy.stats import chi2

WBC_PER_UL_DEFAULT = 8000.0
WHO_WBC_TARGET = 200


@dataclass(frozen=True)
class Density:
    """A parasite density with an exact Poisson interval around it."""

    parasites: int
    wbc_counted: int
    per_ul: float
    low_per_ul: float
    high_per_ul: float

    @property
    def ci_relative_width(self) -> float:
        """Width of the interval as a fraction of the estimate.

        This is what the agent watches. Returns inf when we have no estimate to
        speak of yet, so an empty count never looks precise.
        """
        if self.per_ul <= 0:
            return float("inf")
        return (self.high_per_ul - self.low_per_ul) / self.per_ul


def poisson_interval(count: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact Poisson confidence interval for an observed count.

    Uses the chi-squared relationship rather than a normal approximation, which
    matters here because low parasitaemia is exactly the case we care about and
    it is exactly where the normal approximation falls apart.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    alpha = 1.0 - confidence
    low = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2
    high = chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2
    return float(low), float(high)


def density(
    parasites: int,
    wbc_counted: int,
    wbc_per_ul: float = WBC_PER_UL_DEFAULT,
    confidence: float = 0.95,
) -> Density:
    """Parasites per microlitre, with the interval carried through."""
    if wbc_counted <= 0:
        raise ValueError("wbc_counted must be > 0")

    scale = wbc_per_ul / wbc_counted
    low, high = poisson_interval(parasites, confidence)
    return Density(
        parasites=parasites,
        wbc_counted=wbc_counted,
        per_ul=parasites * scale,
        low_per_ul=low * scale,
        high_per_ul=high * scale,
    )


def should_stop(
    d: Density,
    fields_read: int,
    target_ci_width: float = 0.20,
    min_fields: int = 10,
    max_fields: int = 400,
) -> tuple[bool, str]:
    """The sequential stopping rule. Returns the decision and why.

    Three ways to stop:

    - the interval is tight enough, so more looking will not change the answer;
    - we have read enough white cells to satisfy the WHO protocol on a slide
      that looks negative, which is the honest way to call a negative;
    - we hit the ceiling, which is a flag for a person rather than a result.

    The reason string goes straight into the agent log, so it has to read well
    to a human.
    """
    if fields_read < min_fields:
        return False, f"only {fields_read} fields read, minimum is {min_fields}"

    if d.parasites > 0 and d.ci_relative_width <= target_ci_width:
        return True, (
            f"{d.per_ul:,.0f}/uL, error bar {d.ci_relative_width:.0%} "
            f"is within the {target_ci_width:.0%} target"
        )

    if d.parasites == 0 and d.wbc_counted >= WHO_WBC_TARGET:
        return True, (
            f"no parasites in {d.wbc_counted} white cells, "
            f"which meets the WHO threshold for a negative"
        )

    if fields_read >= max_fields:
        return True, (
            f"stopped at the {max_fields} field ceiling with the error bar still at "
            f"{d.ci_relative_width:.0%}; this needs a person"
        )

    return False, (
        f"error bar {d.ci_relative_width:.0%} is wider than the "
        f"{target_ci_width:.0%} target, keep reading"
    )


# ---------------------------------------------------------------------------
# The decision rule, 5 September 2026.
#
# The rule above stops on a 20% error bar, which 35 of the 50 held-out slides
# can never reach, and calls a negative only at zero parasites in 200 white
# cells, which no clean slide reaches: the pipeline reports about ten false
# parasites per 200 white cells on a clean slide (results-negatives.md). A
# technician does not need a number to 20%; they need a decision: positive or
# negative, and if positive, roughly how much. So:
#
#   negative   enough white cells have gone by (NEGATIVE_WBC_MIN) and the
#              estimate sits at or below what a clean slide produces, the
#              noise floor, measured on the fit half of the uninfected set
#              and never on the frozen half;
#   positive   the lower end of the interval sits above that floor, so the
#              count is more than a clean slide's false alarms;
#   settled    a positive whose interval sits inside one density band, or is
#              within the 20% target, is done;
#   otherwise  keep reading, and at the end of the slide hand it to a person,
#              saying which of the three could not be settled.
#
# The floor is a property of the vision, not of the slide: it moves every time
# the classifier does, and eval/negative_eval.py re-measures it.
# ---------------------------------------------------------------------------

from bisect import bisect_right

BAND_EDGES = (1_000.0, 10_000.0, 100_000.0)
BAND_NAMES = ("under 1 000/uL", "1 000 to 10 000/uL", "10 000 to 100 000/uL", "over 100 000/uL")

# Noise floor: what a clean slide reads at through the pipeline, p95 over the
# fit half of the uninfected set, every field counted (negative_eval.py,
# eval/results/decision_rule.json). 150/uL with the chain-3 network of
# 6 September 2026 (the fit half reads median 15, p90 69, max 223/uL); it was
# 2 610/uL with the chain-2 network. Re-measured after every classifier change.
FLOOR_PER_UL_DEFAULT = 150.0
# White cells to see before a negative can be called. The WHO thick film
# negative is 100 fields, about 0.25 uL of blood (to check against the WHO
# manual before the report); a phone field here carries about 25 white cells.
NEGATIVE_WBC_MIN = 500
# How tight a positive's interval must be before a settled band is enough to
# stop. 1.0 means the band alone decides.
SETTLE_CI_WIDTH = 0.30


def band(per_ul: float) -> int:
    """Index into BAND_NAMES."""
    return bisect_right(BAND_EDGES, per_ul)


@dataclass(frozen=True)
class Decision:
    call: str            # "negative", "positive" or "undecided"
    band: int | None     # set when a positive's interval sits inside one band
    reason: str

    @property
    def band_name(self) -> str | None:
        return BAND_NAMES[self.band] if self.band is not None else None


def decide(d: Density, floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
           negative_wbc_min: int = NEGATIVE_WBC_MIN) -> Decision:
    """What the count says so far, against the noise floor."""
    if d.parasites > 0 and d.low_per_ul > floor_per_ul:
        lo, hi = band(d.low_per_ul), band(d.high_per_ul)
        settled = lo if lo == hi else None
        where = BAND_NAMES[lo] if settled is not None else f"between {BAND_NAMES[lo]} and {BAND_NAMES[hi]}"
        return Decision("positive", settled,
                        f"positive: {d.per_ul:,.0f}/uL ({d.low_per_ul:,.0f} to {d.high_per_ul:,.0f}), "
                        f"above the {floor_per_ul:,.0f}/uL floor, {where}")
    if d.wbc_counted >= negative_wbc_min and d.per_ul <= floor_per_ul:
        return Decision("negative", None,
                        f"negative: {d.parasites} parasites in {d.wbc_counted} white cells reads "
                        f"{d.per_ul:,.0f}/uL, within the {floor_per_ul:,.0f}/uL a clean slide produces")
    if d.wbc_counted < negative_wbc_min and d.per_ul <= floor_per_ul:
        why = f"{d.per_ul:,.0f}/uL is within the clean-slide floor but only {d.wbc_counted} of {negative_wbc_min} white cells seen"
    else:
        why = (f"{d.per_ul:,.0f}/uL with a lower bound of {d.low_per_ul:,.0f} does not clear the "
               f"{floor_per_ul:,.0f}/uL floor yet")
    return Decision("undecided", None, why)


def should_stop_decision(
    d: Density,
    fields_read: int,
    floor_per_ul: float = FLOOR_PER_UL_DEFAULT,
    negative_wbc_min: int = NEGATIVE_WBC_MIN,
    target_ci_width: float = 0.20,
    min_fields: int = 5,
    max_fields: int = 400,
    settle_ci_width: float = SETTLE_CI_WIDTH,
) -> tuple[bool, str, Decision]:
    """The decision-driven stopping rule. Returns stop, why, and the call so far.

    A positive stops when its interval sits inside one band AND is within
    `settle_ci_width` of the estimate, or when it is within `target_ci_width`
    regardless of band. Band alone stopped after five fields on every dense
    slide with a 37% error bar: the band was right and the number was not
    (agent_eval, 5 September 2026), and a clinic wants both.
    """
    if fields_read < min_fields:
        return False, f"only {fields_read} fields read, minimum is {min_fields}", Decision("undecided", None, "too early")
    dec = decide(d, floor_per_ul, negative_wbc_min)
    if dec.call == "negative":
        return True, dec.reason, dec
    if dec.call == "positive":
        if dec.band is not None and d.ci_relative_width <= settle_ci_width:
            return True, dec.reason + f"; error bar {d.ci_relative_width:.0%}", dec
        if d.ci_relative_width <= target_ci_width:
            return True, dec.reason + f"; error bar {d.ci_relative_width:.0%} within the {target_ci_width:.0%} target", dec
    if fields_read >= max_fields:
        return True, f"stopped at the {max_fields} field ceiling, {dec.reason}; this needs a person", dec
    return False, dec.reason + ", keep reading", dec
