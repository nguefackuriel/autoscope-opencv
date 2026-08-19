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
