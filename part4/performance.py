"""Part 4D-F multi-day performance, compounding example and suitability analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceAnalysisError(ValueError):
    """Raised when a return panel cannot support compounding analysis."""


def _validate_return_panel(daily_returns: pd.DataFrame) -> None:
    if daily_returns.empty or daily_returns.isna().any().any():
        raise PerformanceAnalysisError("Daily returns must be complete and non-empty.")
    values = daily_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PerformanceAnalysisError("Daily returns must be finite.")
    if (values <= -1.0).any():
        raise PerformanceAnalysisError("Daily returns must be greater than -100%.")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise PerformanceAnalysisError("Daily returns must use a DatetimeIndex.")


def _comparison_columns(compounded: pd.DataFrame) -> pd.DataFrame:
    result = compounded.loc[:, ["NVDA", "NVDL"]].copy()
    result["Twice NVDA cumulative return"] = 2.0 * result["NVDA"]
    result["NVDL minus twice NVDA"] = (
        result["NVDL"] - result["Twice NVDA cumulative return"]
    )
    return result


def monthly_total_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns into calendar-month total returns."""
    _validate_return_panel(daily_returns)
    monthly = (1.0 + daily_returns).groupby(daily_returns.index.to_period("M")).prod() - 1.0
    monthly.index.name = "Month"
    return _comparison_columns(monthly)


def full_period_total_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound returns and decompose daily-reset compounding and observed drag."""
    _validate_return_panel(daily_returns)
    compounded = pd.DataFrame(
        [(1.0 + daily_returns).prod(axis="index") - 1.0],
        index=pd.Index(["Feb-Aug 2024"], name="Period"),
    )
    result = _comparison_columns(compounded)
    ideal_daily_2x = (1.0 + 2.0 * daily_returns["NVDA"]).prod() - 1.0
    result["Ideal daily 2x NVDA"] = ideal_daily_2x
    result["Compounding effect vs twice NVDA"] = (
        result["Ideal daily 2x NVDA"] - result["Twice NVDA cumulative return"]
    )
    result["NVDL minus ideal daily 2x"] = (
        result["NVDL"] - result["Ideal daily 2x NVDA"]
    )
    return result


def two_day_compounding_example(
    nvda_day_one: float = 0.10,
    nvda_day_two: float = -0.09,
    leverage: float = 2.0,
) -> pd.DataFrame:
    """Calculate the assignment's two-day NVDA and idealized NVDL example."""
    nvdl_day_one = leverage * nvda_day_one
    nvdl_day_two = leverage * nvda_day_two
    nvda_total = (1.0 + nvda_day_one) * (1.0 + nvda_day_two) - 1.0
    nvdl_total = (1.0 + nvdl_day_one) * (1.0 + nvdl_day_two) - 1.0
    return pd.DataFrame(
        {
            "Day 1 return": [nvda_day_one, nvdl_day_one],
            "Day 2 return": [nvda_day_two, nvdl_day_two],
            "Two-day return": [nvda_total, nvdl_total],
        },
        index=pd.Index(["NVDA", "NVDL (ideal 2x, no drag)"], name="Investment"),
    )


def investor_suitability() -> pd.DataFrame:
    """Summarize the Part 4F instrument choice for the two stated use cases."""
    return pd.DataFrame(
        {
            "Preferred instrument": ["NVDL", "NVDA"],
            "Reason": [
                "A one-day catalyst matches NVDL's daily 2x objective, if the trader accepts "
                "roughly doubled downside risk and close monitoring.",
                "NVDA avoids daily leverage resets, volatility drag, financing costs and the "
                "higher probability of large losses over a volatile holding period.",
            ],
        },
        index=pd.Index(
            ["One-day bullish event trade", "One-year bullish investment"],
            name="Use case",
        ),
    )
