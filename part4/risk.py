"""Part 4A daily volatility, historical VaR and historical CVaR."""

from __future__ import annotations

import pandas as pd


class RiskMetricError(ValueError):
    """Raised when returns cannot support the requested downside-risk metrics."""


def historical_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    """Return positive-loss VaR/CVaR, the return cutoff and tail observation count.

    VaR is the negative of the empirical lower-tail quantile. CVaR is the
    negative mean return among observations at or below that cutoff.
    """
    if not 0.0 < confidence < 1.0:
        raise RiskMetricError("Confidence must be strictly between zero and one.")
    if returns.empty or returns.isna().any():
        raise RiskMetricError("Returns must be complete and non-empty.")
    if len(returns) < 2:
        raise RiskMetricError("At least two return observations are required.")

    cutoff = float(returns.quantile(1.0 - confidence))
    tail = returns.loc[returns <= cutoff]
    if tail.empty:
        raise RiskMetricError("The empirical tail contains no observations.")
    return -cutoff, -float(tail.mean()), cutoff, len(tail)


def build_risk_table(
    daily_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate the unannualized Part 4A metrics for NVDA and NVDL."""
    rows: list[dict[str, float | int | str]] = []
    for asset in daily_returns.columns:
        var, cvar, cutoff, tail_count = historical_var_cvar(
            daily_returns[asset], confidence
        )
        rows.append(
            {
                "Asset": asset,
                "Daily standard deviation": float(daily_returns[asset].std(ddof=1)),
                "95% VaR": var,
                "95% CVaR": cvar,
                "5% return cutoff": cutoff,
                "Tail observations": tail_count,
            }
        )
    result = pd.DataFrame(rows).set_index("Asset")
    result.index.name = "Asset"
    return result
