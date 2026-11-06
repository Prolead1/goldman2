"""Part 4B-C regression of NVDL daily returns on NVDA daily returns."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


class RegressionError(ValueError):
    """Raised when a regression input or result is invalid."""


@dataclass(frozen=True)
class RegressionResults:
    """OLS coefficient table and compact fit diagnostics."""

    coefficients: pd.DataFrame
    observations: int
    residual_degrees_of_freedom: int
    r_squared: float
    residual_standard_error: float
    annualized_intercept: float


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Approximate the two-sided 95% Student-t critical value.

    The Cornish-Fisher expansion is highly accurate for this assignment's 145
    residual degrees of freedom and avoids coupling the core calculation to a
    statistical package.
    """
    if degrees_of_freedom < 30:
        raise RegressionError("At least 30 residual degrees of freedom are required.")
    z = NormalDist().inv_cdf(0.975)
    v = float(degrees_of_freedom)
    return float(
        z
        + (z**3 + z) / (4.0 * v)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * v**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * v**3)
    )


def fit_daily_leverage_regression(
    nvda_returns: pd.Series,
    nvdl_returns: pd.Series,
) -> RegressionResults:
    """Fit ``NVDL = alpha + beta * NVDA + epsilon`` with 95% OLS intervals."""
    aligned = pd.concat(
        [nvda_returns.rename("NVDA"), nvdl_returns.rename("NVDL")], axis="columns"
    ).dropna()
    if len(aligned) < 32:
        raise RegressionError("At least 32 paired observations are required.")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RegressionError("Regression returns must be finite.")

    x = aligned["NVDA"].to_numpy(dtype=float)
    y = aligned["NVDL"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    estimates, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        raise RegressionError("The regression design matrix is rank deficient.")

    fitted = design @ estimates
    residuals = y - fitted
    dof = len(y) - design.shape[1]
    residual_sum_squares = float(residuals @ residuals)
    residual_variance = residual_sum_squares / dof
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = estimates / standard_errors
    critical = _student_t_critical_975(dof)
    lower = estimates - critical * standard_errors
    upper = estimates + critical * standard_errors

    total_sum_squares = float((y - y.mean()) @ (y - y.mean()))
    if total_sum_squares <= 0.0:
        raise RegressionError("NVDL returns must have positive variation.")
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    coefficients = pd.DataFrame(
        {
            "Estimate": estimates,
            "Standard error": standard_errors,
            "t-statistic": t_statistics,
            "95% CI lower": lower,
            "95% CI upper": upper,
        },
        index=pd.Index(["Intercept", "NVDA beta"], name="Coefficient"),
    )
    return RegressionResults(
        coefficients=coefficients,
        observations=len(y),
        residual_degrees_of_freedom=dof,
        r_squared=float(r_squared),
        residual_standard_error=float(np.sqrt(residual_variance)),
        annualized_intercept=float(estimates[0] * TRADING_DAYS_PER_YEAR),
    )
