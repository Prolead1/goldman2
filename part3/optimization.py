"""Industry portfolio construction for the FE5108 class project.

The assignment compares three fully invested portfolios estimated from the
January 1995 to December 2004 industry-return covariance matrix:

* unconstrained global minimum variance;
* long-only global minimum variance; and
* inverse volatility.

Expected returns are deliberately absent from every method in this module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Phase1Data, load_phase1_data

UNCONSTRAINED = "unconstrained_minimum_variance"
LONG_ONLY = "long_only_minimum_variance"
INVERSE_VOLATILITY = "inverse_volatility"
PORTFOLIO_NAMES = (UNCONSTRAINED, LONG_ONLY, INVERSE_VOLATILITY)


class PortfolioConstructionError(ValueError):
    """Raised when portfolio inputs or an optimization result are invalid."""


@dataclass(frozen=True)
class Phase2Portfolios:
    """Estimated covariance matrix, portfolio weights, and audit diagnostics."""

    covariance: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def estimate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate the sample covariance matrix from a complete return panel."""
    if not isinstance(returns, pd.DataFrame):
        raise PortfolioConstructionError("Returns must be a pandas DataFrame.")
    if len(returns) < 2:
        raise PortfolioConstructionError("At least two return observations are required.")
    if returns.shape[1] < 1:
        raise PortfolioConstructionError("At least one return series is required.")
    if returns.columns.has_duplicates:
        raise PortfolioConstructionError("Return columns must be unique.")
    if returns.isna().any().any():
        raise PortfolioConstructionError("Returns must not contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes):
        raise PortfolioConstructionError("Every return series must be numeric.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioConstructionError("Returns must contain only finite values.")

    covariance = returns.cov(ddof=1)
    _covariance_array(covariance)
    return covariance


def _covariance_array(covariance: pd.DataFrame) -> np.ndarray:
    """Validate a labelled covariance matrix and return its numeric values."""
    if not isinstance(covariance, pd.DataFrame):
        raise PortfolioConstructionError("Covariance must be a pandas DataFrame.")
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise PortfolioConstructionError("Covariance matrix must be non-empty and square.")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise PortfolioConstructionError("Covariance labels must be unique.")
    if not covariance.index.equals(covariance.columns):
        raise PortfolioConstructionError("Covariance row and column labels must match in order.")

    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioConstructionError("Covariance matrix must contain only finite values.")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise PortfolioConstructionError("Covariance matrix must be symmetric.")
    if np.any(np.diag(values) <= 0.0):
        raise PortfolioConstructionError("Every asset must have positive variance.")

    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = max(float(np.max(np.diag(values))), 1.0) * 1e-10
    if float(eigenvalues.min()) < -tolerance:
        raise PortfolioConstructionError("Covariance matrix must be positive semidefinite.")
    return values


def unconstrained_minimum_variance(covariance: pd.DataFrame) -> pd.Series:
    """Closed-form fully invested global minimum-variance weights.

    The solution is ``Sigma^-1 1 / (1' Sigma^-1 1)``. Short positions are
    allowed, and no expected-return estimate enters the calculation.
    """
    values = _covariance_array(covariance)
    ones = np.ones(len(covariance), dtype=float)
    try:
        raw = np.linalg.solve(values, ones)
    except np.linalg.LinAlgError as exc:
        raise PortfolioConstructionError(
            "Covariance matrix is singular; unconstrained GMV weights are undefined."
        ) from exc
    denominator = float(ones @ raw)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise PortfolioConstructionError("Invalid normalization in unconstrained GMV solution.")
    weights = pd.Series(raw / denominator, index=covariance.index, name=UNCONSTRAINED)
    validate_weights(weights, covariance, long_only=False)
    return weights


def inverse_volatility_weights(covariance: pd.DataFrame) -> pd.Series:
    """Fully invested weights proportional to inverse individual volatility."""
    values = _covariance_array(covariance)
    inverse_volatility = 1.0 / np.sqrt(np.diag(values))
    weights = pd.Series(
        inverse_volatility / inverse_volatility.sum(),
        index=covariance.index,
        name=INVERSE_VOLATILITY,
    )
    validate_weights(weights, covariance, long_only=True)
    return weights


def long_only_minimum_variance(
    covariance: pd.DataFrame,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-13,
) -> pd.Series:
    """Solve long-only minimum variance by projected gradient descent.

    The objective is convex. Each step follows the negative variance gradient
    and projects the candidate weights back onto the probability simplex, which
    enforces ``w >= 0`` and ``sum(w) = 1`` simultaneously.
    """
    values = _covariance_array(covariance)
    weights = inverse_volatility_weights(covariance).to_numpy(dtype=float)
    largest_eigenvalue = float(np.linalg.eigvalsh(values).max())
    if largest_eigenvalue <= 0.0:
        raise PortfolioConstructionError("Covariance matrix has no positive eigenvalue.")
    step_size = 1.0 / (2.0 * largest_eigenvalue)

    for _ in range(max_iterations):
        gradient = 2.0 * values @ weights
        candidate = _project_to_simplex(weights - step_size * gradient)
        if float(np.max(np.abs(candidate - weights))) <= tolerance:
            weights = candidate
            break
        weights = candidate
    else:
        raise PortfolioConstructionError(
            "Long-only minimum-variance optimization did not converge."
        )

    cleaned = np.where(weights < 1e-12, 0.0, weights)
    cleaned = cleaned / cleaned.sum()
    result = pd.Series(cleaned, index=covariance.index, name=LONG_ONLY)
    validate_weights(result, covariance, long_only=True)
    _validate_long_only_kkt(result, covariance)
    return result


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{w: w >= 0, sum(w) = 1}``."""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    positions = np.arange(1, len(values) + 1)
    feasible = ordered - (cumulative - 1.0) / positions > 0.0
    if not feasible.any():
        raise PortfolioConstructionError("Simplex projection failed.")
    rho = int(np.flatnonzero(feasible)[-1])
    threshold = float((cumulative[rho] - 1.0) / (rho + 1))
    return np.maximum(values - threshold, 0.0)


def _validate_long_only_kkt(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Verify first-order optimality for the simplex-constrained solution."""
    values = _covariance_array(covariance)
    vector = weights.reindex(covariance.index).to_numpy(dtype=float)
    gradient = 2.0 * values @ vector
    active = vector > 1e-10
    if not active.any():
        raise PortfolioConstructionError("Long-only solution has no active holdings.")
    multiplier = float(np.mean(gradient[active]))
    active_residual = float(np.max(np.abs(gradient[active] - multiplier)))
    inactive_violation = (
        float(np.maximum(multiplier - gradient[~active], 0.0).max())
        if (~active).any()
        else 0.0
    )
    if active_residual > tolerance or inactive_violation > tolerance:
        raise PortfolioConstructionError(
            "Long-only solution failed optimality checks: "
            f"active residual={active_residual:.3g}, "
            f"inactive violation={inactive_violation:.3g}."
        )


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Return monthly portfolio variance after aligning weights by asset label."""
    values = _covariance_array(covariance)
    if not isinstance(weights, pd.Series):
        raise PortfolioConstructionError("Weights must be a pandas Series.")
    if set(weights.index) != set(covariance.index):
        raise PortfolioConstructionError("Weight and covariance asset labels must match.")
    vector = weights.reindex(covariance.index).to_numpy(dtype=float)
    if not np.isfinite(vector).all():
        raise PortfolioConstructionError("Weights must contain only finite values.")
    return float(vector @ values @ vector)


def validate_weights(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    long_only: bool,
    tolerance: float = 1e-8,
) -> None:
    """Require aligned, finite, fully invested weights and optional long-only bounds."""
    variance = portfolio_variance(weights, covariance)
    total = float(weights.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=tolerance):
        raise PortfolioConstructionError(f"Portfolio weights sum to {total}, not 1.")
    if long_only and float(weights.min()) < -tolerance:
        raise PortfolioConstructionError("Long-only portfolio contains a negative weight.")
    if variance < -tolerance:
        raise PortfolioConstructionError("Portfolio variance must not be negative.")


def weight_diagnostics(weights: pd.DataFrame, covariance: pd.DataFrame) -> pd.DataFrame:
    """Calculate concentration and variance checks for each portfolio column."""
    if tuple(weights.columns) != PORTFOLIO_NAMES:
        raise PortfolioConstructionError(
            f"Expected portfolio columns in this order: {PORTFOLIO_NAMES}."
        )
    rows: dict[str, dict[str, float | int]] = {}
    for name in weights.columns:
        vector = weights[name]
        variance = portfolio_variance(vector, covariance)
        rows[name] = {
            "weight_sum": float(vector.sum()),
            "minimum_weight": float(vector.min()),
            "maximum_weight": float(vector.max()),
            "gross_exposure": float(vector.abs().sum()),
            "nonzero_holdings": int((vector.abs() > 1e-10).sum()),
            "effective_holdings": float(1.0 / vector.pow(2).sum()),
            "monthly_variance": variance,
            "annualized_volatility": float(np.sqrt(variance * 12.0)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def build_phase2_portfolios(seed_returns: pd.DataFrame) -> Phase2Portfolios:
    """Estimate all three Part 3 portfolios from one seed-period return panel."""
    covariance = estimate_covariance(seed_returns)
    weights = pd.concat(
        [
            unconstrained_minimum_variance(covariance),
            long_only_minimum_variance(covariance),
            inverse_volatility_weights(covariance),
        ],
        axis="columns",
    )
    diagnostics = weight_diagnostics(weights, covariance)
    return Phase2Portfolios(covariance=covariance, weights=weights, diagnostics=diagnostics)


def build_phase2_from_workbook(workbook_path: str | Path | None = None) -> Phase2Portfolios:
    """Load Phase 1 and estimate the three portfolios using only ``seed_1``."""
    phase1: Phase1Data = load_phase1_data(workbook_path)
    return build_phase2_portfolios(phase1.windows["seed_1"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate FE5108 industry portfolios.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Path to the Part 3 workbook; defaults to the file beside code/.",
    )
    return parser


def main() -> None:
    """Estimate the three portfolios and print their audit diagnostics."""
    args = _build_parser().parse_args()
    phase2 = build_phase2_from_workbook(args.workbook)
    payload = {
        name: {
            key: int(value) if key == "nonzero_holdings" else float(value)
            for key, value in row.items()
        }
        for name, row in phase2.diagnostics.to_dict(orient="index").items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
