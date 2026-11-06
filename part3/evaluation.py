"""Evaluate the realized volatility of the constructed industry portfolios."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Phase1Data, WORKBOOK_NAME, load_phase1_data
from .optimization import (
    LONG_ONLY,
    Phase2Portfolios,
    build_phase2_portfolios,
    estimate_covariance,
    long_only_minimum_variance,
)
from .reporting import DISPLAY_NAMES

ORIGINAL_LONG_ONLY = "Original long-only (estimated 12/31/2004)"
UPDATED_LONG_ONLY = "Updated long-only (estimated 12/31/2015)"


class PortfolioEvaluationError(ValueError):
    """Raised when returns and portfolio weights cannot be evaluated safely."""


@dataclass(frozen=True)
class Phase3BResults:
    """Part 3B monthly portfolio returns and in-sample volatility comparison."""

    monthly_returns: pd.DataFrame
    comparison: pd.DataFrame


@dataclass(frozen=True)
class Phase3CResults:
    """Part 3C monthly portfolio returns and out-of-sample volatility comparison."""

    monthly_returns: pd.DataFrame
    comparison: pd.DataFrame


@dataclass(frozen=True)
class Phase3DResults:
    """Part 3D re-optimized weights and realized-volatility comparison."""

    weights: pd.DataFrame
    monthly_returns: pd.DataFrame
    comparison: pd.DataFrame
    turnover: float


def portfolio_return_series(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Apply fixed portfolio weights to a monthly asset-return panel."""
    if returns.empty or weights.empty:
        raise PortfolioEvaluationError("Returns and weights must not be empty.")
    if returns.columns.has_duplicates or weights.index.has_duplicates:
        raise PortfolioEvaluationError("Asset labels must be unique.")
    if set(returns.columns) != set(weights.index):
        raise PortfolioEvaluationError("Return columns and weight rows must contain the same assets.")
    if returns.isna().any().any() or weights.isna().any().any():
        raise PortfolioEvaluationError("Returns and weights must not contain missing values.")

    return returns @ weights.reindex(returns.columns)


def annualized_realized_volatility(
    monthly_returns: pd.DataFrame,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualize sample standard deviation from monthly portfolio returns."""
    if periods_per_year < 1:
        raise PortfolioEvaluationError("Periods per year must be positive.")
    if len(monthly_returns) < 2:
        raise PortfolioEvaluationError("At least two monthly observations are required.")
    volatility = monthly_returns.std(axis="index", ddof=1) * np.sqrt(periods_per_year)
    volatility.name = "annualized_volatility"
    return volatility


def build_phase3b_results(
    seed_returns: pd.DataFrame,
    portfolios: Phase2Portfolios,
) -> Phase3BResults:
    """Evaluate the three fixed portfolios over their 1995-2004 seed period."""
    monthly_returns = portfolio_return_series(seed_returns, portfolios.weights)
    volatility = annualized_realized_volatility(monthly_returns)
    best = float(volatility.min())
    comparison = pd.DataFrame(
        {
            "annualized_volatility": volatility,
            "rank_low_to_high": volatility.rank(method="first", ascending=True).astype(int),
            "difference_from_lowest_bps": (volatility - best) * 10_000.0,
        }
    ).sort_values("rank_low_to_high")
    comparison.index = comparison.index.map(DISPLAY_NAMES)
    comparison.index.name = "Portfolio"
    monthly_returns = monthly_returns.rename(columns=DISPLAY_NAMES)
    return Phase3BResults(monthly_returns=monthly_returns, comparison=comparison)


def build_phase3c_results(
    out_of_sample_returns: pd.DataFrame,
    portfolios: Phase2Portfolios,
) -> Phase3CResults:
    """Evaluate the fixed 12/31/2004 portfolios over January 2005-July 2023."""
    monthly_returns = portfolio_return_series(out_of_sample_returns, portfolios.weights)
    volatility = annualized_realized_volatility(monthly_returns)
    best = float(volatility.min())
    comparison = pd.DataFrame(
        {
            "annualized_volatility": volatility,
            "rank_low_to_high": volatility.rank(method="first", ascending=True).astype(int),
            "difference_from_lowest_bps": (volatility - best) * 10_000.0,
        }
    ).sort_values("rank_low_to_high")
    comparison.index = comparison.index.map(DISPLAY_NAMES)
    comparison.index.name = "Portfolio"
    monthly_returns = monthly_returns.rename(columns=DISPLAY_NAMES)
    return Phase3CResults(monthly_returns=monthly_returns, comparison=comparison)


def build_phase3b_from_workbook(
    workbook_path: str | Path | None = None,
) -> Phase3BResults:
    """Load the supplied data, reconstruct portfolios, and run Part 3B."""
    phase1: Phase1Data = load_phase1_data(workbook_path)
    portfolios = build_phase2_portfolios(phase1.windows["seed_1"])
    return build_phase3b_results(phase1.windows["seed_1"], portfolios)


def build_phase3c_from_workbook(
    workbook_path: str | Path | None = None,
) -> Phase3CResults:
    """Load the supplied data, reconstruct 12/31/2004 portfolios, and run Part 3C."""
    phase1: Phase1Data = load_phase1_data(workbook_path)
    portfolios = build_phase2_portfolios(phase1.windows["seed_1"])
    return build_phase3c_results(phase1.windows["oos_1"], portfolios)


def build_phase3d_results(
    original_seed_returns: pd.DataFrame,
    updated_seed_returns: pd.DataFrame,
    evaluation_returns: pd.DataFrame,
) -> Phase3DResults:
    """Re-optimize long-only weights and compare both vintages over 2016-2023."""
    original = build_phase2_portfolios(original_seed_returns).weights[LONG_ONLY].rename(
        ORIGINAL_LONG_ONLY
    )
    updated_covariance = estimate_covariance(updated_seed_returns)
    updated = long_only_minimum_variance(updated_covariance).rename(UPDATED_LONG_ONLY)
    weights = pd.concat([original, updated], axis="columns")

    monthly_returns = portfolio_return_series(evaluation_returns, weights)
    volatility = annualized_realized_volatility(monthly_returns)
    original_volatility = float(volatility.loc[ORIGINAL_LONG_ONLY])
    comparison = pd.DataFrame(
        {
            "estimation_window": ["1995-01 to 2004-12", "2006-01 to 2015-12"],
            "annualized_volatility": [
                original_volatility,
                float(volatility.loc[UPDATED_LONG_ONLY]),
            ],
        },
        index=pd.Index([ORIGINAL_LONG_ONLY, UPDATED_LONG_ONLY], name="Portfolio"),
    )
    comparison["change_vs_original_bps"] = (
        comparison["annualized_volatility"] - original_volatility
    ) * 10_000.0
    comparison["relative_change"] = (
        comparison["annualized_volatility"] / original_volatility - 1.0
    )
    comparison["nonzero_holdings"] = (weights.abs() > 1e-10).sum(axis="index").astype(int)
    turnover = float(0.5 * (updated - original).abs().sum())
    return Phase3DResults(
        weights=weights,
        monthly_returns=monthly_returns,
        comparison=comparison,
        turnover=turnover,
    )


def build_phase3d_from_workbook(
    workbook_path: str | Path | None = None,
) -> Phase3DResults:
    """Load the supplied data and run the Part 3D re-optimization test."""
    phase1: Phase1Data = load_phase1_data(workbook_path)
    return build_phase3d_results(
        phase1.windows["seed_1"],
        phase1.windows["seed_2"],
        phase1.windows["reopt_test"],
    )


def interpretation_markdown(results: Phase3BResults) -> str:
    """Create report-ready interpretation grounded in the calculated ranking."""
    ordered = results.comparison
    lowest = ordered.index[0]
    middle = ordered.index[1]
    highest = ordered.index[2]
    values = ordered["annualized_volatility"]
    return f"""# Part 3B - In-sample effectiveness

Using the fixed portfolio weights estimated from January 1995 through December
2004, the {lowest} portfolio has the lowest annualized realized volatility at
{values.loc[lowest]:.2%}. The {middle} portfolio is next at {values.loc[middle]:.2%},
while the {highest} portfolio has the highest volatility at
{values.loc[highest]:.2%}.

This order is expected in sample. The unconstrained minimum-variance portfolio
is optimized over the broadest feasible set, including short positions, so it
cannot have higher variance than the long-only optimum when both use the same
covariance matrix. The long-only constraint removes some risk-reducing short
positions and therefore raises the minimum attainable variance. Inverse
volatility is a heuristic: it scales holdings by individual volatility but does
not optimize their correlations, so it is not guaranteed to minimize total
portfolio risk.

The comparison is in-sample and therefore gives the optimizer the strongest
possible setting. It does not establish that the unconstrained solution will
remain best out of sample, where covariance-estimation error and unstable short
positions can materially change the ordering.
"""


def build_reporting_payload(results: Phase3BResults) -> dict[str, object]:
    """Create a JSON-safe payload for the Part 3B exhibit."""
    rows = []
    for portfolio, row in results.comparison.iterrows():
        rows.append(
            {
                "portfolio": portfolio,
                "annualized_volatility": float(row["annualized_volatility"]),
                "rank_low_to_high": int(row["rank_low_to_high"]),
                "difference_from_lowest_bps": float(row["difference_from_lowest_bps"]),
            }
        )
    return {
        "title": "Part 3B - In-sample portfolio volatility",
        "subtitle": "Annualized realized volatility, January 1995-December 2004",
        "source": f"Source: {WORKBOOK_NAME}, 49_Industry_Portfolios worksheet",
        "rows": rows,
        "lowest": rows[0]["portfolio"],
        "highest": rows[-1]["portfolio"],
    }


def phase3c_interpretation_markdown(results: Phase3CResults) -> str:
    """Create report-ready Part 3C interpretation from the realized ranking."""
    ordered = results.comparison
    lowest = ordered.index[0]
    middle = ordered.index[1]
    highest = ordered.index[2]
    values = ordered["annualized_volatility"]
    return f"""# Part 3C - Out-of-sample effectiveness

The portfolios retain their 12/31/2004 weights throughout January 2005 through
July 2023. The {lowest} portfolio has the lowest annualized realized volatility
at {values.loc[lowest]:.2%}. The {middle} portfolio is next at
{values.loc[middle]:.2%}, while the {highest} portfolio has the highest volatility
at {values.loc[highest]:.2%}.

Unlike the in-sample comparison, this ordering is not guaranteed by the original
optimization. The covariance matrix was estimated only through December 2004,
whereas industry volatilities and correlations can change over the following
18 years. The unconstrained solution is especially sensitive because its long
and short positions magnify covariance-estimation error. The long-only constraint
acts as regularization by limiting extreme offsetting positions, while inverse
volatility is more diversified but still ignores cross-industry correlations.

The out-of-sample results therefore provide the more meaningful test of whether
the estimated low-volatility structure persisted beyond the seed period.
"""


def build_phase3c_reporting_payload(results: Phase3CResults) -> dict[str, object]:
    """Create a JSON-safe payload for the Part 3C exhibit."""
    rows = []
    for portfolio, row in results.comparison.iterrows():
        rows.append(
            {
                "portfolio": portfolio,
                "annualized_volatility": float(row["annualized_volatility"]),
                "rank_low_to_high": int(row["rank_low_to_high"]),
                "difference_from_lowest_bps": float(row["difference_from_lowest_bps"]),
            }
        )
    return {
        "title": "Part 3C - Out-of-sample portfolio volatility",
        "subtitle": "Fixed 12/31/2004 weights; realized volatility, January 2005-July 2023",
        "source": f"Source: {WORKBOOK_NAME}, 49_Industry_Portfolios worksheet",
        "rows": rows,
        "lowest": rows[0]["portfolio"],
        "highest": rows[-1]["portfolio"],
    }


def phase3d_interpretation_markdown(results: Phase3DResults) -> str:
    """Create report-ready interpretation of the re-optimization result."""
    comparison = results.comparison
    original = float(comparison.loc[ORIGINAL_LONG_ONLY, "annualized_volatility"])
    updated = float(comparison.loc[UPDATED_LONG_ONLY, "annualized_volatility"])
    change_bps = float(comparison.loc[UPDATED_LONG_ONLY, "change_vs_original_bps"])
    relative = float(comparison.loc[UPDATED_LONG_ONLY, "relative_change"])
    if updated < original:
        direction = "lower"
        result_sentence = (
            f"Re-optimization reduced realized volatility by {abs(change_bps):.1f} basis "
            f"points ({abs(relative):.1%} relative)."
        )
    elif updated > original:
        direction = "higher"
        result_sentence = (
            f"Re-optimization increased realized volatility by {abs(change_bps):.1f} basis "
            f"points ({abs(relative):.1%} relative)."
        )
    else:
        direction = "unchanged"
        result_sentence = "Re-optimization left realized volatility unchanged."

    return f"""# Part 3D - Value of re-optimization

Over January 2016 through July 2023, the original 12/31/2004 long-only
minimum-variance portfolio realized {original:.2%} annualized volatility. The
portfolio re-estimated as of 12/31/2015 realized {updated:.2%}, which is
{direction}. {result_sentence}

The updated portfolio uses the most recent ten-year covariance window available
at the rebalance date, so improvement is plausible when industry risks and
correlations have changed since 2004. It is not guaranteed: covariance estimates
remain noisy, and a newer sample can overweight temporary relationships that do
not persist. The two portfolios differ substantially, with one-way turnover of
{results.turnover:.1%}, showing that the estimated risk structure changed enough
to produce a meaningful rebalance.
"""


def build_phase3d_reporting_payload(results: Phase3DResults) -> dict[str, object]:
    """Create a JSON-safe payload for the Part 3D exhibit."""
    rows = []
    for portfolio, row in results.comparison.iterrows():
        rows.append(
            {
                "portfolio": portfolio,
                "estimation_window": row["estimation_window"],
                "annualized_volatility": float(row["annualized_volatility"]),
                "change_vs_original_bps": float(row["change_vs_original_bps"]),
                "relative_change": float(row["relative_change"]),
                "nonzero_holdings": int(row["nonzero_holdings"]),
            }
        )
    weights = [
        {
            "industry": industry,
            "original_weight": float(row[ORIGINAL_LONG_ONLY]),
            "updated_weight": float(row[UPDATED_LONG_ONLY]),
            "change": float(row[UPDATED_LONG_ONLY] - row[ORIGINAL_LONG_ONLY]),
        }
        for industry, row in results.weights.iterrows()
    ]
    return {
        "title": "Part 3D - Value of re-optimization",
        "subtitle": "Realized annualized volatility, January 2016-July 2023",
        "source": f"Source: {WORKBOOK_NAME}, 49_Industry_Portfolios worksheet",
        "rows": rows,
        "turnover": results.turnover,
        "updated_improved": rows[1]["annualized_volatility"] < rows[0]["annualized_volatility"],
        "weights": weights,
    }


def export_phase3b(
    results: Phase3BResults,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the reproducible Part 3B analysis files."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / "part3b"
    )
    destination.mkdir(parents=True, exist_ok=True)

    comparison_path = destination / "in_sample_volatility.csv"
    returns_path = destination / "monthly_portfolio_returns.csv"
    interpretation_path = destination / "interpretation.md"
    payload_path = destination / "workbook_payload.json"

    results.comparison.to_csv(comparison_path)
    results.monthly_returns.to_csv(returns_path, index_label="Date")
    interpretation_path.write_text(interpretation_markdown(results), encoding="utf-8")
    payload_path.write_text(
        json.dumps(build_reporting_payload(results), indent=2), encoding="utf-8"
    )
    return {
        "comparison": comparison_path,
        "monthly_returns": returns_path,
        "interpretation": interpretation_path,
        "payload": payload_path,
    }


def export_phase3c(
    results: Phase3CResults,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the reproducible Part 3C analysis files."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / "part3c"
    )
    destination.mkdir(parents=True, exist_ok=True)

    comparison_path = destination / "out_of_sample_volatility.csv"
    returns_path = destination / "monthly_portfolio_returns.csv"
    interpretation_path = destination / "interpretation.md"
    payload_path = destination / "workbook_payload.json"

    results.comparison.to_csv(comparison_path)
    results.monthly_returns.to_csv(returns_path, index_label="Date")
    interpretation_path.write_text(
        phase3c_interpretation_markdown(results), encoding="utf-8"
    )
    payload_path.write_text(
        json.dumps(build_phase3c_reporting_payload(results), indent=2),
        encoding="utf-8",
    )
    return {
        "comparison": comparison_path,
        "monthly_returns": returns_path,
        "interpretation": interpretation_path,
        "payload": payload_path,
    }


def export_phase3d(
    results: Phase3DResults,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the reproducible Part 3D analysis files."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / "part3d"
    )
    destination.mkdir(parents=True, exist_ok=True)

    comparison_path = destination / "reoptimization_comparison.csv"
    weights_path = destination / "long_only_weight_comparison.csv"
    returns_path = destination / "monthly_portfolio_returns.csv"
    interpretation_path = destination / "interpretation.md"
    payload_path = destination / "workbook_payload.json"

    results.comparison.to_csv(comparison_path)
    results.weights.to_csv(weights_path, index_label="Industry")
    results.monthly_returns.to_csv(returns_path, index_label="Date")
    interpretation_path.write_text(
        phase3d_interpretation_markdown(results), encoding="utf-8"
    )
    payload_path.write_text(
        json.dumps(build_phase3d_reporting_payload(results), indent=2),
        encoding="utf-8",
    )
    return {
        "comparison": comparison_path,
        "weights": weights_path,
        "monthly_returns": returns_path,
        "interpretation": interpretation_path,
        "payload": payload_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Part 3 portfolio volatility.")
    parser.add_argument("--phase", choices=("3b", "3c", "3d"), default="3b")
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Run the requested evaluation phase and export its reproducible files."""
    args = _build_parser().parse_args()
    if args.phase == "3b":
        outputs = export_phase3b(
            build_phase3b_from_workbook(args.workbook),
            output_dir=args.output_dir,
        )
    elif args.phase == "3c":
        outputs = export_phase3c(
            build_phase3c_from_workbook(args.workbook),
            output_dir=args.output_dir,
        )
    else:
        outputs = export_phase3d(
            build_phase3d_from_workbook(args.workbook),
            output_dir=args.output_dir,
        )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
