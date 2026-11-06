"""Build and export the complete FE5108 Part 4 analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import ValidationReport, load_daily_returns, validate_daily_returns
from .performance import (
    full_period_total_returns,
    investor_suitability,
    monthly_total_returns,
    two_day_compounding_example,
)
from .regression import RegressionResults, fit_daily_leverage_regression
from .risk import build_risk_table

FACTSHEET_NET_EXPENSE_RATIO = 0.0115


@dataclass(frozen=True)
class Part4Results:
    """All calculated tables required for Parts 4A-4F."""

    validation: ValidationReport
    daily_returns: pd.DataFrame
    risk_metrics: pd.DataFrame
    regression: RegressionResults
    monthly_returns: pd.DataFrame
    full_period_returns: pd.DataFrame
    two_day_example: pd.DataFrame
    suitability: pd.DataFrame


def build_part4_results(workbook_path: str | Path | None = None) -> Part4Results:
    """Load the source workbook and calculate every Part 4 result."""
    daily = load_daily_returns(workbook_path)
    return Part4Results(
        validation=validate_daily_returns(daily),
        daily_returns=daily,
        risk_metrics=build_risk_table(daily),
        regression=fit_daily_leverage_regression(daily["NVDA"], daily["NVDL"]),
        monthly_returns=monthly_total_returns(daily),
        full_period_returns=full_period_total_returns(daily),
        two_day_example=two_day_compounding_example(),
        suitability=investor_suitability(),
    )


def interpretation_markdown(results: Part4Results) -> str:
    """Create concise, report-ready commentary grounded in calculated results."""
    risk = results.risk_metrics
    coefficients = results.regression.coefficients
    intercept = coefficients.loc["Intercept"]
    beta = coefficients.loc["NVDA beta"]
    full = results.full_period_returns.iloc[0]
    scenario = results.two_day_example["Two-day return"]

    return f"""# Part 4 - Product innovation

## 4A Risk comparison

The daily standard deviation is {risk.loc['NVDA', 'Daily standard deviation']:.2%} for
NVDA and {risk.loc['NVDL', 'Daily standard deviation']:.2%} for NVDL. Historical 95%
VaR is {risk.loc['NVDA', '95% VaR']:.2%} and {risk.loc['NVDL', '95% VaR']:.2%}, while
95% CVaR is {risk.loc['NVDA', '95% CVaR']:.2%} and
{risk.loc['NVDL', '95% CVaR']:.2%}. Each NVDL risk measure is approximately twice the
NVDA measure, consistent with leverage magnifying both ordinary volatility and tail risk.

## 4B Daily leverage regression

The estimated regression is NVDL = {intercept['Estimate']:.6f} +
{beta['Estimate']:.4f} x NVDA. The 95% confidence interval for the NVDA coefficient is
[{beta['95% CI lower']:.4f}, {beta['95% CI upper']:.4f}], which contains the stated
daily target of 2. The regression R-squared is {results.regression.r_squared:.2%}.
The data therefore strongly support delivery of approximately 2x the daily NVDA return
before considering the intercept and residual tracking difference.

## 4C Estimated performance drag

The daily intercept is {intercept['Estimate']:.4%}, with a 95% confidence interval of
[{intercept['95% CI lower']:.4%}, {intercept['95% CI upper']:.4%}]. Multiplying by 252
gives a crude annualized drag estimate of {results.regression.annualized_intercept:.2%},
which is much larger in magnitude than the factsheet's {FACTSHEET_NET_EXPENSE_RATIO:.2%}
net operating expense ratio. The intercept is not a clean fee estimate: it also captures
financing or swap costs, daily rebalancing and trading costs, and market-price tracking
effects during this particular sample.

## 4D Returns beyond one day

From February through August 2024, NVDA returned {full['NVDA']:.2%} and NVDL returned
{full['NVDL']:.2%}. Twice NVDA's cumulative return would be
{full['Twice NVDA cumulative return']:.2%}; NVDL finished
{abs(full['NVDL minus twice NVDA']):.2%} below that simple comparison. The monthly table also
shows that NVDL can finish above or below twice NVDA because daily leverage resets and
daily returns compound along the realized path.

## 4E Two-day compounding example

NVDA's +10% and -9% path produces a two-day return of {scenario['NVDA']:.2%}. An ideal
2x daily product earns +20% and then -18%, producing {scenario['NVDL (ideal 2x, no drag)']:.2%}.
The leveraged ETF loses money even though NVDA finishes slightly positive, illustrating
the volatility drag described in the factsheet.

## 4F Investor suitability

For a one-day catalyst trade, NVDL is the closer match for an investor deliberately seeking
2x exposure and able to tolerate roughly doubled downside. For a one-year bullish view with
expected volatility, NVDA is preferable because it avoids daily leverage resets, volatility
drag, financing costs and the higher probability of a large loss.
"""


def _summary_payload(results: Part4Results) -> dict[str, object]:
    regression = results.regression
    return {
        "validation": asdict(results.validation),
        "risk_metrics": results.risk_metrics.reset_index().to_dict(orient="records"),
        "regression": {
            "observations": regression.observations,
            "residual_degrees_of_freedom": regression.residual_degrees_of_freedom,
            "r_squared": regression.r_squared,
            "residual_standard_error": regression.residual_standard_error,
            "annualized_intercept": regression.annualized_intercept,
            "coefficients": regression.coefficients.reset_index().to_dict(orient="records"),
        },
        "full_period_returns": results.full_period_returns.reset_index().to_dict(
            orient="records"
        ),
        "two_day_example": results.two_day_example.reset_index().to_dict(orient="records"),
    }


def export_part4(
    results: Part4Results,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Export auditable CSV/JSON inputs and report-ready interpretation."""
    output = (
        Path(output_directory)
        if output_directory is not None
        else Path(__file__).resolve().parent / "outputs"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "risk_metrics": output / "risk_metrics.csv",
        "regression": output / "regression_results.csv",
        "monthly_returns": output / "monthly_total_returns.csv",
        "full_period_returns": output / "full_period_total_returns.csv",
        "two_day_example": output / "two_day_example.csv",
        "suitability": output / "investor_suitability.csv",
        "interpretation": output / "interpretation.md",
        "summary": output / "summary.json",
    }
    results.risk_metrics.to_csv(paths["risk_metrics"])
    results.regression.coefficients.to_csv(paths["regression"])
    results.monthly_returns.to_csv(paths["monthly_returns"])
    results.full_period_returns.to_csv(paths["full_period_returns"])
    results.two_day_example.to_csv(paths["two_day_example"])
    results.suitability.to_csv(paths["suitability"])
    paths["interpretation"].write_text(
        interpretation_markdown(results), encoding="utf-8"
    )
    paths["summary"].write_text(
        json.dumps(_summary_payload(results), indent=2), encoding="utf-8"
    )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FE5108 Part 4 analysis.")
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    outputs = export_part4(
        build_part4_results(args.workbook),
        args.output_directory,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
