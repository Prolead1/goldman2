"""Part 3E cumulative performance and crisis-period analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CRISIS_WINDOWS,
    MSCI_WORKBOOK_NAME,
    WORKBOOK_NAME,
    load_msci_usa_returns,
    load_phase1_data,
)
from .optimization import LONG_ONLY, build_phase2_portfolios

LONG_ONLY_LABEL = "Long-only minimum variance"
BENCHMARK_LABEL = "MSCI USA"

TRADITIONAL_DEFENSIVE = {
    "Food",
    "Soda",
    "Beer",
    "Smoke",
    "Hshld",
    "Hlth",
    "MedEq",
    "Drugs",
    "Util",
    "Telcm",
}
COVID_BENEFICIARIES = {
    "Softw",
    "Hardw",
    "Chips",
    "LabEq",
    "BusSv",
    "MedEq",
    "Drugs",
    "Rtail",
}
SPECIAL_SITUATIONS = {"Gold": "Safe-haven / precious metals"}


class PerformanceAnalysisError(ValueError):
    """Raised when Part 3E inputs cannot support the required comparison."""


@dataclass(frozen=True)
class Phase3EResults:
    """Fixed portfolio performance, crisis statistics, and industry rankings."""

    weights: pd.Series
    monthly_returns: pd.DataFrame
    cumulative_wealth: pd.DataFrame
    crisis_summary: pd.DataFrame
    gfc_top10: pd.DataFrame
    covid_top10: pd.DataFrame


def cumulative_wealth(
    monthly_returns: pd.DataFrame,
    inception: str | pd.Timestamp = pd.Timestamp("2004-12-31"),
) -> pd.DataFrame:
    """Convert monthly returns to cumulative growth of one, including inception."""
    if monthly_returns.empty or monthly_returns.isna().any().any():
        raise PerformanceAnalysisError("Monthly returns must be complete and non-empty.")
    if (monthly_returns <= -1.0).any().any():
        raise PerformanceAnalysisError("Monthly returns must be greater than -100%.")
    growth = (1.0 + monthly_returns).cumprod()
    inception_ts = pd.Timestamp(inception)
    if inception_ts >= growth.index[0]:
        raise PerformanceAnalysisError("Inception must precede the first return month.")
    base = pd.DataFrame(
        [np.ones(growth.shape[1])],
        index=pd.DatetimeIndex([inception_ts], name="Date"),
        columns=growth.columns,
    )
    return pd.concat([base, growth], axis="index")


def cumulative_period_returns(returns: pd.DataFrame) -> pd.Series:
    """Compound every column across an inclusive monthly period."""
    if returns.empty or returns.isna().any().any():
        raise PerformanceAnalysisError("Period returns must be complete and non-empty.")
    result = (1.0 + returns).prod(axis="index") - 1.0
    result.name = "cumulative_return"
    return result


def _classification(industry: str, crisis: str) -> str:
    if industry in TRADITIONAL_DEFENSIVE:
        return "Traditional defensive"
    if crisis == "COVID" and industry in COVID_BENEFICIARIES:
        return "COVID / stay-at-home beneficiary"
    return SPECIAL_SITUATIONS.get(industry, "Unexpected / cyclical resilience")


def top_industries_by_cumulative_return(
    industry_returns: pd.DataFrame,
    *,
    crisis: str,
    count: int = 10,
) -> pd.DataFrame:
    """Rank industries by compounded return over a crisis window."""
    if count < 1 or count > industry_returns.shape[1]:
        raise PerformanceAnalysisError("Top-industry count is outside the available universe.")
    ranked = cumulative_period_returns(industry_returns).sort_values(ascending=False).head(count)
    return pd.DataFrame(
        {
            "Industry": ranked.index,
            "Cumulative return": ranked.to_numpy(dtype=float),
            "Classification": [_classification(name, crisis) for name in ranked.index],
        },
        index=pd.Index(range(1, count + 1), name="Rank"),
    )


def build_phase3e_from_workbooks(
    industry_workbook: str | Path | None = None,
    msci_workbook: str | Path | None = None,
) -> Phase3EResults:
    """Load both supplied workbooks and perform the complete Part 3E analysis."""
    phase1 = load_phase1_data(industry_workbook)
    weights = build_phase2_portfolios(phase1.windows["seed_1"]).weights[LONG_ONLY]
    strategy = (phase1.windows["oos_1"] @ weights).rename(LONG_ONLY_LABEL)
    benchmark = load_msci_usa_returns(msci_workbook).reindex(strategy.index)
    if benchmark.isna().any():
        raise PerformanceAnalysisError("MSCI USA returns do not cover every strategy month.")
    monthly = pd.concat([strategy, benchmark.rename(BENCHMARK_LABEL)], axis="columns")

    crisis_rows = []
    labels = {"gfc": "Global Financial Crisis", "covid": "COVID-19 shock"}
    for key, (start, end) in CRISIS_WINDOWS.items():
        period = monthly.loc[start:end]
        if period.empty:
            raise PerformanceAnalysisError(f"No monthly observations found for {key}.")
        compounded = cumulative_period_returns(period)
        relative = float(compounded[LONG_ONLY_LABEL] - compounded[BENCHMARK_LABEL])
        crisis_rows.append(
            {
                "Crisis": labels[key],
                "Window": f"{start:%b %Y}-{end:%b %Y}",
                "Strategy return": float(compounded[LONG_ONLY_LABEL]),
                "MSCI USA return": float(compounded[BENCHMARK_LABEL]),
                "Relative return": relative,
                "Defensive outperformance": relative > 0.0,
            }
        )
    crisis_summary = pd.DataFrame(crisis_rows).set_index("Crisis")

    return Phase3EResults(
        weights=weights.rename(LONG_ONLY_LABEL),
        monthly_returns=monthly,
        cumulative_wealth=cumulative_wealth(monthly),
        crisis_summary=crisis_summary,
        gfc_top10=top_industries_by_cumulative_return(
            phase1.windows["gfc"], crisis="GFC"
        ),
        covid_top10=top_industries_by_cumulative_return(
            phase1.windows["covid"], crisis="COVID"
        ),
    )


def interpretation_markdown(results: Phase3EResults) -> str:
    """Create report-ready Part 3E commentary grounded in calculated results."""
    ending = results.cumulative_wealth.iloc[-1]
    gfc = results.crisis_summary.loc["Global Financial Crisis"]
    covid = results.crisis_summary.loc["COVID-19 shock"]

    def verdict(row: pd.Series) -> str:
        return (
            "behaved defensively and outperformed the broad market"
            if bool(row["Defensive outperformance"])
            else "did not provide defensive outperformance versus the broad market"
        )

    def category_list(frame: pd.DataFrame, category: str) -> str:
        names = frame.loc[frame["Classification"] == category, "Industry"].tolist()
        return ", ".join(names) if names else "none"

    gfc_defensive = category_list(results.gfc_top10, "Traditional defensive")
    covid_defensive = category_list(results.covid_top10, "Traditional defensive")
    covid_beneficiaries = category_list(
        results.covid_top10, "COVID / stay-at-home beneficiary"
    )
    return f"""# Part 3E - Cumulative performance and crisis behavior

One dollar invested on 12/31/2004 grew to ${ending[LONG_ONLY_LABEL]:.2f} in the
fixed long-only minimum-variance portfolio and ${ending[BENCHMARK_LABEL]:.2f} in
the MSCI USA gross total-return index by 7/31/2023. The cumulative chart uses a
base-2 logarithmic vertical axis, so equal vertical distances represent a
doubling or halving of wealth.

During the Global Financial Crisis window (September 2008-February 2009), the
strategy returned {gfc['Strategy return']:.2%} versus {gfc['MSCI USA return']:.2%}
for MSCI USA, a relative result of {gfc['Relative return']:+.2%}. It therefore
{verdict(gfc)}. Traditional defensive industries among the top ten were
{gfc_defensive}; the remaining leaders include safe-haven or unusually resilient
industries and should not all be treated as permanent defensives.

During the COVID shock (February-March 2020), the strategy returned
{covid['Strategy return']:.2%} versus {covid['MSCI USA return']:.2%} for MSCI USA,
a relative result of {covid['Relative return']:+.2%}. It therefore {verdict(covid)}.
Traditional defensives in the top ten were {covid_defensive}. The list also
contains COVID or stay-at-home beneficiaries ({covid_beneficiaries}), illustrating
that this episode rewarded both classic defensives and businesses helped by the
abrupt shift toward remote work, health spending, and at-home consumption.
"""


def build_reporting_payload(results: Phase3EResults) -> dict[str, object]:
    """Create a JSON-safe payload for the Part 3E workbook."""
    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        return [
            {
                "rank": int(rank),
                "industry": row["Industry"],
                "cumulative_return": float(row["Cumulative return"]),
                "classification": row["Classification"],
            }
            for rank, row in frame.iterrows()
        ]

    crisis_rows = []
    for crisis, row in results.crisis_summary.iterrows():
        crisis_rows.append(
            {
                "crisis": crisis,
                "window": row["Window"],
                "strategy_return": float(row["Strategy return"]),
                "benchmark_return": float(row["MSCI USA return"]),
                "relative_return": float(row["Relative return"]),
                "defensive_outperformance": bool(row["Defensive outperformance"]),
            }
        )
    return {
        "title": "Part 3E - Cumulative performance and crisis behavior",
        "subtitle": "Growth of $1, 12/31/2004-7/31/2023; y-axis is log base 2",
        "source": (
            f"Sources: {WORKBOOK_NAME}; {MSCI_WORKBOOK_NAME} "
            "(MSCI USA gross total-return index, USD)"
        ),
        "ending_wealth": {
            name: float(value) for name, value in results.cumulative_wealth.iloc[-1].items()
        },
        "crises": crisis_rows,
        "gfc_top10": records(results.gfc_top10),
        "covid_top10": records(results.covid_top10),
    }


def export_phase3e(
    results: Phase3EResults,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the reproducible Part 3E analysis files."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / "part3e"
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly_returns": destination / "monthly_performance.csv",
        "cumulative_wealth": destination / "cumulative_wealth.csv",
        "crisis_summary": destination / "crisis_summary.csv",
        "gfc_top10": destination / "gfc_top10_industries.csv",
        "covid_top10": destination / "covid_top10_industries.csv",
        "interpretation": destination / "interpretation.md",
        "payload": destination / "workbook_payload.json",
    }
    results.monthly_returns.to_csv(paths["monthly_returns"], index_label="Date")
    results.cumulative_wealth.to_csv(paths["cumulative_wealth"], index_label="Date")
    results.crisis_summary.to_csv(paths["crisis_summary"])
    results.gfc_top10.to_csv(paths["gfc_top10"])
    results.covid_top10.to_csv(paths["covid_top10"])
    paths["interpretation"].write_text(interpretation_markdown(results), encoding="utf-8")
    paths["payload"].write_text(
        json.dumps(build_reporting_payload(results), indent=2), encoding="utf-8"
    )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FE5108 Part 3E performance analysis.")
    parser.add_argument("--industry-workbook", type=Path, default=None)
    parser.add_argument("--msci-workbook", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Run Part 3E and export all reproducible files."""
    args = _build_parser().parse_args()
    outputs = export_phase3e(
        build_phase3e_from_workbooks(args.industry_workbook, args.msci_workbook),
        args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
