"""Load and validate the 49-industry return data required for the project.

The source workbook reports monthly total returns in percentage points. This
module preserves the source file, converts the observations to decimal returns,
and exposes the exact estimation, evaluation, and crisis windows specified in
the assignment.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

WORKBOOK_NAME = "class-project-data-part-3-pf-construction.xlsx"
SHEET_NAME = "49_Industry_Portfolios"
HEADER_ROW_ZERO_BASED = 11
MSCI_WORKBOOK_NAME = "class-project-data-ff-msci-parts-1-and-5.xlsx"
MSCI_SHEET_NAME = "MSCI - factor totret index"
MSCI_HEADER_ROW_ZERO_BASED = 6
MSCI_USA_COLUMN = "USA Standard (Large+Mid Cap)"
EXPECTED_INDUSTRIES = 49
FULL_START = pd.Timestamp("1995-01-31")
FULL_END = pd.Timestamp("2023-07-31")
MISSING_SENTINELS = (-99.99, -999.0)

ANALYSIS_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "seed_1": (pd.Timestamp("1995-01-31"), pd.Timestamp("2004-12-31")),
    "oos_1": (pd.Timestamp("2005-01-31"), pd.Timestamp("2023-07-31")),
    "seed_2": (pd.Timestamp("2006-01-31"), pd.Timestamp("2015-12-31")),
    "reopt_test": (pd.Timestamp("2016-01-31"), pd.Timestamp("2023-07-31")),
}

CRISIS_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "gfc": (pd.Timestamp("2008-09-30"), pd.Timestamp("2009-02-28")),
    "covid": (pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")),
}

EXPECTED_WINDOW_MONTHS = {
    "seed_1": 120,
    "oos_1": 223,
    "seed_2": 120,
    "reopt_test": 91,
    "gfc": 6,
    "covid": 2,
}


class DataValidationError(ValueError):
    """Raised when the supplied workbook cannot support the required analysis."""


@dataclass(frozen=True)
class ValidationReport:
    """Compact audit record for the cleaned return panel."""

    rows: int
    industries: int
    start: str
    end: str
    missing_values: int
    minimum_return: float
    maximum_return: float
    window_months: dict[str, int]


@dataclass(frozen=True)
class Phase1Data:
    """Clean returns, required date slices, and their validation report."""

    returns: pd.DataFrame
    windows: dict[str, pd.DataFrame]
    report: ValidationReport


def default_workbook_path() -> Path:
    """Return the Part 3 workbook from the repository data directory."""
    configured = os.environ.get("FE5108_PART3_WORKBOOK")
    if configured:
        return Path(configured).expanduser()
    return DATA_DIR / WORKBOOK_NAME


def default_msci_workbook_path() -> Path:
    """Return the MSCI workbook from the repository data directory."""
    configured = os.environ.get("FE5108_MSCI_WORKBOOK")
    if configured:
        return Path(configured).expanduser()
    return DATA_DIR / MSCI_WORKBOOK_NAME


def load_msci_usa_returns(workbook_path: str | Path | None = None) -> pd.Series:
    """Load monthly decimal returns for the MSCI USA gross total-return index.

    The source worksheet stores month-end gross total-return index levels in
    US dollars, with comma-separated values and legal notes below the data.
    The December 1994 level is retained only to calculate the January 1995
    return; the returned series spans January 1995 through July 2023.
    """
    path = (
        Path(workbook_path)
        if workbook_path is not None
        else default_msci_workbook_path()
    )
    if not path.is_file():
        raise FileNotFoundError(f"MSCI workbook not found: {path}")

    raw = pd.read_excel(
        path,
        sheet_name=MSCI_SHEET_NAME,
        header=MSCI_HEADER_ROW_ZERO_BASED,
        engine="openpyxl",
    )
    if "Date" not in raw.columns or MSCI_USA_COLUMN not in raw.columns:
        raise DataValidationError(
            f"Expected Date and {MSCI_USA_COLUMN!r} in {MSCI_SHEET_NAME!r}."
        )

    parsed_dates = pd.to_datetime(raw["Date"], errors="coerce")
    valid_rows = parsed_dates.notna()
    dates = pd.DatetimeIndex(
        parsed_dates.loc[valid_rows] + pd.offsets.MonthEnd(0),
        name="Date",
    )
    level_text = raw.loc[valid_rows, MSCI_USA_COLUMN].astype(str).str.replace(
        ",", "", regex=False
    )
    levels = pd.Series(
        pd.to_numeric(level_text, errors="raise").to_numpy(dtype=float),
        index=dates,
        name="MSCI USA gross total-return index",
    ).sort_index()

    if levels.index.has_duplicates:
        raise DataValidationError("MSCI USA index dates must be unique.")
    if not np.isfinite(levels.to_numpy(dtype=float)).all() or (levels <= 0.0).any():
        raise DataValidationError("MSCI USA index levels must be finite and positive.")

    returns = levels.pct_change(fill_method=None).dropna().loc[FULL_START:FULL_END]
    returns.name = "MSCI USA"
    expected = pd.date_range(FULL_START, FULL_END, freq="ME", name="Date")
    if not returns.index.equals(expected):
        missing = expected.difference(returns.index)
        raise DataValidationError(
            "MSCI USA history is not monthly and continuous. Missing months: "
            + ", ".join(date.strftime("%Y-%m") for date in missing)
        )
    return returns


def _parse_months(values: pd.Series) -> pd.DatetimeIndex:
    """Parse numeric or text YYYYMM values as calendar month ends."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.allclose(numeric, np.floor(numeric)):
        raise DataValidationError("The date column must contain integer YYYYMM values.")
    text = numeric.astype("int64").astype(str)
    parsed = pd.to_datetime(text, format="%Y%m", errors="raise")
    return pd.DatetimeIndex(parsed + pd.offsets.MonthEnd(0), name="Date")


def load_industry_returns(workbook_path: str | Path | None = None) -> pd.DataFrame:
    """Load the supplied industry returns as a month-by-industry decimal panel.

    The workbook's header is on Excel row 12. The first column is YYYYMM and
    the remaining 49 columns are monthly value-weighted total returns expressed
    in percentage points.
    """
    path = Path(workbook_path) if workbook_path is not None else default_workbook_path()
    if not path.is_file():
        raise FileNotFoundError(f"Industry workbook not found: {path}")

    raw = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW_ZERO_BASED,
        engine="openpyxl",
    )
    raw = raw.dropna(axis="columns", how="all")
    if raw.shape[1] != EXPECTED_INDUSTRIES + 1:
        raise DataValidationError(
            f"Expected one date column plus {EXPECTED_INDUSTRIES} industries; "
            f"found {raw.shape[1]} columns."
        )

    industry_names = [str(column).strip() for column in raw.columns[1:]]
    if any(not name for name in industry_names):
        raise DataValidationError("Industry names must not be blank.")
    if len(industry_names) != len(set(industry_names)):
        raise DataValidationError("Industry names must be unique after trimming whitespace.")

    dates = _parse_months(raw.iloc[:, 0])
    numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    numeric.columns = industry_names
    numeric.index = dates
    numeric = numeric.replace(list(MISSING_SENTINELS), np.nan) / 100.0
    numeric = numeric.sort_index()
    numeric = numeric.loc[FULL_START:FULL_END]

    validate_returns(numeric)
    return numeric


def slice_window(
    returns: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return an inclusive monthly slice and require every calendar month."""
    start_ts = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    if start_ts > end_ts:
        raise DataValidationError("Window start must not be after window end.")

    window = returns.loc[start_ts:end_ts].copy()
    expected = pd.date_range(start_ts, end_ts, freq="ME", name="Date")
    if not window.index.equals(expected):
        missing = expected.difference(window.index)
        extra = window.index.difference(expected)
        raise DataValidationError(
            "Window does not contain one row per calendar month. "
            f"Missing: {[date.strftime('%Y-%m') for date in missing]}; "
            f"unexpected: {[date.strftime('%Y-%m') for date in extra]}."
        )
    return window


def validate_returns(returns: pd.DataFrame) -> ValidationReport:
    """Validate shape, dates, values, and every assignment-specific window."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise DataValidationError("Returns must use a DatetimeIndex.")
    if returns.index.has_duplicates:
        raise DataValidationError("Return dates must be unique.")
    if not returns.index.is_monotonic_increasing:
        raise DataValidationError("Return dates must be sorted in ascending order.")
    if returns.shape[1] != EXPECTED_INDUSTRIES:
        raise DataValidationError(
            f"Expected {EXPECTED_INDUSTRIES} industry columns; found {returns.shape[1]}."
        )
    if returns.columns.has_duplicates:
        raise DataValidationError("Industry columns must be unique.")
    if returns.empty:
        raise DataValidationError("Return panel is empty.")
    if returns.index[0] != FULL_START or returns.index[-1] != FULL_END:
        raise DataValidationError(
            f"Expected history from {FULL_START:%Y-%m} through {FULL_END:%Y-%m}; "
            f"found {returns.index[0]:%Y-%m} through {returns.index[-1]:%Y-%m}."
        )

    expected_full_index = pd.date_range(FULL_START, FULL_END, freq="ME", name="Date")
    if not returns.index.equals(expected_full_index):
        missing_months = expected_full_index.difference(returns.index)
        raise DataValidationError(
            "Full history is not monthly and continuous. Missing months: "
            + ", ".join(date.strftime("%Y-%m") for date in missing_months)
        )

    missing_values = int(returns.isna().sum().sum())
    if missing_values:
        affected = returns.columns[returns.isna().any()].tolist()
        raise DataValidationError(
            f"Found {missing_values} missing returns in industries: {', '.join(affected)}."
        )
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes):
        raise DataValidationError("Every industry return column must be numeric.")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise DataValidationError("Returns contain an infinite or non-numeric value.")
    if (returns <= -1.0).any().any():
        affected = returns.columns[(returns <= -1.0).any()].tolist()
        raise DataValidationError(
            "Returns at or below -100% remain after sentinel cleaning in: "
            + ", ".join(affected)
        )

    windows = {**ANALYSIS_WINDOWS, **CRISIS_WINDOWS}
    window_months = {
        name: len(slice_window(returns, start, end))
        for name, (start, end) in windows.items()
    }
    incorrect = {
        name: count
        for name, count in window_months.items()
        if count != EXPECTED_WINDOW_MONTHS[name]
    }
    if incorrect:
        raise DataValidationError(f"Unexpected window lengths: {incorrect}")

    return ValidationReport(
        rows=len(returns),
        industries=returns.shape[1],
        start=returns.index[0].strftime("%Y-%m-%d"),
        end=returns.index[-1].strftime("%Y-%m-%d"),
        missing_values=missing_values,
        minimum_return=float(returns.min().min()),
        maximum_return=float(returns.max().max()),
        window_months=window_months,
    )


def load_phase1_data(workbook_path: str | Path | None = None) -> Phase1Data:
    """Load the full panel and construct every required Phase 1 date window."""
    returns = load_industry_returns(workbook_path)
    definitions = {**ANALYSIS_WINDOWS, **CRISIS_WINDOWS}
    windows = {
        name: slice_window(returns, start, end)
        for name, (start, end) in definitions.items()
    }
    report = validate_returns(returns)
    return Phase1Data(returns=returns, windows=windows, report=report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the FE5108 Part 3 workbook.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Path to the Part 3 workbook; defaults to the file beside code/.",
    )
    return parser


def main() -> None:
    """Load the workbook and print a machine-readable validation summary."""
    args = _build_parser().parse_args()
    phase1 = load_phase1_data(args.workbook)
    print(json.dumps(asdict(phase1.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
