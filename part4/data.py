"""Load and validate the NVDA and NVDL daily-return history for Part 4."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WORKBOOK_NAME = "class-project-data-part-4-product-innovation.xlsx"
SHEET_NAME = "NVDL"
HEADER_ROW_ZERO_BASED = 4

EXPECTED_COLUMNS = ("NVDA", "NVDL")
EXPECTED_ROWS = 147
EXPECTED_START = pd.Timestamp("2024-02-01")
EXPECTED_END = pd.Timestamp("2024-08-30")


class DataValidationError(ValueError):
    """Raised when the supplied Part 4 workbook cannot support the analysis."""


@dataclass(frozen=True)
class ValidationReport:
    """Compact audit record for the cleaned daily-return panel."""

    rows: int
    columns: list[str]
    start: str
    end: str
    missing_values: int
    minimum_returns: dict[str, float]
    maximum_returns: dict[str, float]


def default_workbook_path() -> Path:
    """Return the supplied Part 4 workbook from the repository data directory."""
    return DATA_DIR / WORKBOOK_NAME


def validate_daily_returns(returns: pd.DataFrame) -> ValidationReport:
    """Validate labels, dates, units, completeness and finite decimal returns."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise DataValidationError("Daily returns must use a DatetimeIndex.")
    if tuple(returns.columns) != EXPECTED_COLUMNS:
        raise DataValidationError(
            f"Expected columns {EXPECTED_COLUMNS}; found {tuple(returns.columns)}."
        )
    if returns.index.has_duplicates:
        raise DataValidationError("Trade dates must be unique.")
    if not returns.index.is_monotonic_increasing:
        raise DataValidationError("Trade dates must be sorted in ascending order.")
    if len(returns) != EXPECTED_ROWS:
        raise DataValidationError(
            f"Expected {EXPECTED_ROWS} daily observations; found {len(returns)}."
        )
    if returns.index[0] != EXPECTED_START or returns.index[-1] != EXPECTED_END:
        raise DataValidationError(
            f"Expected history from {EXPECTED_START:%Y-%m-%d} through "
            f"{EXPECTED_END:%Y-%m-%d}; found {returns.index[0]:%Y-%m-%d} through "
            f"{returns.index[-1]:%Y-%m-%d}."
        )
    if returns.isna().any().any():
        raise DataValidationError("Daily returns must not contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes):
        raise DataValidationError("Every return column must be numeric.")

    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("Daily returns must contain only finite values.")
    if (values <= -1.0).any():
        raise DataValidationError("Daily returns must be greater than -100%.")
    if np.abs(values).max() >= 1.0:
        raise DataValidationError(
            "The source is expected in decimal returns; an absolute return of 100% or more "
            "suggests an incorrect unit or source range."
        )

    return ValidationReport(
        rows=len(returns),
        columns=list(returns.columns),
        start=returns.index[0].strftime("%Y-%m-%d"),
        end=returns.index[-1].strftime("%Y-%m-%d"),
        missing_values=int(returns.isna().sum().sum()),
        minimum_returns={name: float(value) for name, value in returns.min().items()},
        maximum_returns={name: float(value) for name, value in returns.max().items()},
    )


def load_daily_returns(workbook_path: str | Path | None = None) -> pd.DataFrame:
    """Load ascending NVDA and NVDL daily total returns in decimal form."""
    path = Path(workbook_path) if workbook_path is not None else default_workbook_path()
    if not path.is_file():
        raise FileNotFoundError(f"Part 4 workbook not found: {path}")

    raw = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW_ZERO_BASED,
        engine="openpyxl",
    ).dropna(axis="columns", how="all")
    if raw.shape[1] != 3:
        raise DataValidationError(
            f"Expected one date column plus NVDA and NVDL; found {raw.shape[1]} columns."
        )

    dates = pd.to_datetime(raw.iloc[:, 0], errors="raise")
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    values.columns = list(EXPECTED_COLUMNS)
    values.index = pd.DatetimeIndex(dates, name="Date")
    values = values.sort_index()
    validate_daily_returns(values)
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the FE5108 Part 4 workbook.")
    parser.add_argument("--workbook", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    returns = load_daily_returns(args.workbook)
    print(json.dumps(asdict(validate_daily_returns(returns)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
