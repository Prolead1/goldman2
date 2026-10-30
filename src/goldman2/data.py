"""Loaders for the FE5108 class-project data (Parts 1-5).

Fama-French returns are in percent (0.23 means 0.23%). MSCI values are gross
total-return index levels in USD (convert with :func:`load_msci_returns`). The
Part 2 stock panel and the Part 4 ETF data use decimal returns, consistent with
their source files.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

WORKBOOK = DATA_DIR / "class-project-data-ff-msci-parts-1-and-5.xlsx"
STOCK_PANEL_CSV = DATA_DIR / "class-project-data-part-2-factor-portfolio-construction.csv"
INDUSTRY_XLSX = DATA_DIR / "class-project-data-part-3-pf-construction.xlsx"
ETF_XLSX = DATA_DIR / "class-project-data-part-4-product-innovation.xlsx"

SHEET_FF3 = "FF - 3 factor returns"
SHEET_BETA = "FF - portfolios sorted on beta"
SHEET_PMOM = "FF - portfolios sorted on pmom"
SHEET_MSCI = "MSCI - factor totret index"
SHEET_INDUSTRIES = "49_Industry_Portfolios"

_SHEET_BY_PORTFOLIO = {"beta": SHEET_BETA, "pmom": SHEET_PMOM}

# The provided beta sheet stacks 5 quintile and 10 decile portfolios and has
# its middle decile header labels corrupted into dates.
_BETA_LABELS = (
    ["Lo 20", "Qnt 2 (20)", "Qnt 3 (20)", "Qnt 4 (20)", "Hi 20"]
    + ["Lo 10"]
    + [f"Qnt {i} (10)" for i in range(2, 10)]
    + ["Hi 10"]
)


def _raw(path: Path | str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")


def _ff_sections(raw: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield ``(value|equal, monthly|annual, frame)`` for each FF returns section."""
    for row_pos, heading in raw[0].items():
        if not isinstance(heading, str) or "Returns --" not in heading:
            continue
        kind = "value" if "Value" in heading else "equal"
        freq = "monthly" if "Monthly" in heading else "annual"
        n_digits = 6 if freq == "monthly" else 4

        block = raw.iloc[row_pos + 1 :, :].copy()
        dates = pd.to_numeric(block[0], errors="coerce")
        is_data = (
            dates.astype("Int64").astype(str).str.fullmatch(rf"\d{{{n_digits}}}").fillna(False)
        )
        positions = block.index[is_data]
        if len(positions) == 0:
            continue
        # Keep only the contiguous run of rows belonging to this section.
        run = is_data.loc[positions[0] :].astype(int).cumprod()
        data_rows = run.index[run.eq(1)]

        columns = raw.loc[positions[0] - 1, 1:].dropna().astype(str).str.strip().tolist()
        frame = raw.loc[data_rows, : len(columns) + 1].copy()
        frame.columns = ["date", *columns]
        frame["date"] = dates.loc[data_rows].astype(int)
        if freq == "monthly":
            month = frame["date"].astype(str)
            frame["date"] = pd.PeriodIndex(pd.to_datetime(month, format="%Y%m"), freq="M")
        yield kind, freq, frame.set_index("date").astype(float)


def load_ff3(path: Path | str = WORKBOOK) -> pd.DataFrame:
    """Monthly Fama-French 3-factor returns (percent).

    Columns: ``Mkt-RF``, ``SMB``, ``HML``, ``RF``. Index: month-end
    ``PeriodIndex`` from 1926-07 through the last available month.
    """
    raw = _raw(path, SHEET_FF3)
    header_row = raw.index[raw[1].eq("Mkt-RF")][0]
    data = raw.iloc[header_row + 1 :, :5].copy()
    data.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
    # Monthly rows are YYYYMM; stop before the annual rows further down the sheet.
    data = data[data["date"].astype(str).str.fullmatch(r"\d{6}")]
    data["date"] = pd.PeriodIndex(
        pd.to_datetime(data["date"].astype(int).astype(str), format="%Y%m"), freq="M"
    )
    return data.set_index("date").astype(float)


def load_portfolios(portfolio: str = "beta", path: Path | str = WORKBOOK) -> dict[str, pd.DataFrame]:
    """Fama-French portfolio returns sorted on ``"beta"`` or ``"pmom"`` (percent).

    Returns a dict keyed ``"{value|equal}_{monthly|annual}"``; each frame has
    one column per portfolio (lowest to highest). The beta sheet stacks 5
    quintile portfolios (``Lo 20`` .. ``Hi 20``) next to 10 deciles
    (``Lo 10`` .. ``Hi 10``); the breakpoint is suffixed in brackets to keep
    column names unique and the corrupted decile labels are rebuilt here.
    """
    if portfolio not in _SHEET_BY_PORTFOLIO:
        raise ValueError(f"portfolio must be one of {sorted(_SHEET_BY_PORTFOLIO)}, got {portfolio!r}")

    raw = _raw(path, _SHEET_BY_PORTFOLIO[portfolio])
    sections: dict[str, pd.DataFrame] = {}
    for kind, freq, frame in _ff_sections(raw):
        if portfolio == "beta" and len(frame.columns) == len(_BETA_LABELS):
            frame.columns = _BETA_LABELS
        sections[f"{kind}_{freq}"] = frame
    return sections


def load_industries(path: Path | str = INDUSTRY_XLSX) -> dict[str, pd.DataFrame]:
    """Fama-French 49 industry portfolio returns (percent).

    Returns the same ``"{value|equal}_{monthly|annual}"`` mapping as
    :func:`load_portfolios`. The class extract contains monthly value-weighted
    returns from 1995-01 through 2023-07 only.
    """
    raw = _raw(path, SHEET_INDUSTRIES)
    return {f"{kind}_{freq}": frame for kind, freq, frame in _ff_sections(raw)}


def load_msci_levels(path: Path | str = WORKBOOK) -> pd.DataFrame:
    """Monthly MSCI USA gross total-return index levels (USD, month-end).

    Columns are the five MSCI USA indexes in the workbook: standard, value,
    minimum volatility, equal weighted and momentum. The 1994-12-30 base row
    is included.
    """
    raw = _raw(path, SHEET_MSCI)
    header_row = raw.index[raw[0].eq("Date")][0]
    columns = raw.loc[header_row, 1:].dropna().astype(str).tolist()
    block = raw.iloc[header_row + 1 :, : len(columns) + 1].copy()
    block.columns = ["date", *columns]
    block["date"] = pd.to_datetime(block["date"], format="%b %d, %Y", errors="coerce")
    block = block[block["date"].notna()].dropna(axis=1, how="all")
    for column in columns:
        block[column] = pd.to_numeric(
            block[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    block["date"] = block["date"].dt.to_period("M")
    return block.set_index("date").astype(float)


def load_msci_returns(path: Path | str = WORKBOOK) -> pd.DataFrame:
    """Month-over-month percent returns of the MSCI USA index levels."""
    return load_msci_levels(path).pct_change().dropna(how="all") * 100.0


def load_stock_panel(path: Path | str = STOCK_PANEL_CSV) -> pd.DataFrame:
    """Stock-level monthly panel for Part 2 (mlfactor extract).

    Long format with columns ``stock_id``, ``date`` (month-end ``PeriodIndex``
    of the observation date), the standardized characteristics ``Pb``, ``Bv``,
    ``Vol3Y_Usd`` and ``Mkt_Cap_3M_Usd`` (each on a 0-1 scale), and
    ``R1M_Usd``, the value of that month's forward one-month return in decimal
    form. Covers 1998-11 through 2019-03.

    Note that features dated month *t* line up with the return for month *t+1*
    (e.g. 1998-12 characteristics predict January 1999 returns).
    """
    panel = pd.read_csv(path, parse_dates=["date"])
    panel["date"] = panel["date"].dt.to_period("M")
    return panel.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_etf_daily(path: Path | str = ETF_XLSX, sheet: str = "NVDL") -> pd.DataFrame:
    """Daily total returns (decimals) of NVDL and its underlying NVDA.

    February through August 2024, indexed by trade date in ascending order.
    """
    raw = _raw(path, sheet)
    header_row = raw.index[raw[1].eq("NVDA")][0]
    data = raw.iloc[header_row + 1 :, :3].copy()
    data.columns = ["date", "NVDA", "NVDL"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").astype(float)
    return data.sort_index()
