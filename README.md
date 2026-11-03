# goldman2 — FE5108 class project (Fall 2026)

NUS **FE5108 Portfolio Theory and Investments** practitioner-segment class project
(Fall 2026, Sanne de Boer). The assignment applies factor investing, portfolio
construction and portfolio analytics to Fama-French and MSCI data.

## Contents

| Path | Description |
| --- | --- |
| `docs/class-project-fall-2026.pdf` | Full assignment (Parts 1–5) |
| `docs/class-project-fall-2026.txt` | Plain-text transcription for quick search |
| `docs/nvdl-graniteshares-2x-long-nvda-daily-etf-factsheet.pdf` | NVDL fund factsheet (Part 4) |
| `data/class-project-data-ff-msci-parts-1-and-5.xlsx` | FF factor/portfolio returns and MSCI index levels (Parts 1 and 5) |
| `data/class-project-data-part-2-factor-portfolio-construction.csv` | Stock-level monthly panel (Part 2) |
| `data/class-project-data-part-3-pf-construction.xlsx` | FF 49 industry monthly returns (Part 3) |
| `data/class-project-data-part-4-product-innovation.xlsx` | NVDL and NVDA daily total returns (Part 4) |
| `src/goldman2/data.py` | Loaders for all five parts |
| `part4/` | Self-contained Part 4 analysis, notebook and tests |
| `notebooks/` | Analysis notebooks (one per part) |
| `exports/` | Generated exhibits (tables and charts) for the write-up |

Data overview:

- `FF - 3 factor returns` — monthly `Mkt-RF`, `SMB`, `HML`, `RF` (%), 1926-07 onward
- `FF - portfolios sorted on beta` — value/equal weighted quintile (5) and decile (10)
  returns, monthly and annual, 1963-07 onward
- `FF - portfolios sorted on pmom` — prior-return (momentum) decile returns, monthly and
  annual, 1927-01 onward
- `MSCI - factor totret index` — gross USD index levels since 1994-12-30 for USA standard,
  value, minimum volatility, equal weighted and momentum indexes
- Part 2 CSV — 1,207 anonymized stocks, 1998-11 to 2019-03; `Pb`, `Bv`, `Vol3Y_Usd`,
  `Mkt_Cap_3M_Usd` standardized to 0-1, `R1M_Usd` = next-month forward return in decimals
- Part 3 workbook — 49 FF industries, value-weighted monthly returns (%), 1995-01 to 2023-07
- Part 4 workbook — NVDL/NVDA daily total returns (decimals), 2024-02-01 to 2024-08-30
  (sorted descending in the source file)

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync          # create .venv and install dependencies
uv run jupyter lab
```

## Quick start

```python
from goldman2 import (
    load_etf_daily,
    load_ff3,
    load_industries,
    load_msci_returns,
    load_portfolios,
    load_stock_panel,
)

ff3 = load_ff3()                          # monthly %, PeriodIndex
beta = load_portfolios("beta")            # beta["value_monthly"], beta["equal_monthly"], ...
mom = load_portfolios("pmom")
msci = load_msci_returns()                # month-over-month %, all five indexes
panel = load_stock_panel()                # long stock-level panel, month-end PeriodIndex
industries = load_industries()            # industries["value_monthly"], 49 columns
etf = load_etf_daily()                    # daily NVDA/NVDL decimal returns
```

## Notebooks

- `notebooks/part1-factor-investing.ipynb` — complete Part 1 report with executed code,
  written analysis and seven custom exhibits: factor-premium decay, MSCI momentum pipeline,
  factor diversification, construction-sensitivity correlations, and momentum/volatility
  regimes. Exhibits are also exported to `exports/part1/`.
- `part4/analysis.ipynb` — complete Part 4 workflow covering daily risk, the NVDL-on-NVDA
  leverage regression, multi-day compounding and investor suitability. Supporting modules
  and tests are kept together in `part4/`.

## Assignment map

1. **Part 1 — Factor investing (22 pts)**: factor premia and decay, MSCI factor index
   methodology, factor diversification, sensitivity to construction, regime sensitivity.
2. **Part 2 — Security-level analysis (20 pts)**: data cleaning/validation, cap-weighted
   market proxy, three value portfolio construction rules, tracking error vs. HML.
3. **Part 3 — Portfolio construction (21 pts)**: minimum-variance (long/short and
   long-only), inverse-volatility, in/out-of-sample volatility, re-optimization, crises.
4. **Part 4 — Product innovation (11 pts)**: 2x leveraged single-stock ETF (NVDL vs NVDA):
   risk metrics, regression, drag, compounding.
5. **Part 5 — Portfolio analytics (26 pts)**: factor composite definitions, factor
   exposures, replication, relative performance and tracking-error decomposition.
