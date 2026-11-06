# Part 4 - Product innovation

This directory contains the complete analysis of the GraniteShares 2x Long
NVDA Daily ETF (NVDL) for FE5108 Project 2, Part 4. It uses the supplied daily
NVDA and NVDL total returns in `../data/` and the fund factsheet in `../docs/`.

## Contents

| Path | Purpose |
| --- | --- |
| `analysis.ipynb` | Consolidated Parts 4A-4F walkthrough |
| `data.py` | Workbook loading and validation |
| `risk.py` | Daily volatility, historical 95% VaR and CVaR |
| `regression.py` | OLS leverage regression and 95% confidence intervals |
| `performance.py` | Monthly/full-period compounding, 4E example and 4F suitability |
| `analysis.py` | End-to-end orchestration, interpretation and exports |
| `tests/` | Unit and integration tests |
| `outputs/` | Generated local results, ignored by Git except for `.gitkeep` |

## Method choices

- Returns remain in decimal form, as supplied.
- Daily standard deviation is the sample standard deviation and is not annualized.
- Historical 95% VaR is reported as a positive loss using the empirical 5th
  percentile. Historical CVaR is the positive average loss among observations
  at or below that cutoff.
- The regression includes an intercept: `NVDL = alpha + beta * NVDA + error`.
- The reported 95% coefficient intervals use the Student-t critical value with
  145 residual degrees of freedom.
- Monthly and full-period returns compound the daily observations rather than
  summing them.

## Run

From the repository root:

```bash
uv sync
uv run python -m part4.data
uv run python -m part4.analysis
uv run python -m unittest discover -s part4/tests -v
uv run jupyter lab part4/analysis.ipynb
```

The analysis command writes auditable CSV, JSON and Markdown results to
`part4/outputs/`. Those generated files should not be committed.

## Source scope

- Return sample: 147 trading days, 1 February-30 August 2024.
- Fund objective: 2 times NVDA's daily percentage change before fees and expenses.
- Factsheet net annual operating expense ratio: 1.15% as of 30 June 2024.
- Before 22 January 2024 the fund targeted 1.5 times daily exposure; this does
  not affect the supplied February-August 2024 sample.
