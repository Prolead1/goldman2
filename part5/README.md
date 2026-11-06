# Part 5 - Portfolio analytics

This self-contained folder adds **only Part 5** of FE5108. It reuses the existing `goldman2.data` loaders and the supplied FF/MSCI workbook. The assignment has **six** subsections, 5A-5F (there is no 5G/5H).

## Run (from repository root)

```bash
python -m pip install -e .
python -m pip install reportlab pytest
python -m part5.analysis
python -m part5.reporting
python -m pytest part5/tests -q
```

Results are written to `part5/outputs/`, including the PDF, figures, monthly returns and JSON. The manuscript source is `part5/report.md` and can be pasted into the team's shared Google Doc. This folder can be committed without changing teammates' modules.

## Reproducibility / interpretation

- Sample: Jan 1995-Jul 2023 (343 monthly observations); FF returns and MSCI index returns in monthly **percentage points**.
- OLS: MSCI Value total return on constant, RF, Mkt-RF, HML, SMB; conventional unadjusted t-statistics.
- Factor replication omits the fitted intercept and is evaluated **in-sample**, not an investable backtest.
- Annualized arithmetic return and active return = 12 times monthly mean; volatility and tracking error = sqrt(12) times monthly sample standard deviation.
- The four-factor MSCI blend is rebalanced to equal weights **monthly**. Risk contribution is `Cov(active_i, active_blend)/(4 Var(active_blend))`.
- The report discusses limitations including transaction costs, factor shorting, index methodology and estimation look-ahead.
