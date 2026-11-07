# Part 5 - Portfolio analytics

This directory contains FE5108 Part 5A-5F analysis, manuscript, charts and a generated PDF. The code reuses the repository `goldman2.data` loaders and the supplied FF/MSCI workbook. The factor replication is in-sample and excludes trading costs.

## Reproduce from repository root

```bash
uv sync
uv run python -m part5.analysis
uv run --with reportlab python -m part5.reporting
uv run --with pytest python -m pytest part5/tests -q
```

Without `uv`, create a virtual environment and install `pip install -e . reportlab pytest` before running the same modules with `python`.

The executed notebook lives at `notebooks/part5-portfolio-analytics.ipynb`. Open it with `uv run jupyter lab` if Jupyter is available, or install it with `uv run --with jupyterlab jupyter lab`. The report manuscript is `part5/report.md` and the rendered PDF is `part5/outputs/Part_5_Portfolio_Analytics_Report.pdf`.

The manuscript uses CAGR-based Sharpe ratios to match Part 1, but arithmetic monthly active returns for information ratios. Class lecture notes were not supplied; check the class-specific IR benchmark and limits-to-arbitrage framing before final submission.
