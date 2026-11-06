# Part 3 - Portfolio construction

This directory contains the complete, reproducible analysis for FE5108 Project 2,
Part 3. It uses the repository's supplied Fama-French 49-industry workbook and
MSCI USA benchmark workbook from `../data/`.

## Contents

| Path | Purpose |
| --- | --- |
| `analysis.ipynb` | Consolidated Parts 3A-3E analysis and report walkthrough |
| `data.py` | Data loading, missing-value controls and assignment date windows |
| `optimization.py` | Unconstrained GMV, long-only GMV and inverse-volatility portfolios |
| `reporting.py` | Part 3A weight table and diagnostics |
| `evaluation.py` | Parts 3B-3D realized-volatility analysis |
| `performance.py` | Part 3E cumulative performance and crisis analysis |
| `tests/` | Unit and integration tests for the complete workflow |
| `outputs/` | Generated local artifacts; ignored by Git except for `.gitkeep` |

The loaders explicitly:

- convert source percentage returns to decimal returns;
- treat `-99.99` and `-999` as missing-value sentinels;
- require exactly 49 industries and continuous monthly observations;
- enforce the January 1995-July 2023 assignment cutoff; and
- validate every estimation, evaluation and crisis window.

## Run

From the repository root:

```bash
uv sync
uv run python -m part3.data
uv run python -m part3.optimization
uv run python -m part3.reporting
uv run python -m part3.evaluation
uv run python -m part3.evaluation --phase 3c
uv run python -m part3.evaluation --phase 3d
uv run python -m part3.performance
```

Open the consolidated notebook with:

```bash
uv run jupyter lab part3/analysis.ipynb
```

Run the test suite with:

```bash
uv run python -m unittest discover -s part3/tests -v
```

Alternative workbooks can be passed with command-line options where supported,
or through `FE5108_PART3_WORKBOOK` and `FE5108_MSCI_WORKBOOK`.

## Git policy

Commit the analysis modules, tests, notebook and documentation. Do not commit
generated files under `outputs/`, Python/Jupyter caches, temporary Office lock
files, or local environments. The source workbooks already live in the
repository's shared `data/` directory and must not be copied into this folder.
