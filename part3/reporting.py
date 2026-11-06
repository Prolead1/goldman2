"""Create the reproducible Part 3A weight-table and methodology deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import WORKBOOK_NAME
from .optimization import (
    INVERSE_VOLATILITY,
    LONG_ONLY,
    UNCONSTRAINED,
    Phase2Portfolios,
    build_phase2_from_workbook,
)

DISPLAY_NAMES = {
    UNCONSTRAINED: "Unconstrained minimum variance",
    LONG_ONLY: "Long-only minimum variance",
    INVERSE_VOLATILITY: "Inverse volatility",
}
EXTREME_COUNT = 5


def weight_table(portfolios: Phase2Portfolios) -> pd.DataFrame:
    """Return presentation-ready weights with a final portfolio-total row."""
    table = portfolios.weights.rename(columns=DISPLAY_NAMES).copy()
    table.index.name = "Industry"
    totals = pd.DataFrame([table.sum()], index=pd.Index(["Total"], name="Industry"))
    return pd.concat([table, totals])


def weight_extremes(
    weights: pd.DataFrame,
    count: int = EXTREME_COUNT,
) -> dict[str, dict[str, list[str]]]:
    """Select exactly ``count`` largest and smallest industries per portfolio.

    Stable sorting makes ties deterministic: tied values retain workbook order.
    The top set is selected first and excluded from the bottom set, ensuring
    that one cell cannot receive both colours.
    """
    if count < 1 or 2 * count > len(weights):
        raise ValueError("Extreme count must be positive and fit twice in the universe.")

    result: dict[str, dict[str, list[str]]] = {}
    for column in weights.columns:
        ranked_high = weights[column].sort_values(ascending=False, kind="stable")
        top = ranked_high.index[:count].tolist()
        eligible_bottom = weights.loc[~weights.index.isin(top), column]
        bottom = (
            eligible_bottom.sort_values(ascending=True, kind="stable")
            .index[:count]
            .tolist()
        )
        result[column] = {"top": top, "bottom": bottom}
    return result


def methodology_markdown() -> str:
    """Return concise report text describing the Part 3A construction methods."""
    return """# Part 3A portfolio-construction methodology

All three portfolios use monthly value-weighted total returns for the 49 Fama–French
industries from January 1995 through December 2004 (120 observations). Returns are
converted from percentage points to decimals before estimation. The sample covariance
matrix uses the usual `T - 1` denominator; expected returns are not used.

## Unconstrained global minimum variance

Minimize `w'Σw` subject to `1'w = 1`, allowing positive and negative weights. The
closed-form solution is `w = Σ⁻¹1 / (1'Σ⁻¹1)`.

## Long-only global minimum variance

Minimize `w'Σw` subject to `1'w = 1` and `w ≥ 0`. The implementation uses projected
gradient descent on this convex problem, projects every step onto the probability
simplex, and verifies the first-order optimality conditions after convergence.

## Inverse volatility

Set each raw weight to the reciprocal of that industry's sample volatility and
normalize the vector to sum to one: `wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)`.

In the weight exhibit, green marks the five largest weights and red marks the five
smallest weights within each portfolio. Ties are resolved in the original workbook
industry order so that exactly five cells receive each colour.
"""


def build_reporting_payload(portfolios: Phase2Portfolios) -> dict[str, object]:
    """Create the serializable payload used by the formatted workbook builder."""
    weights = portfolios.weights
    diagnostics = portfolios.diagnostics.rename(index=DISPLAY_NAMES)
    return {
        "title": "Part 3A — Industry portfolio weights",
        "subtitle": "Estimated from monthly returns, January 1995–December 2004",
        "source": f"Source: {WORKBOOK_NAME}, 49_Industry_Portfolios worksheet",
        "weights": [
            {
                "industry": industry,
                **{DISPLAY_NAMES[name]: float(value) for name, value in row.items()},
            }
            for industry, row in weights.iterrows()
        ],
        "totals": {
            DISPLAY_NAMES[name]: float(value) for name, value in weights.sum().items()
        },
        "extremes": {
            DISPLAY_NAMES[name]: selections
            for name, selections in weight_extremes(weights).items()
        },
        "diagnostics": [
            {"portfolio": name, **{key: float(value) for key, value in row.items()}}
            for name, row in diagnostics.iterrows()
        ],
    }


def export_part3a(
    portfolios: Phase2Portfolios,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write reproducible CSV, JSON, and Markdown inputs for the final exhibit."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / "part3a"
    )
    destination.mkdir(parents=True, exist_ok=True)

    weights_path = destination / "weights.csv"
    diagnostics_path = destination / "diagnostics.csv"
    methodology_path = destination / "methodology.md"
    payload_path = destination / "workbook_payload.json"

    weight_table(portfolios).to_csv(weights_path)
    portfolios.diagnostics.rename(index=DISPLAY_NAMES).to_csv(
        diagnostics_path,
        index_label="Portfolio",
    )
    methodology_path.write_text(methodology_markdown(), encoding="utf-8")
    payload_path.write_text(
        json.dumps(build_reporting_payload(portfolios), indent=2),
        encoding="utf-8",
    )
    return {
        "weights": weights_path,
        "diagnostics": diagnostics_path,
        "methodology": methodology_path,
        "payload": payload_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Part 3A deliverables.")
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Estimate portfolios and export all reproducible Part 3A inputs."""
    args = _build_parser().parse_args()
    outputs = export_part3a(
        build_phase2_from_workbook(args.workbook),
        output_dir=args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
