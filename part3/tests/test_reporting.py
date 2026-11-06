from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from part3.data import load_phase1_data
from part3.optimization import (
    PORTFOLIO_NAMES,
    build_phase2_portfolios,
)
from part3.reporting import (
    DISPLAY_NAMES,
    export_part3a,
    weight_extremes,
    weight_table,
)


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        phase1 = load_phase1_data()
        cls.portfolios = build_phase2_portfolios(phase1.windows["seed_1"])

    def test_weight_table_has_total_row(self) -> None:
        table = weight_table(self.portfolios)
        self.assertEqual(table.shape, (50, 3))
        np.testing.assert_allclose(table.loc["Total"].to_numpy(), 1.0, atol=1e-8)

    def test_extremes_select_exactly_five_disjoint_industries(self) -> None:
        extremes = weight_extremes(self.portfolios.weights)
        self.assertEqual(set(extremes), set(PORTFOLIO_NAMES))
        for selections in extremes.values():
            self.assertEqual(len(selections["top"]), 5)
            self.assertEqual(len(selections["bottom"]), 5)
            self.assertTrue(set(selections["top"]).isdisjoint(selections["bottom"]))

    def test_export_writes_all_reproducible_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = export_part3a(self.portfolios, temporary_directory)
            self.assertEqual(
                set(outputs), {"weights", "diagnostics", "methodology", "payload"}
            )
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))
            diagnostics = Path(outputs["diagnostics"]).read_text(encoding="utf-8")
            self.assertIn(DISPLAY_NAMES[PORTFOLIO_NAMES[0]], diagnostics)


if __name__ == "__main__":
    unittest.main()
