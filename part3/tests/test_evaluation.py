from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from part3.data import load_phase1_data
from part3.evaluation import (
    PortfolioEvaluationError,
    ORIGINAL_LONG_ONLY,
    UPDATED_LONG_ONLY,
    build_phase3b_results,
    build_phase3c_results,
    build_phase3d_results,
    export_phase3b,
    export_phase3c,
    export_phase3d,
    portfolio_return_series,
)
from part3.optimization import build_phase2_portfolios
from part3.reporting import DISPLAY_NAMES


class Phase3BEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_phase1_data()
        cls.portfolios = build_phase2_portfolios(cls.phase1.windows["seed_1"])
        cls.results = build_phase3b_results(
            cls.phase1.windows["seed_1"], cls.portfolios
        )

    def test_realized_volatility_matches_covariance_diagnostics(self) -> None:
        expected = self.portfolios.diagnostics["annualized_volatility"].rename(
            index=DISPLAY_NAMES
        )
        actual = self.results.comparison["annualized_volatility"].reindex(expected.index)
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-12)

    def test_in_sample_order_is_expected(self) -> None:
        self.assertEqual(
            self.results.comparison.index.tolist(),
            [
                "Unconstrained minimum variance",
                "Long-only minimum variance",
                "Inverse volatility",
            ],
        )
        self.assertEqual(self.results.comparison["rank_low_to_high"].tolist(), [1, 2, 3])

    def test_monthly_returns_cover_exact_seed_period(self) -> None:
        returns = self.results.monthly_returns
        self.assertEqual(returns.shape, (120, 3))
        self.assertEqual(returns.index[0], pd.Timestamp("1995-01-31"))
        self.assertEqual(returns.index[-1], pd.Timestamp("2004-12-31"))

    def test_asset_mismatch_is_rejected(self) -> None:
        weights = self.portfolios.weights.drop(index=self.portfolios.weights.index[-1])
        with self.assertRaisesRegex(PortfolioEvaluationError, "same assets"):
            portfolio_return_series(self.phase1.windows["seed_1"], weights)

    def test_export_writes_all_phase3b_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = export_phase3b(self.results, temporary_directory)
            self.assertEqual(
                set(outputs),
                {"comparison", "monthly_returns", "interpretation", "payload"},
            )
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))


class Phase3CEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_phase1_data()
        cls.portfolios = build_phase2_portfolios(cls.phase1.windows["seed_1"])
        cls.results = build_phase3c_results(
            cls.phase1.windows["oos_1"], cls.portfolios
        )

    def test_monthly_returns_cover_exact_out_of_sample_period(self) -> None:
        returns = self.results.monthly_returns
        self.assertEqual(returns.shape, (223, 3))
        self.assertEqual(returns.index[0], pd.Timestamp("2005-01-31"))
        self.assertEqual(returns.index[-1], pd.Timestamp("2023-07-31"))

    def test_volatility_is_calculated_from_out_of_sample_monthly_returns(self) -> None:
        expected = self.results.monthly_returns.std(ddof=1) * np.sqrt(12.0)
        actual = self.results.comparison["annualized_volatility"].reindex(expected.index)
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-12)

    def test_ranks_are_complete_and_ordered(self) -> None:
        self.assertEqual(self.results.comparison["rank_low_to_high"].tolist(), [1, 2, 3])
        self.assertTrue(
            self.results.comparison["annualized_volatility"].is_monotonic_increasing
        )

    def test_export_writes_all_phase3c_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = export_phase3c(self.results, temporary_directory)
            self.assertEqual(
                set(outputs),
                {"comparison", "monthly_returns", "interpretation", "payload"},
            )
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))


class Phase3DReoptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_phase1_data()
        cls.results = build_phase3d_results(
            cls.phase1.windows["seed_1"],
            cls.phase1.windows["seed_2"],
            cls.phase1.windows["reopt_test"],
        )

    def test_both_weight_vectors_are_long_only_and_fully_invested(self) -> None:
        np.testing.assert_allclose(self.results.weights.sum(axis=0), 1.0, atol=1e-8)
        self.assertGreaterEqual(float(self.results.weights.min().min()), 0.0)
        self.assertEqual(
            self.results.weights.columns.tolist(),
            [ORIGINAL_LONG_ONLY, UPDATED_LONG_ONLY],
        )

    def test_monthly_returns_cover_exact_reoptimization_test_period(self) -> None:
        returns = self.results.monthly_returns
        self.assertEqual(returns.shape, (91, 2))
        self.assertEqual(returns.index[0], pd.Timestamp("2016-01-31"))
        self.assertEqual(returns.index[-1], pd.Timestamp("2023-07-31"))

    def test_comparison_matches_realized_monthly_returns(self) -> None:
        expected = self.results.monthly_returns.std(ddof=1) * np.sqrt(12.0)
        actual = self.results.comparison["annualized_volatility"]
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-12)

    def test_turnover_is_one_way_weight_change(self) -> None:
        expected = 0.5 * (
            self.results.weights[UPDATED_LONG_ONLY]
            - self.results.weights[ORIGINAL_LONG_ONLY]
        ).abs().sum()
        self.assertAlmostEqual(self.results.turnover, float(expected))
        self.assertGreaterEqual(self.results.turnover, 0.0)
        self.assertLessEqual(self.results.turnover, 1.0)

    def test_export_writes_all_phase3d_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = export_phase3d(self.results, temporary_directory)
            self.assertEqual(
                set(outputs),
                {"comparison", "weights", "monthly_returns", "interpretation", "payload"},
            )
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))


if __name__ == "__main__":
    unittest.main()
