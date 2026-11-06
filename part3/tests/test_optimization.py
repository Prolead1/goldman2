from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from part3.data import load_phase1_data
from part3.optimization import (
    INVERSE_VOLATILITY,
    LONG_ONLY,
    UNCONSTRAINED,
    build_phase2_portfolios,
    estimate_covariance,
    inverse_volatility_weights,
    long_only_minimum_variance,
    portfolio_variance,
    unconstrained_minimum_variance,
)


class PortfolioFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagonal_covariance = pd.DataFrame(
            [[1.0, 0.0], [0.0, 4.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )

    def test_unconstrained_solution_on_diagonal_covariance(self) -> None:
        weights = unconstrained_minimum_variance(self.diagonal_covariance)
        np.testing.assert_allclose(weights.to_numpy(), [0.8, 0.2], atol=1e-10)

    def test_long_only_solution_on_diagonal_covariance(self) -> None:
        weights = long_only_minimum_variance(self.diagonal_covariance)
        np.testing.assert_allclose(weights.to_numpy(), [0.8, 0.2], atol=1e-7)

    def test_inverse_volatility_uses_standard_deviation_not_variance(self) -> None:
        weights = inverse_volatility_weights(self.diagonal_covariance)
        np.testing.assert_allclose(weights.to_numpy(), [2.0 / 3.0, 1.0 / 3.0])

    def test_portfolio_variance_aligns_labels(self) -> None:
        weights = pd.Series({"B": 0.2, "A": 0.8})
        self.assertTrue(
            math.isclose(portfolio_variance(weights, self.diagonal_covariance), 0.8)
        )


class RealIndustryPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_phase1_data()
        cls.phase2 = build_phase2_portfolios(cls.phase1.windows["seed_1"])

    def test_covariance_uses_only_first_seed_window(self) -> None:
        expected = estimate_covariance(self.phase1.windows["seed_1"])
        pd.testing.assert_frame_equal(self.phase2.covariance, expected)
        self.assertEqual(len(self.phase1.windows["seed_1"]), 120)

    def test_all_weight_vectors_are_fully_invested(self) -> None:
        np.testing.assert_allclose(self.phase2.weights.sum(axis=0), 1.0, atol=1e-8)

    def test_long_only_and_inverse_volatility_are_nonnegative(self) -> None:
        self.assertGreaterEqual(float(self.phase2.weights[LONG_ONLY].min()), 0.0)
        self.assertGreaterEqual(float(self.phase2.weights[INVERSE_VOLATILITY].min()), 0.0)

    def test_unconstrained_variance_is_a_lower_bound(self) -> None:
        covariance = self.phase2.covariance
        variances = {
            name: portfolio_variance(self.phase2.weights[name], covariance)
            for name in self.phase2.weights
        }
        self.assertLessEqual(variances[UNCONSTRAINED], variances[LONG_ONLY] + 1e-12)
        self.assertLessEqual(
            variances[UNCONSTRAINED], variances[INVERSE_VOLATILITY] + 1e-12
        )

    def test_gmv_portfolios_beat_equal_weight_in_sample(self) -> None:
        covariance = self.phase2.covariance
        equal = pd.Series(1.0 / len(covariance), index=covariance.index)
        equal_variance = portfolio_variance(equal, covariance)
        self.assertLessEqual(
            portfolio_variance(self.phase2.weights[UNCONSTRAINED], covariance),
            equal_variance + 1e-12,
        )
        self.assertLessEqual(
            portfolio_variance(self.phase2.weights[LONG_ONLY], covariance),
            equal_variance + 1e-12,
        )

    def test_diagnostics_cover_all_three_portfolios(self) -> None:
        self.assertEqual(
            list(self.phase2.diagnostics.index),
            [UNCONSTRAINED, LONG_ONLY, INVERSE_VOLATILITY],
        )
        self.assertTrue((self.phase2.diagnostics["monthly_variance"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
