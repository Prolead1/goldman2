from __future__ import annotations

import unittest

import numpy as np

from part4.data import load_daily_returns
from part4.regression import fit_daily_leverage_regression


class DailyLeverageRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        returns = load_daily_returns()
        cls.results = fit_daily_leverage_regression(returns["NVDA"], returns["NVDL"])

    def test_sample_and_fit_diagnostics(self) -> None:
        self.assertEqual(self.results.observations, 147)
        self.assertEqual(self.results.residual_degrees_of_freedom, 145)
        self.assertGreater(self.results.r_squared, 0.999)

    def test_coefficient_estimates_match_direct_ols(self) -> None:
        estimates = self.results.coefficients["Estimate"]
        self.assertAlmostEqual(estimates.loc["Intercept"], -0.0006116439012)
        self.assertAlmostEqual(estimates.loc["NVDA beta"], 1.9959749225727)

    def test_beta_interval_contains_stated_two_times_target(self) -> None:
        beta = self.results.coefficients.loc["NVDA beta"]
        self.assertLess(beta["95% CI lower"], 2.0)
        self.assertGreater(beta["95% CI upper"], 2.0)

    def test_intercept_interval_is_negative(self) -> None:
        intercept = self.results.coefficients.loc["Intercept"]
        self.assertLess(intercept["95% CI upper"], 0.0)
        self.assertAlmostEqual(self.results.annualized_intercept, -0.1541342631033)

    def test_standard_errors_are_positive(self) -> None:
        self.assertTrue(
            np.all(self.results.coefficients["Standard error"].to_numpy() > 0.0)
        )


if __name__ == "__main__":
    unittest.main()
