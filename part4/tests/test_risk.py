from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from part4.data import load_daily_returns
from part4.risk import RiskMetricError, build_risk_table, historical_var_cvar


class RiskMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = load_daily_returns()
        cls.metrics = build_risk_table(cls.returns)

    def test_metrics_match_expected_values(self) -> None:
        expected = {
            "NVDA": (0.0374600398544, 0.0612099307130, 0.0707251566807),
            "NVDL": (0.0747948679550, 0.1202879688740, 0.1416172928560),
        }
        for asset, values in expected.items():
            actual = self.metrics.loc[
                asset, ["Daily standard deviation", "95% VaR", "95% CVaR"]
            ].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, values, rtol=0.0, atol=1e-12)

    def test_leverage_approximately_doubles_each_risk_measure(self) -> None:
        for column in ["Daily standard deviation", "95% VaR", "95% CVaR"]:
            ratio = self.metrics.loc["NVDL", column] / self.metrics.loc["NVDA", column]
            self.assertGreater(ratio, 1.9)
            self.assertLess(ratio, 2.1)

    def test_tail_contains_eight_observations(self) -> None:
        self.assertEqual(self.metrics["Tail observations"].tolist(), [8, 8])

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(RiskMetricError):
            historical_var_cvar(pd.Series([0.01, -0.02]), confidence=1.0)


if __name__ == "__main__":
    unittest.main()
