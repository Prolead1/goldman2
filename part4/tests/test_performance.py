from __future__ import annotations

import unittest

import numpy as np

from part4.data import load_daily_returns
from part4.performance import (
    full_period_total_returns,
    investor_suitability,
    monthly_total_returns,
    two_day_compounding_example,
)


class MultiDayPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = load_daily_returns()

    def test_monthly_table_covers_february_through_august(self) -> None:
        monthly = monthly_total_returns(self.returns)
        self.assertEqual(len(monthly), 7)
        self.assertEqual(str(monthly.index[0]), "2024-02")
        self.assertEqual(str(monthly.index[-1]), "2024-08")

    def test_full_period_returns_match_direct_compounding(self) -> None:
        expected = (1.0 + self.returns).prod(axis="index") - 1.0
        actual = full_period_total_returns(self.returns).iloc[0]
        np.testing.assert_allclose(
            actual[["NVDA", "NVDL"]].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            atol=1e-12,
        )
        self.assertAlmostEqual(actual["NVDL minus twice NVDA"], -0.0656622697159)
        self.assertAlmostEqual(actual["Ideal daily 2x NVDA"], 2.0837224672, places=9)
        self.assertAlmostEqual(
            actual["Compounding effect vs twice NVDA"], 0.2027275071, places=9
        )
        self.assertAlmostEqual(
            actual["NVDL minus ideal daily 2x"], -0.2683897768, places=9
        )
        self.assertAlmostEqual(
            actual["Compounding effect vs twice NVDA"]
            + actual["NVDL minus ideal daily 2x"],
            actual["NVDL minus twice NVDA"],
        )

    def test_two_day_example_shows_volatility_drag(self) -> None:
        scenario = two_day_compounding_example()
        self.assertAlmostEqual(scenario.loc["NVDA", "Two-day return"], 0.001)
        self.assertAlmostEqual(
            scenario.loc["NVDL (ideal 2x, no drag)", "Two-day return"], -0.016
        )

    def test_suitability_distinguishes_trading_and_investing(self) -> None:
        suitability = investor_suitability()
        self.assertEqual(
            suitability["Preferred instrument"].tolist(), ["NVDL", "NVDA"]
        )


if __name__ == "__main__":
    unittest.main()
