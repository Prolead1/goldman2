import unittest

import numpy as np
import pandas as pd

from part3.data import default_msci_workbook_path, default_workbook_path, load_msci_usa_returns
from part3.performance import (
    BENCHMARK_LABEL,
    LONG_ONLY_LABEL,
    build_phase3e_from_workbooks,
    cumulative_wealth,
    export_phase3e,
)


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.industry_path = default_workbook_path()
        cls.msci_path = default_msci_workbook_path()
        cls.results = build_phase3e_from_workbooks(cls.industry_path, cls.msci_path)

    def test_msci_loader_has_required_continuous_history(self):
        returns = load_msci_usa_returns(self.msci_path)
        self.assertEqual(returns.name, BENCHMARK_LABEL)
        self.assertEqual(len(returns), 343)
        self.assertEqual(returns.index[0], pd.Timestamp("1995-01-31"))
        self.assertEqual(returns.index[-1], pd.Timestamp("2023-07-31"))
        self.assertFalse(returns.isna().any())

    def test_monthly_comparison_uses_exact_out_of_sample_window(self):
        monthly = self.results.monthly_returns
        self.assertEqual(monthly.shape, (223, 2))
        self.assertEqual(list(monthly.columns), [LONG_ONLY_LABEL, BENCHMARK_LABEL])
        self.assertEqual(monthly.index[0], pd.Timestamp("2005-01-31"))
        self.assertEqual(monthly.index[-1], pd.Timestamp("2023-07-31"))

    def test_cumulative_wealth_starts_at_one_and_compounds_returns(self):
        wealth = self.results.cumulative_wealth
        self.assertEqual(wealth.shape, (224, 2))
        self.assertEqual(wealth.index[0], pd.Timestamp("2004-12-31"))
        np.testing.assert_allclose(wealth.iloc[0].to_numpy(), np.ones(2))
        expected = (1.0 + self.results.monthly_returns).prod(axis="index")
        np.testing.assert_allclose(wealth.iloc[-1].to_numpy(), expected.to_numpy())

    def test_crisis_returns_match_direct_compounding(self):
        windows = {
            "Global Financial Crisis": ("2008-09-30", "2009-02-28", 6),
            "COVID-19 shock": ("2020-02-29", "2020-03-31", 2),
        }
        for label, (start, end, months) in windows.items():
            period = self.results.monthly_returns.loc[start:end]
            self.assertEqual(len(period), months)
            expected = (1.0 + period).prod(axis="index") - 1.0
            actual = self.results.crisis_summary.loc[label]
            self.assertAlmostEqual(actual["Strategy return"], expected[LONG_ONLY_LABEL])
            self.assertAlmostEqual(actual["MSCI USA return"], expected[BENCHMARK_LABEL])

    def test_top_industry_tables_are_complete_and_descending(self):
        for frame in (self.results.gfc_top10, self.results.covid_top10):
            self.assertEqual(len(frame), 10)
            self.assertTrue(frame["Cumulative return"].is_monotonic_decreasing)
            self.assertFalse(frame.isna().any().any())

    def test_cumulative_wealth_rejects_impossible_returns(self):
        returns = pd.DataFrame(
            {"Portfolio": [-1.0]}, index=pd.DatetimeIndex(["2005-01-31"], name="Date")
        )
        with self.assertRaises(ValueError):
            cumulative_wealth(returns)

    def test_export_writes_all_expected_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = export_phase3e(self.results, temp_dir)
            self.assertEqual(len(outputs), 8)
            self.assertTrue(all(path.is_file() for path in outputs.values()))


if __name__ == "__main__":
    unittest.main()
