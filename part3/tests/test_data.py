from __future__ import annotations

import math
import unittest

import pandas as pd

from part3.data import (
    EXPECTED_INDUSTRIES,
    DataValidationError,
    default_workbook_path,
    load_industry_returns,
    load_phase1_data,
    slice_window,
    validate_returns,
)


class IndustryReturnDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = load_industry_returns()

    def test_default_workbook_exists(self) -> None:
        self.assertTrue(default_workbook_path().is_file())

    def test_loader_returns_expected_clean_panel(self) -> None:
        returns = self.returns
        self.assertEqual(returns.shape, (343, EXPECTED_INDUSTRIES))
        self.assertEqual(returns.index[0], pd.Timestamp("1995-01-31"))
        self.assertEqual(returns.index[-1], pd.Timestamp("2023-07-31"))
        self.assertFalse(returns.isna().any().any())
        self.assertTrue(math.isclose(float(returns.min().min()), -0.4099))
        self.assertTrue(math.isclose(float(returns.max().max()), 0.6898))

    def test_industry_names_are_normalized(self) -> None:
        returns = self.returns
        self.assertEqual(returns.columns[0], "Agric")
        self.assertEqual(returns.columns[-1], "Other")
        self.assertTrue(all(name == name.strip() for name in returns.columns))
        self.assertTrue(returns.columns.is_unique)

    def test_required_windows_have_exact_month_counts(self) -> None:
        phase1 = load_phase1_data()
        self.assertEqual(
            {name: len(frame) for name, frame in phase1.windows.items()},
            {
                "seed_1": 120,
                "oos_1": 223,
                "seed_2": 120,
                "reopt_test": 91,
                "gfc": 6,
                "covid": 2,
            },
        )

    def test_validation_rejects_a_missing_month(self) -> None:
        incomplete = self.returns.drop(pd.Timestamp("2000-06-30"))
        with self.assertRaisesRegex(DataValidationError, "Missing months: 2000-06"):
            validate_returns(incomplete)

    def test_validation_rejects_missing_return(self) -> None:
        incomplete = self.returns.copy()
        incomplete.iloc[0, 0] = float("nan")
        with self.assertRaisesRegex(DataValidationError, "missing returns"):
            validate_returns(incomplete)

    def test_slice_window_is_inclusive_and_continuous(self) -> None:
        crisis = slice_window(self.returns, "2008-09", "2009-02")
        self.assertEqual(len(crisis), 6)
        self.assertEqual(crisis.index[0], pd.Timestamp("2008-09-30"))
        self.assertEqual(crisis.index[-1], pd.Timestamp("2009-02-28"))


if __name__ == "__main__":
    unittest.main()
