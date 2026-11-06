from __future__ import annotations

import unittest

import pandas as pd

from part4.data import (
    DataValidationError,
    default_workbook_path,
    load_daily_returns,
    validate_daily_returns,
)


class DailyReturnDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns = load_daily_returns()

    def test_default_workbook_exists(self) -> None:
        self.assertTrue(default_workbook_path().is_file())

    def test_loader_returns_expected_panel(self) -> None:
        self.assertEqual(self.returns.shape, (147, 2))
        self.assertEqual(self.returns.columns.tolist(), ["NVDA", "NVDL"])
        self.assertEqual(self.returns.index[0], pd.Timestamp("2024-02-01"))
        self.assertEqual(self.returns.index[-1], pd.Timestamp("2024-08-30"))
        self.assertTrue(self.returns.index.is_monotonic_increasing)
        self.assertFalse(self.returns.isna().any().any())

    def test_returns_are_decimal_not_percentage_points(self) -> None:
        self.assertLess(float(self.returns.abs().max().max()), 1.0)
        self.assertGreater(float(self.returns.abs().max().max()), 0.10)

    def test_validation_rejects_missing_observation(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "147"):
            validate_daily_returns(self.returns.iloc[:-1])

    def test_validation_rejects_invalid_return_unit(self) -> None:
        invalid = self.returns.copy()
        invalid.iloc[0, 0] = 1.5
        with self.assertRaisesRegex(DataValidationError, "incorrect unit"):
            validate_daily_returns(invalid)


if __name__ == "__main__":
    unittest.main()
