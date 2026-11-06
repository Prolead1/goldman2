from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from part4.analysis import build_part4_results, export_part4, interpretation_markdown


class CompletePart4AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = build_part4_results()

    def test_all_assignment_tables_are_present(self) -> None:
        self.assertEqual(self.results.risk_metrics.shape[0], 2)
        self.assertEqual(self.results.regression.coefficients.shape[0], 2)
        self.assertEqual(self.results.monthly_returns.shape[0], 7)
        self.assertEqual(self.results.full_period_returns.shape[0], 1)
        self.assertEqual(self.results.two_day_example.shape[0], 2)
        self.assertEqual(self.results.suitability.shape[0], 2)

    def test_interpretation_covers_parts_4a_through_4f(self) -> None:
        text = interpretation_markdown(self.results)
        for label in ["4A", "4B", "4C", "4D", "4E", "4F"]:
            self.assertIn(label, text)

    def test_export_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = export_part4(self.results, temporary_directory)
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(payload["regression"]["observations"], 147)


if __name__ == "__main__":
    unittest.main()
