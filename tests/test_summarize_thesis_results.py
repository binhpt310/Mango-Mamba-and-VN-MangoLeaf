"""Verify reported four-seed statistics and reject incomplete result sets."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summary', ROOT / 'code/summarize_thesis_results.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_retained_four_seed_statistics(self):
        rows = {row['case']: row for row in summary.summarize(ROOT / 'experiments')}
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row['n'] == 4 for row in rows.values()))
        self.assertAlmostEqual(rows['01_phase1_mangoleafbd_320']['test_accuracy_mean_pct'], 99.6875)
        self.assertAlmostEqual(rows['01_phase1_mangoleafbd_320']['test_accuracy_sample_sd_pct'], 0.2393567769)
        self.assertAlmostEqual(rows['04_phase3_vn_mangoleaf_320']['test_accuracy_mean_pct'], 98.8928571429)
        self.assertAlmostEqual(rows['05_direct_vn_from_phase1_320']['test_accuracy_mean_pct'], 98.1071428571)

    def test_missing_seed_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                summary.summarize(Path(directory))


if __name__ == '__main__':
    unittest.main()
