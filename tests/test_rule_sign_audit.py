"""Локальные тесты расчёта признаков (без необходимости загружать Petrobras)."""
import unittest

import numpy as np
import pandas as pd

from experiments.audit_class2_rule_signals import audit_window, slope_pa_per_hour, summarize


class TestClass2RuleSignAudit(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2020-01-01", periods=300, freq="s")
        x = np.arange(300, dtype=float)
        self.df = pd.DataFrame({
            "P-PDG": x * 5,
            "P-TPT": 10000 - x * 4,
            "P-MON-CKP": 12000 - x * 2,
        }, index=self.times)

    def test_expected_direction(self):
        result = audit_window(self.df)
        self.assertTrue(result["three_sensors_usable"])
        self.assertTrue(result["sign_pattern_matches"])
        self.assertAlmostEqual(result["pdg_slope_pa_h"], 18000.0)

    def test_opposite_direction(self):
        df = self.df.copy()
        df["P-PDG"] = -df["P-PDG"]
        result = audit_window(df)
        self.assertFalse(result["sign_pattern_matches"])

    def test_missing_sensor_means_no_match_not_false_negative(self):
        df = self.df.drop(columns=["P-PDG"])
        result = audit_window(df)
        self.assertFalse(result["three_sensors_usable"])
        self.assertIsNone(result["sign_pattern_matches"])

    def test_microsecond_resolution_is_accepted(self):
        # В pandas 3 временной индекс может иметь разрешение datetime64[us].
        # Проверка шага должна одинаково работать для us и ns.
        df = self.df.copy()
        df.index = df.index.as_unit("us")
        result = audit_window(df)
        self.assertTrue(result["three_sensors_usable"])
        self.assertTrue(result["sign_pattern_matches"])

    def test_non_consecutive_time_is_rejected(self):
        df = self.df.copy()
        times = list(df.index)
        times[-1] += pd.Timedelta(seconds=1)
        df.index = pd.DatetimeIndex(times)
        with self.assertRaises(ValueError):
            audit_window(df)

    def test_summary_has_plain_phase_labels(self):
        windows = pd.DataFrame([
            {"phase": "normal_0", "three_sensors_usable": True, "sign_pattern_matches": False},
            {"phase": "transition_102", "three_sensors_usable": True, "sign_pattern_matches": True},
        ])
        summary = summarize(windows, ["phase"])
        self.assertEqual(set(summary["phase"]), {"normal_0", "transition_102"})

    def test_insufficient_coverage_is_nan(self):
        data = np.arange(300, dtype=float)
        data[:31] = np.nan
        slope, coverage = slope_pa_per_hour(data)
        self.assertTrue(np.isnan(slope))
        self.assertAlmostEqual(coverage, 269 / 300)


if __name__ == "__main__":
    unittest.main()
