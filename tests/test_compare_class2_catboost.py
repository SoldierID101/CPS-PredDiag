"""Checks for the research comparison; does not require a real Petrobras dataset."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from compare_class2_catboost import (
    read_points, prepare_frame, past_features, compare_historical_rf,
    evaluate_binary, event_table, train_leave_one_well_out,
)
from ml_agent import FEATURES


class TestCatBoostComparison(unittest.TestCase):
    def test_original_253_cohort_and_previous_rf_metrics(self):
        pts = read_points(ROOT / "experiments/inputs/class2_horizon_points.csv")
        self.assertEqual(len(pts), 253)
        self.assertEqual(int(pts.target.sum()), 33)
        self.assertEqual(pts.well_id.nunique(), 7)
        old = compare_historical_rf(pts, ROOT / "experiments/inputs/class2_rf_baseline_oof.csv")
        joined = pts.merge(old, on=["episode_id", "file", "well_id", "decision_time"])
        result = evaluate_binary(joined.target.to_numpy(), joined.score_rf_previous.to_numpy())
        self.assertAlmostEqual(result["ap"], 0.200563, places=5)
        self.assertAlmostEqual(result["f1_05"], 0.283582, places=5)
        self.assertEqual(result["fp_windows"], 82)

    def test_features_see_only_past_sensors_not_class(self):
        idx = pd.date_range("2020-01-01", periods=400, freq="s")
        data = pd.DataFrame({"P-TPT": np.arange(400, dtype=float),
                             "T-TPT": np.arange(400, dtype=float) * 0.05,
                             "class": [102]*400}, index=idx)
        a = past_features(prepare_frame(data), idx[299])
        data.iloc[300:, data.columns.get_loc("P-TPT")] = 1e9
        data["class"] = 0
        b = past_features(prepare_frame(data), idx[299])
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            if np.isnan(a[k]):
                self.assertTrue(np.isnan(b[k]))
            else:
                self.assertAlmostEqual(a[k], b[k])
        self.assertEqual(a["P-TPT_mean"], 149.5)

    def test_irregular_time_rejected(self):
        ix = pd.date_range("2020-01-01", periods=301, freq="s")
        ix = ix.delete(120).append(pd.DatetimeIndex([ix[-1] + pd.Timedelta(seconds=1)]))
        f = prepare_frame(pd.DataFrame({"P-TPT": np.ones(301)}, index=ix))
        with self.assertRaises(ValueError):
            past_features(f, ix[-1])

    def test_old_oof_wrong_target_refused(self):
        pts = read_points(ROOT / "experiments/inputs/class2_horizon_points.csv")
        old = pd.read_csv(ROOT / "experiments/inputs/class2_rf_baseline_oof.csv")
        old.loc[0, "target"] = 1 - old.loc[0, "target"]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"changed.csv"
            old.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                compare_historical_rf(pts, path)

    def test_leakage_control_well_folds(self):
        # Synthetic data for the entire LOOW training path.
        rng = np.random.default_rng(42)
        rows = []
        for well in range(1, 5):
            for i in range(8):
                record = {k: float(v) for k, v in zip(FEATURES, rng.normal(size=len(FEATURES)))}
                record.update({
                    "well_id": well, "file": f"WELL-{well:05d}.parquet",
                    "episode_id": f"WELL-{well:05d}__run1",
                    "decision_time": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=5*i),
                    "event_time": pd.Timestamp("2020-01-01 01:00"),
                    "minutes_until_event": float(60-5*i),
                    "target": i % 4 == 0,
                })
                rows.append(record)
        table = pd.DataFrame(rows)
        table.loc[0, FEATURES[0]] = np.nan
        out = train_leave_one_well_out(table, cat_iterations=15)
        self.assertEqual(len(out), 32)
        self.assertFalse(out[["score_rf_refit", "score_catboost"]].isna().any().any())
        self.assertTrue(out.score_catboost.between(0, 1).all())

    def test_event_detection_counts_positive_windows_only(self):
        d = pd.DataFrame({"episode_id": ["a", "a", "b", "b"],
                          "well_id": [1, 1, 2, 2],
                          "target": [1, 0, 1, 0],
                          "decision_time": pd.date_range("2020-01-01", periods=4, freq="5min"),
                          "minutes_until_event": [10, 40, 5, 30],
                          "score_catboost": [0.6, 0.99, 0.1, 0.99]})
        events = event_table(d, "score_catboost")
        self.assertEqual(int(events.detected.sum()), 1)
        self.assertEqual(float(events.loc[events.detected, "lead_minutes_first_correct_alert"].iloc[0]), 10)


if __name__ == "__main__":
    unittest.main()
