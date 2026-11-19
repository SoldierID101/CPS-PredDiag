"""Tests for exploration-only model benchmark; no Petrobras downloads required."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import benchmark_class2_models as bench


def synthetic():
    rows = []
    for w in (2, 3, 9, 10):
        base = pd.Timestamp("2020-01-01")
        for i in range(12):
            row = {
                "file": f"WELL-{w:05d}_20200101000000.parquet", "well_id": w,
                "horizon_minutes": 30, "decision_time": base + pd.Timedelta(minutes=5*i),
                "event_time": base + pd.Timedelta(minutes=65) if i >= 9 else pd.NaT,
                "target": int(i>=9),
            }
            for j, f in enumerate(bench.FEATURES):
                row[f] = float(0.07*j + (i>=9)*5 + w*0.01 + i*0.1)
            rows.append(row)
    return pd.DataFrame(rows)


class BenchmarkTests(unittest.TestCase):
    def test_features_explicit_and_labels_forbidden(self):
        self.assertEqual(len(bench.FEATURES), 12)
        for x in ("target", "class", "state", "event_time", "well_id", "future_status"):
            self.assertNotIn(x, bench.FEATURES)

    def test_split_is_disjoint_and_complete(self):
        f = synthetic()
        indexes = []
        for w, train, test in bench.well_folds(f):
            self.assertFalse(set(f.iloc[train].well_id) & set(f.iloc[test].well_id))
            indexes.extend(test)
        self.assertEqual(sorted(indexes), list(range(len(f))))

    def test_duplicate_time_refused(self):
        f = synthetic()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "cohort.csv"
            pd.concat([f, f.iloc[[0]]], ignore_index=True).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "Дублирующиеся"):
                bench.load_cohort(p)

    def test_positive_event_after_decision_required(self):
        f = synthetic()
        f.loc[f.target.eq(1), "event_time"] = pd.Timestamp("2019-01-01")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "cohort.csv"
            f.to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "после наступления"):
                bench.load_cohort(p)

    def test_binary_event_and_window_metrics_separate(self):
        f = synthetic().iloc[:12].copy()
        f["score"] = np.where(f.target.eq(1), 0.2, 0.1)
        f.loc[f.target.eq(1)].index  # positive windows present
        f.loc[f.index[-1], "score"] = 0.7
        r = bench.evaluate(f)
        self.assertEqual(r["events_detected_05"], 1)
        self.assertEqual(r["events_total"], 1)
        self.assertAlmostEqual(r["recall_05"], 1/3)

    def test_predict_oof_two_models_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "cohort.csv"
            out = Path(t) / "results"
            synthetic().to_csv(p, index=False)
            manifest = bench.run(p, out, ("dummy_prior", "logreg"))
            self.assertEqual(manifest["n"], 48)
            self.assertEqual(manifest["distinct_events"], 4)
            oof = pd.read_csv(out / "oof_predictions.csv")
            self.assertEqual(len(oof), 96)
            self.assertFalse(oof.score.isna().any())
            self.assertEqual(len(pd.read_csv(out / "model_comparison.csv")), 2)
            self.assertEqual(len(pd.read_csv(out / "by_well.csv")), 8)

    def test_horizon_select_only_30(self):
        f = synthetic()
        older = f.copy(); older["horizon_minutes"] = 15
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "cohort.csv"
            pd.concat([f, older], ignore_index=True).to_csv(p, index=False)
            self.assertEqual(len(bench.load_cohort(p)), len(f))

    def test_invalid_model_refused(self):
        with self.assertRaises(ValueError):
            bench.make_model("oracle")


if __name__ == "__main__":
    unittest.main()
