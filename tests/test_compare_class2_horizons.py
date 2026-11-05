"""Тесты горизонтов. Синтетические данные служат только проверке кода."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import blind_class2_forecast as b
from experiments.compare_class2_horizons import (
    DEFAULT_HORIZONS, build_cohorts, check_horizons, common_decision_subset,
    create_report, model_summary, train_catboost_oof, verify_same_past_features,
)


def synthetic_frame(n=6100, event_at=5000):
    idx = pd.date_range("2020-01-01", periods=n, freq="s")
    labels = np.zeros(n, dtype=float)
    labels[event_at-600:event_at] = 102
    labels[event_at:] = 2
    seconds = np.arange(n)
    return pd.DataFrame({
        "class": pd.array(labels, dtype="Float64"), "state": 77,
        "P-TPT": 13e6 - seconds * 17,
        "T-TPT": 130 - seconds * 0.001,
        "P-MON-CKP": np.full(n, np.nan),
        "P-PDG": np.full(n, 0.0),
    }, index=idx)


def fake_features(frame, i):
    # Подмена исключительно для быстрого теста оркестрации. Настоящее
    # извлечение признаков уже проверено в test_blind_class2_forecast.py.
    x = float(i % 100) / 100
    return {f: x for f in (*b.BASELINE_FEATURES, *b.EXTENDED_FEATURES)}, 2


def small_cohorts():
    base_time = pd.Timestamp("2020-01-01 12:00:00")
    rows = []
    for h in DEFAULT_HORIZONS:
        for well in (2, 3, 9):
            for j in range(20):
                minute = 5*j
                target = int(100 - minute <= h)
                row = {
                    "horizon_minutes": h, "file": f"WELL-{well:05d}_synthetic.parquet",
                    "well_id": well, "decision_time": base_time + pd.Timedelta(minutes=minute),
                    "target": target, "phase_for_audit_only": 102 if minute >= 70 else 0,
                    "event_time": base_time + pd.Timedelta(minutes=100) if target else pd.NaT,
                }
                for k, feat in enumerate(b.BASELINE_FEATURES):
                    row[feat] = minute / 100 + k / 1000 + well / 5000
                rows.append(row)
    return pd.DataFrame(rows)


class TestClass2Horizons(unittest.TestCase):
    def test_horizons_validated(self):
        self.assertEqual(check_horizons((60, 15, 30)), (15, 30, 60))
        for hs in ((), (15, 15), (0, 15), (-1, 15), (True, 15), (15.0, 30)):
            with self.subTest(hs=hs), self.assertRaises(ValueError):
                check_horizons(hs)

    def test_15m_reuses_original_target_and_schedule(self):
        frame = b.prepare_frame(synthetic_frame())
        with patch("experiments.blind_class2_forecast.past_only_features", side_effect=fake_features):
            old, _ = b.select_decision_points(frame)
            new, _ = b.select_decision_points(frame, horizon_seconds=15*60)
        pd.testing.assert_frame_equal(old, new)

    def test_horizon_expansion_not_a_label_gate(self):
        frame = b.prepare_frame(synthetic_frame())
        changed = frame.copy()
        changed["class"] = 0
        with patch("experiments.blind_class2_forecast.past_only_features", side_effect=fake_features):
            _, a = b.select_decision_points(frame, horizon_seconds=3600)
            _, c = b.select_decision_points(changed, horizon_seconds=3600)
        self.assertEqual(a["candidate_times"], c["candidate_times"])

    def test_longer_horizon_positive_earlier(self):
        frame = b.prepare_frame(synthetic_frame())
        with patch("experiments.blind_class2_forecast.past_only_features", side_effect=fake_features):
            c15, _ = b.select_decision_points(frame, horizon_seconds=900)
            c60, _ = b.select_decision_points(frame, horizon_seconds=3600)
        self.assertGreater(int(c60.target.sum()), int(c15.target.sum()))
        self.assertEqual(c15.event_time.dropna().unique().tolist(),
                         c60.event_time.dropna().unique().tolist())

    def test_censored_longer_horizon_not_assumed_negative(self):
        labels = np.zeros(3500, dtype=float)
        y15, _, _ = b.classify_future(labels, 1000, len(labels), 900)
        y60, _, _ = b.classify_future(labels, 1000, len(labels), 3600)
        self.assertEqual(y15, 0)
        self.assertTrue(np.isnan(y60))

    def test_feature_consistency_pass_and_fail(self):
        x = pd.DataFrame([
            {"file": "A", "decision_time": pd.Timestamp("2020-01-01"),
             "horizon_minutes": h, **{k: 1.0 for k in b.BASELINE_FEATURES}}
            for h in DEFAULT_HORIZONS
        ])
        verify_same_past_features(x)
        x.loc[2, b.BASELINE_FEATURES[0]] = 2.0
        with self.assertRaisesRegex(ValueError, "Признаки меняются"):
            verify_same_past_features(x)

    def test_common_only_shared_decisions(self):
        t = pd.Timestamp("2020-01-01")
        p = pd.DataFrame([
            {"horizon_minutes": h, "file": "a", "decision_time": t, "score": 0.3}
            for h in DEFAULT_HORIZONS
        ] + [{"horizon_minutes": 15, "file": "a",
              "decision_time": t + pd.Timedelta(minutes=5), "score": 0.1}])
        c = common_decision_subset(p, DEFAULT_HORIZONS)
        self.assertEqual(len(c), 3)
        self.assertTrue((c.decision_time == t).all())

    def test_common_rejects_no_overlap(self):
        p = pd.DataFrame([{"horizon_minutes": h, "file": "a",
                           "decision_time": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=h)}
                          for h in DEFAULT_HORIZONS])
        with self.assertRaises(ValueError):
            common_decision_subset(p, DEFAULT_HORIZONS)

    def test_data_discovery_single_read_for_three_horizons(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for well in (2, 3, 9):
                (root / f"WELL-{well:05d}_20170101000000.parquet").write_bytes(b"synthetic")
            with patch("pandas.read_parquet", side_effect=lambda _: synthetic_frame()) as mocked_read, \
                 patch("experiments.blind_class2_forecast.past_only_features", side_effect=fake_features):
                cohort, audit, provenance = build_cohorts(root)
            self.assertEqual(mocked_read.call_count, 3)
            self.assertEqual(set(cohort.horizon_minutes), set(DEFAULT_HORIZONS))
            self.assertEqual(len(audit), 9)
            self.assertEqual(len(provenance), 3)
            self.assertEqual(cohort.well_id.nunique(), 3)
            verify_same_past_features(cohort)

    def test_oof_no_leakage_and_reports(self):
        df = small_cohorts()
        self.assertTrue(df.groupby("horizon_minutes").target.nunique().eq(2).all())
        all_oof = []
        for h in DEFAULT_HORIZONS:
            oof = train_catboost_oof(df[df.horizon_minutes.eq(h)], iterations=7)
            self.assertTrue(oof.score.between(0, 1).all())
            m, e = model_summary(oof)
            self.assertEqual(m["n"], 60)
            self.assertEqual(m["eligible_completed_events"], 3)
            all_oof.append(oof)
        paired = common_decision_subset(pd.concat(all_oof), DEFAULT_HORIZONS)
        self.assertEqual(len(paired), len(df))

    def test_end_to_end_creates_reports(self):
        cohort = small_cohorts()
        audit = pd.DataFrame({"file": ["synthetic"], "horizon_minutes": [15]})
        with TemporaryDirectory() as td:
            out = Path(td)
            create_report(cohort, audit, DEFAULT_HORIZONS, 5, out, [])
            expected = ["comparison.csv", "comparison_common_times.csv", "cohort_audit.csv",
                        "cohort_features.csv", "oof_predictions.csv", "common_oof_predictions.csv",
                        "by_well.csv", "by_event.csv", "experiment.json"]
            for fname in expected:
                self.assertTrue((out/fname).is_file(), fname)
            comp = pd.read_csv(out / "comparison.csv")
            self.assertEqual(comp.horizon_minutes.tolist(), [15, 30, 60])
            self.assertEqual(comp.eligible_completed_events.tolist(), [3, 3, 3])
            manifest = (out/"experiment.json").read_text(encoding="utf-8")
            self.assertIn('"oracle_gate": false', manifest)


if __name__ == "__main__":
    unittest.main()
