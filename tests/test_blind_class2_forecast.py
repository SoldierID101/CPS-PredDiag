"""Проверки новой постановки без oracle-гейта. Не требует локальных Parquet."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.blind_class2_forecast import (
    BASELINE_FEATURES, EXTENDED_FEATURES, classify_future,
    contiguous_segments, event_summary, past_only_features, prepare_frame,
    select_decision_points, summarize_binary, train_oof, read_cohort,
)


def fake_episode(*, n: int = 3100, event_at: int | None = 2500,
                 well_noise: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2017-03-10 12:00:00", periods=n, freq="s")
    cls = np.zeros(n, dtype=float)
    if event_at is not None:
        cls[max(0, event_at - 700):event_at] = 102
        cls[event_at:] = 2
    time = np.arange(n, dtype=float)
    # Динамика сенсоров имитируется для теста функций, НЕ для научной оценки.
    return pd.DataFrame({
        "class": pd.array(cls, dtype="Float64"),
        "state": pd.array(np.full(n, 123), dtype="Int64"),
        "P-MON-CKP": 5e6 - 50 * time + well_noise,
        "P-TPT": 1.8e7 - 500 * time + well_noise,
        "T-TPT": 100.0 - 0.03 * time,
        "P-PDG": np.full(n, 0.0),
    }, index=index)


class TestBlindClass2(unittest.TestCase):
    def test_right_censor_is_not_negative(self):
        a = np.zeros(1400, dtype=float)
        y, event, reason = classify_future(a, 899, len(a))
        self.assertTrue(np.isnan(y))
        self.assertIsNone(event)
        self.assertEqual(reason, "censored_or_unknown")

    def test_negative_needs_known_full_horizon(self):
        a = np.zeros(2000, dtype=float)
        y, event, reason = classify_future(a, 899, len(a))
        self.assertEqual(y, 0)
        self.assertIsNone(event)
        self.assertEqual(reason, "negative")
        a[1200] = np.nan
        self.assertTrue(np.isnan(classify_future(a, 899, len(a))[0]))

    def test_positive_onset_in_next_15min(self):
        a = np.zeros(3000, dtype=float)
        a[2000:2400] = 102
        a[2400:] = 2
        y, event, _ = classify_future(a, 1500, len(a))
        self.assertEqual(y, 1)
        self.assertEqual(event, 2400)
        self.assertEqual(classify_future(a, 1200, len(a))[0], 0)
        # Метки неизвестны до самого события — нельзя считать положительным.
        a[2300] = np.nan
        self.assertTrue(np.isnan(classify_future(a, 1500, len(a))[0]))

    def test_no_oracle_schedule_and_normal_windows(self):
        frame = fake_episode()
        samples, audit = select_decision_points(frame)
        self.assertGreater(audit["eligible_negative"], 0)
        self.assertGreater(audit["eligible_positive"], 0)
        self.assertSetEqual(set(samples["target"].astype(int)), {0, 1})
        self.assertIn(0, samples["phase_for_audit_only"].to_list())
        # Расписание от начала записи, а не от момента label 102.
        self.assertEqual(samples.decision_time.iloc[0], frame.index[899])

    def test_class_and_state_cannot_enter_features(self):
        frame = fake_episode()
        a, _ = past_only_features(frame, 1199)
        changed = frame.copy()
        changed.loc[changed.index[:1200], "class"] = 102
        changed.loc[changed.index[:1200], "state"] = 999
        b, _ = past_only_features(changed, 1199)
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(np.isclose(a[key], b[key], equal_nan=True), key)
        self.assertFalse(any(k.startswith(("class", "state", "well_id")) for k in a))
        self.assertSetEqual(set(BASELINE_FEATURES + EXTENDED_FEATURES), set(a))

    def test_future_sensor_values_do_not_change_features(self):
        frame = fake_episode()
        a, _ = past_only_features(frame, 1199)
        newer = frame.copy()
        newer.iloc[1200:, newer.columns.get_loc("P-TPT")] = 1e12
        newer.iloc[1200:, newer.columns.get_loc("T-TPT")] = -1e12
        b, _ = past_only_features(newer, 1199)
        for k in a:
            self.assertTrue(np.isclose(a[k], b[k], equal_nan=True), k)

    def test_missing_one_sensor_allowed_but_two_not(self):
        frame = fake_episode()
        frame["P-MON-CKP"] = np.nan
        s, _ = select_decision_points(frame)
        self.assertFalse(s.empty)  # есть TPT + T-TPT
        frame["T-TPT"] = np.nan
        s, audit = select_decision_points(frame)
        self.assertTrue(s.empty)
        self.assertGreater(audit["insufficient_sensors"], 0)

    def test_gap_resets_history_and_future(self):
        a = fake_episode(n=4200, event_at=None)
        a = a.drop(a.index[2000])
        a = prepare_frame(a)
        intervals = contiguous_segments(a.index)
        self.assertEqual(len(intervals), 2)
        # После разрыва нужно снова накопить 900 секунд истории.
        samples, _ = select_decision_points(a)
        times = set(samples.decision_time)
        self.assertNotIn(a.index[2000], times)
        for t in samples.decision_time:
            pos = a.index.get_loc(t)
            self.assertTrue(np.all(np.diff(a.index[pos-899:pos+1].asi8) == 1_000_000_000))

    def test_nullable_na_is_censor(self):
        a = fake_episode()
        a.loc[a.index[2250], "class"] = pd.NA
        samples, audit = select_decision_points(a)
        self.assertGreater(audit["censored_or_unknown_future"], 0)
        # Нет 0 на тех горизонтах, где было неизвестное значение.
        affected = samples.loc[
            (samples.decision_time < a.index[2250]) &
            (samples.decision_time >= a.index[2250] - pd.Timedelta(seconds=900))]
        self.assertTrue(affected.empty)

    def test_microsecond_index_is_normalized(self):
        f = fake_episode()
        f.index = f.index.as_unit("us")
        normalized = prepare_frame(f)
        self.assertEqual(normalized.index.unit, "ns")
        samples, audit = select_decision_points(normalized)
        self.assertGreater(audit["eligible_negative"], 0)
        self.assertFalse(samples.empty)

    def test_after_first_event_no_extra_negative_windows(self):
        f = fake_episode(n=3900, event_at=2400)
        f.loc[f.index[2600:], "class"] = 0  # якобы восстановилась норма
        samples, audit = select_decision_points(f)
        self.assertGreater(audit["excluded_after_first_event"], 0)
        self.assertTrue((samples.decision_time < f.index[2400]).all())

    def test_missing_time_or_unsorted_rejected(self):
        frame = fake_episode()
        with self.assertRaises(ValueError):
            prepare_frame(frame.iloc[::-1])
        with self.assertRaises(ValueError):
            prepare_frame(frame.reset_index(drop=True))

    def test_summary_counts_and_event_lead(self):
        t = pd.Timestamp("2018-01-01 00:00:00")
        oof = pd.DataFrame({
            "file": ["A", "A", "A"], "well_id": [1, 1, 1],
            "decision_time": [t, t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=10)],
            "event_time": [pd.NaT, t + pd.Timedelta(minutes=19), t + pd.Timedelta(minutes=19)],
            "target": [0, 1, 1], "catboost_12": [0.9, 0.2, 0.8],
        })
        m = summarize_binary(oof, "catboost_12")
        self.assertEqual(m["fp_windows"], 1)
        self.assertEqual(m["tp_windows"], 1)
        e = event_summary(oof, "catboost_12")
        self.assertEqual(len(e), 1)
        self.assertTrue(e.detected.iloc[0])
        self.assertAlmostEqual(e.lead_minutes.iloc[0], 9)

    def test_label_change_does_not_change_decision_schedule(self):
        a = fake_episode()
        b = a.copy()
        b["class"] = pd.array(np.full(len(b), 0), dtype="Float64")
        _, report_a = select_decision_points(a)
        _, report_b = select_decision_points(b)
        self.assertEqual(report_a["candidate_times"], report_b["candidate_times"])

    def test_file_discovery_and_events_audit_without_real_parquet(self):
        with TemporaryDirectory() as temp:
            dataset = Path(temp)
            for well in (2, 3, 9):
                (dataset / f"WELL-{well:05d}_20170310120000.parquet").write_bytes(b"stub")
            with patch("pandas.read_parquet", side_effect=lambda path: fake_episode()):
                cohort, audit, prov = read_cohort(dataset)
            self.assertEqual(len(audit), 3)
            self.assertEqual(len(prov), 3)
            self.assertEqual(int(audit.observed_class2_onsets.sum()), 3)
            self.assertEqual(cohort.well_id.nunique(), 3)

    def test_training_folds_on_small_synthetic_data(self):
        try:
            import catboost  # noqa: F401
        except ImportError:
            self.skipTest("CatBoost не установлен")
        frames = []
        for well in (2, 3, 9):
            f = fake_episode(n=3200, event_at=2500, well_noise=well * 1e4)
            rows, _ = select_decision_points(f)
            rows.insert(0, "file", f"WELL-{well:05d}_test.parquet")
            rows.insert(1, "well_id", well)
            frames.append(rows)
        df = pd.concat(frames, ignore_index=True)
        self.assertEqual(df.well_id.nunique(), 3)
        pred = train_oof(df, cb_iterations=10)
        self.assertEqual(len(pred), len(df))
        self.assertTrue(pred[["rf_12", "catboost_12", "catboost_multiscale"]].notna().all().all())
        self.assertTrue(pred[["rf_12", "catboost_12", "catboost_multiscale"]].ge(0).all().all())
        self.assertTrue(pred[["rf_12", "catboost_12", "catboost_multiscale"]].le(1).all().all())


if __name__ == "__main__":
    unittest.main()
