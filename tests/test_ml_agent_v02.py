"""Контрактные тесты shadow ML Agent v0.2; реальных ML-метрик не вычисляют."""
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_agent_v02 import ShadowConfig, ShadowMLAgentV02


class FakeModel:
    classes_ = [0, 1]

    def __init__(self, score=0.7):
        self.score = score
        self.last_features = None
        self.calls = 0

    def predict_proba(self, features):
        self.last_features = features.copy()
        self.calls += 1
        return np.array([[1 - self.score, self.score]])


def history(start="2017-03-20 03:00:00", *, missing=()):
    idx = pd.date_range(start, periods=900, freq="s")
    x = np.arange(900, dtype=float)
    f = pd.DataFrame({
        "P-MON-CKP": 12e6 - x * 0.5,
        "P-TPT": 19e6 - x * 0.7,
        "T-TPT": 115 - x * 0.001,
        "class": np.zeros(900),
        "state": np.ones(900),
    }, index=idx)
    return f.drop(columns=list(missing))


def agent(*, score=0.7, confirms=2):
    model = FakeModel(score)
    return ShadowMLAgentV02(model=model, config=ShadowConfig(
        model_version="synthetic-untrained", threshold=0.5, confirmations=confirms,
    )), model


class ContractTests(unittest.TestCase):
    def test_without_oracle_gating_and_shadow_only(self):
        a, _ = agent(confirms=1)
        h = history()
        res = a.predict(h, well_id="12", decision_time=h.index[-1])
        self.assertEqual(res["status"], "experimental_prediction")
        self.assertEqual(res["event_class"], 2)
        self.assertEqual(res["horizon_minutes"], 30)
        self.assertTrue(res["uncalibrated"])
        self.assertTrue(res["shadow_mode"])
        self.assertTrue(res["alarm_new_start"])

    def test_two_consecutive_confirmations(self):
        a, _ = agent()
        h1 = history()
        h2 = history(start="2017-03-20 03:05:00")
        first = a.predict(h1, well_id="12", decision_time=h1.index[-1])
        second = a.predict(h2, well_id="12", decision_time=h2.index[-1])
        self.assertFalse(first["alarm_active"])
        self.assertFalse(first["alarm_new_start"])
        self.assertTrue(second["alarm_active"])
        self.assertTrue(second["alarm_new_start"])

    def test_duplicate_time_is_idempotent(self):
        a, m = agent()
        h = history()
        one = a.predict(h, well_id="12", decision_time=h.index[-1])
        two = a.predict(h, well_id="12", decision_time=h.index[-1])
        self.assertEqual(one, two)
        self.assertEqual(m.calls, 1)
        self.assertFalse(two["alarm_active"])

    def test_state_isolated_between_wells(self):
        a, _ = agent()
        h1, h2 = history(), history(start="2017-03-20 03:05:00")
        a.predict(h1, well_id="12", decision_time=h1.index[-1])
        a.predict(h1, well_id="13", decision_time=h1.index[-1])
        self.assertTrue(a.predict(h2, well_id="12", decision_time=h2.index[-1])["alarm_active"])
        self.assertTrue(a.predict(h2, well_id="13", decision_time=h2.index[-1])["alarm_active"])

    def test_gap_resets_confirmation_streak(self):
        a, _ = agent()
        h1, h3 = history(), history(start="2017-03-20 03:10:00")
        a.predict(h1, well_id="12", decision_time=h1.index[-1])
        self.assertFalse(a.predict(h3, well_id="12", decision_time=h3.index[-1])["alarm_active"])

    def test_reset_between_replay_files(self):
        a, _ = agent()
        h1, h2 = history(), history(start="2017-03-20 03:05:00")
        a.predict(h1, well_id="12", decision_time=h1.index[-1])
        a.reset("12")
        self.assertFalse(a.predict(h2, well_id="12", decision_time=h2.index[-1])["alarm_active"])

    def test_future_row_refused(self):
        a, _ = agent()
        h = history()
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-301])["status"], "invalid_time")

    def test_missing_history_refused(self):
        a, _ = agent()
        h = history().iloc[-300:]
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "insufficient_data")

    def test_timestamp_gap_refused(self):
        a, _ = agent()
        h = history().drop(index=history().index[300])
        h.loc[h.index[-1] + pd.Timedelta(seconds=1)] = h.iloc[-1]
        self.assertEqual(len(h), 900)
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "invalid_time")

    def test_two_missing_sensors(self):
        a, _ = agent()
        h = history(missing=("P-TPT", "T-TPT"))
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "insufficient_data")

    def test_missing_model_is_explicit(self):
        a = ShadowMLAgentV02(model=None, config=ShadowConfig(model_version="none", threshold=0.5, confirmations=2))
        h = history()
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "model_unavailable")

    def test_ignores_class_and_state(self):
        a, m1 = agent()
        h1 = history()
        a.predict(h1, well_id="12", decision_time=h1.index[-1])
        a2, m2 = agent()
        h2 = h1.copy()
        h2["class"] = 2
        h2["state"] = 102
        a2.predict(h2, well_id="12", decision_time=h2.index[-1])
        pd.testing.assert_frame_equal(m1.last_features, m2.last_features)

    def test_future_change_does_not_modify_earlier_features(self):
        a, m1 = agent()
        full = history(start="2017-03-20 02:55:00")
        now = full.index[-1]
        a.predict(full, well_id="12", decision_time=now)
        first = m1.last_features.copy()
        future = history(start="2017-03-20 03:10:00")
        future.loc[:, "P-TPT"] = 1e10  # Произвольное изменение только после t
        combined = pd.concat([full, future])
        a2, m2 = agent()
        a2.predict(combined.loc[:now], well_id="12", decision_time=now)
        pd.testing.assert_frame_equal(first, m2.last_features)

    def test_no_positive_when_score_low(self):
        a, _ = agent(score=0.1, confirms=1)
        h = history()
        r = a.predict(h, well_id="12", decision_time=h.index[-1])
        self.assertFalse(r["alarm_active"])
        self.assertFalse(r["alarm_new_start"])

    def test_nonfinite_score_is_error(self):
        a, _ = agent(score=float("nan"))
        h = history()
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "model_error")

    def test_unordered_time_is_refused(self):
        a, _ = agent()
        h = history().iloc[::-1]
        self.assertEqual(a.predict(h, well_id="12", decision_time=h.index[-1])["status"], "invalid_time")

    def test_bad_data_resets_confirmation_streak(self):
        a, _ = agent()
        first = history()
        second = history(start="2017-03-20 03:05:00", missing=("P-TPT", "T-TPT"))
        third = history(start="2017-03-20 03:10:00")
        a.predict(first, well_id="12", decision_time=first.index[-1])
        self.assertEqual(a.predict(second, well_id="12", decision_time=second.index[-1])["status"], "insufficient_data")
        r = a.predict(third, well_id="12", decision_time=third.index[-1])
        self.assertFalse(r["alarm_active"])
        self.assertFalse(r["alarm_new_start"])

    def test_stale_time_does_not_change_state(self):
        a, _ = agent()
        h1, h2 = history(), history(start="2017-03-20 03:05:00")
        a.predict(h1, well_id="12", decision_time=h1.index[-1])
        r = a.predict(h1, well_id="12", decision_time=h1.index[-1] - pd.Timedelta(seconds=300))
        self.assertEqual(r["status"], "invalid_time")
        self.assertTrue(a.predict(h2, well_id="12", decision_time=h2.index[-1])["alarm_active"])

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            ShadowConfig(model_version="", threshold=0.5, confirmations=2)
        with self.assertRaises(ValueError):
            ShadowConfig(model_version="v1", threshold=0.0, confirmations=2)
        with self.assertRaises(ValueError):
            ShadowConfig(model_version="v1", threshold=0.5, confirmations=3)

    def test_microsecond_index_allows_prediction(self):
        # pandas 3.x can store DatetimeIndex in microseconds by default.
        # The legacy featurizer assumes nanoseconds when reading .asi8.
        a, m = agent(confirms=1)
        h = history()
        h.index = h.index.as_unit("us")
        self.assertEqual(str(h.index.dtype), "datetime64[us]")
        result = a.predict(h, well_id="12", decision_time=h.index[-1])
        self.assertEqual(result["status"], "experimental_prediction")
        self.assertTrue(result["alarm_active"])
        self.assertEqual(m.calls, 1)
        self.assertEqual(m.last_features.shape, (1, 12))
        self.assertEqual(str(h.index.dtype), "datetime64[us]")  # input unchanged

    def test_microsecond_and_nanosecond_indexes_yield_same_features(self):
        h_ns = history()
        h_ns.index = h_ns.index.as_unit("ns")
        h_us = h_ns.copy()
        h_us.index = h_us.index.as_unit("us")
        a_ns, m_ns = agent(confirms=1)
        a_us, m_us = agent(confirms=1)
        r_ns = a_ns.predict(h_ns, well_id="12", decision_time=h_ns.index[-1])
        r_us = a_us.predict(h_us, well_id="12", decision_time=h_us.index[-1])
        self.assertEqual(r_ns["status"], "experimental_prediction")
        self.assertEqual(r_us["status"], "experimental_prediction")
        pd.testing.assert_frame_equal(m_ns.last_features, m_us.last_features)
        self.assertEqual(r_ns["model_score"], r_us["model_score"])

    def test_microsecond_index_two_confirmations(self):
        a, m = agent(confirms=2)
        h1 = history()
        h2 = history(start="2017-03-20 03:05:00")
        h1.index = h1.index.as_unit("us")
        h2.index = h2.index.as_unit("us")
        first = a.predict(h1, well_id="12", decision_time=h1.index[-1])
        second = a.predict(h2, well_id="12", decision_time=h2.index[-1])
        self.assertEqual(first["status"], "experimental_prediction")
        self.assertFalse(first["alarm_active"])
        self.assertEqual(second["status"], "experimental_prediction")
        self.assertTrue(second["alarm_active"])
        self.assertEqual(m.calls, 2)

    def test_model_feature_order_matches_legacy_featurizer(self):
        from ml_agent import FEATURES
        a, model = agent()
        h = history()
        a.predict(h, well_id="12", decision_time=h.index[-1])
        self.assertEqual(model.last_features.columns.tolist(), FEATURES)
        self.assertEqual(model.last_features.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
