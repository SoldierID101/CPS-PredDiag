"""Тесты сборки Pipeline без реального ML-артефакта и реальных порогов."""
import unittest

import numpy as np
import pandas as pd

from diagnostic_pipeline import DiagnosticPipeline
from prognostic_agent import PrognosticAgent
from rule_based_agent import RuleBasedAgent


class StubMLAgent:
    """Шпион для проверки передачи данных, не замена реальной ML-модели."""
    def __init__(self):
        self.calls = []

    def predict(self, history, *, transition_102_confirmed, decision_time=None):
        self.calls.append({
            "columns": list(history.columns),
            "last": history.index[-1],
            "len": len(history),
            "flag": transition_102_confirmed,
            "time": decision_time,
        })
        if transition_102_confirmed is not True:
            return {"agent": "ml", "status": "not_applicable"}
        return {"agent": "ml", "status": "experimental_prediction",
                "model_score": 0.83, "above_research_threshold": True,
                "uncalibrated": True, "horizon_minutes": 15}


class TestDiagnosticPipeline(unittest.TestCase):
    def setUp(self):
        self.spy = StubMLAgent()
        self.pipeline = DiagnosticPipeline(
            rule_agent=RuleBasedAgent(),
            prognostic_agent=PrognosticAgent(),
            ml_agent=self.spy,
        )
        idx = pd.date_range("2020-01-01", periods=700, freq="s")
        self.df = pd.DataFrame({
            "P-PDG": np.linspace(1, 3, len(idx)),
            "P-TPT": np.linspace(5, 2, len(idx)),
            "T-TPT": np.linspace(30, 31, len(idx)),
            "class": [102] * 700,
            "state": [102] * 700,
            "target": [1] * 700,
        }, index=idx)

    def test_blind_never_uses_class_102_even_if_all_labels_are_102(self):
        result = self.pipeline.run(self.df, mode="blind")
        self.assertEqual(result["agents"]["ml"]["status"], "not_applicable")
        self.assertEqual(result["decision"]["decision_level"], "unknown")
        self.assertEqual(result["transition_evidence"]["source"], "unavailable")
        self.assertIsNone(self.spy.calls[-1]["flag"])
        self.assertFalse(set(self.spy.calls[-1]["columns"]) & {"class", "state", "target"})

    def test_oracle_replay_can_use_only_complete_past_102_window(self):
        result = self.pipeline.run(self.df, mode="oracle_replay")
        self.assertTrue(result["transition_evidence"]["confirmed_102"])
        self.assertEqual(result["agents"]["ml"]["status"], "experimental_prediction")
        self.assertTrue(result["decision"]["ml_research_alert"])
        self.assertEqual(result["decision"]["decision_level"], "unknown")
        self.assertTrue(result["research_only"])
        self.assertIn("разметка", result["transition_evidence"]["note"].lower())
        self.assertNotIn("class", self.spy.calls[-1]["columns"])

    def test_oracle_does_not_confirm_102_if_window_spans_normal(self):
        df = self.df.copy()
        df.loc[df.index[-100:], "class"] = 0
        result = self.pipeline.run(df, mode="oracle_replay")
        self.assertFalse(result["transition_evidence"]["confirmed_102"])
        self.assertEqual(result["agents"]["ml"]["status"], "not_applicable")

    def test_oracle_does_not_confirm_102_when_timestamps_broken(self):
        df = self.df.drop(self.df.index[-20]).copy()
        result = self.pipeline.run(df, mode="oracle_replay")
        self.assertFalse(result["transition_evidence"]["confirmed_102"])

    def test_oracle_does_not_confirm_nullable_missing_label(self):
        df = self.df.copy()
        df["class"] = pd.array([102] * len(df), dtype="Int64")
        df.loc[df.index[-20], "class"] = pd.NA
        result = self.pipeline.run(df, mode="oracle_replay")
        self.assertFalse(result["transition_evidence"]["confirmed_102"])
        self.assertEqual(result["agents"]["ml"]["status"], "not_applicable")

    def test_oracle_ignores_missing_label_outside_current_300s_window(self):
        df = self.df.copy()
        df["class"] = pd.array([102] * len(df), dtype="Int64")
        df.loc[df.index[1], "class"] = pd.NA
        result = self.pipeline.run(df, mode="oracle_replay")
        self.assertTrue(result["transition_evidence"]["confirmed_102"])

    def test_oracle_requires_class(self):
        with self.assertRaisesRegex(ValueError, "class"):
            self.pipeline.run(self.df.drop(columns="class"), mode="oracle_replay")

    def test_future_rows_are_cut_before_all_agents(self):
        point = self.df.index[325]
        result = self.pipeline.run(self.df, decision_time=point, mode="blind")
        self.assertEqual(self.spy.calls[-1]["len"], 326)
        self.assertEqual(self.spy.calls[-1]["last"], point)
        self.assertEqual(result["decision_time"], point.isoformat())
        self.assertEqual(result["agents"]["rule_based"]["status"], "unknown")

    def test_missing_rules_and_thresholds_not_normal(self):
        result = self.pipeline.run(self.df)
        self.assertEqual(result["agents"]["rule_based"]["configured_rules"], 0)
        self.assertEqual(result["agents"]["prognostic"], [])
        self.assertEqual(result["decision"]["action"], "no_automatic_action")

    def test_microsecond_index_supported_and_normalized(self):
        df = self.df.copy()
        df.index = df.index.as_unit("us")
        result = self.pipeline.run(df, mode="oracle_replay")
        self.assertTrue(result["transition_evidence"]["confirmed_102"])
        self.assertEqual(self.spy.calls[-1]["last"], df.index[-1])
        self.assertEqual(self.spy.calls[-1]["last"].asm8.dtype.name, "datetime64[ns]")

    def test_rule_alert_is_independent_of_ml(self):
        synthetic_rules = RuleBasedAgent([{
            "id": "TEST-1", "sensor": "P-PDG", "condition": ">",
            "threshold": 2.0, "severity": "warning", "unit": "Pa",
            "source": "synthetic unit test, NOT a Petrobras diagnostic limit",
            "message": "synthetic high level",
        }])
        pipeline = DiagnosticPipeline(rule_agent=synthetic_rules,
                                      prognostic_agent=PrognosticAgent(), ml_agent=self.spy)
        result = pipeline.run(self.df, mode="blind")
        self.assertEqual(result["decision"]["decision_level"], "warning")
        self.assertFalse(result["decision"]["ml_research_alert"])
        self.assertEqual(result["agents"]["rule_based"]["triggered_rules"][0]["rule_id"], "TEST-1")

    def test_prognostic_threshold_only_synthetic(self):
        synthetic_prognostic = PrognosticAgent([{
            "id": "TEST-PROG", "sensor": "P-PDG", "threshold": 3.05,
            "direction": "above", "severity": "warning", "unit": "Pa",
            "source": "synthetic unit test, NOT a Petrobras diagnostic limit",
        }])
        pipeline = DiagnosticPipeline(rule_agent=RuleBasedAgent(),
                                      prognostic_agent=synthetic_prognostic, ml_agent=self.spy)
        result = pipeline.run(self.df, mode="blind")
        self.assertEqual(result["agents"]["prognostic"][0]["forecast_status"], "predicted_crossing")
        self.assertEqual(result["decision"]["forecast_status"], "predicted_crossing")
        self.assertEqual(result["decision"]["decision_level"], "warning")
        self.assertIn("не оценка времени до отказа", result["explanation"])

    def test_bad_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline.run(self.df.iloc[::-1])
        with self.assertRaises(ValueError):
            self.pipeline.run(self.df.iloc[[0, 0, 1]])
        with self.assertRaises(ValueError):
            self.pipeline.run(self.df, decision_time=pd.Timestamp("1999-01-01"))

    def test_invalid_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            self.pipeline.run(self.df, mode="made_up")


if __name__ == "__main__":
    unittest.main()
