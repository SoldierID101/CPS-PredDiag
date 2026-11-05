"""Проверки последовательного воспроизведения без чтения Petrobras/ML-артефакта."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.replay_pipeline import (
    iter_decision_positions, replay_pipeline, timeline_row, write_replay_outputs,
)
from diagnostic_pipeline import DiagnosticPipeline
from prognostic_agent import PrognosticAgent
from rule_based_agent import RuleBasedAgent


class TrackingML:
    def __init__(self):
        self.seen = []

    def predict(self, history, *, transition_102_confirmed, decision_time=None):
        self.seen.append((history.index[-1], len(history), transition_102_confirmed,
                          list(history.columns)))
        if transition_102_confirmed is True:
            return {"agent": "ml", "status": "experimental_prediction",
                    "model_score": 0.8, "above_research_threshold": True,
                    "uncalibrated": True, "horizon_minutes": 15}
        return {"agent": "ml", "status": "not_applicable"}


class TestReplayPipeline(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", periods=900, freq="s").as_unit("us")
        self.frame = pd.DataFrame({
            "P-PDG": [100.0] * 900,
            "P-TPT": [90.0] * 900,
            "T-TPT": [30.0] * 900,
            "class": pd.array([102] * 900, dtype="Int64"),
        }, index=self.index)
        self.spy = TrackingML()
        self.pipeline = DiagnosticPipeline(rule_agent=RuleBasedAgent(),
                                           prognostic_agent=PrognosticAgent(),
                                           ml_agent=self.spy)

    def test_first_at_300th_then_every_five_minutes(self):
        self.assertEqual(list(iter_decision_positions(self.frame)), [299, 599, 899])

    def test_gaps_reset_window(self):
        frame = self.frame.drop(self.index[400])
        self.assertEqual(list(iter_decision_positions(frame)), [299, 699])

    def test_blind_excludes_labels_from_all_agents_and_no_future(self):
        results = list(replay_pipeline(self.frame, pipeline=self.pipeline, mode="blind"))
        self.assertEqual(len(results), 3)
        self.assertEqual([call[1] for call in self.spy.seen], [300, 600, 900])
        self.assertTrue(all(call[2] is None for call in self.spy.seen))
        self.assertTrue(all("class" not in call[3] for call in self.spy.seen))
        self.assertEqual([r["decision"]["decision_level"] for r in results],
                         ["unknown"] * 3)

    def test_oracle_only_uses_past_complete_window(self):
        frame = self.frame.copy()
        frame.loc[self.index[450], "class"] = pd.NA
        results = list(replay_pipeline(frame, pipeline=self.pipeline, mode="oracle_replay"))
        flags = [r["transition_evidence"]["confirmed_102"] for r in results]
        self.assertEqual(flags, [True, False, True])
        self.assertEqual(self.spy.seen[1][2], False)

    def test_same_schedule_regardless_of_class(self):
        altered = self.frame.copy()
        altered["class"] = [0] * len(altered)
        self.assertEqual(list(iter_decision_positions(self.frame)),
                         list(iter_decision_positions(altered)))

    def test_max_points_and_bad_parameters(self):
        results = list(replay_pipeline(self.frame, pipeline=self.pipeline, max_points=2))
        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.spy.seen), 2)
        with self.assertRaises(ValueError):
            list(iter_decision_positions(self.frame, stride_seconds=0))
        with self.assertRaises(ValueError):
            list(replay_pipeline(self.frame, pipeline=self.pipeline, max_points=0))
        with self.assertRaises(ValueError):
            list(replay_pipeline(self.frame, pipeline=self.pipeline, mode="invalid"))

    def test_reject_non_monotonic_and_duplicates(self):
        with self.assertRaises(ValueError):
            list(iter_decision_positions(self.frame.iloc[::-1]))
        with self.assertRaises(ValueError):
            list(iter_decision_positions(self.frame.iloc[[0, 0, 1]]))

    def test_short_episode_no_decisions(self):
        short = self.frame.iloc[:299]
        self.assertEqual(list(iter_decision_positions(short)), [])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "300"):
                write_replay_outputs(replay_pipeline(short, pipeline=self.pipeline),
                                     folder, source_filename="example.parquet", mode="blind",
                                     stride_seconds=300)

    def test_jsonl_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = write_replay_outputs(
                replay_pipeline(self.frame, pipeline=self.pipeline, mode="oracle_replay"),
                folder, source_filename="example.parquet", mode="oracle_replay",
                stride_seconds=300,
            )
            self.assertEqual(summary["decisions_count"], 3)
            self.assertEqual(summary["decision_levels"], {"unknown": 3})
            self.assertEqual(summary["ml_research_alerts_count"], 3)
            self.assertEqual(summary["oracle_confirmed_102_count"], 3)
            lines = (Path(folder) / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["mode"], "oracle_replay")
            tab = pd.read_csv(Path(folder) / "timeline.csv")
            self.assertEqual(len(tab), 3)
            self.assertEqual(float(tab.iloc[0]["ml_score_uncalibrated"]), 0.8)
            self.assertEqual(json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))["decisions_count"], 3)


if __name__ == "__main__":
    unittest.main()
