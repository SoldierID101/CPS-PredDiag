"""Тесты чистых функций Streamlit UI, без зависимости от пакета streamlit."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

from streamlit_app import (
    list_class2_files, build_ui_timeline, ml_chart_data, sensor_chart_data,
    output_directory, run_selected_file,
)


class TestStreamlitApp(unittest.TestCase):
    def _result(self, *, timestamp="2020-01-01T00:04:59", status="not_applicable",
                score=None, oracle=False):
        ml = {"status": status}
        if score is not None:
            ml["model_score"] = score
            ml["above_research_threshold"] = score >= 0.5
        return {
            "decision_time": timestamp,
            "mode": "oracle_replay" if oracle else "blind",
            "transition_evidence": {"confirmed_102": oracle},
            "agents": {"ml": ml, "rule_based": {"status": "unknown"}, "prognostic": []},
            "decision": {"decision_level": "unknown", "action": "no_automatic_action"},
        }

    def test_list_only_real_class2_parquet_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("WELL-b.parquet", "WELL-a.parquet", "SIMULATED-x.parquet", "WELL-q.csv"):
                (root / name).touch()
            self.assertEqual([f.name for f in list_class2_files(root)],
                             ["WELL-a.parquet", "WELL-b.parquet"])

    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list_class2_files(Path(folder)), [])

    def test_missing_ml_score_remains_missing_not_zero(self):
        timeline = build_ui_timeline([self._result()])
        self.assertTrue(pd.isna(timeline["ml_score_uncalibrated"].iloc[0]))
        chart = ml_chart_data(timeline)
        self.assertTrue(pd.isna(chart["ML-оценка (некалибр.)"].iloc[0]))
        self.assertTrue(pd.isna(chart["Порог 0,5"].iloc[0]))

    def test_chart_threshold_only_during_evaluated_ml(self):
        timeline = build_ui_timeline([
            self._result(),
            self._result(timestamp="2020-01-01T00:09:59", status="experimental_prediction", score=0.7, oracle=True),
        ])
        chart = ml_chart_data(timeline)
        self.assertTrue(np.isnan(chart["Порог 0,5"].iloc[0]))
        self.assertEqual(chart["Порог 0,5"].iloc[1], 0.5)
        self.assertEqual(chart["ML-оценка (некалибр.)"].iloc[1], 0.7)

    def test_empty_results(self):
        self.assertTrue(build_ui_timeline([]).empty)
        self.assertTrue(ml_chart_data(build_ui_timeline([])).empty)

    def test_sensor_chart_uses_real_telemetry_only(self):
        idx = pd.date_range("2020-01-01", periods=120, freq="s")
        df = pd.DataFrame({"P-TPT": np.r_[np.ones(60), 3 * np.ones(60)], "class": [102] * 120}, index=idx)
        chart = sensor_chart_data(df, "P-TPT")
        self.assertEqual(len(chart), 2)
        self.assertEqual(chart["P-TPT"].tolist(), [1.0, 3.0])
        self.assertNotIn("class", chart.columns)

    def test_missing_sensor_not_fabricated(self):
        idx = pd.date_range("2020-01-01", periods=60, freq="s")
        df = pd.DataFrame({"P-TPT": [1.0] * 60}, index=idx)
        self.assertTrue(sensor_chart_data(df, "P-MON-CKP").empty)
        with self.assertRaises(ValueError):
            sensor_chart_data(df, "class")

    def test_output_directory_isolated_by_file_and_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "WELL-00002_example.parquet"
            base = Path(folder) / "out"
            a = output_directory(file, "blind", base)
            b = output_directory(file, "oracle_replay", base)
            self.assertEqual(a, base / "blind" / "WELL-00002_example")
            self.assertNotEqual(a, b)
            with self.assertRaises(ValueError):
                output_directory(file, "invalid", base)
            with self.assertRaises(ValueError):
                output_directory(Path(folder) / "SIMULATED-01.parquet", "blind", base)


    def test_full_replay_uses_existing_pipeline_and_writes_separate_folders(self):
        """Интеграционная проверка с синтетическим ML, без артефакта Petrobras."""
        class TestML:
            def predict(self, history, *, transition_102_confirmed, decision_time=None):
                if transition_102_confirmed is True:
                    return {"agent": "ml", "status": "experimental_prediction",
                            "model_score": 0.8, "above_research_threshold": True}
                return {"agent": "ml", "status": "not_applicable"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (model / "manifest.json").write_text("{}", encoding="utf-8")
            rules = root / "rules.json"
            rules.write_text(json.dumps({
                "schema_version": 1, "dataset": "Petrobras 3W Dataset v2.0.0",
                "event_class": 2, "approval_status": "draft", "rules": [],
            }), encoding="utf-8")
            source = root / "WELL-test.parquet"
            source.touch()
            index = pd.date_range("2020-01-01", periods=900, freq="s")
            frame = pd.DataFrame({
                "class": pd.array([102] * 900, dtype="Int64"),
                "P-TPT": np.ones(900), "T-TPT": np.ones(900),
            }, index=index)
            with patch("streamlit_app.pd.read_parquet", return_value=frame), \
                 patch("streamlit_app.MLAgent", return_value=TestML()):
                _, blind_results, blind_table, blind_summary, blind_dir = run_selected_file(
                    source, mode="blind", model_dir=model, rules_file=rules,
                    output_root=root / "out")
                _, oracle_results, oracle_table, oracle_summary, oracle_dir = run_selected_file(
                    source, mode="oracle_replay", model_dir=model, rules_file=rules,
                    output_root=root / "out")
            self.assertEqual(blind_summary["decisions_count"], 3)
            self.assertEqual(oracle_summary["decisions_count"], 3)
            self.assertEqual(blind_summary["ml_research_alerts_count"], 0)
            self.assertEqual(oracle_summary["ml_research_alerts_count"], 3)
            self.assertTrue(blind_table["ml_score_uncalibrated"].isna().all())
            self.assertTrue(oracle_table["ml_score_uncalibrated"].eq(0.8).all())
            self.assertTrue(all(r["decision"]["decision_level"] == "unknown" for r in oracle_results))
            self.assertNotEqual(blind_dir, oracle_dir)
            for dest in (blind_dir, oracle_dir):
                for name in ("decisions.jsonl", "timeline.csv", "summary.json"):
                    self.assertTrue((dest / name).is_file())

    def test_missing_model_fails_with_readable_message(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "WELL-test.parquet"
            source.touch()
            with self.assertRaisesRegex(FileNotFoundError, "артефакты ML-модели"):
                run_selected_file(source, mode="blind", model_dir=root / "missing")


if __name__ == "__main__":
    unittest.main()
