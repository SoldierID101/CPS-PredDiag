"""Тесты Prognostic Agent. Все пороги здесь искусственные и только для тестов."""
from __future__ import annotations
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prognostic_agent import PrognosticAgent
from coordinator import coordinate_decision


def synthetic_rule(threshold=32.0, direction="above"):
    return {"id": "SYNTHETIC-ONLY", "sensor": "P-TPT", "threshold": threshold,
            "direction": direction, "severity": "warning", "unit": "Pa",
            "source": "SYNTHETIC unit test; not a Petrobras threshold"}


def synthetic_history(values=None, freq="s"):
    if values is None:
        values = np.linspace(20, 30, 300)
    idx = pd.date_range("2026-01-01T00:00:00", periods=len(values), freq=freq)
    return pd.DataFrame({"P-TPT": values}, index=idx)


class TestPrognosticAgent(unittest.TestCase):
    def test_default_has_no_threshold_and_no_prediction(self):
        self.assertEqual(PrognosticAgent().predict(synthetic_history()), [])

    def test_predicts_trend_crossing_not_failure(self):
        result = PrognosticAgent([synthetic_rule()]).predict(synthetic_history())[0]
        self.assertEqual(result["forecast_status"], "predicted_crossing")
        self.assertTrue(0 < result["time_to_threshold_h"] <= 0.25)
        self.assertTrue(result["research_only"])
        self.assertIn("НЕ время до отказа", result["note"])

    def test_crossing_integrates_with_coordinator(self):
        prediction = PrognosticAgent([synthetic_rule()]).predict(synthetic_history())
        decision = coordinate_decision(predictions=prediction)
        self.assertEqual(decision["forecast_status"], "predicted_crossing")
        self.assertEqual(decision["decision_level"], "warning")
        self.assertEqual(decision["nearest_sensor"], "P-TPT")

    def test_already_exceeded(self):
        result = PrognosticAgent([synthetic_rule(25)]).predict(synthetic_history())[0]
        self.assertEqual(result["forecast_status"], "already_exceeded")
        self.assertEqual(result["time_to_threshold_h"], 0)

    def test_away_from_threshold_no_forecast(self):
        result = PrognosticAgent([synthetic_rule(40)]).predict(synthetic_history(np.linspace(30, 20, 300)))[0]
        self.assertEqual(result["forecast_status"], "no_forecast")
        self.assertIsNone(result["time_to_threshold_h"])

    def test_missing_sensor_is_not_normal(self):
        df = synthetic_history().rename(columns={"P-TPT": "P-PDG"})
        result = PrognosticAgent([synthetic_rule()]).predict(df)[0]
        self.assertEqual(result["forecast_status"], "insufficient_data")
        self.assertEqual(coordinate_decision(predictions=[result])["decision_level"], "unknown")

    def test_missing_last_value_is_insufficient(self):
        df = synthetic_history(); df.iloc[-1, 0] = np.nan
        self.assertEqual(PrognosticAgent([synthetic_rule()]).predict(df)[0]["forecast_status"], "insufficient_data")

    def test_irregular_timestamps_rejected(self):
        df = synthetic_history(freq="2s")
        self.assertEqual(PrognosticAgent([synthetic_rule()]).predict(df)[0]["forecast_status"], "insufficient_data")

    def test_microsecond_datetime_resolution_supported(self):
        df = synthetic_history(); df.index = df.index.as_unit("us")
        self.assertEqual(PrognosticAgent([synthetic_rule()]).predict(df)[0]["forecast_status"], "predicted_crossing")

    def test_future_rows_never_read(self):
        df = synthetic_history(np.linspace(10, 20, 350))
        first = PrognosticAgent([synthetic_rule(25)]).predict(df)[0]
        df.iloc[:50, 0] = -1e8  # история ДО последних 300 секунд
        second = PrognosticAgent([synthetic_rule(25)]).predict(df)[0]
        self.assertEqual(first["forecast_status"], second["forecast_status"])
        self.assertAlmostEqual(first["trend_slope_per_hour"], second["trend_slope_per_hour"])

    def test_label_column_not_used(self):
        df = synthetic_history(); df["class"] = 2; df["state"] = 102
        a = PrognosticAgent([synthetic_rule()]).predict(df)[0]
        df["class"] = 0; df["state"] = -999
        b = PrognosticAgent([synthetic_rule()]).predict(df)[0]
        self.assertEqual(a["forecast_status"], b["forecast_status"])
        self.assertAlmostEqual(a["time_to_threshold_h"], b["time_to_threshold_h"])

    def test_no_source_or_oracle_sensor_rejected(self):
        config = synthetic_rule(); del config["source"]
        with self.assertRaises(ValueError):
            PrognosticAgent([config])
        config = synthetic_rule(); config["sensor"] = "class"
        with self.assertRaises(ValueError):
            PrognosticAgent([config])

    def test_outside_horizon_no_forecast(self):
        result = PrognosticAgent([synthetic_rule(threshold=1000)]).predict(synthetic_history())[0]
        self.assertEqual(result["forecast_status"], "no_forecast")

if __name__ == "__main__":
    unittest.main()
