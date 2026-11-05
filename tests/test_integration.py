"""Запуск из корня CPS-PredDiag: python -m unittest discover -s tests -v"""
import unittest

from coordinator import coordinate_decision
from explanation import generate_explanation


class TestCoordinator(unittest.TestCase):
    def test_no_agents_is_unknown_not_normal(self):
        decision = coordinate_decision()
        self.assertEqual(decision["decision_level"], "unknown")
        self.assertEqual(decision["action"], "no_automatic_action")

    def test_ml_only_is_research_signal_not_authority(self):
        ml = {"status": "experimental_prediction", "model_score": 0.8361,
              "above_research_threshold": True, "horizon_minutes": 15}
        decision = coordinate_decision(ml_result=ml)
        self.assertEqual(decision["decision_level"], "unknown")
        self.assertTrue(decision["ml_research_alert"])
        explanation = generate_explanation(decision=decision)
        self.assertIn("не калибрована", explanation)

    def test_ml_not_applicable(self):
        decision = coordinate_decision(ml_result={"status": "not_applicable"})
        self.assertFalse(decision["ml_research_alert"])
        self.assertEqual(decision["decision_level"], "unknown")

    def test_rule_critical_not_changed_by_ml(self):
        decision = coordinate_decision({"status": "critical", "critical_count": 1}, [],
                                       ml_result={"status": "experimental_prediction",
                                                  "above_research_threshold": False})
        self.assertEqual(decision["decision_level"], "critical")
        self.assertEqual(decision["action"], "immediate_attention")

    def test_forecast_threshold_crossing_not_failure(self):
        decision = coordinate_decision(
            {"status": "normal"},
            [{"forecast_status": "predicted_crossing", "sensor": "P-TPT",
              "severity": "critical", "time_to_threshold_h": 1.5}],
        )
        self.assertEqual(decision["decision_level"], "warning")
        self.assertEqual(decision["action"], "plan_maintenance")
        self.assertIn("НЕ прогноз времени отказа", decision["reasons"][0])

    def test_rule_normal_does_not_misstate_future(self):
        decision = coordinate_decision({"status": "normal"})
        exp = generate_explanation({"status": "normal"}, [], decision)
        self.assertEqual(decision["decision_level"], "normal")
        self.assertIn("не означает отсутствия", exp)


if __name__ == "__main__":
    unittest.main()
