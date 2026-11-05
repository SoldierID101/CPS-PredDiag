"""Синтетические тесты контракта Rule-Based Agent. Это НЕ реальные пороги Petrobras."""
import unittest

from coordinator import coordinate_decision
from explanation import generate_explanation
from rule_based_agent import RuleBasedAgent


def example_rule(sensor="P-TPT", severity="warning"):
    return {"id": "SYNTHETIC_TEST", "sensor": sensor, "condition": ">",
            "threshold": 10.0, "severity": severity, "unit": "unit-test",
            "source": "synthetic unit test only", "message": "Тестовое превышение"}


class TestRuleBasedAgent(unittest.TestCase):
    def test_without_rules_is_unknown(self):
        result = RuleBasedAgent().evaluate({"P-TPT": 30.0})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(coordinate_decision(result)["decision_level"], "unknown")

    def test_below_threshold_is_normal_if_all_available(self):
        result = RuleBasedAgent([example_rule()]).evaluate({"P-TPT": 8.0})
        self.assertEqual(result["status"], "normal")
        self.assertTrue(result["complete"])

    def test_warning_reaches_coordinator(self):
        result = RuleBasedAgent([example_rule()]).evaluate({"P-TPT": 12.0})
        decision = coordinate_decision(rule_summary=result)
        self.assertEqual(decision["decision_level"], "warning")
        self.assertEqual(len(result["triggered_rules"]), 1)
        self.assertIn("Тестовое превышение", generate_explanation(result, [], decision))

    def test_critical_reaches_coordinator(self):
        result = RuleBasedAgent([example_rule(severity="critical")]).evaluate({"P-TPT": 12.0})
        self.assertEqual(coordinate_decision(rule_summary=result)["decision_level"], "critical")

    def test_missing_sensor_does_not_mean_normal(self):
        result = RuleBasedAgent([example_rule()]).evaluate({"T-TPT": 7.0})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["unavailable_rule_ids"], ["SYNTHETIC_TEST"])

    def test_partial_data_does_not_hide_observed_violation(self):
        other = dict(example_rule(sensor="T-TPT"), id="SECOND_RULE")
        result = RuleBasedAgent([example_rule(), other]).evaluate({"P-TPT": 12.0})
        self.assertEqual(result["status"], "warning")
        self.assertFalse(result["complete"])

    def test_oracle_labels_cannot_be_rules(self):
        with self.assertRaises(ValueError):
            RuleBasedAgent([example_rule(sensor="class")])

    def test_non_finite_reading_is_unavailable(self):
        result = RuleBasedAgent([example_rule()]).evaluate({"P-TPT": float("nan")})
        self.assertEqual(result["status"], "unknown")

    def test_threshold_requires_provenance_and_units(self):
        rule = example_rule()
        rule.pop("source")
        with self.assertRaises(ValueError):
            RuleBasedAgent([rule])


if __name__ == "__main__":
    unittest.main()
