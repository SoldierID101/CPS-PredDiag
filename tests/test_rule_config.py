"""Проверка загрузчика JSON, только синтетические пороги в тестах."""

import json
import tempfile
import unittest
from pathlib import Path

from coordinator import coordinate_decision
from rule_config import load_rule_agent, load_rule_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "rules" / "class2_dhsv.json"


def synthetic_config(*, status="approved", rules=None, event_class=2):
    return {
        "schema_version": 1,
        "dataset": "synthetic unit test (not Petrobras thresholds)",
        "event_class": event_class,
        "approval_status": status,
        "rules": [] if rules is None else rules,
    }


def synthetic_rule(**changes):
    data = {
        "id": "TEST_ONLY", "sensor": "P-TPT", "condition": ">",
        "threshold": 10, "severity": "warning", "unit": "synthetic",
        "source": "synthetic unit test only", "message": "Тестовое превышение",
    }
    data.update(changes)
    return data


class TestRuleConfig(unittest.TestCase):
    def write_config(self, folder, config):
        path = Path(folder) / "config.json"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return path

    def test_draft_has_no_active_rules_and_returns_unknown(self):
        agent = load_rule_agent(DEFAULT_JSON, expected_event_class=2)
        self.assertEqual(agent.rules, ())
        result = agent.evaluate({"P-TPT": 1e8, "class": 102})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(coordinate_decision(result)["decision_level"], "unknown")

    def test_loading_valid_synthetic_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(rules=[synthetic_rule()]))
            agent = load_rule_agent(path, expected_event_class=2)
            self.assertEqual(agent.evaluate({"P-TPT": 11})["status"], "warning")
            self.assertEqual(agent.evaluate({"P-TPT": 9})["status"], "normal")

    def test_draft_with_active_rules_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(status="draft", rules=[synthetic_rule()]))
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_wrong_event_class_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(event_class=8))
            with self.assertRaises(ValueError):
                load_rule_agent(path, expected_event_class=2)

    def test_invalid_threshold_rejected_by_existing_agent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(rules=[synthetic_rule(threshold=None)]))
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_oracle_class_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(rules=[synthetic_rule(sensor="class")]))
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_duplicate_rule_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_config(temp, synthetic_config(rules=[synthetic_rule(), synthetic_rule()]))
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_invalid_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"rules":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Некорректный JSON"):
                load_rule_agent(path)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                load_rule_agent(Path(temp) / "missing.json")

    def test_wrong_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = synthetic_config()
            cfg["schema_version"] = 2
            path = self.write_config(temp, cfg)
            with self.assertRaises(ValueError):
                load_rule_config(path)

    def test_invalid_approval_status_type_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = synthetic_config()
            cfg["approval_status"] = []
            path = self.write_config(temp, cfg)
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_rules_must_be_list(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = synthetic_config()
            cfg["rules"] = "not a list"
            path = self.write_config(temp, cfg)
            with self.assertRaises(ValueError):
                load_rule_agent(path)

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = synthetic_config()
            cfg["candidate_rules"] = [{"id": "UNVALIDATED"}]
            path = self.write_config(temp, cfg)
            with self.assertRaises(ValueError):
                load_rule_agent(path)


if __name__ == "__main__":
    unittest.main()
