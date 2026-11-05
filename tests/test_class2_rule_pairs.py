"""Тесты аудита двухдатчиковых признаков без разметки в формуле."""
from __future__ import annotations
import unittest
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_class2_rule_pairs import evaluate_pairs, summarize_pairs

class TestClass2RulePairs(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "file": ["a", "a", "b"], "well": ["W1", "W1", "W2"],
            "phase": ["transition_102", "normal_0", "transition_102"],
            "decision_time": ["t1", "t2", "t3"],
            "pdg_slope_pa_h": [2.0, -1.0, 1.0],
            "tpt_slope_pa_h": [-1.0, -1.0, 0.0],
            "mon_ckp_slope_pa_h": [-2.0, float("nan"), -2.0],
        })

    def test_expected_signs(self):
        result = evaluate_pairs(self.df)
        pdg_tpt = result[result["pair"] == "PDG_up_TPT_down"].reset_index(drop=True)
        self.assertEqual(pdg_tpt["sign_match"].tolist(), [True, False, False])

    def test_unavailable_is_not_false(self):
        result = evaluate_pairs(self.df)
        pdg_mon = result[result["pair"] == "PDG_up_MON_down"].reset_index(drop=True)
        self.assertTrue(pd.isna(pdg_mon.loc[1, "sign_match"]))
        self.assertFalse(bool(pdg_mon.loc[1, "eligible"]))

    def test_zero_slope_does_not_match(self):
        result = evaluate_pairs(self.df)
        pdg_tpt = result[result["pair"] == "PDG_up_TPT_down"].reset_index(drop=True)
        self.assertFalse(bool(pdg_tpt.loc[2, "sign_match"]))
        self.assertTrue(bool(pdg_tpt.loc[2, "eligible"]))

    def test_summary_uses_eligible_denominator(self):
        result = evaluate_pairs(self.df)
        summary = summarize_pairs(result, ["pair", "phase"])
        x = summary[(summary["pair"] == "PDG_up_MON_down") & (summary["phase"] == "normal_0")].iloc[0]
        self.assertEqual(int(x["eligible_windows"]), 0)
        self.assertTrue(pd.isna(x["fraction_given_eligible"]))
        self.assertEqual(int(x["all_windows"]), 1)

    def test_labels_do_not_change_predictions(self):
        a = evaluate_pairs(self.df)
        modified = self.df.copy(); modified["phase"] = "event_2"
        b = evaluate_pairs(modified)
        self.assertEqual(a["sign_match"].tolist(), b["sign_match"].tolist())

    def test_missing_field_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_pairs(self.df.drop(columns=["pdg_slope_pa_h"]))

if __name__ == "__main__":
    unittest.main()
