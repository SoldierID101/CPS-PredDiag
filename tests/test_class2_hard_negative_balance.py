"""Проверка протокола Hard Negative Balance на искусственных данных, не Petrobras."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import evaluate_class2_hard_negative_balance as h


def synth():
    events, controls = [], []
    start = pd.Timestamp("2020-01-01")
    for well in (2, 3, 9, 10):
        for i in range(9):
            target = int(i >= 6)
            row = {"file": f"WELL-{well:05d}_20200101010000.parquet", "well_id": well,
                   "horizon_minutes": 30, "decision_time": start + pd.Timedelta(minutes=5*i),
                   "event_time": str(start + pd.Timedelta(minutes=50)) if target else None,
                   "target": target}
            row.update({k: float(i * 0.07 + well * .01 + target * .4) for k in h.FEATURES})
            events.append(row)
    for well in (2, 3, 9, 10, 4, 6, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29,
                 30, 33, 35, 36, 37, 38, 39, 41, 1, 5, 7, 8, 28, 31, 32, 34, 40, 42):
        # Данные искусственные; ни одного DHSV события в этих контролях нет.
        for i in range(5):
            class_folder = 0 if i % 2 == 0 else 8
            path = f"{class_folder}/WELL-{well:05d}_20200101000000.parquet"
            row = {"file": path, "well_id": well, "event_class_folder": class_folder,
                   "control_phase": "normal_only_file" if class_folder == 0 else "other_event",
                   "decision_time": start + pd.Timedelta(minutes=i*5)}
            row.update({k: float(well * .01 + .005*i) for k in h.FEATURES})
            controls.append(row)
    return pd.DataFrame(events), pd.DataFrame(controls)


class TestHardNegativeBalance(unittest.TestCase):
    def setUp(self):
        self.events, self.controls = synth()

    def test_partition_deterministic_and_complete(self):
        wells = set(range(20, 65))
        a, b = h.partition_development_controls(wells)
        c, d = h.partition_development_controls(set(reversed(sorted(wells))))
        self.assertEqual((a, b), (c, d))
        self.assertFalse(a & b)
        self.assertEqual(a | b, wells)

    def test_partition_invalid(self):
        with self.assertRaises(ValueError):
            h.partition_development_controls({1, 2})
        with self.assertRaises(ValueError):
            h.partition_development_controls({1, 2, 3}, 1.0)

    def test_stratified_tuning_covers_normal_and_class8(self):
        s = pd.DataFrame([{"well_id":4,"event_class_folder":0},
                          {"well_id":6,"event_class_folder":0},
                          {"well_id":19,"event_class_folder":0},
                          {"well_id":19,"event_class_folder":8},
                          {"well_id":25,"event_class_folder":8},
                          {"well_id":14,"event_class_folder":9}])
        fit, tune = h.ensure_tuning_class_coverage({4,6,19,25}, {14}, s)
        self.assertEqual(tune, {14,19})
        self.assertEqual(fit, {4,6,25})
        self.assertFalse(fit & tune)

    def test_nested_ratio_selected_using_only_inner_metrics(self):
        choices = pd.DataFrame([
            {"ratio":0., "threshold":.5,"confirmations":1,
             "inner_target_reached":True,"inner_event_recall":.9,"inner_combined_burden":.4},
            {"ratio":.25, "threshold":.3,"confirmations":2,
             "inner_target_reached":True,"inner_event_recall":.85,"inner_combined_burden":.1},
            {"ratio":.50, "threshold":.2,"confirmations":2,
             "inner_target_reached":False,"inner_event_recall":.7,"inner_combined_burden":.0},
        ])
        selected = h.choose_nested_variant(choices)
        self.assertEqual(selected['ratio'], .25)
        self.assertEqual(selected['threshold'], .3)
        self.assertEqual(selected['confirmations'], 2)

    def test_fractions_are_nested(self):
        sample = h.hn.sample_train_controls(self.controls, set(self.controls.well_id), 12, 24)
        zero = h.fraction_of_controls(sample, 0.0)
        one = h.fraction_of_controls(sample, 0.25)
        two = h.fraction_of_controls(sample, 0.5)
        full = h.fraction_of_controls(sample, 1.0)
        self.assertEqual(len(zero), 0)
        self.assertTrue(set(one.index).issubset(set(range(len(one)))))
        ids = lambda df: set(map(tuple, df[["file", "decision_time"]].to_numpy()))
        self.assertTrue(ids(one).issubset(ids(two)))
        self.assertTrue(ids(two).issubset(ids(full)))
        self.assertLess(len(one), len(two))
        self.assertLess(len(two), len(full))

    def test_fractions_reject_unlisted_ratio(self):
        with self.assertRaises(ValueError):
            h.fraction_of_controls(self.controls, 0.75)

    def test_inner_oof_no_test_well_leakage_and_numerical(self):
        pool = self.controls.loc[self.controls.well_id.isin({2, 3, 9, 10, 4, 6})]
        inner = h.inner_predictions(self.events, pool, ratio=0.25, iterations=12)
        self.assertEqual(len(inner), len(self.events))
        self.assertTrue(inner.score.between(0, 1).all())
        self.assertFalse(inner.duplicated(["file", "decision_time"]).any())

    def test_policy_grid_target_and_control_tradeoff(self):
        start = pd.Timestamp('2020-01-01')
        ev = pd.DataFrame({"file": ["event-A"]*6, "well_id": [2]*6,
                           "decision_time": pd.date_range(start, periods=6, freq="5min"),
                           "target": [0, 0, 0, 1, 1, 1],
                           "event_time": [None]*3 + [start + pd.Timedelta(minutes=35)]*3,
                           "score": [.2, .3, .4, .7, .8, .9]})
        ct = pd.DataFrame({"file": ["other-B"]*6, "well_id": [19]*6,
                           "decision_time": pd.date_range(start, periods=6, freq="5min"),
                           "score": [.05, .15, .3, .5, .6, .2],
                           "event_class_folder": [0, 0, 0, 8, 8, 8]})
        p, grid = h.policy_grid(ev, ct)
        self.assertEqual(len(grid), len(h.THRESHOLDS)*len(h.CONFIRMATIONS))
        self.assertTrue(p["inner_target_reached"])
        self.assertGreaterEqual(p["inner_event_recall"], .8)
        self.assertIn(p["confirmations"], (1, 2))

    def test_policy_control_overlap_rejected(self):
        ev = pd.DataFrame({"file": ["a", "a"], "well_id": [2, 2],
                           "decision_time": pd.date_range('2020-01-01', periods=2, freq='5min'),
                           "target": [0, 1], "event_time": [None, '2020-01-02'],
                           "score": [.3, .7]})
        ct = pd.DataFrame({"file": ['b'], "well_id": [2],
                           "decision_time": ['2020-01-01'], "score": [.8]})
        with self.assertRaises(ValueError):
            h.policy_grid(ev, ct)

    def test_policy_cannot_read_labels_or_future(self):
        rows = self.events.loc[self.events.well_id.eq(2)].copy()
        model = h._model(self.events.loc[self.events.well_id.ne(2)],
                         self.controls.iloc[0:0], 12)
        scores = h._score_class2(model, rows)
        p = {"threshold": 0.5, "confirmations": 2}
        a = h.apply_rule(scores, p)
        b = scores.copy()
        b['target'] = 1 - b['target']
        b['event_time'] = '1900-01-01'
        b['class'] = 102
        b['state'] = 'oracle'
        applied = h.apply_rule(b, p)
        np.testing.assert_array_equal(a.alert_active.to_numpy(), applied.alert_active.to_numpy())
        np.testing.assert_array_equal(a.alert_start.to_numpy(), applied.alert_start.to_numpy())

    def test_alarm_starts_not_same_as_active_fraction(self):
        control = pd.DataFrame({"file": ['z']*5, "well_id": [19]*5,
                                "decision_time": pd.date_range('2020-01-01', periods=5, freq='5min'),
                                "score": [.9,.9,.9,.9,.0]})
        checked = h.apply_rule(control, {"threshold": .5, "confirmations": 2})
        counts = h.control_row(checked)
        self.assertEqual(counts['alarm_starts'], 1)
        self.assertEqual(round(counts['active_fraction'], 3), .6)

    def test_event_metrics_count_episodes(self):
        ev = self.events.loc[self.events.well_id.eq(2)].copy()
        ev['score'] = [.1]*6 + [.9]*3
        scored = h.apply_rule(ev, {"threshold": .5, "confirmations": 1})
        result = h.event_row(scored)
        self.assertEqual(result['events_total'], 1)
        self.assertEqual(result['events_detected'], 1)
        self.assertEqual(result['fp_windows'], 0)

    def test_synthetic_run_outputs_and_separate_test_wells(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data_dir = root / 'dataset'
            selected = []
            for (file, well, event_class), group in self.controls.groupby(
                    ['file', 'well_id', 'event_class_folder']):
                p = data_dir / file
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                selected.append({'file': file, 'well_id': well,
                                 'event_class_folder': event_class})
            selected_path = root / 'selected.csv'
            pd.DataFrame(selected).to_csv(selected_path, index=False)
            train_path = root / 'training.csv'
            self.events.to_csv(train_path, index=False)
            audit = pd.DataFrame([{'file': 'synthetic', 'error': None}])
            with patch.object(h.ctl, 'build_controls', return_value=(self.controls, audit)):
                events, controls = h.run(train_path, selected_path, data_dir,
                                         root/'out', iterations=12, per_file=3,
                                         per_well_class=6)
            self.assertEqual(len(events), 12)
            self.assertEqual(len(controls), 24)
            self.assertEqual(set(events.ratio), set(h.RATIOS))
            self.assertTrue((events.events_total == 4).all())
            self.assertTrue((controls.windows > 0).all())
            meta = pd.read_json(root/'out'/'experiment.json', typ='series')
            self.assertEqual(len(meta['class2_wells']), 4)
            self.assertFalse(set(meta['fit_dev_wells']) & set(meta['external_control_wells']))
            self.assertFalse(set(meta['tune_dev_wells']) & set(meta['external_control_wells']))


if __name__ == '__main__':
    unittest.main()
