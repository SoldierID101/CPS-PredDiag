"""Локальные инварианты эксперимента hard negatives (без реального 3W)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import evaluate_class2_hard_negatives as h


def make_cohorts():
    positive, controls = [], []
    start = pd.Timestamp('2020-01-01')
    for well in (2, 3, 10):
        event_time = start + pd.Timedelta(minutes=80)
        for i in range(12):
            target = int(i >= 8)
            record = {
                'file': f'WELL-{well:05d}_event.parquet', 'well_id': well,
                'decision_time': start + pd.Timedelta(minutes=5*i),
                'event_time': event_time if target else pd.NaT,
                'target': target, 'horizon_minutes': 30,
            }
            record.update({key: float(0.05*i + (target * 4) + well * .002)
                           for key in h.FEATURES})
            positive.append(record)
    for well in (2, 3, 10, 19, 25, 28):
        for event_class in (0, 8):
            name = f'{event_class}/WELL-{well:05d}_control.parquet'
            for i in range(10):
                record = {'file': name, 'well_id': well, 'event_class_folder': event_class,
                          'control_phase': 'normal_only_file' if event_class == 0 else 'other_transition',
                          'decision_time': start + pd.Timedelta(minutes=5*i)}
                record.update({key: float(well*0.005 + i * .002)
                               for key in h.FEATURES})
                controls.append(record)
    return pd.DataFrame(positive), pd.DataFrame(controls)


class TestHardNegatives(unittest.TestCase):
    def setUp(self):
        self.train, self.controls = make_cohorts()

    def test_holdout_is_deterministic_and_disjoint(self):
        a, b = h.assign_external_holdout(set(range(1, 45)), {2, 3, 10})
        c, d = h.assign_external_holdout(set(range(44, 0, -1)), {2, 3, 10})
        self.assertEqual((a, b), (c, d))
        self.assertFalse(a & b)
        self.assertFalse(b & {2, 3, 10})
        self.assertEqual(a | b, set(range(1, 45)) - {2, 3, 10})

    def test_invalid_holdout_config_rejected(self):
        with self.assertRaises(ValueError):
            h.assign_external_holdout({1,2}, {2}, 0.0)
        with self.assertRaises(ValueError):
            h.assign_external_holdout({2}, {2})

    def test_training_controls_sample_caps(self):
        s = h.sample_train_controls(self.controls, {2,3,10,19,25,28}, per_file=4, per_well_class=5)
        self.assertTrue((s.groupby('file').size() <= 4).all())
        self.assertTrue((s.groupby(['well_id', 'event_class_folder']).size() <= 5).all())
        self.assertLess(len(s), len(self.controls))
        self.assertTrue(set(s.well_id).issubset({2,3,10,19,25,28}))

    def test_empty_training_controls_rejected(self):
        with self.assertRaises(ValueError):
            h.sample_train_controls(self.controls, {999})
        with self.assertRaises(ValueError):
            h.sample_train_controls(self.controls, {2}, per_file=0)

    def test_same_file_or_class2_in_controls_rejected(self):
        bad = self.controls.copy()
        bad.loc[0, 'file'] = self.train.file.iloc[0]
        with self.assertRaises(ValueError):
            h.validate_controls(bad, self.train)
        bad = self.controls.copy()
        bad.loc[0, 'event_class_folder'] = 2
        with self.assertRaises(ValueError):
            h.validate_controls(bad, self.train)

    def test_duplicate_control_window_rejected(self):
        bad = pd.concat([self.controls, self.controls.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            h.validate_controls(bad, self.train)

    def test_selected_file_rejects_path_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'0').mkdir()
            (root/'0'/'WELL-00002_20200101000000.parquet').touch()
            path = root/'selected.csv'
            pd.DataFrame([{'event_class_folder':0,'well_id':3,
                           'file':'0/WELL-00002_20200101000000.parquet'}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                h.load_selected(path, root)
            pd.DataFrame([{'event_class_folder':2,'well_id':2,
                           'file':'2/WELL-00002_20200101000000.parquet'}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                h.load_selected(path, root)

    def test_training_fold_does_not_train_on_heldout_well(self):
        # Реальный CatBoost: 6 внешних моделей и 2 модели для контрольных скважин.
        scores, ctrl_scores, sampled, folds = h.make_outer_predictions(
            self.train, self.controls, {19,25}, {28}, iterations=12, per_file=4)
        self.assertEqual(len(scores), len(self.train)*2)
        self.assertEqual(len(folds), 3)
        self.assertTrue((folds.train_control_rows > 0).all())
        self.assertEqual(set(ctrl_scores.cohort),
                         {'loow_class2_wells','heldout_control_only_wells'})
        self.assertEqual(set(ctrl_scores.loc[ctrl_scores.cohort.eq('heldout_control_only_wells'), 'well_id']), {28})
        self.assertFalse(sampled.well_id.eq(28).any())
        self.assertTrue(scores.score.between(0,1).all())
        self.assertTrue(ctrl_scores.score.between(0,1).all())
        self.assertFalse(scores.duplicated(['variant','file','decision_time']).any())

    def test_horizon_event_metrics_and_controls(self):
        scores, ctrl_scores, _, _ = h.make_outer_predictions(
            self.train, self.controls, {19,25}, {28}, iterations=12, per_file=4)
        event, event_wells = h.score_event_metrics(scores)
        controls, per_class, per_well = h.score_control_metrics(ctrl_scores)
        self.assertEqual(set(event.policy), {'fixed05_1','fixed05_2'})
        self.assertEqual(set(event.variant), {'baseline','hard_negative'})
        self.assertTrue((event.events_total == 3).all())
        self.assertTrue(event.ap.between(0,1).all())
        self.assertGreater(len(event_wells), 0)
        self.assertTrue(controls.active_fraction.between(0,1).all())
        self.assertTrue((controls.eligible_negative_hours > 0).all())
        self.assertGreater(len(per_class), 0)
        self.assertGreater(len(per_well), 0)

    def test_false_alarm_starts_distinct_from_active_time(self):
        r = pd.DataFrame({'variant':['baseline']*7,'file':['0/WELL-00019_20200101000000.parquet']*7,
                          'well_id':[19]*7,'cohort':['heldout_control_only_wells']*7,
                          'control_phase':['normal_only_file']*7,'event_class_folder':[0]*7,
                          'decision_time':pd.date_range('2020-01-01', periods=7, freq='5min'),
                          'score':[.9,.8,.8,.8,.8,.1,.1]})
        outcome = h.apply_fixed_policy(r, 2)
        self.assertEqual(int(outcome.alert_start.sum()),1)
        self.assertEqual(int(outcome.alert_active.sum()),4)

    def test_target_and_state_cannot_change_control_predictions(self):
        a = self.controls.copy()
        a['class'] = 102
        a['state'] = 'oracle'
        b = self.controls.copy()
        b['class'] = -1
        b['state'] = 'alternate'
        model = h.fit_catboost(self.train, None, iterations=12)
        np.testing.assert_allclose(h.score_model(model,a),h.score_model(model,b))


if __name__ == '__main__':
    unittest.main()
