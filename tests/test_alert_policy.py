"""Проверки без Petrobras parquet; запускаются вместе со старым unittest."""
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
from experiments import evaluate_class2_alert_policy as p


class TestAlertPolicy(unittest.TestCase):
    @staticmethod
    def series(scores, start='2020-01-01', label=None):
        t = pd.date_range(start, periods=len(scores), freq='5min')
        return pd.DataFrame({'file': 'WELL-00001_a.parquet', 'decision_time': t,
                             'score': scores,
                             'target': label if label is not None else [0]*len(scores),
                             'event_time': [pd.NaT]*len(scores)})

    def test_one_confirmation_starts_on_first_high_score(self):
        got = p.trigger_policy(self.series([.1, .8, .9, .1, .9]), .5, 1)
        self.assertEqual(got.alert_active.tolist(), [False, True, True, False, True])
        self.assertEqual(got.alert_start.tolist(), [False, True, False, False, True])

    def test_two_confirmations_rejects_single_spikes(self):
        got = p.trigger_policy(self.series([.1, .8, .1, .9, .8]), .5, 2)
        self.assertEqual(got.alert_active.tolist(), [False, False, False, False, True])
        self.assertEqual(got.alert_start.tolist(), [False, False, False, False, True])

    def test_gap_resets_confirmation(self):
        data = self.series([.9, .9, .9, .9])
        data.loc[2:, 'decision_time'] += pd.Timedelta(minutes=5)
        got = p.trigger_policy(data, .5, 2)
        self.assertEqual(got.alert_active.tolist(), [False, True, False, True])
        self.assertEqual(got.alert_start.tolist(), [False, True, False, True])

    def test_file_boundary_resets_confirmation(self):
        data = self.series([.9, .9, .9, .9])
        data.loc[2:, 'file'] = 'WELL-00002_b.parquet'
        got = p.trigger_policy(data, .5, 2)
        self.assertEqual(got.alert_active.tolist(), [False, True, False, True])

    def test_future_values_do_not_change_previous_alert(self):
        data = self.series([.7, .8, .1, .9])
        first = p.trigger_policy(data, .5, 2)
        data.loc[3, 'score'] = 0.0
        second = p.trigger_policy(data, .5, 2)
        self.assertEqual(first.alert_active.iloc[:3].tolist(), second.alert_active.iloc[:3].tolist())

    def test_target_and_event_labels_do_not_change_policy_triggers(self):
        data = self.series([.2, .8, .9, .3])
        a = p.trigger_policy(data, .5, 2)
        data['target'] = [1, 0, 1, 1]
        data['event_time'] = pd.Timestamp('2020-01-01 00:30:00')
        b = p.trigger_policy(data, .5, 2)
        self.assertEqual(a.alert_start.tolist(), b.alert_start.tolist())
        self.assertEqual(a.alert_active.tolist(), b.alert_active.tolist())

    def test_metrics_count_distinct_events_not_positive_windows(self):
        data = self.series([.9, .9, .8, .1, .8], label=[0, 1, 1, 0, 0])
        data.loc[1:2, 'event_time'] = pd.Timestamp('2020-01-01 00:20:00')
        got = p.trigger_policy(data, .5, 1)
        stats = p.policy_metrics(got)
        self.assertEqual(stats['events_total'], 1)
        self.assertEqual(stats['events_detected_active'], 1)
        self.assertEqual(stats['events_detected_new_alarm'], 0) # early warning active already
        self.assertEqual(stats['out_of_horizon_alarm_starts'], 2)

    def test_inner_choice_only_depends_on_inner_dataframe(self):
        # Достаточно двух событий во внутреннем OOF, не для теста алгоритма обучения.
        data = self.series([.1, .3, .6, .9, .1, .2, .8, .9], label=[0,0,1,1,0,0,1,1])
        data.loc[2:3, 'event_time'] = pd.Timestamp('2020-01-01 00:25:00')
        data.loc[6:7, 'event_time'] = pd.Timestamp('2020-01-01 00:45:00')
        choice, grid = p.choose_policy(data, min_event_recall=.5)
        self.assertIn(choice['threshold'], p.THRESHOLDS)
        self.assertIn(choice['confirmations'], p.CONFIRMATIONS)
        self.assertEqual(len(grid), len(p.THRESHOLDS)*len(p.CONFIRMATIONS))

    def test_bad_parameters_rejected(self):
        with self.assertRaises(ValueError):
            p.trigger_policy(self.series([.3]), .0, 1)
        with self.assertRaises(ValueError):
            p.trigger_policy(self.series([.3]), .5, 3)
        with self.assertRaises(ValueError):
            p.choose_policy(self.series([.3]), 1.1)

    def test_duplicate_timestamp_rejected(self):
        data = self.series([.2,.3])
        data.loc[1, 'decision_time'] = data.loc[0, 'decision_time']
        with self.assertRaises(ValueError):
            p.trigger_policy(data, .5, 1)

    def test_loader_rejects_target_mismatch(self):
        cols = {x: [0.1] * 4 for x in p.FEATURES}
        feature = pd.DataFrame({'file': ['a','b','c','d'], 'well_id': [1,2,3,4],
                                'horizon_minutes': [30]*4,
                                'decision_time': pd.date_range('2020-01-01', periods=4),
                                'target': [0,0,1,1],
                                'event_time': [pd.NaT, pd.NaT, pd.Timestamp('2020-01-05'),pd.Timestamp('2020-01-06')],
                                **cols})
        pred = feature[list(p.KEYS) + ['target','event_time']].copy()
        pred['score'] = [.1,.2,.3,.4]
        pred.loc[0, 'target'] = 1
        with tempfile.TemporaryDirectory() as tmp:
            f, o = Path(tmp)/'features.csv', Path(tmp)/'oof.csv'
            feature.to_csv(f, index=False)
            pred.to_csv(o, index=False)
            with self.assertRaisesRegex(ValueError, 'Несовпадение целевых меток'):
                p.load_and_match(f,o)

    def test_inner_training_receives_only_training_wells(self):
        # Обход тяжёлых реальных моделей: проверяем *именно* границу доступа.
        all_rows=[]
        for well in (1,2,3,4):
            for j in range(8):
                event = pd.Timestamp('2020-01-01 00:45:00')
                record = {'file': f'WELL-{well:05d}_f.parquet', 'well_id': well,
                          'decision_time': pd.Timestamp('2020-01-01') + pd.Timedelta(minutes=5*j),
                          'target': int(j>=6), 'event_time': event if j>=6 else pd.NaT,
                          'score': .05 + j*.1}
                record.update({x: float(j + well) for x in p.FEATURES})
                all_rows.append(record)
        cohort = pd.DataFrame(all_rows)
        seen=[]
        def stub_inner(train, iterations):
            # Nested call can't see the actual outer test well.
            excluded = ({1,2,3,4}-set(train.well_id.unique())).pop()
            seen.append(excluded)
            return train[['file','well_id','decision_time','target','event_time','score']].copy()
        with patch.object(p, 'inner_oof', side_effect=stub_inner):
            out, selection, grid = p.nested_evaluation(cohort, min_event_recall=.5)
        self.assertEqual(sorted(seen), [1,2,3,4])
        self.assertEqual(len(out), len(cohort))
        self.assertEqual(len(selection), 4)

    def test_event_table_separates_early_start_and_active_horizon(self):
        d = self.series([.8, .9, .9, .9, .9, .9, .9], label=[0,0,0,1,1,1,1])
        d['event_time'] = [pd.NaT]*3 + [pd.Timestamp('2020-01-01 00:45:00')]*4
        d['well_id'] = 1
        got = p.trigger_policy(d, .5, 2)
        summary = p.event_table(got).iloc[0]
        self.assertTrue(summary['detected_active'])
        self.assertFalse(summary['detected_new_alarm'])
        self.assertTrue(summary['had_any_pre_event_alarm'])
        self.assertEqual(summary['lead_first_alarm_before_event_minutes'], 40.0)

    def test_chosen_policy_does_not_change_when_outer_labels_change(self):
        rows=[]
        for well in (1,2,3,4):
            for j in range(8):
                event = pd.Timestamp('2020-01-01 00:45:00')
                rows.append({'file': f'WELL-{well:05d}_f.parquet', 'well_id': well,
                             'decision_time': pd.Timestamp('2020-01-01') + pd.Timedelta(minutes=5*j),
                             'target': int(j>=6), 'event_time': event if j>=6 else pd.NaT,
                             'score': .05+j*.1})
        cohort = pd.DataFrame(rows)
        def stub(train, iterations):
            return train[['file','well_id','decision_time','target','event_time','score']].copy()
        with patch.object(p, 'inner_oof', side_effect=stub):
            _, a, _ = p.nested_evaluation(cohort)
            mutated=cohort.copy()
            mutated.loc[mutated.well_id.eq(4), 'target'] = 0
            mutated.loc[mutated.well_id.eq(4), 'event_time'] = pd.NaT
            _, b, _ = p.nested_evaluation(mutated)
        fields=['threshold','confirmations','inner_event_recall_active']
        self.assertEqual(a.loc[a.outer_test_well.eq(4),fields].to_dict('records'),
                         b.loc[b.outer_test_well.eq(4),fields].to_dict('records'))


if __name__ == '__main__':
    unittest.main()
