"""Unit checks for blind replay; no Petrobras data or fitted model artifact required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.offline_replay_ml_v02 import (
    telemetry_only, decision_positions, parse_well_id, load_cohort_for_well,
    load_reference, replay, _max_feature_difference,
)
from ml_agent import FEATURES, SENSORS, features_from_history


class ConstantModel:
    classes_ = [0, 1]

    def __init__(self, score=.7):
        self.score = score
        self.calls = 0

    def predict_proba(self, features):
        self.calls += 1
        assert features.columns.tolist() == FEATURES
        return np.array([[1-self.score, self.score]])


def _data(n=1800, *, start='2017-03-20 00:00:00'):
    time = pd.date_range(start, periods=n, freq='s').as_unit('us')
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        'timestamp': time,
        'P-MON-CKP': 10_000_000 + 0.1*x,
        'P-TPT': 20_000_000 - 0.5*x,
        'T-TPT': 100 + 0.01*x,
        'class': np.full(n, 102),
        'state': np.full(n, 102),
    })


def _expected(raw, indices):
    tel = telemetry_only(raw)
    rows = []
    for pos in indices:
        values, _ = features_from_history(tel.iloc[pos-299:pos+1].copy().set_axis(
            tel.index[pos-299:pos+1].as_unit('ns'), axis=0))
        row = {'file': 'WELL-00012_20170320011000.parquet',
               'well_id': 12, 'decision_time': tel.index[pos], 'target': int(pos % 2 == 0),
               **values.iloc[0].to_dict()}
        rows.append(row)
    return pd.DataFrame(rows)


class ReplayTests(unittest.TestCase):
    def test_schedule_ignores_class_state(self):
        raw = _data()
        one = telemetry_only(raw)
        changed = raw.copy()
        changed['class'] = np.where(np.arange(len(changed)) % 2, 2, np.nan)
        changed['state'] = 777
        two = telemetry_only(changed)
        pd.testing.assert_frame_equal(one, two)
        self.assertEqual(decision_positions(one.index), [899, 1199, 1499, 1799])

    def test_schedule_uses_new_segment_after_gap(self):
        raw = _data(1000)
        b = _data(1000, start='2017-03-21 00:00:00')
        both = pd.concat([raw, b], ignore_index=True)
        idx = telemetry_only(both).index
        self.assertEqual(decision_positions(idx), [899, 1899])

    def test_missing_label_columns_supported(self):
        raw = _data().drop(columns=['class', 'state'])
        self.assertEqual(len(decision_positions(telemetry_only(raw).index)), 4)

    def test_time_index_microseconds_kept(self):
        t = telemetry_only(_data())
        self.assertEqual(str(t.index.dtype), 'datetime64[ns]')
        self.assertEqual(t.columns.tolist(), list(SENSORS))

    def test_duplicate_time_rejected(self):
        raw = _data()
        raw.loc[12, 'timestamp'] = raw.loc[11, 'timestamp']
        with self.assertRaises(ValueError):
            telemetry_only(raw)

    def test_unordered_time_rejected(self):
        raw = _data().iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            telemetry_only(raw)

    def test_filename_parser(self):
        self.assertEqual(parse_well_id('WELL-00012_20170320011000.parquet'), 12)
        with self.assertRaises(ValueError):
            parse_well_id('../WELL-00012_20170320011000.parquet')

    def test_shadow_replay_feature_and_score_parity(self):
        raw = _data()
        cohort = _expected(raw, [899, 1199, 1499])
        scores = cohort[['file', 'well_id', 'decision_time']].copy()
        scores['score'] = .7
        scores['alert_active'] = True
        out, summary = replay(raw, well=12, file=cohort.file.iloc[0], model=ConstantModel(),
                              test_cohort=cohort, reference=scores)
        self.assertEqual(len(out), 4)
        self.assertEqual(summary['archived_eligible_decisions'], 3)
        self.assertEqual(summary['non_comparable_decisions'], 1)
        self.assertTrue(summary['eligible_feature_match'])
        self.assertTrue(summary['eligible_score_match'])
        self.assertTrue(summary['eligible_alarm_match'])

    def test_replay_future_change_not_affect_past(self):
        raw = _data()
        cohort = _expected(raw, [899, 1199])
        a, _ = replay(raw, well=12, file=cohort.file.iloc[0], model=ConstantModel(),
                      test_cohort=cohort, reference=None)
        raw2 = raw.copy()
        raw2.loc[raw2.index >= 1200, 'P-TPT'] = 1e10
        b, _ = replay(raw2, well=12, file=cohort.file.iloc[0], model=ConstantModel(),
                      test_cohort=cohort, reference=None)
        pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])

    def test_unknown_retrospective_period_not_negative(self):
        raw = _data()
        cohort = _expected(raw, [1199, 1499])
        out, sm = replay(raw, well=12, file=cohort.file.iloc[0], model=ConstantModel(),
                         test_cohort=cohort, reference=None)
        self.assertFalse(bool(out.iloc[0].in_archived_risk_set))
        self.assertEqual(sm['non_comparable_decisions'], 2)
        self.assertIsNone(sm['eligible_score_match'])

    def test_reference_must_have_matching_moments(self):
        raw = _data()
        cohort = _expected(raw, [899, 1199])
        reference = cohort[['file', 'well_id', 'decision_time']].iloc[:1].copy()
        reference['score'] = .7
        reference['alert_active'] = True
        with self.assertRaisesRegex(ValueError, 'разные моменты'):
            replay(raw, well=12, file=cohort.file.iloc[0], model=ConstantModel(),
                   test_cohort=cohort, reference=reference)

    def test_well_held_out_from_training(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cohort_features.csv'
            examples = []
            for w in (2, 3, 12):
                for target in (0, 1):
                    examples.append({'well_id': w, 'horizon_minutes': 30,
                        'file': f'WELL-{w:05d}_20170320011000.parquet',
                        'decision_time': pd.Timestamp('2017-03-20')+pd.Timedelta(minutes=target*5),
                        'target': target, **{f: 1. for f in FEATURES}})
            pd.DataFrame(examples).to_csv(path, index=False)
            train, heldout = load_cohort_for_well(path, 12)
            self.assertEqual(set(train.well_id), {2, 3})
            self.assertEqual(set(heldout.well_id), {12})

    def test_reference_filter_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'scores.csv'
            df = pd.DataFrame([{'file':'WELL-00012_20170320011000.parquet',
                'well_id':12, 'outer_well':12, 'ratio':r, 'policy':p,
                'score':v, 'decision_time':'2017-03-20 03:29:59', 'alert_active':False}
                for r,p,v in [(0.,'fixed05_1',.1),(.25,'fixed05_1',.2),(0.,'fixed05_2',.3)]])
            df.to_csv(path, index=False)
            r = load_reference(path, 12, 'WELL-00012_20170320011000.parquet')
            self.assertEqual(len(r), 1)
            self.assertAlmostEqual(r.iloc[0].score, .1)

    def test_feature_nan_mismatch_is_detected(self):
        a = pd.DataFrame([{f: 1. for f in FEATURES}])
        a.loc[0, FEATURES[0]] = np.nan
        s = pd.Series({f: 1. for f in FEATURES})
        _, matched = _max_feature_difference(a, s)
        self.assertFalse(matched)


if __name__ == '__main__':
    unittest.main()
