"""Unit tests for entire WELL-by-WELL replay, no genuine Petrobras data required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_agent import FEATURES
from experiments.offline_replay_ml_v02_all import (
    file_inventory, _verify_preflight, _score_summary, run_all,
)
from experiments.offline_replay_ml_v02 import telemetry_only
from ml_agent import features_from_history


class ConstantModel:
    classes_ = [0, 1]

    def predict_proba(self, x):
        assert x.columns.tolist() == FEATURES
        return np.tile([.3, .7], (len(x), 1))


def _raw(n=1200):
    t = pd.date_range('2017-03-20', periods=n, freq='s').as_unit('us')
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        'timestamp': t, 'P-MON-CKP': 10_000_000 + x,
        'P-TPT': 20_000_000 - x, 'T-TPT': 100 + x/1000,
        'class': [102] * n, 'state': [102] * n,
    })


class AllWellReplayTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.dataset = self.root / 'dataset' / '2'
        self.dataset.mkdir(parents=True)
        self.cohort = self.root / 'cohort.csv'
        self.reference = self.root / 'reference.csv'
        self.out = self.root / 'outputs'
        self.raw = _raw()
        tel = telemetry_only(self.raw)
        feat, _ = features_from_history(tel.iloc[600:900])
        self.features = feat.iloc[0].to_dict()
        self.rows = []
        self.refs = []
        for w in (2, 3, 12):
            for i in (0, 1):
                name = f'WELL-{w:05d}_20170320{i+1:02d}1000.parquet'
                # Full Parquet I/O is deliberately replaced by a mock.
                (self.dataset / name).write_text('placeholder', encoding='utf8')
                self.rows.append({
                    'file': name, 'well_id': w, 'horizon_minutes': 30,
                    'decision_time': tel.index[899], 'target': i, **self.features,
                })
                self.refs.append({
                    'file': name, 'well_id': w, 'decision_time': tel.index[899],
                    'outer_well': w, 'ratio': 0.0, 'policy': 'fixed05_1',
                    'score': 0.7, 'alert_active': True,
                })
        pd.DataFrame(self.rows).to_csv(self.cohort, index=False)
        pd.DataFrame(self.refs).to_csv(self.reference, index=False)

    def test_inventory_covers_all_wells_and_files(self):
        pairs, n = file_inventory(self.cohort)
        self.assertEqual(n, 6)
        self.assertEqual(len(pairs), 6)
        self.assertEqual({w for w, _ in pairs}, {2, 3, 12})

    def test_invalid_well_id_or_filename_is_refused(self):
        data = pd.read_csv(self.cohort)
        data.loc[0, 'well_id'] = 123
        data.to_csv(self.cohort, index=False)
        with self.assertRaisesRegex(ValueError, 'WELL'):
            file_inventory(self.cohort)

    def test_duplicates_are_refused(self):
        data = pd.read_csv(self.cohort)
        pd.concat([data, data.iloc[:1]], ignore_index=True).to_csv(self.cohort, index=False)
        with self.assertRaisesRegex(ValueError, 'дубликаты'):
            file_inventory(self.cohort)

    def test_preflight_does_not_silently_skip_missing(self):
        pairs, _ = file_inventory(self.cohort)
        (self.dataset / pairs[0][1]).unlink()
        with self.assertRaises(FileNotFoundError):
            _verify_preflight(self.dataset, pairs, self.reference)

    def test_aggregate_fail_on_single_bad_file(self):
        a = {'status': 'checked', 'parity_pass': True, 'well_id': 2,
             'archived_eligible_decisions': 2, 'all_blind_replay_decisions': 4,
             'non_comparable_decisions': 2}
        b = dict(a, well_id=3, parity_pass=False)
        s = _score_summary([a, b], 2, 2, 4, 2)
        self.assertFalse(s['parity_pass'])
        self.assertEqual(s['files_failed'], 1)

    def test_end_to_end_per_well_training_per_file_reset_and_summary(self):
        seen = []

        def fake_train(train, iterations=300):
            self.assertEqual(len(train), 4)
            self.assertEqual(train['well_id'].nunique(), 2)
            seen.append(tuple(sorted(train['well_id'].unique())))
            return ConstantModel()

        with patch('experiments.offline_replay_ml_v02_all.train_baseline', side_effect=fake_train), \
             patch('experiments.offline_replay_ml_v02_all.load_raw', return_value=self.raw):
            s = run_all(self.dataset, self.cohort, self.reference, self.out)

        self.assertTrue(s['parity_pass'])
        self.assertEqual(s['files_checked'], 6)
        self.assertEqual(s['wells_checked'], 3)
        self.assertEqual(s['archived_eligible_expected'], 6)
        self.assertEqual(s['archived_eligible_checked'], 6)
        self.assertEqual(s['all_blind_replay_decisions_checked'], 12)
        self.assertEqual(s['non_comparable_decisions_checked'], 6)
        self.assertEqual(len(seen), 3)  # One fit per WELL, not one fit per file.
        self.assertIn((2, 3), seen)
        self.assertIn((2, 12), seen)
        self.assertIn((3, 12), seen)
        by_file = pd.read_csv(self.out/'file_parity.csv')
        self.assertTrue(by_file['parity_pass'].all())
        self.assertEqual(len(pd.read_csv(self.out/'well_parity.csv')), 3)
        self.assertEqual(len(list(self.out.glob('*_replay.csv'))), 6)
        on_disk = json.loads((self.out/'all_wells_replay_summary.json').read_text(encoding='utf8'))
        self.assertTrue(on_disk['parity_pass'])

    def test_mismatch_reported_not_hidden(self):
        wrong = pd.read_csv(self.reference)
        wrong.loc[0, 'score'] = 0.9
        wrong.to_csv(self.reference, index=False)
        with patch('experiments.offline_replay_ml_v02_all.train_baseline', return_value=ConstantModel()), \
             patch('experiments.offline_replay_ml_v02_all.load_raw', return_value=self.raw):
            with self.assertRaisesRegex(RuntimeError, 'не прошёл'):
                run_all(self.dataset, self.cohort, self.reference, self.out)
        summary = json.loads((self.out/'all_wells_replay_summary.json').read_text(encoding='utf8'))
        self.assertFalse(summary['parity_pass'])
        self.assertEqual(summary['files_failed'], 1)
        self.assertEqual(summary['files_checked'], 6)

    def test_missing_file_refused_even_with_mocked_reader(self):
        for p in self.dataset.iterdir():
            p.unlink()
        with patch('experiments.offline_replay_ml_v02_all.train_baseline', return_value=ConstantModel()), \
             patch('experiments.offline_replay_ml_v02_all.load_raw', return_value=self.raw):
            with self.assertRaises(FileNotFoundError):
                run_all(self.dataset, self.cohort, self.reference, self.out)


if __name__ == '__main__':
    unittest.main()
