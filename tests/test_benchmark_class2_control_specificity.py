"""Control-specificity contract tests; no Petrobras files or pyarrow required."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import benchmark_class2_control_specificity as specificity


def index():
    return pd.DataFrame({'file':['0/WELL-00001_20200101000000.parquet'],
                         'well_id':[1],'decision_time':[pd.Timestamp('2020-01-01 00:14:59')],
                         'event_class_folder':[0],'control_phase':['normal_only_file']})


def raw():
    t=pd.date_range('2020-01-01',periods=900,freq='s').as_unit('us')
    return pd.DataFrame({'timestamp':t,'P-MON-CKP':np.arange(900,dtype=float)*2,
                         'P-TPT':np.arange(900,dtype=float)*4,'T-TPT':np.arange(900,dtype=float)*.1,
                         'class':[0]*900,'state':[0]*900})


class ControlSpecificityTests(unittest.TestCase):
    def test_index_rejects_class2(self):
        d=index();d['event_class_folder']=2;d['file']='2/WELL-00001_20200101000000.parquet'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'index.csv';d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'DHSV'):
                specificity.load_index(p)

    def test_index_rejects_train_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'index.csv';index().to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'пересекаются'):
                specificity.load_index(p,{1,2})

    def test_index_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'index.csv';pd.concat([index(),index()]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Дубли'):
                specificity.load_index(p)

    def test_extract_features_with_microsecond_timestamp(self):
        result=specificity.features_for_index(raw(),index())
        self.assertEqual(len(result),1)
        self.assertEqual(len(set(specificity.ALL_FEATURES)-set(result.columns)),0)
        self.assertAlmostEqual(float(result.iloc[0]['P-TPT_delta']),4*299)

    def test_future_class_change_does_not_affect_features(self):
        a=raw()
        b=raw()
        b['class']=102;b['state']=102
        one=specificity.features_for_index(a,index())
        two=specificity.features_for_index(b,index())
        pd.testing.assert_frame_equal(one,two)

    def test_gapped_history_cannot_be_scored(self):
        d=raw().drop(index=100).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'истории'):
            specificity.features_for_index(d,index())

    def test_missing_control_file_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                specificity.extract_controls(index(),Path(td))

    def test_control_metrics_not_field_uptime(self):
        d=index();d['score']=[.7];d['alert_active']=[True];d['alarm_start']=[True]
        m=specificity.control_metrics(d)
        self.assertAlmostEqual(m['eligible_sampled_hours'],5/60)
        self.assertEqual(m['alarm_starts'],1)
        self.assertAlmostEqual(m['starts_per_eligible_sampled_hour'],12)

    def test_all_model_control_scores_are_recomputed_from_same_features(self):
        # A fake learner sees only DHSV training rows, never control targets.
        from unittest.mock import patch
        import benchmark_class2_models as bench
        train_rows=[]
        for w in (2,3,9):
            for i in range(3):
                row=dict(file=f"WELL-{w:05d}.parquet",well_id=w,
                         decision_time=pd.Timestamp("2020-01-01")+pd.Timedelta(minutes=5*i),
                         event_time=pd.Timestamp("2020-01-01 00:20") if i==2 else pd.NaT,
                         target=int(i==2), horizon_minutes=30)
                row.update({f:float(i) for f in bench.FEATURES})
                train_rows.append(row)
        cohort=pd.DataFrame(train_rows)
        oof=cohort[["file","well_id","decision_time","event_time","target"]].copy()
        oof["score"]=[0.1,0.2,0.9]*3
        oof["model"]="dummy_prior"
        controls=index().copy()
        controls.update({f:7.0 for f in []})
        for f in bench.FEATURES:controls[f]=7.0
        class Recorder:
            def fit(self,X,y):
                self.fit_rows=len(X)
                self.fit_labels=y.tolist()
                self.train_last=float(X.iloc[-1][bench.FEATURES[0]])
                return self
            def predict_proba(self,X):
                assert self.fit_rows==9
                assert self.fit_labels==[0,0,1]*3
                assert float(X.iloc[0][bench.FEATURES[0]])==7.0
                assert self.train_last==2.0
                return np.array([[0.25,0.75]]*len(X))
        with patch.object(bench,'make_model',return_value=Recorder()):
            scores,choices,by_class,by_well=specificity.score_controls(cohort,oof,controls,('dummy_prior',))
        self.assertEqual(scores.score.tolist(),[0.75])
        self.assertEqual(len(choices),1)
        self.assertEqual(len(by_class),1)
        self.assertEqual(len(by_well),1)

    def test_index_full_artifact_inventory(self):
        reference=ROOT/'reference'/'control_heldout_index.csv'
        d=specificity.load_index(reference,expected_wells={2,3,9,10,11,12,13})
        self.assertEqual(len(d),4056)
        self.assertEqual(d.file.nunique(),23)
        self.assertEqual(d.well_id.nunique(),9)


if __name__=='__main__':unittest.main()
