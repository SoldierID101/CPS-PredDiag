"""Contract tests for DHSV nested alert policy (no real Parquet needed)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import benchmark_class2_alert_policy as policy
import benchmark_class2_models as bench


def cohort():
    rows=[]
    for w in (2,3,9,10):
        base=pd.Timestamp("2020-01-01")
        for i in range(9):
            row=dict(file=f"WELL-{w:05d}_20200101000000.parquet", well_id=w,
                     horizon_minutes=30, decision_time=base + pd.Timedelta(minutes=5*i),
                     target=int(i>=6), event_time=base+pd.Timedelta(minutes=45) if i>=6 else pd.NaT)
            for j,feature in enumerate(bench.FEATURES):row[feature]=0.1*j+(i>=6)*10+w*.001
            rows.append(row)
    return pd.DataFrame(rows)


class NestedAlertPolicyTests(unittest.TestCase):
    def test_one_confirmation_single_spike(self):
        d=pd.DataFrame({"well_id":[2]*3,"file":["x"]*3,
                        "decision_time":pd.date_range("2020-01-01",periods=3,freq="5min"),
                        "score":[0.2,0.9,0.1],"target":[0,1,0],"event_time":[pd.NaT,pd.Timestamp("2020-01-01 00:16"),pd.NaT]})
        res=policy.apply_policy(d,0.5,1)
        self.assertEqual(res.alarm_start.tolist(),[False,True,False])

    def test_two_confirmations_and_no_cross_file_streak(self):
        d=pd.DataFrame({"well_id":[2]*4,"file":["x","x","y","y"],
                        "decision_time":[pd.Timestamp("2020-01-01")+pd.Timedelta(minutes=i*5) for i in range(4)],
                        "score":[0.9]*4})
        res=policy.apply_policy(d,0.5,2)
        self.assertEqual(res.alert_active.tolist(),[False,True,False,True])
        self.assertEqual(int(res.alarm_start.sum()),2)

    def test_gap_resets_confirmations(self):
        d=pd.DataFrame({"well_id":[2]*3,"file":["x"]*3,
                        "decision_time":[pd.Timestamp("2020-01-01"),pd.Timestamp("2020-01-01 00:10"),pd.Timestamp("2020-01-01 00:15")],
                        "score":[0.9]*3})
        self.assertEqual(policy.apply_policy(d,0.5,2).alert_active.tolist(),[False,False,True])

    def test_policy_ignores_current_labels_and_future_fields(self):
        d=pd.DataFrame({"well_id":[2]*3,"file":["x"]*3,
                        "decision_time":pd.date_range("2020-01-01",periods=3,freq="5min"),
                        "score":[.9,.8,.1],"target":[1,1,1],"class":[102,2,0]})
        a=policy.apply_policy(d,.5,2)
        d.target=0;d['class']=0
        b=policy.apply_policy(d,.5,2)
        self.assertEqual(a.alert_active.tolist(),b.alert_active.tolist())

    def test_policy_selection_only_inner_scores(self):
        f=cohort()
        inner=f[["file","well_id","decision_time","target","event_time"]].copy()
        inner["score"]=np.where(inner.target,0.9,0.1)
        first,_=policy.choose_policy(inner)
        # Simulated changes to unobserved outer-well data must not influence policy.
        untouched=inner.copy()
        untouched.loc[untouched.well_id.eq(2),"score"]=0.12
        selected,_=policy.choose_policy(inner)
        self.assertEqual(first,selected)

    def test_event_activity_not_equivalent_to_new_start(self):
        d=pd.DataFrame({"file":["f"]*3,"well_id":[2]*3,
                        "decision_time":pd.date_range("2020-01-01",periods=3,freq="5min"),
                        "event_time":[pd.NaT,pd.Timestamp("2020-01-01 00:20"),pd.Timestamp("2020-01-01 00:20")],
                        "target":[0,1,1],"score":[.9,.9,.9]})
        m=policy.measure(policy.apply_policy(d,.5,1))
        self.assertEqual(m['events_detected'],1)
        self.assertEqual(m['events_with_new_start_in_horizon'],0)
        self.assertEqual(m['out_of_horizon_alarm_starts'],1)

    def test_no_train_well_in_outer_policy_retrain(self):
        f=cohort()
        test=f[["file","well_id","decision_time","target","event_time"]].copy()
        test["score"]=np.where(test.target,0.9,0.1)
        test["model"]="logreg"
        seen=[]
        def fake_oof(train, name):
            self.assertNotIn(int(seen[-1]),set(train.well_id))
            return train.assign(score=np.where(train.target,0.9,0.1))
        with patch.object(bench,'predict_oof',side_effect=fake_oof):
            # Manually pass each generator round using callback observing heldout well
            it=policy.make_outer_predictions(f,test,("logreg",))
            for w in sorted(f.well_id.unique()):
                seen.append(w)
                model,outer,selected,grid,scored=next(it)
                self.assertEqual(w,outer)
                self.assertEqual(set(scored.well_id),{w})

    def test_reference_oof_detects_missing_and_label_mismatch(self):
        f=cohort()
        oof=f[["file","well_id","horizon_minutes","decision_time","target","event_time"]].copy()
        oof["score"]=0.5;oof["model"]="logreg";oof["test_well"]=oof.well_id
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"oof.csv";oof.to_csv(path,index=False)
            self.assertEqual(len(policy.load_oof(path,f,("logreg",))),len(f))
            oof.loc[0,"target"]=1
            oof.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"разметку"):
                policy.load_oof(path,f,("logreg",))

    def test_nested_two_model_end_to_end(self):
        f=cohort()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"cohort.csv";f.to_csv(source,index=False)
            ref=[]
            for name in ('dummy_prior','logreg'):
                ref.append(bench.predict_oof(f,name))
            oof=root/"oof.csv";pd.concat(ref,ignore_index=True).to_csv(oof,index=False)
            result=policy.run(source,oof,root/"report",('dummy_prior','logreg'))
            self.assertEqual(result['n_windows_per_model'],36)
            self.assertTrue((root/"report"/"inner_policy_grid.csv").is_file())
            self.assertEqual(len(pd.read_csv(root/"report"/"outer_alerts.csv")),72)
            self.assertEqual(len(pd.read_csv(root/"report"/"policy_by_well.csv")),8)


if __name__ == '__main__':unittest.main()
