"""Стресс-тесты к проверке специфичности класса 2; только синтетические данные."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import evaluate_class2_controls as c


def frame(n=3300, mode="normal"):
    t = pd.date_range("2021-01-01", periods=n, freq="s")
    second = np.arange(n, dtype=float)
    labels = np.zeros(n, dtype=float)
    if mode == "other_event":
        labels[2200:2600] = 104
        labels[2600:] = 4
    elif mode == "unknown":
        labels[2100:2150] = np.nan
    elif mode == "unexpected2":
        labels[2100:2150] = 102
    return pd.DataFrame({
        "class": pd.array(labels, dtype="Float64"),
        "state": 999,
        "P-TPT": 15e6 - second * 2,
        "T-TPT": 90 - second * .001,
        "P-MON-CKP": np.full(n, np.nan),
    }, index=t)


def synthetic_training():
    rows = []
    for well in (2, 3, 9, 10):
        for k in range(12):
            target = int(k >= 8)
            row = {"horizon_minutes": 30, "file": f"WELL-{well:05d}_TRAIN.parquet",
                   "well_id": well, "decision_time": f"2020-01-01 00:{k*5 % 60:02d}:00",
                   "target": target}
            for j, key in enumerate(c.FEATURES):
                row[key] = k/10 + j/100 + well/10000
            rows.append(row)
    return pd.DataFrame(rows)


class TestSpecificityControls(unittest.TestCase):
    def test_class2_not_allowed_and_no_invalid_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "класс 2"):
                c.choose_control_files(Path(td), classes=(0, 2))
            with self.assertRaises(ValueError):
                c.choose_control_files(Path(td), classes=(0,), per_class_well=0)

    def test_deterministic_selection_across_wells(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "0").mkdir()
            for well in (2, 3):
                for k in range(4):
                    (root / "0" / f"WELL-{well:05d}_20170101000{k}00.parquet").write_bytes(b"x")
            a = c.choose_control_files(root, classes=(0,), per_class_well=2)
            b = c.choose_control_files(root, classes=(0,), per_class_well=2)
            self.assertEqual(a.file.tolist(), b.file.tolist())
            self.assertEqual(a.groupby("well_id").size().to_dict(), {2:2, 3:2})
            self.assertTrue(a.file.str.startswith("0/").all())

    def test_negative_requires_complete_known_future(self):
        y = np.zeros(3000)
        self.assertEqual(c.future_is_verifiable_negative(y, 899, len(y), 0), "known_no_class2")
        self.assertEqual(c.future_is_verifiable_negative(y, 2500, len(y), 0), "right_censored")
        y[1800] = np.nan
        self.assertEqual(c.future_is_verifiable_negative(y, 899, len(y), 0), "unknown_labels")
        y[1800] = 2
        self.assertEqual(c.future_is_verifiable_negative(y, 899, len(y), 0), "contains_class2_or_102")
        y[1800] = 104
        self.assertEqual(c.future_is_verifiable_negative(y, 899, len(y), 0), "class0_file_not_all_normal")
        self.assertEqual(c.future_is_verifiable_negative(y, 899, len(y), 4), "known_no_class2")

    def test_classification_for_other_class(self):
        self.assertEqual(c.current_phase(0, 0), "normal_only_file")
        self.assertEqual(c.current_phase(4, 0), "pre_other_event")
        self.assertEqual(c.current_phase(4, 104), "other_transition")
        self.assertEqual(c.current_phase(4, 4), "other_event")

    def test_normal_file_extracts_only_negative_windows(self):
        s, audit = c.extract_negative_controls(frame(), 0, "0/WELL-00002_20210101000000.parquet", 2)
        self.assertGreater(len(s), 0)
        self.assertGreater(audit["right_censored"], 0)
        self.assertTrue(s.control_phase.eq("normal_only_file").all())
        self.assertTrue(s.columns.isin(["class", "state", "target"]).sum() == 0)

    def test_label_alteration_does_not_change_extracted_features(self):
        a = frame()
        b = a.copy()
        b.loc[b.index[2000:], "class"] = 4
        x, _ = c.blind.past_only_features(c.blind.prepare_frame(a), 1199)
        z, _ = c.blind.past_only_features(c.blind.prepare_frame(b), 1199)
        for key in c.FEATURES:
            np.testing.assert_allclose(x[key], z[key], equal_nan=True)

    def test_missing_sensors_excluded_not_silently_normal(self):
        f = frame()
        f["T-TPT"] = np.nan
        s, audit = c.extract_negative_controls(f, 0, "0/WELL-00002_20210101000000.parquet", 2)
        self.assertTrue(s.empty)
        self.assertGreater(audit["unusable_sensors"], 0)

    def test_no_scoring_across_time_gaps(self):
        f = frame()
        f = f.drop(f.index[1750])
        s, audit = c.extract_negative_controls(f, 0, "0/WELL-00002_20210101000000.parquet", 2)
        self.assertTrue(s.empty)
        self.assertTrue(audit["right_censored"] >= 0)

    def test_data_errors_are_reported_not_zero_alarm(self):
        selected = pd.DataFrame([{"file": "0/WELL-00002_20210101000000.parquet",
                                  "path": "missing", "well_id": 2, "event_class_folder": 0}])
        with patch("pandas.read_parquet", side_effect=OSError("broken")):
            data, audit = c.build_controls(selected)
        self.assertTrue(data.empty)
        self.assertIn("OSError", audit.error.iloc[0])

    def test_train_model_known_and_unseen_not_mixed(self):
        x = synthetic_training()
        data = pd.DataFrame([{
            "file": f"0/WELL-{well:05d}_20210101000000.parquet", "well_id": well,
            "event_class_folder": 0, "control_phase": "normal_only_file",
            "decision_time": pd.Timestamp("2021-01-01"),
            **{k: 0.5 for k in c.FEATURES}}
            for well in (2, 20)])
        scores = c.train_and_score(x, data, iterations=5)
        self.assertEqual(scores.set_index("well_id").training_mode.to_dict(),
                         {2:"loow_heldout_well", 20:"new_well_fullfit"})
        self.assertTrue(scores.score.between(0, 1).all())

    def test_training_duplicate_from_control_forbidden(self):
        train = synthetic_training()
        data = pd.DataFrame([{"file": "0/WELL-00002_TRAIN.parquet", "well_id": 2,
                              "decision_time": pd.Timestamp("2021-01-01"),
                              **{k: 0.5 for k in c.FEATURES}}])
        with self.assertRaisesRegex(ValueError, "Файл контроля присутствует"):
            c.train_and_score(train, data, iterations=5)

    def test_nested_known_and_unseen_fallback(self):
        t = pd.date_range("2021-01-01", periods=3, freq="300s")
        records=[]
        for well, mode in ((2, "loow_heldout_well"), (20, "new_well_fullfit")):
            for ts in t:
                records.append({"file": f"0/WELL-{well:05d}_20210101000000.parquet",
                                "well_id": well, "decision_time": ts, "score": 0.4,
                                "training_mode": mode, "event_class_folder": 0,
                                "control_phase": "normal_only_file"})
        outcome=c.apply_policies(pd.DataFrame(records), {2:(0.3,2)})
        known=outcome[outcome.well_id.eq(2)]
        unseen=outcome[outcome.well_id.eq(20)]
        self.assertEqual(known.nested_start.sum(), 1)
        self.assertEqual(unseen.nested_start.sum(), 0)
        self.assertTrue(known.nested_source.eq("inner_train_only").all())
        self.assertTrue(unseen.nested_source.eq("fixed_fallback_unseen").all())
        summary=c.summarize_controls(outcome)
        self.assertIn("starts_per_eligible_negative_hour", summary.columns)
        self.assertEqual(summary.training_mode.nunique(), 2)

    def test_whole_experiment_with_mocked_parquet(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/"dataset"
            for cls,well in ((0,2),(4,20)):
                (root/str(cls)).mkdir(parents=True)
                (root/str(cls)/f"WELL-{well:05d}_20210101000000.parquet").write_bytes(b"fake")
            training = Path(td)/"train.csv"
            synth=synthetic_training()
            synth.to_csv(training,index=False)
            policy = Path(td)/"policy.csv"
            pd.DataFrame([{"outer_test_well":w, "threshold":0.3,"confirmations":2}
                          for w in (2,3,9,10)]).to_csv(policy,index=False)
            with patch("pandas.read_parquet", side_effect=lambda path:
                       frame(mode="normal" if "/0/" in str(path).replace("\\", "/") else "other_event")):
                result,audit,summary=c.run(root,training,policy,Path(td)/"results",
                                          (0,4),1,5)
            self.assertTrue(result.score.between(0,1).all())
            self.assertEqual(result.training_mode.nunique(),2)
            self.assertEqual(summary.training_mode.nunique(),2)
            for name in ("selected_files.csv","control_predictions.csv","by_class_and_phase.csv",
                         "file_audit.csv","experiment.json"):
                self.assertTrue((Path(td)/"results"/name).is_file(),name)


if __name__ == "__main__":
    unittest.main()
