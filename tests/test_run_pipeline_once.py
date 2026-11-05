"""Проверка выбора момента воспроизведения без чтения файлов Parquet."""
import unittest

import pandas as pd

from experiments.run_pipeline_once import first_complete_102_time


class TestRunPipelineOnce(unittest.TestCase):
    def test_selects_first_completed_300_seconds(self):
        index = pd.date_range("2020-01-01", periods=630, freq="s")
        frame = pd.DataFrame({"class": [0] * 10 + [102] * 620}, index=index)
        self.assertEqual(first_complete_102_time(frame), index[309])

    def test_stops_at_gap(self):
        index = pd.date_range("2020-01-01", periods=310, freq="s").delete(150)
        frame = pd.DataFrame({"class": [102] * len(index)}, index=index)
        with self.assertRaisesRegex(ValueError, "300"):
            first_complete_102_time(frame)

    def test_nullable_missing_label_breaks_102_run(self):
        index = pd.date_range("2020-01-01", periods=301, freq="s")
        labels = pd.array([102] * 301, dtype="Int64")
        labels[1] = pd.NA
        frame = pd.DataFrame({"class": labels}, index=index)
        with self.assertRaisesRegex(ValueError, "300"):
            first_complete_102_time(frame)

    def test_nullable_missing_label_before_run_is_allowed(self):
        index = pd.date_range("2020-01-01", periods=301, freq="s")
        labels = pd.array([102] * 301, dtype="Int64")
        labels[0] = pd.NA
        frame = pd.DataFrame({"class": labels}, index=index)
        self.assertEqual(first_complete_102_time(frame), index[300])

    def test_missing_class_rejected(self):
        frame = pd.DataFrame(index=pd.date_range("2020-01-01", periods=300, freq="s"))
        with self.assertRaisesRegex(ValueError, "class"):
            first_complete_102_time(frame)


if __name__ == "__main__":
    unittest.main()
