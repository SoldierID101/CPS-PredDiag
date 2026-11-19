"""Offline replay CatBoost-12 / ML Agent v0.2, only for research.

Re-trains *one* LOOW CatBoost model on archived 30-minute cohort features
(excludes the entire held-out WELL), streams all telemetry-only decisions, and
compares *only* matching archived eligible moments to fixed05_1 baseline scores.

No oracle gating, no post-onset interpretation as negative, no Coordinator calls.
Does not load or write model pickles. CatBoost scores are uncalibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_agent import FEATURES, SENSORS
from ml_agent_v02 import REQUIRED_HISTORY_SECONDS, STRIDE_SECONDS, ShadowConfig, ShadowMLAgentV02

DEFAULT_FILE = "WELL-00012_20170320011000.parquet"
DEFAULT_COHORT = ROOT / "outputs/class2_horizons/cohort_features.csv"
DEFAULT_REFERENCE = ROOT / "outputs/class2_hard_negative_balance/class2_scores_and_alerts.csv"
DEFAULT_DATASET = ROOT / "dataset/2" / DEFAULT_FILE
DEFAULT_OUTPUT = ROOT / "outputs/ml_agent_v02_replay"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for b in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_well_id(filename: str) -> int:
    match = re.fullmatch(r"WELL-(\d{5})_\d{14}\.(?:parquet|csv)", filename)
    if not match:
        raise ValueError(f"Ожидалось имя WELL-00012_YYYYMMDDHHMMSS.parquet: {filename}")
    return int(match.group(1))


def _to_time_axis(raw: pd.DataFrame) -> pd.DatetimeIndex:
    if "timestamp" in raw.columns:
        stamp = raw["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(stamp):
            times = pd.DatetimeIndex(stamp)
        elif pd.api.types.is_numeric_dtype(stamp):
            # In Petrobras 3W, timestamp is Unix nanoseconds. Do NOT infer units.
            numeric = pd.to_numeric(stamp, errors="raise")
            if not np.isfinite(numeric).all():
                raise ValueError("NaN/inf во временных метках")
            times = pd.DatetimeIndex(pd.to_datetime(numeric.astype("int64"), unit="ns", errors="raise"))
        else:
            times = pd.DatetimeIndex(pd.to_datetime(stamp, errors="raise"))
    elif isinstance(raw.index, pd.DatetimeIndex):
        times = raw.index
    else:
        raise ValueError("Нет временной оси: timestamp или DatetimeIndex")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Временная ось содержит NaT, дубликаты или немонотонные метки")
    if times.tz is not None:
        raise ValueError("Неожиданная временная зона; ожидаются метки 3W без timezone")
    return times.as_unit("ns")


def telemetry_only(raw: pd.DataFrame) -> pd.DataFrame:
    """Erase label/state columns BEFORE scheduling and model entry."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Ожидался pandas DataFrame")
    times = _to_time_axis(raw)
    telemetry = raw.reindex(columns=SENSORS).copy()
    telemetry.index = times
    return telemetry


def load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдена телеметрия: {path}")
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError("Для Parquet нужна библиотека pyarrow: python -m pip install pyarrow") from exc
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Поддерживаются только Parquet и CSV с полем timestamp")


def decision_positions(index: pd.DatetimeIndex) -> list[int]:
    """Every 300s from 900th sample of each contiguous 1-Hz segment.

    Uses only timestamps. No label, model score, cohort, or event time.
    """
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Неупорядоченные временные метки")
    ns = index.as_unit("ns").asi8
    cuts = np.flatnonzero(np.diff(ns) != 1_000_000_000) + 1
    bounds = [0, *cuts.tolist(), len(index)]
    positions = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        positions.extend(range(a + REQUIRED_HISTORY_SECONDS - 1, b, STRIDE_SECONDS))
    return positions


def load_cohort_for_well(path: Path, well: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдена когорта: {path}")
    data = pd.read_csv(path)
    needed = {"well_id", "horizon_minutes", "file", "decision_time", "target", *FEATURES}
    if not needed.issubset(data.columns):
        raise ValueError(f"В когорте не хватает полей: {sorted(needed-set(data.columns))}")
    data = data.loc[data.horizon_minutes.eq(30)].copy()
    if data.empty or not set(data.target.unique()) <= {0, 1}:
        raise ValueError("Нет валидной 30-минутной когорты")
    data["decision_time"] = pd.to_datetime(data["decision_time"], errors="raise").dt.as_unit("ns")
    if data.duplicated(["file", "decision_time"]).any():
        raise ValueError("В когорте повторяются file/decision_time")
    train = data.loc[data.well_id.ne(well)].copy()
    test = data.loc[data.well_id.eq(well)].copy()
    if test.empty or train.empty or train.target.nunique() != 2 or len(set(train.well_id)) < 2:
        raise ValueError("Невозможна LOOW-валидация: недостаточно обучающих скважин/классов")
    if int(well) in set(map(int, train.well_id)):
        raise AssertionError("Утечка test WELL в обучение")
    return train, test


def load_reference(path: Path, well: int, file: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдены архивные прогнозы: {path}")
    ref = pd.read_csv(path)
    needed = {"file", "well_id", "ratio", "policy", "score", "decision_time", "outer_well", "alert_active"}
    if not needed.issubset(ref.columns):
        raise ValueError(f"Нет полей архива: {sorted(needed - set(ref.columns))}")
    # Historical 30m baseline: hard-negative ratio 0, single confirmation, threshold 0.5.
    r = ref.loc[ref.file.eq(file) & ref.well_id.eq(well) & ref.outer_well.eq(well) &
                np.isclose(ref.ratio, 0.0) & ref.policy.eq("fixed05_1")].copy()
    if r.empty:
        raise ValueError("Нет сохранённого baseline LOOW для файла и скважины")
    r["decision_time"] = pd.to_datetime(r["decision_time"], errors="raise").dt.as_unit("ns")
    if r.decision_time.duplicated().any() or not np.isfinite(r.score.to_numpy(float)).all():
        raise ValueError("Дубли или нечисловые оценки в архиве")
    r["alert_active"] = r["alert_active"].map({True: True, False: False, "True": True, "False": False})
    if r.alert_active.isna().any():
        raise ValueError("Некорректный alert_active в архиве")
    return r


def train_baseline(train: pd.DataFrame, iterations: int = 300):
    """Same model family/config as experiments/evaluate_class2_alert_policy.py."""
    from experiments.evaluate_class2_alert_policy import catboost_model
    if iterations != 300:
        # iterations override is useful in tests, but not a comparable historical model.
        pass
    model = catboost_model(iterations)
    x = train[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    y = train.target.to_numpy(dtype=int)
    model.fit(x, y)
    return model


class CaptureModel:
    """Adapts a trained estimator, exposing the actual v0.2 feature vector."""
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        self.last_features: pd.DataFrame | None = None

    def predict_proba(self, data: pd.DataFrame):
        self.last_features = data.copy()
        return self.model.predict_proba(data)


def _max_feature_difference(actual: pd.DataFrame, expected: pd.Series) -> tuple[float, bool]:
    left = actual.iloc[0].to_numpy(dtype=float)
    right = expected[FEATURES].to_numpy(dtype=float)
    bad_nan = bool((np.isnan(left) != np.isnan(right)).any())
    diff = np.abs(left - right)
    finite = diff[np.isfinite(diff)]
    max_abs = float(finite.max()) if len(finite) else 0.0
    # Relative check accounts for pressure values in millions and small slopes.
    close = bool(np.allclose(left, right, rtol=1e-10, atol=1e-5, equal_nan=True))
    return max_abs, close and not bad_nan


def replay(raw: pd.DataFrame, *, well: int, file: str, model, test_cohort: pd.DataFrame,
           reference: pd.DataFrame | None, threshold: float = 0.5, confirmations: int = 1,
           tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict]:
    """Replay by telemetry-only decisions; compare archived points without oracle gating."""
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("Нужна положительная конечная tolerance")
    telemetry = telemetry_only(raw)
    audit_cohort = test_cohort.loc[test_cohort.file.eq(file)].copy()
    if audit_cohort.empty:
        raise ValueError("У test WELL нет строк исследуемого файла")
    if audit_cohort.decision_time.duplicated().any():
        raise ValueError("Повторение времени в архиве когорты")
    audit_map = {pd.Timestamp(r.decision_time): r for _, r in audit_cohort.iterrows()}
    if reference is not None:
        ref_map = {pd.Timestamp(r.decision_time): r for _, r in reference.iterrows()}
        if set(ref_map) != set(audit_map):
            raise ValueError("В архиве прогнозов и когорте разные моменты для этого файла")
    else:
        ref_map = {}
    wrapped = CaptureModel(model)
    agent = ShadowMLAgentV02(model=wrapped, config=ShadowConfig(
        model_version=f"catboost-12-30m-baseline-loow-well-{well:05d}",
        threshold=float(threshold), confirmations=int(confirmations),
    ))
    agent.reset(str(well))
    rows = []
    for pos in decision_positions(telemetry.index):
        t = telemetry.index[pos]
        # Exactly the 900 last observed seconds; no label or future rows.
        past = telemetry.iloc[pos - REQUIRED_HISTORY_SECONDS + 1:pos + 1]
        wrapped.last_features = None
        result = agent.predict(past, well_id=str(well), decision_time=t)
        comparable = t in audit_map
        row = {
            "file": file, "well_id": well, "decision_time": t.isoformat(),
            "status": result["status"], "model_score": result["model_score"],
            "alarm_active": result["alarm_active"], "alarm_new_start": result["alarm_new_start"],
            "in_archived_risk_set": comparable, "reference_score": np.nan,
            "score_abs_diff": np.nan, "feature_max_abs_diff": np.nan,
            "feature_matches": None, "reference_alarm_active": None, "alarm_matches": None,
        }
        if comparable:
            if result["status"] != "experimental_prediction" or wrapped.last_features is None:
                raise RuntimeError(f"Не выполнен прогноз для архивного момента {t}: {result}")
            row["feature_max_abs_diff"], row["feature_matches"] = _max_feature_difference(
                wrapped.last_features, audit_map[t])
            if t in ref_map:
                r = ref_map[t]
                row["reference_score"] = float(r.score)
                row["score_abs_diff"] = abs(float(row["model_score"]) - float(r.score))
                if threshold == 0.5 and confirmations == 1:
                    row["reference_alarm_active"] = bool(r.alert_active)
                    row["alarm_matches"] = bool(result["alarm_active"] == bool(r.alert_active))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Файл слишком короткий для 15-минутной истории")
    matched = out.loc[out.in_archived_risk_set]
    if len(matched) != len(audit_map):
        raise RuntimeError(f"Не все моменты архива воспроизведены: {len(matched)}/{len(audit_map)}")
    if ref_map and matched.score_abs_diff.isna().any():
        raise RuntimeError("Не для всех архивных моментов сопоставлены оценки")
    score_ok = bool(matched.score_abs_diff.le(tolerance).all()) if ref_map else None
    feature_ok = bool(matched.feature_matches.eq(True).all())
    alarm_comparison = matched.alarm_matches.dropna()
    alarm_ok = bool(alarm_comparison.eq(True).all()) if not alarm_comparison.empty else None
    summary = {
        "file": file, "well_id": well, "source_rows": len(telemetry),
        "all_blind_replay_decisions": len(out),
        "archived_eligible_decisions": len(matched),
        "non_comparable_decisions": len(out) - len(matched),
        "eligible_feature_match": feature_ok,
        "max_abs_feature_difference": float(matched.feature_max_abs_diff.max()),
        "score_comparison_performed": bool(ref_map),
        "max_abs_score_difference": float(matched.score_abs_diff.max()) if ref_map else None,
        "max_allowed_score_difference": tolerance if ref_map else None,
        "eligible_score_match": score_ok,
        "eligible_alarm_match": alarm_ok,
        "compared_alarm_rows": int(len(alarm_comparison)),
        "important": "Exploratory parity only. Not independent validation or field false-alarm rate. "
                     "Predictions outside retrospective risk set MUST NOT be counted as verified negatives.",
    }
    return out, summary


def run(telemetry_path: Path = DEFAULT_DATASET, cohort_path: Path = DEFAULT_COHORT,
        reference_path: Path = DEFAULT_REFERENCE, output_dir: Path = DEFAULT_OUTPUT,
        iterations: int = 300, tolerance: float = 1e-6) -> dict:
    if iterations != 300:
        raise ValueError("С архивными прогнозами сравнима только модель на 300 итераций")
    well = parse_well_id(telemetry_path.name)
    train, test = load_cohort_for_well(cohort_path, well)
    compare_file = telemetry_path.name.removesuffix(".csv") + ".parquet" if telemetry_path.suffix == ".csv" else telemetry_path.name
    ref = load_reference(reference_path, well, compare_file)
    raw = load_raw(telemetry_path)
    model = train_baseline(train, iterations=iterations)
    predictions, summary = replay(raw, well=well, file=compare_file, model=model, test_cohort=test,
                                  reference=ref, tolerance=tolerance)
    summary.update({
        "experiment": "ml_agent_v02_offline_replay_v0.1", "horizon_minutes": 30,
        "history_admission_seconds": REQUIRED_HISTORY_SECONDS, "features_window_seconds": 300,
        "decision_stride_seconds": STRIDE_SECONDS, "baseline_hard_negative_ratio": 0,
        "threshold": 0.5, "confirmations": 1,
        "train_wells": sorted(map(int, train.well_id.unique())),
        "train_rows": len(train),
        "model_retrained": True,
        "telemetry_sha256": _sha256(telemetry_path),
        "cohort_sha256": _sha256(cohort_path),
        "reference_sha256": _sha256(reference_path),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "catboost": __import__("catboost").__version__},
        "parity_pass": bool(summary["eligible_feature_match"] and summary["eligible_score_match"]
                            and summary["eligible_alarm_match"] is True),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"{Path(compare_file).stem}_replay.csv", index=False)
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "file", "all_blind_replay_decisions", "archived_eligible_decisions", "non_comparable_decisions",
        "eligible_feature_match", "max_abs_feature_difference", "eligible_score_match",
        "max_abs_score_difference", "eligible_alarm_match", "parity_pass")}, ensure_ascii=False, indent=2))
    print(f"Отчёты: {output_dir}")
    if not summary["parity_pass"]:
        raise RuntimeError("Воспроизведение не совпало с архивом; проверьте CSV/версии пакетов. "
                           "Сохранён подробный отчёт, исходные модули не изменены.")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--telemetry", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tolerance", type=float, default=1e-6)
    args = p.parse_args()
    run(args.telemetry, args.cohort, args.reference, args.output, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
