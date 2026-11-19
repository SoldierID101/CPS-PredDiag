"""Control-only well specificity for the fixed 30-minute DHSV classifiers.

Controls are an *already audited* held-out risk-set index from an earlier
experiment. We re-extract historical sensor features from raw Parquet, then
score every model with the SAME 12-feature input and a policy selected using
DHSV OOF only. No control labels enter training or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_class2_models as bench
from benchmark_class2_alert_policy import apply_policy, choose_policy, load_oof

INDEX_COLUMNS = ("file", "well_id", "decision_time", "event_class_folder", "control_phase")
ALL_FEATURES = tuple(bench.FEATURES)
STEP = pd.Timedelta(minutes=5)


def load_index(path: Path, expected_wells: set[int] | None = None) -> pd.DataFrame:
    control = pd.read_csv(path)
    missing = set(INDEX_COLUMNS) - set(control)
    if missing:
        raise ValueError(f"В индексе нет полей: {missing}")
    control = control[list(INDEX_COLUMNS)].copy()
    control.decision_time = pd.to_datetime(control.decision_time, errors="raise")
    control.well_id = pd.to_numeric(control.well_id, errors="raise").astype(int)
    control.event_class_folder = pd.to_numeric(control.event_class_folder, errors="raise").astype(int)
    if control.isna().any().any():
        raise ValueError("Пропуски метаданных контроля")
    if control.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли контрольных решений")
    if control.event_class_folder.eq(2).any():
        raise ValueError("Контроль содержит класс DHSV")
    if not control.apply(lambda x: x['file'].startswith(str(x['event_class_folder']) + '/WELL-' + f"{x['well_id']:05d}"), axis=1).all():
        raise ValueError("Неверная связь файла, класса и скважины")
    if not all(Path(f).suffix.lower() == ".parquet" and len(Path(f).parts) == 2 for f in control.file):
        raise ValueError("Индекс должен содержать относительные пути вида 0/WELL-...parquet")
    if expected_wells and set(control.well_id) & set(expected_wells):
        raise ValueError("Контрольные скважины пересекаются с исследуемыми DHSV WELL")
    return control.sort_values(["well_id", "file", "decision_time"], kind="stable").reset_index(drop=True)


def normalize_parquet(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("Неверное содержимое Parquet")
    if "timestamp" in raw.columns:
        ts = raw["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            dt = pd.to_datetime(ts, unit="ns", errors="raise")
        else:
            dt = pd.to_datetime(ts, errors="raise")
        result = raw.drop(columns=["timestamp"]).copy()
        result.index = pd.DatetimeIndex(dt).as_unit("ns")
    elif isinstance(raw.index, pd.DatetimeIndex):
        result = raw.copy()
        result.index = raw.index.as_unit("ns")
    else:
        raise ValueError("Parquet без временного индекса/столбца timestamp")
    if result.index.hasnans or result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("Parquet: некорректные/повторные временные точки")
    return result


def features_for_index(raw: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Require full 900s history and >=2 usable sensors as in ML Agent v0.2."""
    from ml_agent import SENSORS, features_from_history, MIN_COVERAGE
    clean = normalize_parquet(raw)
    rows = []
    for record in decisions.itertuples(index=False):
        t = pd.Timestamp(record.decision_time)
        pos = int(clean.index.searchsorted(t, side="left"))
        if pos >= len(clean) or clean.index[pos] != t or pos < 899:
            raise ValueError(f"Нет 900s истории для {record.file} {t}")
        history = clean.iloc[pos-899:pos+1]
        if len(history) != 900 or not np.all(np.diff(history.index.asi8) == 1_000_000_000):
            raise ValueError(f"Непрерывная 900s история недоступна: {record.file} {t}")
        features, coverage = features_from_history(history)
        if sum(coverage[s] >= MIN_COVERAGE for s in SENSORS) < 2:
            raise ValueError(f"Недоступны два датчика: {record.file} {t}")
        row = {k: getattr(record, k) for k in INDEX_COLUMNS}
        row.update(features.iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def extract_controls(index: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    frames = []
    for f, group in index.groupby("file", sort=True):
        source = data_root / f
        if not source.is_file():
            raise FileNotFoundError(f"Не найден контрольный Parquet: {source}")
        frames.append(features_for_index(pd.read_parquet(source), group))
        print(f"  {f}: {len(group)} контрольных решений", flush=True)
    all_rows = pd.concat(frames, ignore_index=True)
    if len(all_rows) != len(index) or all_rows.duplicated(["file", "decision_time"]).any():
        raise AssertionError("Контроли не покрыты полностью")
    return all_rows


def score_controls(cohort: pd.DataFrame, reference_oof: pd.DataFrame, controls: pd.DataFrame,
                   models: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    choices, all_alerts = [], []
    for model_name in models:
        # OOF predictions on all seven DHSV wells: heldout controls have DIFFERENT wells.
        inner = reference_oof.loc[reference_oof.model.eq(model_name)].copy()
        choice, _ = choose_policy(inner)
        model = bench.make_model(model_name)
        model.fit(cohort[list(ALL_FEATURES)], cohort.target)
        score = np.asarray(model.predict_proba(controls[list(ALL_FEATURES)])[:, 1], dtype=float)
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError(f"Недопустимые контрольные оценки модели {model_name}")
        records = controls[list(INDEX_COLUMNS)].copy()
        records["model"] = model_name
        records["score"] = score
        scored = apply_policy(records, choice["threshold"], choice["confirmations"])
        scored["threshold"] = choice["threshold"]
        scored["confirmations"] = choice["confirmations"]
        choices.append({"model": model_name, **choice})
        all_alerts.append(scored)
        print(f"{model_name:>13s}: control threshold={choice['threshold']:.2f}, "
              f"confirms={choice['confirmations']}", flush=True)
    combined = pd.concat(all_alerts, ignore_index=True)
    by_class = []
    by_well = []
    for (model, cls), group in combined.groupby(["model", "event_class_folder"]):
        by_class.append({"model": model, "event_class_folder": int(cls), **control_metrics(group)})
    for (model, well), group in combined.groupby(["model", "well_id"]):
        by_well.append({"model": model, "well_id": int(well), **control_metrics(group)})
    return combined, pd.DataFrame(choices), pd.DataFrame(by_class), pd.DataFrame(by_well)


def control_metrics(g: pd.DataFrame) -> dict:
    """Eligible sampled windows only: NOT uninterrupted clock hours of plant operation."""
    n = int(len(g))
    hours = n * 5/60
    return {"wells": int(g.well_id.nunique()), "files": int(g.file.nunique()), "windows": n,
            "eligible_sampled_hours": hours, "active_fraction": float(g.alert_active.mean()),
            "alarm_starts": int(g.alarm_start.sum()),
            "starts_per_eligible_sampled_hour": int(g.alarm_start.sum())/hours if hours else None}


def run(cohort_path: Path, oof_path: Path, index_path: Path, data_root: Path, out: Path,
        models: tuple[str,...] = ("logreg", "random_forest", "catboost", "lightgbm")) -> dict:
    if not cohort_path.is_file() or not oof_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Отсутствуют cohort, OOF или индекс контрольных окон")
    cohort = bench.load_cohort(cohort_path)
    oof = load_oof(oof_path, cohort, models)
    index = load_index(index_path, expected_wells=set(cohort.well_id))
    out.mkdir(parents=True, exist_ok=True)
    features = extract_controls(index, data_root)
    features.to_csv(out / "control_features.csv", index=False)
    scored, choices, by_class, by_well = score_controls(cohort, oof, features, models)
    scored.to_csv(out / "control_alerts.csv", index=False)
    choices.to_csv(out / "control_selected_policies.csv", index=False)
    by_class.to_csv(out / "control_by_class.csv", index=False)
    by_well.to_csv(out / "control_by_well.csv", index=False)
    totals = pd.DataFrame([{"model": name, **control_metrics(g)} for name,g in scored.groupby("model")])
    totals.to_csv(out / "control_comparison.csv", index=False)
    manifest = {
        "experiment": "class2_control_specificity_v0.1_exploratory",
        "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        "control_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "models": list(models), "n_control_wells": int(index.well_id.nunique()),
        "n_control_files": int(index.file.nunique()), "n_control_windows": int(len(index)),
        "data_root": str(data_root), "control_features_from_raw": True,
        "control_target_used_to_fit": False, "control_target_used_to_tune": False,
        "threshold_tuned_on_class2_loow": True,
        "scope": "Preselected eligible negative windows in control-only wells",
        "limitations": ["Control-only wells have been examined previously: NOT independent final validation",
                        "No hard negative retraining in this experiment",
                        "Sampled eligible negative hours are NOT full industrial uptime or field FAR",
                        "Control index selection inherited from earlier audited experiment; generalize cautiously",
                        "Full-fit model controls and LOOW class2 have different training-set sizes"]}
    (out / "control_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(totals.round(4).to_string(index=False))
    print(f"\nОтчёты: {out}")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, default=Path("outputs/class2_horizons/cohort_features.csv"))
    p.add_argument("--oof", type=Path, default=Path("outputs/class2_model_benchmark/oof_predictions.csv"))
    p.add_argument("--index", type=Path, default=Path("reference/control_heldout_index.csv"))
    p.add_argument("--data-root", type=Path, default=Path("dataset/3W"))
    p.add_argument("--out", type=Path, default=Path("outputs/class2_control_specificity_benchmark"))
    p.add_argument("--models", nargs="+", choices=bench.MODEL_NAMES, default=["logreg","random_forest","catboost","lightgbm"])
    args=p.parse_args()
    run(args.cohort,args.oof,args.index,args.data_root,args.out,tuple(dict.fromkeys(args.models)))

if __name__=="__main__":
    main()
