"""Petrobras 3W class=2: CatBoost vs RandomForest on EXACT original 253 windows.

Research only. The evaluation cohort is preselected by true transient class 102
(oracle). It is NOT a blind/online model evaluation. No changes to MLAgent.

From the project root:
  python -m pip install catboost
  python experiments/compare_class2_catboost.py

Expected files:
  dataset/2/WELL-*.parquet                 (original real Petrobras files)
  experiments/inputs/class2_horizon_points.csv
  experiments/inputs/class2_rf_baseline_oof.csv
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_agent import FEATURES, SENSORS, WINDOW_SECONDS, features_from_history

KEYS = ["episode_id", "file", "well_id", "decision_time"]
FIXED_THRESHOLD = 0.5   # historical research cut-off, NOT tuned / calibrated
SEED = 42


def read_points(path: Path) -> pd.DataFrame:
    """Lock the cohort to the previously audited h=15 labeled 102 windows."""
    points = pd.read_csv(path)
    required = set(KEYS + ["horizon_minutes", "target", "event_time", "minutes_until_event"])
    absent = required - set(points.columns)
    if absent:
        raise ValueError(f"Отсутствуют столбцы таблицы окон: {sorted(absent)}")
    points = points.loc[points["horizon_minutes"].eq(15)].copy()
    # Right-censored / uncertain outcomes stay excluded, never labeled normal.
    points = points.loc[points["target"].isin([0, 1])].copy()
    points["target"] = points["target"].astype(int)
    points["well_id"] = points["well_id"].astype(int)
    points["decision_time"] = pd.to_datetime(points["decision_time"], errors="raise")
    if points[KEYS].isna().any().any() or points.duplicated(KEYS).any():
        raise ValueError("Пропущенные или повторные идентификаторы окон")
    if points.empty or points["target"].nunique() < 2:
        raise ValueError("Для сравнения нужны размеченные окна обоих классов")
    if points["well_id"].nunique() < 3:
        raise ValueError("Для проверки по скважинам нужны минимум три скважины")
    points = points.sort_values(KEYS, kind="stable").reset_index(drop=True)
    return points


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare second-resolution timestamps without dropping/moving observations."""
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(frame.index.dtype):
            raise ValueError("Нет timestamp / DatetimeIndex в Parquet (числовой индекс нельзя трактовать как время)")
        try:
            frame.index = pd.to_datetime(frame.index, errors="raise")
        except Exception as exc:
            raise ValueError("Нет корректной временной оси в Parquet") from exc
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Временная ось должна быть DatetimeIndex")
    frame = frame.copy()
    frame.index = frame.index.as_unit("ns")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("В Parquet повторяющиеся или неупорядоченные моменты")
    return frame


def past_features(frame: pd.DataFrame, moment: pd.Timestamp) -> dict[str, float]:
    """No label/state/future reaches feature extraction."""
    moment = pd.Timestamp(moment).as_unit("ns")
    position = frame.index.get_indexer([moment])[0]
    if position < WINDOW_SECONDS - 1:
        raise ValueError(f"Не хватает истории для момента {moment}")
    # Explicitly pass ONLY sensor columns, and only rows available by moment.
    sensor_data = frame.reindex(columns=list(SENSORS))
    history = sensor_data.iloc[position - WINDOW_SECONDS + 1 : position + 1]
    values, _ = features_from_history(history)
    result = values.iloc[0].to_dict()
    return {name: float(result[name]) for name in FEATURES}


def build_feature_table(points: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """Extract precisely the 12 features already used in ML Agent v0.1."""
    records: list[dict] = []
    for filename, subset in points.groupby("file", sort=True):
        if Path(filename).name != filename or not filename.endswith(".parquet"):
            raise ValueError(f"Некорректное имя файла: {filename}")
        path = dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Не найден Parquet: {path}")
        frame = prepare_frame(pd.read_parquet(path))
        for row in subset.itertuples(index=False):
            d = row._asdict()
            feat = past_features(frame, d["decision_time"])
            records.append({**d, **feat})
        print(f"{filename}: окон {len(subset)}")
    table = pd.DataFrame.from_records(records)
    table = table.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if len(table) != len(points) or table[KEYS].duplicated().any():
        raise RuntimeError("Потеря или дублирование окон при построении признаков")
    return table


def compare_historical_rf(points: pd.DataFrame, oof_path: Path) -> pd.DataFrame:
    """Use the previously saved OUT-OF-FOLD RF scores, never full-fit model."""
    old = pd.read_csv(oof_path)
    required = set(KEYS + ["target", "prob_rf_all12"])
    if not required.issubset(old.columns):
        raise ValueError("В старом OOF отсутствуют обязательные столбцы")
    old = old.copy()
    old["decision_time"] = pd.to_datetime(old["decision_time"], errors="raise")
    old["well_id"] = old["well_id"].astype(int)
    if old.duplicated(KEYS).any():
        raise ValueError("Повторные ключи в сохранённом OOF")
    merged = points[KEYS + ["target"]].merge(
        old[KEYS + ["target", "prob_rf_all12"]], on=KEYS,
        how="outer", validate="one_to_one", indicator=True,
        suffixes=("_points", "_old"),
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(points):
        raise ValueError("Старый RF OOF относится к другой выборке окон")
    if not (merged["target_points"] == merged["target_old"]).all():
        raise ValueError("Метки старого эксперимента расходятся")
    score = pd.to_numeric(merged["prob_rf_all12"], errors="raise")
    if score.isna().any() or not score.between(0, 1).all():
        raise ValueError("Некорректные оценки старого RF")
    return merged[KEYS + ["prob_rf_all12"]].rename(
        columns={"prob_rf_all12": "score_rf_previous"}
    )


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score,
        recall_score, roc_auc_score, confusion_matrix,
    )
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Некорректные или неполные прогнозы")
    predicted = scores >= FIXED_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "positive": int(y_true.sum()),
        "ap": float(average_precision_score(y_true, scores)) if y_true.sum() else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "precision_05": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_05": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_05": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn), "fp_windows": int(fp), "fn": int(fn), "tp_windows": int(tp),
    }


def train_leave_one_well_out(table: pd.DataFrame, cat_iterations: int = 300) -> pd.DataFrame:
    """Both algorithms receive identical training/testing folds, no well_id feature."""
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    needed = set(FEATURES + KEYS + ["target"])
    if not needed.issubset(table.columns):
        raise ValueError("Недостаточно признаков или метаданных")
    values = table[FEATURES].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).astype(float)
    y = table["target"].to_numpy(dtype=int)
    wells = table["well_id"].to_numpy(dtype=int)
    out = table[KEYS + ["target", "event_time", "minutes_until_event"]].copy()
    for c in ("score_rf_refit", "score_catboost"):
        out[c] = np.nan

    for well in sorted(np.unique(wells)):
        train = wells != well
        test = wells == well
        if not train.any() or not test.any() or len(np.unique(y[train])) != 2:
            raise ValueError(f"Необучаемый fold скважины {well}: нужен 0 и 1 в train")
        Xtr, Xte = values.loc[train], values.loc[test]
        # Imputation is fitted strictly on TRAINING wells (no leakage).
        rf = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=70, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=SEED,
                n_jobs=1,
            ),
        )
        rf.fit(Xtr, y[train])
        cb = CatBoostClassifier(
            iterations=cat_iterations, depth=4, learning_rate=0.03,
            l2_leaf_reg=5, loss_function="Logloss",
            auto_class_weights="Balanced", nan_mode="Min",
            random_seed=SEED, thread_count=2,
            verbose=False, allow_writing_files=False,
        )
        cb.fit(Xtr, y[train])
        out.loc[test, "score_rf_refit"] = rf.predict_proba(Xte)[:, 1]
        out.loc[test, "score_catboost"] = cb.predict_proba(Xte)[:, 1]
        print(f"  Test WELL-{well:05d}: {test.sum()} окон; "
              f"train {train.sum()}, положительных в train: {y[train].sum()}")
    if out[["score_rf_refit", "score_catboost"]].isna().any().any():
        raise RuntimeError("Не все окна получили out-of-fold прогноз")
    return out


def event_table(oof: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Detection only for episodes with an eligible positive window."""
    items = []
    for episode, group in oof.groupby("episode_id", sort=True):
        positives = group[group["target"].eq(1)].copy()
        if positives.empty:
            continue
        alerts = positives[positives[model_col] >= FIXED_THRESHOLD]
        earliest = alerts.sort_values("decision_time").head(1)
        lead = earliest["minutes_until_event"].iloc[0] if len(earliest) else np.nan
        items.append({
            "model": model_col, "episode_id": episode,
            "well_id": int(group["well_id"].iloc[0]),
            "n_positive_windows": len(positives),
            "detected": bool(len(earliest)),
            "lead_minutes_first_correct_alert": float(lead),
        })
    return pd.DataFrame(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "2")
    parser.add_argument("--points", type=Path, default=ROOT / "experiments" / "inputs" / "class2_horizon_points.csv")
    parser.add_argument("--previous-oof", type=Path, default=ROOT / "experiments" / "inputs" / "class2_rf_baseline_oof.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "catboost_compare")
    args = parser.parse_args()

    points = read_points(args.points)
    print(f"Исходный протокол: {len(points)} окон; {points.target.sum()} положительных; "
          f"скважин {points.well_id.nunique()}")
    previous = compare_historical_rf(points, args.previous_oof)
    # Save the historical RF summary even if parquet isn't accessible.
    old_join = points[KEYS + ["target"]].merge(previous, on=KEYS, validate="one_to_one")
    old_stats = evaluate_binary(old_join.target.to_numpy(), old_join.score_rf_previous.to_numpy())
    print("Исторический RF OOF: AP = {:.4f}, F1@0.5 = {:.4f}".format(
        old_stats["ap"], old_stats["f1_05"]
    ))

    print("Извлекаем те же 12 признаков из реальных Parquet...")
    features_table = build_feature_table(points, args.dataset_dir)
    oof = train_leave_one_well_out(features_table)
    oof = oof.merge(previous, on=KEYS, how="left", validate="one_to_one")
    models = ["score_rf_previous", "score_rf_refit", "score_catboost"]
    summary = []
    events_all = []
    per_well = []
    for col in models:
        metrics = evaluate_binary(oof["target"].to_numpy(), oof[col].to_numpy())
        event_df = event_table(oof, col)
        events_all.append(event_df)
        metrics.update({
            "model": col,
            "eligible_events": len(event_df),
            "detected_events": int(event_df["detected"].sum()) if len(event_df) else 0,
            "median_lead_detected_min": (float(event_df.loc[event_df.detected, "lead_minutes_first_correct_alert"].median())
                                         if len(event_df) and event_df.detected.any() else None),
        })
        summary.append(metrics)
        for well, part in oof.groupby("well_id"):
            per_well.append({"model": col, "well_id": int(well),
                             **evaluate_binary(part.target.to_numpy(), part[col].to_numpy())})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(per_well).to_csv(args.output_dir / "by_well.csv", index=False)
    pd.concat(events_all, ignore_index=True).to_csv(args.output_dir / "by_event.csv", index=False)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    features_table.to_csv(args.output_dir / "features_audit.csv", index=False)

    import catboost, sklearn
    metadata = {
        "task": "oracle_gated_class2_h15", "source": "Petrobras 3W Dataset v2.0.0, real WELL",
        "evaluation": "Leave-One-Well-Out", "n_windows": len(oof),
        "n_positive": int(oof.target.sum()), "n_wells": int(oof.well_id.nunique()),
        "feature_names": FEATURES, "threshold": FIXED_THRESHOLD,
        "random_seed": SEED, "catboost_iterations": 300,
        "sklearn_version": sklearn.__version__, "catboost_version": catboost.__version__,
        "pandas_version": pd.__version__, "python_version": platform.python_version(),
        "notes": [
            "Uses oracle-labelled transient 102 window selection; NOT blind predictive accuracy.",
            "No model selected or tuned on held-out wells in this script; feature design previously explored on these wells.",
            "Old RF scores from saved OOF, not from full-fit model.",
            "No deployment inference or live transition detection.",
            "Class2 baseline features extracted from past 300 seconds only.",
            "Censored/unknown future labels excluded using previously audited points table.",
        ],
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for col in models:
            p, r, _ = precision_recall_curve(oof.target, oof[col])
            ax.plot(r, p, label=col)
        ax.axhline(oof.target.mean(), color="gray", linestyle=":", label="Positive prevalence")
        ax.set(xlabel="Recall", ylabel="Precision", title="Class 2, 15 min, LOOW (oracle gated)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "pr_curves.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib не установлен — график PR пропущен; CSV сформированы")

    print("\n=== Сравнение (все оценки OOF; порог 0.5) ===")
    print(pd.DataFrame(summary)[["model", "n", "positive", "ap", "precision_05",
                                 "recall_05", "f1_05", "fp_windows", "detected_events"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Файлы результатов: {args.output_dir}")
    print("ОГРАНИЧЕНИЕ: протокол с oracle 102; не переносить эти метрики в blind-режим.")


if __name__ == "__main__":
    main()
