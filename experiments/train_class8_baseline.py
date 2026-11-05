"""First reproducible baseline for 3W class 8, 6-hour horizon.

Run from CPS-PredDiag root:
    python experiments/train_class8_baseline.py

Scientific scope: predicts occurrence of class 8 within 6 hours conditioned
on the evaluated period being labelled 108. This is NOT a forecast of the
original onset 0 -> 108 or an overall industrial failure probability.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "outputs" / "features" / "class8_6h_ml_features.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "models" / "class8_baseline"

SENSORS = ("P-TPT", "P-MON-CKP", "T-TPT")
ALL_FEATURES = [
    f"{sensor}_{stat}"
    for sensor in SENSORS
    for stat in ("mean", "std", "delta", "slope_per_hour")
]
PRESSURE_ONLY_FEATURES = [
    f"P-MON-CKP_{stat}"
    for stat in ("mean", "std", "delta", "slope_per_hour")
]


def build_estimator(kind: str):
    if kind == "dummy_prior":
        return DummyClassifier(strategy="prior")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("rf", RandomForestClassifier(
            n_estimators=70,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )),
    ])


def score_binary(y_true: np.ndarray, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    result = {
        "samples": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc_ap": average_precision_score(y_true, probability)
        if np.any(y_true == 1) else float("nan"),
        "roc_auc": roc_auc_score(y_true, probability)
        if len(np.unique(y_true)) == 2 else float("nan"),
        "precision_0.5": precision_score(y_true, predicted, zero_division=0),
        "recall_0.5": recall_score(y_true, predicted, zero_division=0),
        "f1_0.5": f1_score(y_true, predicted, zero_division=0),
        "false_alarms_windows": int(((predicted == 1) & (y_true == 0)).sum()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-8 grouped RF baseline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"file", "well_id", "decision_time", "target", *ALL_FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")

    if data.empty:
        raise ValueError("Обучающая таблица пуста")
    if set(data["target"].dropna().unique()) != {0, 1} or data["target"].isna().any():
        raise ValueError("Ожидаются две непустые целевые категории 0 и 1")
    if data["well_id"].isna().any() or data["file"].isna().any():
        raise ValueError("В метаданных отсутствует well_id или file")
    if data.duplicated(["file", "decision_time"]).any():
        raise ValueError("Обнаружены дубли точек прогнозирования")

    # Сохраняем исключительно числовые признаки, не исходные идентификаторы.
    numeric = data[ALL_FEATURES].apply(pd.to_numeric, errors="raise")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    presence = pd.DataFrame({
        f"has_{sensor}": numeric[f"{sensor}_mean"].notna().astype(int)
        for sensor in SENSORS
    })
    models = {
        "dummy_prior": numeric[PRESSURE_ONLY_FEATURES],
        "rf_pressure_only": numeric[PRESSURE_ONLY_FEATURES],
        "rf_all_12": numeric[ALL_FEATURES],
        "rf_availability_only": presence,
    }

    target = data["target"].to_numpy(dtype=int)
    groups = data["well_id"].to_numpy()
    splits = list(LeaveOneGroupOut().split(numeric, target, groups))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = data[["file", "well_id", "decision_time", "target"]].copy()
    overview = []
    by_well = []
    by_file = []

    for name, features in models.items():
        probabilities = np.full(len(data), np.nan, dtype=float)

        for train_idx, test_idx in splits:
            estimator = build_estimator(name)
            estimator.fit(features.iloc[train_idx], target[train_idx])
            probabilities[test_idx] = estimator.predict_proba(features.iloc[test_idx])[:, 1]

        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Не все OOF-прогнозы получены для {name}")

        oof[f"prob_{name}"] = probabilities
        result = {"model": name, **score_binary(target, probabilities)}
        overview.append(result)

        for well, frame in data.groupby("well_id", sort=True):
            indices = frame.index.to_numpy()
            by_well.append({
                "model": name,
                "well_id": well,
                **score_binary(target[indices], probabilities[indices]),
            })

        # Иллюстративная эпизодная метрика: хотя бы одно корректное
        # предупреждение в положительных 15-минутных окнах данного файла.
        for file, frame in data.groupby("file", sort=True):
            indices = frame.index.to_numpy()
            y = target[indices]
            predicted = probabilities[indices] >= 0.5
            by_file.append({
                "model": name,
                "file": file,
                "well_id": frame["well_id"].iloc[0],
                "has_event": bool((y == 1).any()),
                "detected_event": bool((predicted & (y == 1)).any()),
                "false_alarm_windows": int((predicted & (y == 0)).sum()),
                "total_windows": len(y),
            })

    summary = pd.DataFrame(overview)
    well_scores = pd.DataFrame(by_well)
    file_scores = pd.DataFrame(by_file)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    well_scores.to_csv(args.output_dir / "by_well.csv", index=False)
    file_scores.to_csv(args.output_dir / "by_file.csv", index=False)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    print("\nLeave-One-Well-Out: 9 групп, каждый тест на неиспользованной скважине")
    print("PR-AUC здесь означает Average Precision (AP). Порог срабатывания: 0.5 (не оптимизирован).")
    print(summary[["model", "pr_auc_ap", "roc_auc", "precision_0.5", "recall_0.5", "f1_0.5", "false_alarms_windows"]].round(3).to_string(index=False))
    print("\nЭпизоды с событием (12 файлов, группировка по file):")
    print(file_scores[file_scores.has_event].groupby("model").agg(
        events=("has_event", "size"), detected=("detected_event", "sum")
    ).to_string())
    print("\nСводка AP по отдельным скважинам:")
    print(well_scores.pivot(index="well_id", columns="model", values="pr_auc_ap").round(3).to_string())
    print(f"\nОтчёты: {args.output_dir}")
    print("Важно: это предварительная офлайн-проверка для переходного класса 108, не реальная эксплуатационная надёжность.")


if __name__ == "__main__":
    main()
