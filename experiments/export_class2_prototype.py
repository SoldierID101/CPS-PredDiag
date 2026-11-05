"""Обучить и сохранить ДЕМОНСТРАЦИОННУЮ модель класса 2 (H=15 минут).

Источники: outputs/features/class2_5min_ml_features.csv,
           outputs/inventory/class2_horizon_points.csv
Модель обучается на ВСЕХ размеченных окнах для последующего replay.
Оценка качества — только в отдельно сохранённых Leave-One-Well-Out отчётах!

Запуск: python experiments/export_class2_prototype.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SENSORS = ("P-MON-CKP", "P-TPT", "T-TPT")
FEATURES = [f"{s}_{stat}" for s in SENSORS for stat in (
    "mean", "std", "delta", "slope_per_hour"
)]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs/features/class2_5min_ml_features.csv")
    parser.add_argument("--points", type=Path,
                        default=ROOT / "outputs/inventory/class2_horizon_points.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/models/class2_prototype")
    parser.add_argument("--force", action="store_true", help="Разрешить перезаписать модель")
    args = parser.parse_args()

    points = pd.read_csv(args.points, dtype={"well_id": str},
                         parse_dates=["decision_time"])
    features = pd.read_csv(args.features, dtype={"well_id": str},
                           parse_dates=["decision_time"])
    keys = ["episode_id", "file", "well_id", "decision_time"]
    needed = set(keys + FEATURES)
    if missing := sorted(needed - set(features.columns)):
        raise ValueError(f"В таблице признаков отсутствуют колонки: {missing}")
    if not set(keys + ["horizon_minutes", "target"]).issubset(points.columns):
        raise ValueError("Не хватает столбцов в class2_horizon_points.csv")
    if features.duplicated(keys).any():
        raise ValueError("Повторяющиеся ключи в таблице признаков")
    selected = points.loc[
        (points.horizon_minutes == 15) & (points.target.isin([0, 1])),
        keys + ["target"]
    ].copy()
    if selected.duplicated(keys).any() or selected.empty:
        raise ValueError("Повторяющиеся или отсутствующие целевые метки")
    data = selected.merge(features[keys + FEATURES], on=keys,
                          how="left", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Не для всех размеченных точек есть признаки")
    x = data[FEATURES].apply(pd.to_numeric, errors="raise").replace(
        [np.inf, -np.inf], np.nan
    )
    y = data.target.astype(int).to_numpy()
    if np.unique(y).tolist() != [0, 1]:
        raise ValueError("Нет обоих классов")
    if x.notna().sum(axis=1).eq(0).any():
        raise ValueError("Есть окна без единого пригодного признака")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("forest", RandomForestClassifier(
            n_estimators=70, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=1,
        )),
    ])
    pipeline.fit(x, y)
    model_file = "rf_all12_h15_v0_1.joblib"
    model_path = args.output / model_file
    manifest_path = args.output / "manifest.json"
    if not args.force and (model_path.exists() or manifest_path.exists()):
        raise FileExistsError("Файлы модели уже существуют. Для перезаписи --force")
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    manifest = {
        "model_version": "class2_rf_all12_h15_v0.1",
        "model_file": model_file,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Petrobras 3W v2.0.0, real WELL files, event class 2",
        "event_class": 2,
        "transition_class": 102,
        "requires_oracle_or_independently_detected_transition": True,
        "horizon_minutes": 15,
        "window_seconds": 300,
        "min_coverage_per_sensor": 0.9,
        "features": FEATURES,
        "training_rows": int(len(y)),
        "training_positive": int(y.sum()),
        "training_wells": int(data.well_id.nunique()),
        "estimator": "RandomForestClassifier",
        "estimator_params": {"n_estimators": 70, "min_samples_leaf": 5,
                             "class_weight": "balanced_subsample", "random_state": 42},
        "sklearn_version": sklearn.__version__,
        "source_sha256": {"features": file_hash(args.features),
                          "points": file_hash(args.points)},
        "scores_calibrated": False,
        "usage": "Research replay only. Not validated for production safety control.",
        "evaluation": "Independent LOOW reported separately; full-fit model not evaluated on new wells.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    reloaded = joblib.load(model_path)
    original_score = pipeline.predict_proba(x.iloc[:1])[0, 1]
    loaded_score = reloaded.predict_proba(x.iloc[:1])[0, 1]
    if not np.isclose(original_score, loaded_score, rtol=0, atol=1e-12):
        raise RuntimeError("После сохранения модель не воспроизводит прогноз")
    print(f"Сохранено: {model_path}")
    print(f"Метаданные: {manifest_path}")
    print(f"Обучающих окон: {len(y)}, положительных: {y.sum()}, скважин: {data.well_id.nunique()}")
    print("ВНИМАНИЕ: модель обучена на всех размеченных окнах и НЕ проверялась отдельно на этих же данных.")
    print("Требует внешнего подтверждения class=102; оценки не калиброваны.")


if __name__ == "__main__":
    main()
