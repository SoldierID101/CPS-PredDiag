"""CPS-PredDiag: первый ML baseline для 3W, класс 2 (DHSV).

Задача: при ИЗВЕСТНОМ по офлайн-разметке состоянии 102 предсказать
появление класса 2 в следующие 15 или 30 минут. Это НЕ прогноз начала 102.

Запуск из корня проекта:
    python experiments/train_class2_baseline.py

Вход:
    outputs/inventory/class2_horizon_points.csv
    dataset/2/WELL-*.parquet

Требуются pandas, numpy, pyarrow, scikit-learn.
Скрипт не изменяет исходный датасет и не перезаписывает результаты класса 8.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SENSORS = ("P-MON-CKP", "P-TPT", "T-TPT")
STATS = ("mean", "std", "delta", "slope_per_hour")
FEATURES = [f"{sensor}_{stat}" for sensor in SENSORS for stat in STATS]
CKP_FEATURES = [f"P-MON-CKP_{stat}" for stat in STATS]
WINDOW = 300  # 5 минут × 1 Гц
TRANSIENT_CLASS = 102
RANDOM_STATE = 42


def describe_sensor(series: pd.Series | None) -> dict[str, float]:
    if series is None:
        return {stat: np.nan for stat in STATS}
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    valid = np.isfinite(values)
    # Для модели разрешаем не более 10% отсутствующих показаний внутри окна.
    if valid.sum() < int(WINDOW * 0.9):
        return {stat: np.nan for stat in STATS}
    x = np.flatnonzero(valid).astype(np.float64)
    y = values[valid]
    return {
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)),
        "delta": float(y[-1] - y[0]),
        "slope_per_hour": float(np.polyfit(x, y, 1)[0] * 3600),
    }


def extract_features(points: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    unique_points = points.drop_duplicates(["file", "decision_time"])
    rows = []
    errors = []
    for filename, subset in unique_points.groupby("file", sort=True):
        path = dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Не найден исходный Parquet: {path}")
        schema = set(pq.read_schema(path).names)
        if "class" not in schema:
            raise ValueError(f"Отсутствует столбец class: {filename}")
        available_sensors = [name for name in SENSORS if name in schema]
        df = pd.read_parquet(path, columns=["class", *available_sensors])
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Ожидался DatetimeIndex: {filename}")
        if df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise ValueError(f"Некорректный порядок времени: {filename}")
        labels = pd.to_numeric(df["class"], errors="coerce")

        calculated = 0
        for point in subset.itertuples(index=False):
            moment = pd.Timestamp(point.decision_time)
            pos = int(df.index.get_indexer([moment])[0])
            if pos < WINDOW - 1:
                errors.append(f"{filename}: нет 300 измерений до {moment}")
                continue
            start = pos - WINDOW + 1
            window = df.iloc[start:pos + 1]
            times = window.index.asi8
            expected_start = moment - pd.Timedelta(seconds=WINDOW - 1)
            if (window.index[0] != expected_start
                    or not np.all(np.diff(times) == 1_000_000_000)
                    or window.index[0] < pd.Timestamp(point.transient_start)
                    or not bool(labels.iloc[start:pos + 1].eq(TRANSIENT_CLASS).all())):
                errors.append(f"{filename}: некорректная история 102 к {moment}")
                continue

            result = {
                "episode_id": point.episode_id,
                "file": filename,
                "well_id": str(point.well_id),
                "decision_time": moment,
            }
            for sensor in SENSORS:
                stats = describe_sensor(window[sensor] if sensor in window else None)
                for stat, value in stats.items():
                    result[f"{sensor}_{stat}"] = value
            rows.append(result)
            calculated += 1
        print(f"{filename}: признаков рассчитано для {calculated} точек")

    if errors:
        print("\nНекорректные окна (первые 10):")
        for message in errors[:10]:
            print(" -", message)
        raise RuntimeError(
            f"Обнаружено {len(errors)} некорректных историй. "
            "Обучение остановлено: нельзя незаметно менять состав выборки."
        )
    out = pd.DataFrame(rows)
    if out.empty or out.duplicated(["file", "decision_time"]).any():
        raise ValueError("Пустые или повторяющиеся признаки")
    return out


def build_model(name: str):
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior")
    if name == "logreg_elapsed_oracle":
        return Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("forest", RandomForestClassifier(
            n_estimators=70, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    two_classes = np.unique(y).size == 2
    return {
        "n_windows": len(y),
        "n_positive": int(y.sum()),
        "ap": float(average_precision_score(y, p)) if two_classes else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else np.nan,
        "precision_05": float(precision_score(y, pred, zero_division=0)),
        "recall_05": float(recall_score(y, pred, zero_division=0)),
        "f1_05": float(f1_score(y, pred, zero_division=0)),
        "fp_windows": int(((pred == 1) & (y == 0)).sum()),
        "tp_windows": int(((pred == 1) & (y == 1)).sum()),
    }


def evaluate_horizon(horizon: int, points: pd.DataFrame,
                     features: pd.DataFrame, output_dir: Path) -> None:
    selected = points.loc[
        (points["horizon_minutes"] == horizon) & points["target"].isin([0, 1])
    ].copy()
    if selected.empty:
        raise ValueError(f"Нет размеченных точек для H={horizon}")
    data = selected.merge(
        features, on=["episode_id", "file", "well_id", "decision_time"],
        how="left", validate="one_to_one", indicator=True,
    )
    if not data["_merge"].eq("both").all():
        raise ValueError(f"Часть точек H={horizon} не имеет истории датчиков")
    data.drop(columns="_merge", inplace=True)
    data = data.reset_index(drop=True)
    x_all = data[FEATURES].apply(pd.to_numeric, errors="raise")
    x_all = x_all.replace([np.inf, -np.inf], np.nan)
    x_ckp = x_all[CKP_FEATURES]
    # Контроль: время начала 102 взято из ИСТИННОЙ офлайн-разметки.
    # Ни event_time, ни minutes_until_event, ни ID скважины сюда не входят.
    elapsed = pd.to_numeric(data["minutes_since_transient_start"], errors="raise")
    x_elapsed = pd.DataFrame({"log1p_elapsed_min_oracle": np.log1p(elapsed)})
    if not np.isfinite(x_elapsed.to_numpy()).all():
        raise ValueError("Некорректное время от начала перехода")

    inputs = {
        "dummy_prior": x_ckp,
        "rf_ckp_only": x_ckp,
        "rf_all12": x_all,
        "logreg_elapsed_oracle": x_elapsed,
    }
    y = data["target"].to_numpy(dtype=np.int64)
    groups = data["well_id"].to_numpy()
    if np.unique(groups).size < 3 or np.unique(y).size != 2:
        raise ValueError("Недостаточно групп или классов")
    splits = list(LeaveOneGroupOut().split(x_all, y, groups))
    horizon_dir = output_dir / f"horizon_{horizon}min"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    oof = data[[
        "episode_id", "file", "well_id", "decision_time", "target",
        "event_time", "minutes_until_event",
    ]].copy()
    all_metrics = []
    per_well = []
    per_episode = []
    for name, x in inputs.items():
        prob = np.full(len(data), np.nan)
        for train, test in splits:
            if np.unique(y[train]).size < 2:
                raise ValueError("В обучающей части отсутствует один из классов")
            model = build_model(name)
            model.fit(x.iloc[train], y[train])
            classes = model.classes_
            positive_class = np.flatnonzero(classes == 1)
            if len(positive_class) != 1:
                raise ValueError(f"Не найдён положительный класс: {name}")
            prob[test] = model.predict_proba(x.iloc[test])[:, positive_class[0]]
        if not np.isfinite(prob).all():
            raise ValueError(f"Прогнозы неполные: {name}")
        oof[f"prob_{name}"] = prob
        all_metrics.append({"model": name, "horizon_min": horizon,
                            **metrics(y, prob)})

        for well_id, idx in data.groupby("well_id", sort=True).indices.items():
            per_well.append({
                "model": name, "horizon_min": horizon, "well_id": well_id,
                **metrics(y[idx], prob[idx]),
            })
        for episode, idx in data.groupby("episode_id", sort=True).indices.items():
            yt = y[idx]
            predicted = prob[idx] >= 0.5
            correct = (yt == 1) & predicted
            leads = pd.to_numeric(
                data.iloc[idx].loc[correct, "minutes_until_event"],
                errors="coerce",
            )
            per_episode.append({
                "model": name,
                "horizon_min": horizon,
                "episode_id": episode,
                "well_id": data.iloc[idx[0]]["well_id"],
                "file": data.iloc[idx[0]]["file"],
                "has_positive_windows": bool((yt == 1).any()),
                "detected": bool(correct.any()),
                "first_correct_lead_min": float(leads.max()) if len(leads) else np.nan,
                "fp_windows": int((predicted & (yt == 0)).sum()),
            })

    summary = pd.DataFrame(all_metrics)
    wells = pd.DataFrame(per_well)
    episodes = pd.DataFrame(per_episode)
    summary.to_csv(horizon_dir / "summary.csv", index=False, encoding="utf-8-sig")
    wells.to_csv(horizon_dir / "by_well.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(horizon_dir / "by_episode.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(horizon_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Горизонт {horizon} мин ===")
    print(f"Размеченных окон: {len(data)}; target=1: {y.sum()}; "
          f"target=0: {(y == 0).sum()}; скважин: {np.unique(groups).size}")
    print(summary[["model", "ap", "roc_auc", "precision_05", "recall_05",
                   "f1_05", "fp_windows"]].round(3).to_string(index=False))
    detected = episodes.loc[episodes["has_positive_windows"]].groupby(
        "model", sort=False
    ).agg(detected=("detected", "sum"), episodes=("detected", "size"))
    print("Эпизоды с зарегистрированным событием:")
    print(detected.to_string())
    print(f"Отчёты: {horizon_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path,
                        default=ROOT / "outputs/inventory/class2_horizon_points.csv")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/2")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "outputs/features")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/models/class2_baseline")
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30])
    args = parser.parse_args()

    points = pd.read_csv(args.points, dtype={"well_id": str},
                         parse_dates=["decision_time", "transient_start", "event_time"])
    required = {
        "episode_id", "file", "well_id", "decision_time", "horizon_minutes",
        "target", "transient_start", "event_time", "minutes_since_transient_start",
        "minutes_until_event",
    }
    if missing := sorted(required - set(points.columns)):
        raise ValueError(f"Не хватает столбцов в аудите: {missing}")
    if points.empty or points[list(required - {"target", "event_time", "minutes_until_event"})].isna().any().any():
        raise ValueError("Пустой отчёт или пропуски обязательных метаданных")
    if points.duplicated(["episode_id", "decision_time", "horizon_minutes"]).any():
        raise ValueError("Дубли точек прогнозирования")
    for horizon in args.horizons:
        if horizon not in set(points["horizon_minutes"]):
            raise ValueError(f"Нет горизонта {horizon} минут в аудите")

    # Извлечение датчиков делаем один раз для ВСЕХ уникальных точек.
    # Впоследствии неизвестные метки просто не попадают в обучение.
    features = extract_features(points, args.dataset_dir)
    args.features_dir.mkdir(parents=True, exist_ok=True)
    output_features = args.features_dir / "class2_5min_ml_features.csv"
    features.to_csv(output_features, index=False, encoding="utf-8-sig")
    print(f"\nИзвлечено {len(features)} уникальных окон; признаков: {len(FEATURES)}")
    print("Доля пропусков в признаках (%):")
    print((features[FEATURES].isna().mean() * 100).round(1).to_string())
    print(f"Файл признаков: {output_features}")

    for horizon in sorted(set(args.horizons)):
        evaluate_horizon(horizon, points, features, args.output_dir)

    print("\nОграничения: oracle-метка 102 известна заранее; контроль по времени")
    print("использует истинный момент её начала. Метрики — офлайн-оценка")
    print("на уже исследованном датасете; порог 0.5 не настроен.")
    print("Окна не являются независимыми отказами, а пропуски")
    print("заполняются только по обучающей части каждой проверки.")


if __name__ == "__main__":
    main()
