"""Exploratory 30m DHSV cross-well ML benchmark on an already-fixed cohort.

No raw parquet reading, no app changes, no test-well data in training.
Do not interpret results as a pristine independent final validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS = ("P-MON-CKP", "P-TPT", "T-TPT")
STATS = ("mean", "std", "delta", "slope_per_hour")
FEATURES = tuple(f"{sensor}_{stat}" for sensor in SENSORS for stat in STATS)
META = ("file", "well_id", "horizon_minutes", "decision_time", "target", "event_time")
MODEL_NAMES = ("dummy_prior", "logreg", "random_forest", "catboost", "lightgbm")
SEED = 42

def load_cohort(path: Path, horizon_minutes: int = 30) -> pd.DataFrame:
    src = pd.read_csv(path)
    missing = [col for col in (*META, *FEATURES) if col not in src.columns]
    if missing:
        raise ValueError(f"Когорте не хватает столбцов: {missing}")
    frame = src.loc[src["horizon_minutes"].eq(horizon_minutes), list(META) + list(FEATURES)].copy()
    if frame.empty:
        raise ValueError(f"Нет окон для горизонта {horizon_minutes} минут")
    if frame["well_id"].isna().any() or frame["file"].isna().any():
        raise ValueError("Нет WELL ID / имени файла")
    frame["well_id"] = pd.to_numeric(frame["well_id"], errors="raise").astype(int)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], errors="raise")
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    if frame["decision_time"].isna().any():
        raise ValueError("Есть пустые моменты решений")
    if frame.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дублирующиеся решения по файлу и времени")
    if frame["target"].isna().any() or not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("target должен быть бинарным и известным")
    frame["target"] = frame["target"].astype(int)
    if frame.loc[frame.target.eq(1), "event_time"].isna().any():
        raise ValueError("У положительных окон неизвестен момент события")
    if ((frame.target.eq(1)) & (frame.event_time <= frame.decision_time)).any():
        raise ValueError("Необоснованное положительное окно после наступления события")
    frame[list(FEATURES)] = frame[list(FEATURES)].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame[list(FEATURES)].to_numpy(dtype=float)).any():
        raise ValueError("Бесконечные значения признаков")
    return frame.sort_values(["well_id", "file", "decision_time"], kind="stable").reset_index(drop=True)


def make_model(name: str):
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, random_state=SEED),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=2),
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3,
            loss_function="Logloss", random_seed=SEED, thread_count=2,
            verbose=False, allow_writing_files=False,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=4,
            min_child_samples=10, reg_lambda=3.0, class_weight="balanced",
            random_state=SEED, verbosity=-1, n_jobs=2,
        )
    raise ValueError(f"Неизвестный алгоритм: {name}")


def well_folds(frame: pd.DataFrame):
    """Return (test_well, train_index, test_index); entire wells are disjoint."""
    wells = sorted(frame.well_id.unique())
    if len(wells) < 3:
        raise ValueError("Для сравнения требуется не менее трёх WELL")
    for w in wells:
        train = frame.index[frame.well_id.ne(w)].to_numpy()
        test = frame.index[frame.well_id.eq(w)].to_numpy()
        assert not set(frame.iloc[train].well_id).intersection(set(frame.iloc[test].well_id))
        if len(np.unique(frame.loc[train, "target"])) != 2:
            raise ValueError(f"Train без обоих классов при исключении WELL-{w:05d}")
        yield int(w), train, test


def predict_oof(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    results = []
    for well, train_idx, test_idx in well_folds(frame):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        learner = make_model(model_name)
        learner.fit(train[list(FEATURES)], train.target)
        proba = learner.predict_proba(test[list(FEATURES)])[:, 1]
        if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
            raise ValueError(f"Недопустимые оценки {model_name}, WELL-{well:05d}")
        out = test[list(META)].copy()
        out["score"] = np.asarray(proba, dtype=float)
        out["model"] = model_name
        out["test_well"] = well
        results.append(out)
        print(f"{model_name:>13s} | WELL-{well:05d} | train={len(train):3d} test={len(test):3d}", flush=True)
    predictions = pd.concat(results, ignore_index=True)
    if len(predictions) != len(frame):
        raise AssertionError("LOOW не покрывает все строки")
    return predictions.sort_values(["well_id", "file", "decision_time"], kind="stable").reset_index(drop=True)


def evaluate(preds: pd.DataFrame, threshold: float = 0.5) -> dict:
    target = preds.target.to_numpy(dtype=int)
    scores = preds.score.to_numpy(dtype=float)
    binary = scores >= threshold
    positives = preds.loc[preds.target.eq(1)]
    events_total = positives[["file", "event_time"]].drop_duplicates().shape[0]
    events_detected = positives.loc[positives.score.ge(threshold), ["file", "event_time"]].drop_duplicates().shape[0]
    both_classes = len(np.unique(target)) == 2
    return {
        "n": int(len(preds)), "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "ap": float(average_precision_score(target, scores)) if target.sum() else None,
        "roc_auc": float(roc_auc_score(target, scores)) if both_classes else None,
        "precision_05": float(precision_score(target, binary, zero_division=0)),
        "recall_05": float(recall_score(target, binary, zero_division=0)),
        "f1_05": float(f1_score(target, binary, zero_division=0)),
        "fp_windows_05": int(((target == 0) & binary).sum()),
        "events_detected_05": int(events_detected), "events_total": int(events_total),
        "event_recall_05": float(events_detected / events_total) if events_total else None,
    }


def run(input_path: Path, out_dir: Path, model_names=MODEL_NAMES) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Не найдена фиксированная выборка: {input_path}")
    frame = load_cohort(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pred, metrics, by_well = [], [], []
    for name in model_names:
        preds = predict_oof(frame, name)
        all_pred.append(preds)
        metrics.append({"model": name, **evaluate(preds)})
        for well, group in preds.groupby("well_id"):
            by_well.append({"model": name, "well_id": int(well), **evaluate(group)})
    combined = pd.concat(all_pred, ignore_index=True)
    combined.to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(metrics).sort_values("ap", ascending=False).to_csv(out_dir / "model_comparison.csv", index=False)
    pd.DataFrame(by_well).to_csv(out_dir / "by_well.csv", index=False)
    versions = {}
    for package in ("scikit-learn", "catboost", "lightgbm", "numpy", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    estimator_specs = {name: repr(make_model(name)) for name in model_names}
    manifest = {
        "experiment": "class2_model_benchmark_v0.1_exploratory",
        "horizon_minutes": 30, "feature_window_seconds": 300,
        "model_names": list(model_names), "n": len(frame), "positives": int(frame.target.sum()),
        "wells": sorted(map(int, frame.well_id.unique())),
        "distinct_events": int(frame.loc[frame.target.eq(1), ["file", "event_time"]].drop_duplicates().shape[0]),
        "features": list(FEATURES), "threshold_for_secondary_metrics": 0.5,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "random_seed": SEED, "estimator_specs": estimator_specs, "package_versions": versions,
        "software": {"python": __import__('sys').version.split()[0], "pandas": pd.__version__,
                     "sklearn": __import__('sklearn').__version__},
        "limitations": [
            "Already investigated WELL cohort, not an untouched final test.",
            "Fixed hyperparameters chosen for exploratory reproducibility, not optimized.",
            "0.5 threshold across models is descriptive only: scores are not comparably calibrated.",
            "These are retrospectively eligible risk-set windows, not continuous field monitoring.",
            "No additional hard negatives; class2-only comparisons do not establish specificity on other failure classes.",
            "Only 11 positive episodes; correlated windows do not mean independent observations.",
        ],
    }
    (out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- Исследовательское сравнение: AP — основная метрика ---")
    print(pd.DataFrame(metrics).sort_values("ap", ascending=False).round(4).to_string(index=False))
    print(f"\nОтчёты: {out_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/class2_horizons/cohort_features.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/class2_model_benchmark"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    args = parser.parse_args()
    run(args.input, args.out, tuple(dict.fromkeys(args.models)))


if __name__ == "__main__":
    main()
