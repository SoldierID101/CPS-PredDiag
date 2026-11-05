"""CPS-PredDiag: exploratory feature-expansion experiment for 3W class 8.

Requires the outputs of:
    experiments/build_class8_features.py
    experiments/audit_class8_transient_windows.py
    experiments/train_class8_baseline.py

Run from the project root:
    python experiments/train_class8_expanded_features.py

Only features from the 15 minutes ending at decision_time are used.
All metrics are exploratory because these wells have already informed feature selection.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from train_class8_baseline import ALL_FEATURES, build_estimator, score_binary # Reuse the existing baseline's SAME RF configuration and scoring.


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "features" / "class8_6h_ml_features.csv"
AUDIT_PATH = ROOT / "outputs" / "inventory" / "class8_transient_sensor_windows.csv"
OUT_DIR = ROOT / "outputs" / "models" / "class8_feature_expansion"

NEW_SENSORS = ("QGL", "P-JUS-CKGL", "T-MON-CKP")
NEW_STATS = ("std", "abs_delta")
META = ["file", "well_id", "decision_time", "target"]


def load_joined_data():
    baseline = pd.read_csv(BASE_PATH, dtype={"well_id": str})
    audit = pd.read_csv(AUDIT_PATH, dtype={"well_id": str})

    expected_base = set(META + ALL_FEATURES)
    expected_audit = set(META + ["sensor", "usable", "std", "abs_delta"])
    if missing := expected_base - set(baseline.columns):
        raise ValueError(f"В базовой таблице нет столбцов: {sorted(missing)}")
    if missing := expected_audit - set(audit.columns):
        raise ValueError(f"В аудите окон нет столбцов: {sorted(missing)}")

    if baseline.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли окон в базовой таблице")
    if audit.duplicated(["file", "decision_time", "sensor"]).any():
        raise ValueError("Дубли датчиков для точки прогнозирования")
    if baseline["target"].isna().any() or not set(baseline["target"].unique()) <= {0, 1}:
        raise ValueError("Некорректные целевые метки baseline")

    selected = audit.loc[audit["sensor"].isin(NEW_SENSORS)].copy()
    expected_rows = len(baseline) * len(NEW_SENSORS)
    if len(selected) != expected_rows:
        raise ValueError(
            f"Ожидается {expected_rows} записей по новым датчикам, получено {len(selected)}"
        )
    # Check that the source tables describe exactly the same decision points and labels.
    audit_meta = selected[META].drop_duplicates()
    if audit_meta.duplicated(["file", "decision_time"]).any():
        raise ValueError("Противоречивые метаданные аудита")
    validation = baseline[META].merge(
        audit_meta,
        on=["file", "decision_time"],
        how="outer", validate="one_to_one", indicator=True,
        suffixes=("_base", "_audit"),
    )
    if not validation["_merge"].eq("both").all():
        raise ValueError("Точки прогнозирования в двух CSV не совпадают")
    for column in ("well_id", "target"):
        if not validation[f"{column}_base"].eq(validation[f"{column}_audit"]).all():
            raise ValueError(f"Не совпадает {column} между двумя CSV")

    # For unusable windows, do not treat even a numeric value as valid.
    selected.loc[~selected["usable"].astype(bool), list(NEW_STATS)] = np.nan
    long = selected.melt(
        id_vars=["file", "decision_time", "sensor"],
        value_vars=list(NEW_STATS),
        var_name="stat", value_name="value",
    )
    long["feature"] = long["sensor"] + "_" + long["stat"]
    wide = long.pivot(
        index=["file", "decision_time"], columns="feature", values="value"
    ).reset_index()
    wide.columns.name = None

    joined = baseline.merge(wide, on=["file", "decision_time"],
                            how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Потеряны окна при соединении признаков")
    joined = joined.drop(columns="_merge")
    return joined


def main():
    joined = load_joined_data()
    qgl = [f"QGL_{stat}" for stat in NEW_STATS]
    extra = [f"{sensor}_{stat}" for sensor in NEW_SENSORS[1:] for stat in NEW_STATS]
    all_new = qgl + extra

    experiments = {
        "rf_old12": ALL_FEATURES,
        "rf_old12_plus_qgl2": ALL_FEATURES + qgl,
        "rf_old12_plus_other4": ALL_FEATURES + extra,
        "rf_old12_plus_all6": ALL_FEATURES + all_new,
    }

    y = joined["target"].to_numpy(dtype=int)
    groups = joined["well_id"].to_numpy()
    folds = list(LeaveOneGroupOut().split(joined, y, groups=groups))
    if len(folds) != joined["well_id"].nunique():
        raise RuntimeError("Некорректное групповое разбиение")

    scores, file_scores, well_scores = [], [], []
    predictions = joined[META].copy()
    for name, columns in experiments.items():
        X = joined[columns].apply(pd.to_numeric, errors="raise")
        X = X.replace([np.inf, -np.inf], np.nan)
        probabilities = np.full(len(joined), np.nan, dtype=float)

        for train_i, test_i in folds:
            model = build_estimator(name)
            model.fit(X.iloc[train_i], y[train_i])
            probabilities[test_i] = model.predict_proba(X.iloc[test_i])[:, 1]

        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Не все прогнозы получены: {name}")
        predictions[f"prob_{name}"] = probabilities
        scores.append({"model": name, "n_features": len(columns),
                       **score_binary(y, probabilities)})

        for well, rows in joined.groupby("well_id", sort=True):
            idx = rows.index.to_numpy()
            well_scores.append({"model": name, "well_id": well,
                                **score_binary(y[idx], probabilities[idx])})

        for file, rows in joined.groupby("file", sort=True):
            idx = rows.index.to_numpy()
            flagged = probabilities[idx] >= 0.5
            positive = y[idx] == 1
            file_scores.append({
                "model": name, "file": file, "well_id": rows["well_id"].iloc[0],
                "has_event": bool(positive.any()),
                "detected_event": bool((flagged & positive).any()),
                "false_alarm_windows": int((flagged & ~positive).sum()),
            })
        print(f"Готово: {name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(scores)
    by_file = pd.DataFrame(file_scores)
    by_well = pd.DataFrame(well_scores)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    by_well.to_csv(OUT_DIR / "by_well.csv", index=False)
    by_file.to_csv(OUT_DIR / "by_file.csv", index=False)
    predictions.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    print("\nЭксперимент: Leave-One-Well-Out, неизменные настройки Random Forest")
    print("Каждая модель проверяется на тех же 4006 окнах и 9 скважинах.")
    print(summary[["model", "n_features", "pr_auc_ap", "roc_auc",
                   "precision_0.5", "recall_0.5", "f1_0.5",
                   "false_alarms_windows"]].round(3).to_string(index=False))
    print("\nОбнаруженные эпизоды (из 12):")
    print(by_file[by_file["has_event"]].groupby("model").agg(
        detected=("detected_event", "sum"),
        total=("has_event", "size")
    ).to_string())
    print(f"\nРезультаты: {OUT_DIR}")
    print("Это исследовательское сравнение на уже изученных скважинах,"
          " НЕ независимое подтверждение улучшения.")


if __name__ == "__main__":
    main()
