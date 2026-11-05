"""CPS-PredDiag: conditional time-since-transition baseline for 3W class 8.

Runs a Leave-One-Well-Out comparison of the existing Random Forest out-of-fold
predictions and a simple logistic regression using elapsed time since the
annotated start of state 108. The start time is an ORACLE from the dataset:
this is an *offline control*, not an operational prediction pipeline.

Run from project root: python experiments/compare_class8_time_baseline.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HORIZON = ROOT / "outputs" / "inventory" / "class8_horizon_audit.csv"
RF_PRED = ROOT / "outputs" / "models" / "class8_feature_expansion" / "oof_predictions.csv"
OUT = ROOT / "outputs" / "models" / "class8_time_control"


def load_data():
    audit = pd.read_csv(HORIZON, dtype={"well_id": "string"})
    rf = pd.read_csv(RF_PRED, dtype={"well_id": "string"})
    required_a = {"file", "well_id", "decision_time", "horizon_hours", "target", "transient_start"}
    required_r = {"file", "well_id", "decision_time", "target", "prob_rf_old12"}
    if required_a - set(audit.columns) or required_r - set(rf.columns):
        raise ValueError("В исходных таблицах не хватает обязательных столбцов")

    audit = audit[(audit["horizon_hours"] == 6) & audit["target"].notna()].copy()
    audit["decision_time"] = pd.to_datetime(audit["decision_time"], errors="raise")
    audit["transient_start"] = pd.to_datetime(audit["transient_start"], errors="raise")
    rf["decision_time"] = pd.to_datetime(rf["decision_time"], errors="raise")

    if audit.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли в аудите горизонта")
    if rf.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли в прогнозах RF")
    # Do not include 'event_time': this is a future outcome, not an input.
    d = rf[["file", "well_id", "decision_time", "target", "prob_rf_old12"]].merge(
        audit[["file", "well_id", "decision_time", "target", "transient_start"]],
        on=["file", "decision_time"], how="outer", validate="one_to_one",
        suffixes=("_rf", "_audit"), indicator=True,
    )
    if not d["_merge"].eq("both").all():
        raise ValueError("Несовпадение точек прогнозирования")
    if not d["well_id_rf"].eq(d["well_id_audit"]).all():
        raise ValueError("Несовпадение идентификаторов скважин")
    if not d["target_rf"].eq(d["target_audit"]).all():
        raise ValueError("Несовпадение целевых меток")

    d["well_id"] = d.pop("well_id_rf")
    d["target"] = d.pop("target_rf").astype(int)
    d["elapsed_hours"] = (d["decision_time"] - d["transient_start"]).dt.total_seconds() / 3600
    if not np.isfinite(d["elapsed_hours"]).all() or (d["elapsed_hours"] < 0).any():
        raise ValueError("Некорректная длительность переходного состояния")
    if len(d) != 4006:
        raise ValueError(f"Ожидалось 4006 строк, найдено {len(d)}")
    return d.sort_values(["file", "decision_time"]).reset_index(drop=True)


def scores(y, p):
    pred = p >= 0.5
    return {
        "pr_auc_ap": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision_0.5": precision_score(y, pred, zero_division=0),
        "recall_0.5": recall_score(y, pred, zero_division=0),
        "f1_0.5": f1_score(y, pred, zero_division=0),
        "false_alarm_windows": int(((pred == 1) & (y == 0)).sum()),
        "true_positive_windows": int(((pred == 1) & (y == 1)).sum()),
    }


def main():
    d = load_data()
    x = np.log1p(d[["elapsed_hours"]])
    y = d["target"].to_numpy(dtype=int)
    groups = d["well_id"].to_numpy()
    prob = np.full(len(d), np.nan)
    splits = list(LeaveOneGroupOut().split(x, y, groups))
    for train, test in splits:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
            )),
        ])
        model.fit(x.iloc[train], y[train])
        prob[test] = model.predict_proba(x.iloc[test])[:, 1]
    if not np.isfinite(prob).all():
        raise ValueError("Некорректные прогнозы временной модели")

    d["prob_elapsed_only"] = prob
    summary = pd.DataFrame([
        {"model": "rf_old12", **scores(y, d["prob_rf_old12"].to_numpy())},
        {"model": "logreg_elapsed_only", **scores(y, prob)},
    ])
    detections = []
    for model, col in [("rf_old12", "prob_rf_old12"), ("logreg_elapsed_only", "prob_elapsed_only")]:
        detected = sum(
            bool(((g[col] >= 0.5) & (g["target"] == 1)).any())
            for _, g in d.groupby("file") if (g["target"] == 1).any()
        )
        detections.append((model, detected))
    summary["detected_events"] = summary["model"].map(dict(detections))

    by_well = []
    for well, g in d.groupby("well_id"):
        for model, col in [("rf_old12", "prob_rf_old12"), ("logreg_elapsed_only", "prob_elapsed_only")]:
            by_well.append({"well_id": well, "model": model,
                            "samples": len(g), "positives": int(g["target"].sum()),
                            "ap": average_precision_score(g["target"], g[col])})

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(by_well).to_csv(OUT / "by_well.csv", index=False)
    d[["file", "well_id", "decision_time", "target", "elapsed_hours", "prob_rf_old12", "prob_elapsed_only"]].to_csv(
        OUT / "oof_predictions.csv", index=False,
    )
    print("Контроль: время с начала 108, Leave-One-Well-Out, 9 скважин")
    print(summary.round(3).to_string(index=False))
    print("\nОтчёты:", OUT)
    print("ВАЖНО: начало 108 взято из офлайн-разметки (oracle); при эксплуатации оно должно обнаруживаться отдельно.")
    print("Сравнение исследовательское: эти скважины уже использовались для анализа моделей.")


if __name__ == "__main__":
    main()
