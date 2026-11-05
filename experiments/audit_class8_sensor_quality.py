"""Аудит качества признаков Petrobras 3W, класс 8.

Запуск из корня CPS-PredDiag:
    python experiments/audit_class8_sensor_quality.py

Читает только готовые CSV. Не изменяет обучение и порог Random Forest.
"""
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs" / "features" / "class8_6h_ml_features.csv"
PREDICTIONS = ROOT / "outputs" / "models" / "class8_baseline" / "oof_predictions.csv"
OUTPUT = ROOT / "outputs" / "inventory" / "class8_sensor_quality_by_well.csv"
SENSORS = ("P-TPT", "P-MON-CKP", "T-TPT")


def main():
    if not FEATURES.exists():
        raise FileNotFoundError(f"Нет файла признаков: {FEATURES}")
    data = pd.read_csv(FEATURES)
    required = {"file", "well_id", "decision_time", "target"}
    for sensor in SENSORS:
        required.update([f"{sensor}_mean", f"{sensor}_std"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Не хватает столбцов: {sorted(missing)}")
    if data.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли точек прогнозирования")

    if PREDICTIONS.exists():
        preds = pd.read_csv(PREDICTIONS)
        needed = {"file", "decision_time", "target", "prob_rf_all_12"}
        if not needed.issubset(preds.columns):
            raise ValueError("В файле прогнозов нет обязательных колонок")
        if preds.duplicated(["file", "decision_time"]).any():
            raise ValueError("Дубли точек прогнозирования в прогнозах")
        data = data.merge(
            preds[["file", "decision_time", "target", "prob_rf_all_12"]],
            on=["file", "decision_time"], how="left", validate="one_to_one",
            suffixes=("", "_pred"),
        )
        if data["prob_rf_all_12"].isna().any() or not data["target"].eq(data["target_pred"]).all():
            raise ValueError("Прогнозы не полностью соответствуют признакам")
        data = data.drop(columns="target_pred")
    else:
        print("Прогнозы не найдены: считаем только качество датчиков.")

    rows = []
    for well_id, part in data.groupby("well_id", sort=True):
        for sensor in SENSORS:
            values = pd.to_numeric(part[f"{sensor}_mean"], errors="coerce")
            stds = pd.to_numeric(part[f"{sensor}_std"], errors="coerce")
            available = values.dropna()
            std_available = stds.dropna()
            uniq = int(available.round(6).nunique())
            row = {
                "well_id": well_id,
                "sensor": sensor,
                "windows": len(part),
                "availability_pct": round(100 * available.size / len(part), 2),
                "unique_window_means_6dp": uniq,
                "median_window_mean": float(available.median()) if not available.empty else np.nan,
                "std_zero_pct": round(100 * (std_available == 0).mean(), 2)
                if not std_available.empty else np.nan,
                "status": "MISSING" if available.empty else (
                    "CONSTANT_IN_WINDOWS" if uniq == 1 and (std_available == 0).all()
                    else "VARIABLE_OR_MIXED"
                ),
            }
            if "prob_rf_all_12" in part:
                predicted = part["prob_rf_all_12"] >= 0.5
                row["fp_windows_in_well"] = int((predicted & part["target"].eq(0)).sum())
                row["tp_windows_in_well"] = int((predicted & part["target"].eq(1)).sum())
            rows.append(row)

    report = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print("Проблемы доступности и неизменные сигналы (по готовым 15-минутным окнам):")
    subset = report[
        (report["availability_pct"] < 100)
        | (report["status"] == "CONSTANT_IN_WINDOWS")
    ]
    print(subset[["well_id", "sensor", "availability_pct", "median_window_mean", "status"]].to_string(index=False))
    if "fp_windows_in_well" in report:
        print("\nЛожноположительные окна по скважинам (rf_all_12):")
        print(report.drop_duplicates("well_id")[["well_id", "fp_windows_in_well", "tp_windows_in_well"]].to_string(index=False))
    print(f"\nСохранено: {OUTPUT}")
    print("CONSTANT_IN_WINDOWS означает постоянство по рассчитанным окнам, но не доказывает дефект датчика.")


if __name__ == "__main__":
    main()
