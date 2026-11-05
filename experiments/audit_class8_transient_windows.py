"""CPS-PredDiag — доступность и динамика новых датчиков до события класса 8.

Исследуется только 15-минутная ИСТОРИЯ перед точками прогнозирования,
расположенными внутри размеченного переходного состояния 108.

Входы:
    dataset/8/WELL-*.parquet
    outputs/inventory/class8_horizon_audit.csv

Выходы:
    outputs/inventory/class8_transient_sensor_windows.csv
    outputs/inventory/class8_transient_sensor_summary.csv

Не обучает модель и не меняет исходный baseline. Метки target используются
ТОЛЬКО для описательных группировок, а не для вычисления признаков.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "8"
LABELS_PATH = ROOT / "outputs" / "inventory" / "class8_horizon_audit.csv"
OUT_DIR = ROOT / "outputs" / "inventory"

SENSORS = ("P-MON-CKP", "QGL", "P-JUS-CKGL", "T-MON-CKP")
HORIZON_HOURS = 6
WINDOW_SIZE = 900     # 15 минут при 1 Гц
EXPECTED_STEP_NS = 1_000_000_000
MIN_VALID_FRACTION = 0.90


def window_stats(values: pd.Series) -> dict:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    finite = numbers[np.isfinite(numbers)]
    n_valid = len(finite)
    usable = n_valid >= WINDOW_SIZE * MIN_VALID_FRACTION
    if not usable:
        return {
            "valid_pct": 100 * n_valid / WINDOW_SIZE,
            "usable": False,
            "varied": False,
            "std": np.nan,
            "abs_delta": np.nan,
        }
    return {
        "valid_pct": 100 * n_valid / WINDOW_SIZE,
        "usable": True,
        "varied": bool(np.max(finite) > np.min(finite)),
        "std": float(np.std(finite, ddof=1)),
        "abs_delta": float(abs(finite[-1] - finite[0])),
    }


def analyze_file(filename: str, points: pd.DataFrame) -> list[dict]:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)

    columns = set(pq.read_schema(path).names)
    selected = [sensor for sensor in SENSORS if sensor in columns]
    if not selected:
        raise ValueError(f"Нет ни одного выбранного датчика: {filename}")

    df = pd.read_parquet(path, columns=selected)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Нет DatetimeIndex: {filename}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"Некорректный временной индекс: {filename}")

    results = []
    for point in points.itertuples(index=False):
        t = pd.Timestamp(point.decision_time)
        start_of_transient = pd.Timestamp(point.transient_start)
        end_idx = df.index.get_indexer([t])[0]
        if end_idx < WINDOW_SIZE - 1:
            continue

        window = df.iloc[end_idx - WINDOW_SIZE + 1:end_idx + 1]
        if window.index[0] < start_of_transient:
            continue
        if window.index[0] != t - pd.Timedelta(seconds=WINDOW_SIZE - 1):
            continue
        if not (np.diff(window.index.asi8) == EXPECTED_STEP_NS).all():
            continue

        for sensor in SENSORS:
            if sensor in window.columns:
                stats = window_stats(window[sensor])
            else:
                stats = {
                    "valid_pct": 0.0, "usable": False, "varied": False,
                    "std": np.nan, "abs_delta": np.nan,
                }
            results.append({
                "file": filename,
                "well_id": str(point.well_id),
                "decision_time": t,
                "target": int(point.target),
                "sensor": sensor,
                **stats,
            })
    return results


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Сначала нужен отчёт горизонтов: {LABELS_PATH}")

    labels = pd.read_csv(
        LABELS_PATH, dtype={"well_id": str},
        parse_dates=["decision_time", "transient_start"]
    )
    labels = labels.loc[
        (labels["horizon_hours"] == HORIZON_HOURS)
        & labels["target"].isin([0, 1])
    ].copy()
    if labels.empty:
        raise ValueError("Нет размеченных точек для горизонта 6 часов")
    if labels.duplicated(["file", "decision_time"]).any():
        raise ValueError("Повторяющиеся точки прогнозирования")

    all_rows = []
    for filename, points in labels.groupby("file", sort=True):
        rows = analyze_file(filename, points)
        all_rows.extend(rows)
        print(f"{filename}: {len(rows) // len(SENSORS)} окон")

    if not all_rows:
        raise ValueError("Не найдено подходящих временных окон")

    details = pd.DataFrame(all_rows)
    expected = len(labels) * len(SENSORS)
    if len(details) != expected:
        print(f"ВНИМАНИЕ: получено {len(details)} записей, ожидалось {expected}. "
              "Некоторые окна могли быть пропущены из-за разрывов времени.")

    summary_rows = []
    for sensor, group in details.groupby("sensor", sort=False):
        usable = group.loc[group["usable"]]
        row = {
            "sensor": sensor,
            "windows": len(group),
            "wells_with_usable_windows": usable["well_id"].nunique(),
            "usable_pct": round(100 * len(usable) / len(group), 2),
            "varied_among_usable_pct": (
                round(100 * usable["varied"].mean(), 2) if len(usable) else np.nan
            ),
            "median_window_std": usable["std"].median(),
            "median_window_abs_delta": usable["abs_delta"].median(),
        }
        for target in (0, 1):
            subset = usable.loc[usable["target"] == target]
            row[f"usable_target_{target}"] = len(subset)
            row[f"median_std_target_{target}"] = subset["std"].median()
            row[f"median_abs_delta_target_{target}"] = subset["abs_delta"].median()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "class8_transient_sensor_windows.csv"
    summary_path = OUT_DIR / "class8_transient_sensor_summary.csv"
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nИзменчивость в 15-минутных окнах состояния 108:")
    cols = ["sensor", "windows", "wells_with_usable_windows", "usable_pct",
            "varied_among_usable_pct", "median_window_std", "median_window_abs_delta"]
    print(summary[cols].round(4).to_string(index=False))
    print("\nДетально:", detail_path)
    print("Сводка:", summary_path)
    print("Медианы по target — описательные; НЕ измеряют качество прогноза.")
    print("Для модели можно использовать лишь историю до decision_time;")
    print("target и идентификаторы файлов/скважин являются метаданными.")


if __name__ == "__main__":
    main()
