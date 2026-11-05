"""CPS-PredDiag: first ML feature table for the 3W class-8, 6-hour task.

Inputs:
  outputs/inventory/class8_horizon_audit.csv
  dataset/8/WELL-*.parquet

A feature row uses *only* 900 consecutive sensor samples ending at decision_time.
Labels are taken from the horizon audit; event times are NOT ML features.
No train/test split or model training is performed here.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parents[1]
HORIZON_CSV = PROJECT_DIR / "outputs" / "inventory" / "class8_horizon_audit.csv"
DATASET_DIR = PROJECT_DIR / "dataset" / "8"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "features"

HORIZON_HOURS = 6
WINDOW_SECONDS = 900  # 15 min, 1 Hz
SENSORS = ("P-TPT", "P-MON-CKP", "T-TPT")
SUFFIXES = ("mean", "std", "delta", "slope_per_hour")
FEATURE_COLUMNS = [f"{sensor}_{suffix}" for sensor in SENSORS for suffix in SUFFIXES]


def sensor_features(values: pd.Series) -> dict:
    """Use only observed values; if <90% observed, mark all features NaN."""
    names = ("mean", "std", "delta", "slope_per_hour")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    is_valid = np.isfinite(numeric)
    if is_valid.sum() < WINDOW_SECONDS * 0.90:
        return {name: np.nan for name in names}

    x = np.flatnonzero(is_valid).astype("float64")  # elapsed seconds since window start
    y = numeric[is_valid]
    slope_hour = float(np.polyfit(x, y, deg=1)[0] * 3600) if len(x) > 1 else np.nan
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        "delta": float(y[-1] - y[0]),
        "slope_per_hour": slope_hour,
    }


def process_file(filename: str, points: pd.DataFrame) -> tuple[list[dict], int]:
    path = DATASET_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    available = set(pq.read_schema(path).names)
    present_sensors = [name for name in SENSORS if name in available]
    if not present_sensors:
        raise ValueError(f"Нет нужных датчиков: {filename}")

    df = pd.read_parquet(path, columns=present_sensors)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Нет DatetimeIndex: {filename}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"Некорректный индекс времени: {filename}")

    rows = []
    skipped = 0
    for point in points.itertuples(index=False):
        decision_time = pd.Timestamp(point.decision_time)
        location = df.index.get_indexer([decision_time])[0]
        if location < WINDOW_SECONDS - 1:
            skipped += 1
            continue
        start_pos = location - WINDOW_SECONDS + 1
        window = df.iloc[start_pos:location + 1]
        expected_start = decision_time - pd.Timedelta(seconds=WINDOW_SECONDS - 1)

        if window.index[0] != expected_start or not (np.diff(window.index.asi8) == 1_000_000_000).all():
            skipped += 1
            continue
        # A predictive window must be entirely within the already known transient period.
        if window.index[0] < pd.Timestamp(point.transient_start):
            skipped += 1
            continue

        result = {
            "file": filename,              # grouping/traceability only
            "well_id": str(point.well_id),  # grouping only (not an ML feature)
            "decision_time": decision_time, # metadata only (not an ML feature)
            "target": int(point.target),
        }
        for sensor in SENSORS:
            statistics = (
                sensor_features(window[sensor]) if sensor in window.columns
                else {suffix: np.nan for suffix in SUFFIXES}
            )
            for suffix, value in statistics.items():
                result[f"{sensor}_{suffix}"] = value
        rows.append(result)
    return rows, skipped


def main() -> None:
    if not HORIZON_CSV.is_file():
        raise FileNotFoundError(f"Нет отчёта горизонтов: {HORIZON_CSV}")

    labels = pd.read_csv(HORIZON_CSV, dtype={"well_id": str}, parse_dates=["decision_time", "transient_start"])
    labels = labels.loc[(labels["horizon_hours"] == HORIZON_HOURS) & labels["target"].isin([0, 1])].copy()
    if labels.empty:
        raise ValueError("Нет размеченных точек для выбранного горизонта")
    if labels.duplicated(["file", "decision_time"]).any():
        raise ValueError("Повторяющиеся точки прогнозирования в исходной таблице")

    results: list[dict] = []
    skipped = 0
    for filename, points in labels.groupby("file", sort=True):
        file_rows, file_skipped = process_file(filename, points)
        results.extend(file_rows)
        skipped += file_skipped
        print(f"{filename}: рассчитано {len(file_rows)} окон; пропущено {file_skipped}")

    if not results:
        raise ValueError("Не удалось сформировать ни одного окна")

    data = pd.DataFrame(results)
    data = data[["file", "well_id", "decision_time", "target", *FEATURE_COLUMNS]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "class8_6h_ml_features.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nПроверка подготовленной выборки:")
    print("Строк:", len(data))
    print("Признаков:", len(FEATURE_COLUMNS))
    print("Скважин:", data["well_id"].nunique())
    print("Файлов:", data["file"].nunique())
    print("target=0:", int((data["target"] == 0).sum()))
    print("target=1:", int((data["target"] == 1).sum()))
    print("Пропущено окон из-за некорректной истории:", skipped)
    print("\nДоля пропусков в признаках (%):")
    print((data[FEATURE_COLUMNS].isna().mean() * 100).round(2).to_string())
    print("\nЗаписано:", output_path)
    print("ВНИМАНИЕ: file, well_id и decision_time — метаданные, а не признаки ML.")


if __name__ == "__main__":
    main()
