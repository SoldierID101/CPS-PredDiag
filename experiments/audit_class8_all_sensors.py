"""CPS-PredDiag: class-8 availability/variability of ALL 3W telemetry channels.

Reads one real Parquet file at a time. No model training, no modifications
of source data. Intended to choose the next sensor set BEFORE new ML tests.

Run: python experiments/audit_class8_all_sensors.py
Output:
  outputs/inventory/class8_all_sensors_by_file.csv
  outputs/inventory/class8_all_sensors_summary.csv
"""
from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "8"
OUT_DIR = PROJECT_ROOT / "outputs" / "inventory"
NON_SENSORS = {"class", "state", "timestamp"}


def well_from_filename(name: str) -> str:
    match = re.match(r"WELL-(\d+)_", name)
    return match.group(1) if match else "UNKNOWN"


def inspect_file(path: Path, sensors: list[str]) -> list[dict]:
    names_in_file = set(pq.read_schema(path).names)
    used = [s for s in sensors if s in names_in_file]
    df = pd.read_parquet(path, columns=used)
    rows = []
    for sensor in sensors:
        if sensor not in df.columns:
            present = 0
            distinct = 0
            status = "MISSING"
        else:
            series = pd.to_numeric(df[sensor], errors="coerce")
            present = int(series.notna().sum())
            distinct = int(series.nunique(dropna=True))
            if distinct == 0:
                status = "MISSING"
            elif distinct == 1:
                status = "CONSTANT"
            else:
                status = "VARIABLE"
        rows.append({
            "file": path.name,
            "well_id": well_from_filename(path.name),
            "sensor": sensor,
            "rows": len(df),
            "valid_rows": present,
            "availability_pct": round(100.0 * present / len(df), 3) if len(df) else 0.0,
            "unique_values": distinct,
            "status": status,
        })
    return rows


def main() -> None:
    files = sorted(DATASET_DIR.glob("WELL-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Не найдены реальные файлы класса 8: {DATASET_DIR}")

    # Union of observed Parquet schemas: sensors need not exist in every file.
    sensors = sorted({
        name
        for f in files
        for name in pq.read_schema(f).names
        if name not in NON_SENSORS and not name.startswith("__index_level_")
    })
    print(f"Файлов: {len(files)}, обнаружено имен датчиков: {len(sensors)}")

    records = []
    for idx, file_path in enumerate(files, start=1):
        records.extend(inspect_file(file_path, sensors))
        print(f"[{idx:2d}/{len(files)}] {file_path.name}")

    detail = pd.DataFrame(records)
    summary_rows = []
    for sensor, group in detail.groupby("sensor", sort=True):
        usable = group[(group["availability_pct"] >= 95.0) & (group["status"] == "VARIABLE")]
        summary_rows.append({
            "sensor": sensor,
            "files_variable": int((group["status"] == "VARIABLE").sum()),
            "files_constant": int((group["status"] == "CONSTANT").sum()),
            "files_missing": int((group["status"] == "MISSING").sum()),
            "files_variable_and_95pct_available": len(usable),
            "wells_variable_and_95pct_available": usable["well_id"].nunique(),
            "median_availability_pct": round(float(group["availability_pct"].median()), 2),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["files_variable_and_95pct_available", "wells_variable_and_95pct_available", "sensor"],
        ascending=[False, False, True],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "class8_all_sensors_by_file.csv"
    summary_path = OUT_DIR / "class8_all_sensors_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nДоступность и изменчивость по всем 14 реальным файлам:")
    print(summary.to_string(index=False))
    print("\nПо файлам:", detail_path)
    print("Сводный отчёт:", summary_path)
    print("ВАЖНО: VARIABLE не означает прогностическую полезность. Это описательная проверка полного файла.")


if __name__ == "__main__":
    main()
