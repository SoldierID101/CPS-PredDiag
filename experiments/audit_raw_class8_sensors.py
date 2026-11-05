"""Проверка исходных сигналов Petrobras 3W класса 8 без обучения ML.

Запуск из корня CPS-PredDiag:
    python experiments/audit_raw_class8_sensors.py

Выходные файлы:
    outputs/inventory/class8_raw_sensor_by_file.csv
    outputs/inventory/class8_raw_sensor_by_state.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset" / "8"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inventory"
SENSORS = ("P-TPT", "P-MON-CKP", "T-TPT")


def describe_signal(values: np.ndarray) -> dict:
    """Сводка исходных значений; NaN и бесконечности считаются недоступными."""
    valid = values[np.isfinite(values)]
    total = len(values)
    count = len(valid)

    if count == 0:
        return {
            "rows": total, "valid_rows": 0, "availability_pct": 0.0,
            "unique_values": 0, "min": np.nan, "median": np.nan,
            "max": np.nan, "std": np.nan, "status": "MISSING"
        }

    unique_count = int(np.unique(valid).size)
    return {
        "rows": total,
        "valid_rows": count,
        "availability_pct": round(100 * count / total, 2),
        "unique_values": unique_count,
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "status": "CONSTANT" if unique_count == 1 else "VARIABLE",
    }


def inspect_file(path: Path):
    match = re.match(r"WELL-(\d+)_", path.name)
    if not match:
        return [], []

    well_id = match.group(1)
    schema_columns = set(pq.read_schema(path).names)
    if "class" not in schema_columns:
        raise ValueError(f"Нет столбца class: {path}")

    present_sensors = [name for name in SENSORS if name in schema_columns]
    frame = pd.read_parquet(path, columns=["class", *present_sensors])
    labels = pd.to_numeric(frame["class"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    n = len(frame)

    state_masks = {
        "NORMAL_0": labels == 0,
        "TRANSIENT_108": labels == 108,
        "EVENT_8": labels == 8,
        "UNLABELED": ~np.isfinite(labels),
        "OTHER_LABEL": (np.isfinite(labels)
                        & ~np.isin(labels, [0, 108, 8])),
    }

    file_rows = []
    state_rows = []
    for sensor in SENSORS:
        present = sensor in present_sensors
        values = (
            pd.to_numeric(frame[sensor], errors="coerce")
            .to_numpy(dtype="float64", na_value=np.nan)
            if present else np.full(n, np.nan, dtype=np.float64)
        )
        common = {
            "well_id": well_id,
            "file": path.name,
            "sensor": sensor,
            "column_present": present,
        }
        file_rows.append({**common, **describe_signal(values)})

        for state, mask in state_masks.items():
            if not mask.any():
                continue
            state_rows.append({
                **common,
                "state": state,
                **describe_signal(values[mask]),
            })

    return file_rows, state_rows


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с данными: {DATA_DIR}")

    paths = sorted(DATA_DIR.rglob("WELL-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"В {DATA_DIR} нет реальных WELL-файлов")

    file_rows = []
    state_rows = []
    errors = []

    for path in paths:
        try:
            by_file, by_state = inspect_file(path)
            file_rows.extend(by_file)
            state_rows.extend(by_state)
            print(f"Проверен: {path.name}")
        except Exception as exc:
            errors.append((path.name, str(exc)))
            print(f"ОШИБКА: {path.name} — {exc}")

    if not file_rows:
        raise RuntimeError("Не удалось прочитать ни одного файла")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_report = OUTPUT_DIR / "class8_raw_sensor_by_file.csv"
    state_report = OUTPUT_DIR / "class8_raw_sensor_by_state.csv"
    by_file_df = pd.DataFrame(file_rows)
    by_state_df = pd.DataFrame(state_rows)
    by_file_df.to_csv(file_report, index=False, encoding="utf-8-sig")
    by_state_df.to_csv(state_report, index=False, encoding="utf-8-sig")

    suspicious = by_file_df[
        by_file_df["status"].isin(["MISSING", "CONSTANT"])
        | ((by_file_df["sensor"] == "P-TPT") & (by_file_df["min"] < 0))
    ]

    print("\nПотенциально проблемные сигналы в исходных Parquet:")
    if suspicious.empty:
        print("Не найдены по выбранным критериям")
    else:
        columns = [
            "well_id", "file", "sensor", "availability_pct",
            "unique_values", "min", "max", "status"
        ]
        print(suspicious[columns].to_string(index=False))

    print(f"\nФайлов проверено: {len(paths)}; ошибок: {len(errors)}")
    print(f"По файлам: {file_report}")
    print(f"По состояниям: {state_report}")
    print("CONSTANT — неизменное значение в исходном файле, не обязательно дефект.")
    if errors:
        raise RuntimeError("Часть файлов не прочитана; см. сообщения выше")


if __name__ == "__main__":
    main()
