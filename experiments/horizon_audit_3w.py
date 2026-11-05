"""CPS-PredDiag: audit of forecast horizons for Petrobras 3W class 8.
Counts potential decision points in the TRANSIENT state (108). It does not
train a model, extract sensor features, or measure predictive performance.
A target of 1 means a recorded class-8 onset occurs within the next H hours.
A target of 0 means the next H hours are observed without class 8.
An unknown target means the recording ends/changes state before H can be checked.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset" / "8"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inventory"

NORMAL_CLASS = 0
TRANSIENT_CLASS = 108
EVENT_CLASS = 8
WINDOW_MINUTES = 15
STEP_MINUTES = 15
HORIZONS_HOURS = (3, 6, 12)
NANOSECOND = 1_000_000_000


def audit_file(path: Path) -> list[dict]:
    df = pd.read_parquet(path, columns=["class"])
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return []
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        print(f"Пропущен файл с некорректным временем: {path.name}")
        return []

    labels = pd.to_numeric(df["class"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    times = df.index.asi8
    boundaries = np.r_[0, np.flatnonzero(
        (labels[1:] != labels[:-1]) | (np.diff(times) != NANOSECOND)
    ) + 1, len(labels)]
    runs = [(int(start), int(end), int(labels[start])) for start, end in zip(boundaries[:-1], boundaries[1:])]

    well_match = re.match(r"WELL-(\d+)_", path.name)
    well_id = well_match.group(1) if well_match else "unknown"
    rows = []
    window_points = WINDOW_MINUTES * 60
    step_points = STEP_MINUTES * 60

    for run_i, (start, end, state) in enumerate(runs):
        if state != TRANSIENT_CLASS or run_i == 0:
            continue
        # Only transient episodes that actually follow a contiguous normal run.
        if runs[run_i - 1][2] != NORMAL_CLASS:
            continue
        preceding_end = runs[run_i - 1][1]
        if preceding_end != start or times[start] - times[start - 1] != NANOSECOND:
            continue

        event_time_ns = None
        if run_i + 1 < len(runs) and runs[run_i + 1][2] == EVENT_CLASS:
            next_start = runs[run_i + 1][0]
            if next_start == end and times[next_start] - times[end - 1] == NANOSECOND:
                event_time_ns = int(times[next_start])

        # All history points for a prospective decision are from state 108.
        for end_idx in range(start + window_points - 1, end, step_points):
            t_ns = int(times[end_idx])
            t = df.index[end_idx]
            for horizon in HORIZONS_HOURS:
                future_ns = t_ns + int(horizon * 3600) * NANOSECOND
                if event_time_ns is not None:
                    target = 1 if event_time_ns <= future_ns else 0
                elif int(times[end - 1]) >= future_ns:
                    target = 0
                else:
                    target = None  # Censored: future beyond observed state.
                rows.append({
                    "file": path.name,
                    "well_id": well_id,
                    "decision_time": t,
                    "horizon_hours": horizon,
                    "target": target,
                    "event_time": pd.Timestamp(event_time_ns) if event_time_ns is not None else pd.NaT,
                    "transient_start": df.index[start],
                })
    return rows


def main() -> None:
    files = sorted(DATA_DIR.glob("WELL-*.parquet"))
    print(f"Реальных файлов класса 8: {len(files)}")
    all_rows = []
    for path in files:
        try:
            file_rows = audit_file(path)
            all_rows.extend(file_rows)
            print(f"  {path.name}: {len(file_rows)} точек (для всех горизонтов)")
        except Exception as exc:
            print(f"  Ошибка: {path.name} — {exc}")

    if not all_rows:
        print("Подходящие наблюдения не найдены.")
        return

    result = pd.DataFrame(all_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "class8_horizon_audit.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")

    print("\nПотенциальные точки прогнозирования (НЕ независимые события):")
    for horizon in HORIZONS_HOURS:
        sub = result[result["horizon_hours"] == horizon]
        pos = sub[sub["target"] == 1]
        neg = sub[sub["target"] == 0]
        unknown = sub[sub["target"].isna()]
        print(f"Горизонт {horizon:2d} ч: положительных={len(pos):5d}, "
              f"отрицательных={len(neg):5d}, неизвестных={len(unknown):5d}; "
              f"скважин с положительными={pos['well_id'].nunique()}, "
              f"файлов с положительными={pos['file'].nunique()}")
    print(f"\nПодробная таблица: {output}")
    print("Важно: target=0 значит только отсутствие события в наблюдаемые H часов; "
          "не значит, что событие никогда не произойдёт.")


if __name__ == "__main__":
    main()
