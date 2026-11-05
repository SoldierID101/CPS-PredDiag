"""Сравнение поведения датчиков перед началом переходного состояния и после него.
Первый исследовательский аудит для реальных эпизодов классов 2 и 8.
Не обучает модель и не использует будущие метки для вычисления признаков
в нормальных окнах. Границы состояний используются только для анализа.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inventory"
EVENT_CLASSES = (2, 8)
SENSORS = ("P-TPT", "P-MON-CKP", "T-TPT", "P-PDG")
WINDOW_SECONDS = 15 * 60
MIN_VALID_FRACTION = 0.90


def continuous_seconds(index: pd.DatetimeIndex) -> bool:
    if len(index) < 2:
        return False
    return bool((index.to_series().diff().iloc[1:] == pd.Timedelta(seconds=1)).all())


def process_file(path: Path, event_class: int) -> tuple[list[dict], str | None]:
    df = pd.read_parquet(path)
    if "class" not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return [], "нет class или DatetimeIndex"

    labels = pd.to_numeric(df["class"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    transition_class = 100 + event_class
    transient_positions = np.flatnonzero(labels == transition_class)
    event_positions = np.flatnonzero(labels == event_class)

    if not len(transient_positions) or not len(event_positions):
        return [], "нет перехода или завершённого события"

    onset = int(transient_positions[0])
    failure = int(event_positions[0])
    n = WINDOW_SECONDS

    if onset < 2 * n or failure - onset < n:
        return [], "недостаточно двух нормальных или одного переходного окна"

    # Требуется ровно два предшествующих нормальных окна и одно
    # начальное переходное окно; unknown-метки не заполняем.
    if not (
        np.all(labels[onset - 2 * n:onset] == 0)
        and np.all(labels[onset:onset + n] == transition_class)
        and np.all(labels[onset:failure] == transition_class)
    ):
        return [], "нарушен порядок 0 -> 10X -> X"

    times = df.index[onset - 2 * n:onset + n]
    if not continuous_seconds(times):
        return [], "нерегулярные временные метки в исследуемых окнах"

    baseline = df.iloc[onset - 2 * n:onset - n]
    pre = df.iloc[onset - n:onset]
    early = df.iloc[onset:onset + n]
    match = re.search(r"WELL-(\d+)_", path.name)
    well_id = match.group(1) if match else "UNKNOWN"

    rows = []
    for sensor in SENSORS:
        if sensor not in df.columns:
            continue

        parts = [pd.to_numeric(part[sensor], errors="coerce").dropna()
                 for part in (baseline, pre, early)]
        if any(len(part) < MIN_VALID_FRACTION * n for part in parts):
            continue

        med_baseline, med_pre, med_early = (float(part.median()) for part in parts)
        rows.append({
            "event_class": event_class,
            "well_id": well_id,
            "file": path.name,
            "sensor": sensor,
            "transient_start": str(df.index[onset]),
            "event_start": str(df.index[failure]),
            "transient_to_event_hours": round((df.index[failure] - df.index[onset]).total_seconds() / 3600, 3),
            "baseline_median": med_baseline,
            "pre_median": med_pre,
            "early_transient_median": med_early,
            "change_before_onset": med_pre - med_baseline,
            "change_after_onset": med_early - med_pre,
            "pre_std": float(parts[1].std()),
            "early_transient_std": float(parts[2].std()),
        })

    return rows, None if rows else "нет датчиков с достаточным покрытием"


def main() -> None:
    all_rows = []
    skipped = []

    for event_class in EVENT_CLASSES:
        files = sorted((DATASET_DIR / str(event_class)).glob("WELL-*.parquet"))
        print(f"\nКласс {event_class}: найдено {len(files)} реальных файлов")
        for path in files:
            try:
                rows, reason = process_file(path, event_class)
                all_rows.extend(rows)
                if reason:
                    skipped.append((event_class, path.name, reason))
            except Exception as exc:
                skipped.append((event_class, path.name, f"ошибка: {exc}"))

    if not all_rows:
        print("Нет пригодных окон. Причины пропуска:")
        for item in skipped:
            print(*item, sep=" | ")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(all_rows)
    output = OUTPUT_DIR / "pre_onset_sensor_audit.csv"
    report.to_csv(output, index=False, encoding="utf-8-sig")

    summary = (
        report.assign(
            abs_before=report["change_before_onset"].abs(),
            abs_after=report["change_after_onset"].abs(),
        )
        .groupby(["event_class", "sensor"])
        .agg(
            episodes=("file", "nunique"),
            wells=("well_id", "nunique"),
            median_abs_before=("abs_before", "median"),
            median_abs_after=("abs_after", "median"),
        )
    )

    print("\nСравнение модулей изменений медианы (единицы исходного датчика):")
    print(summary.round(3).to_string())
    print(f"\nПропущено файлов: {len(skipped)}")
    for item in skipped[:12]:
        print("  ", *item, sep=" | ")
    print("\nДетальный отчёт:", output)
    print("Важно: это описательная статистика, не оценка качества прогноза.")


if __name__ == "__main__":
    main()
