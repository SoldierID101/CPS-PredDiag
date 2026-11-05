"""Evaluate the *offline* 6-hour class-8 alert quality at the fixed 0.5 threshold.

Input files are outputs from the existing CPS-PredDiag scripts.
This script never fits a model or tunes its threshold on held-out predictions.
It uses the ground-truth event_time ONLY for offline evaluation.

Run from the project root:
    python experiments/evaluate_class8_alerts.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

PROJECT_DIR = Path(__file__).resolve().parents[1]
OOF_PATH = PROJECT_DIR / "outputs" / "models" / "class8_baseline" / "oof_predictions.csv"
HORIZON_PATH = PROJECT_DIR / "outputs" / "inventory" / "class8_horizon_audit.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "models" / "class8_baseline" / "event_alert_audit.csv"
MODEL_COLUMN = "prob_rf_all_12"
THRESHOLD = 0.5
HORIZON_HOURS = 6
STEP = timedelta(minutes=15)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_target(value: str) -> int:
    number = float(value)
    if number not in (0.0, 1.0):
        raise ValueError(f"Некорректная метка: {value!r}")
    return int(number)


def count_false_alert_series(rows: list[dict]) -> int:
    """Count distinct adjacent 15-min false-alert WINDOWS (not actual alarms)."""
    count = 0
    previous = None
    for row in rows:
        is_false = (row["target"] == 0 and row["alarm"])
        if is_false and (previous is None or row["time"] - previous != STEP):
            count += 1
        previous = row["time"] if is_false else None
    return count


def main() -> None:
    predictions = read_csv(OOF_PATH)
    horizons = read_csv(HORIZON_PATH)

    # The horizon file contains information about the future, for EVALUATION ONLY.
    event_by_point: dict[tuple[str, str], str] = {}
    truth_by_point: dict[tuple[str, str], int] = {}
    for row in horizons:
        if float(row["horizon_hours"]) != HORIZON_HOURS or not row["target"]:
            continue
        key = (row["file"], row["decision_time"])
        if key in truth_by_point:
            raise ValueError(f"Дублирующаяся точка аудита: {key}")
        truth_by_point[key] = parse_target(row["target"])
        event_by_point[key] = row["event_time"]

    per_file = defaultdict(list)
    for row in predictions:
        key = (row["file"], row["decision_time"])
        if key not in truth_by_point:
            raise ValueError(f"Не найдена точка в аудите горизонтов: {key}")
        target = parse_target(row["target"])
        if target != truth_by_point[key]:
            raise ValueError(f"Метки не совпадают: {key}")
        prob = float(row[MODEL_COLUMN])
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Некорректная вероятность для {key}: {prob}")
        per_file[row["file"]].append({
            "file": row["file"],
            "well_id": row["well_id"],
            "time": datetime.fromisoformat(row["decision_time"]),
            "target": target,
            "alarm": prob >= THRESHOLD,
            "event_time": event_by_point[key],
        })

    results = []
    leads = []
    for file, records in sorted(per_file.items()):
        records.sort(key=lambda item: item["time"])
        event_timestamps = {r["event_time"] for r in records if r["event_time"]}
        if len(event_timestamps) > 1:
            raise ValueError(f"Несколько event_time внутри {file}")
        event_time = (
            datetime.fromisoformat(next(iter(event_timestamps)))
            if event_timestamps else None
        )
        has_event = event_time is not None
        true_positive = [r for r in records if r["target"] == 1 and r["alarm"]]
        false_positive = [r for r in records if r["target"] == 0 and r["alarm"]]
        false_negative = [r for r in records if r["target"] == 1 and not r["alarm"]]
        first_lead_h = None
        if true_positive:
            if event_time is None:
                raise ValueError(f"Положительная метка без event_time: {file}")
            first_lead_h = (event_time - true_positive[0]["time"]).total_seconds() / 3600
            if not (0 < first_lead_h <= HORIZON_HOURS):
                raise ValueError(f"Подозрительная заблаговременность: {file}, {first_lead_h}")
            leads.append(first_lead_h)
        results.append({
            "file": file,
            "well_id": records[0]["well_id"],
            "has_recorded_event": int(has_event),
            "event_detected": int(bool(true_positive)),
            "tp_windows": len(true_positive),
            "fp_windows": len(false_positive),
            "fn_windows": len(false_negative),
            "false_alert_series": count_false_alert_series(records),
            "first_correct_warning_lead_h": (
                f"{first_lead_h:.3f}" if first_lead_h is not None else ""
            ),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    total_events = sum(r["has_recorded_event"] for r in results)
    detected_events = sum(r["event_detected"] for r in results)
    print(f"Модель: {MODEL_COLUMN}, порог: {THRESHOLD}")
    print(f"События с подтверждённым наступлением класса 8: {detected_events}/{total_events}")
    print(f"TP-окон: {sum(r['tp_windows'] for r in results)}")
    print(f"FP-окон: {sum(r['fp_windows'] for r in results)}")
    print(f"Серий FP-окон (условно, с шагом 15 мин): {sum(r['false_alert_series'] for r in results)}")
    if leads:
        print(f"Медиана заблаговременности первого верного сигнала: {median(leads):.2f} ч")
    print("\nПо файлам:")
    for r in results:
        print(
            f"{r['file']}: событие={r['has_recorded_event']}, "
            f"обнаружено={r['event_detected']}, FP={r['fp_windows']}, "
            f"серий_FP={r['false_alert_series']}, "
            f"lead_h={r['first_correct_warning_lead_h'] or '-'}"
        )
    print(f"\nОтчёт: {OUTPUT_PATH}")
    print("ВНИМАНИЕ: это офлайн-анализ размеченных окон класса 108, не метрика реальной системы тревог.")


if __name__ == "__main__":
    main()
