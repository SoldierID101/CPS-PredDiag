"""CPS-PredDiag — аудит завершённых и правоцензурированных эпизодов класса 8.

Источники (уже созданные скриптами проекта):
    outputs/inventory/real_label_audit.csv
    outputs/inventory/class8_horizon_audit.csv

Результат:
    outputs/inventory/class8_episode_structure.csv

Ничего не обучает, не читает Parquet, не меняет исходные файлы.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "outputs" / "inventory"
LABELS_PATH = INVENTORY / "real_label_audit.csv"
HORIZONS_PATH = INVENTORY / "class8_horizon_audit.csv"
OUTPUT_PATH = INVENTORY / "class8_episode_structure.csv"
HORIZON_HOURS = 6


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Отсутствует файл: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def basename(value):
    """Нормализуем и прямые, и обратные разделители путей из CSV."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def read_target(value):
    if value is None or str(value).strip() == "":
        return "unknown"
    number = float(value)
    if number == 0.0:
        return "negative"
    if number == 1.0:
        return "positive"
    raise ValueError(f"Неизвестная целевая метка: {value!r}")


def main():
    label_rows = [row for row in read_csv(LABELS_PATH)
                  if int(row["event_class"]) == 8]
    horizon_rows = [row for row in read_csv(HORIZONS_PATH)
                    if float(row["horizon_hours"]) == HORIZON_HOURS]
    if not label_rows or not horizon_rows:
        raise ValueError("Не обнаружены эпизоды класса 8 для горизонта 6 часов")

    labels_by_file = {}
    for row in label_rows:
        name = basename(row["file"])
        if name in labels_by_file:
            raise ValueError(f"Дублирование файла в аудите меток: {name}")
        labels_by_file[name] = row

    points_by_file = defaultdict(list)
    for row in horizon_rows:
        points_by_file[basename(row["file"])].append(row)

    if set(labels_by_file) != set(points_by_file):
        print("ВНИМАНИЕ: файлы двух аудитов не совпадают")
        print("  Только в аудите меток:", sorted(set(labels_by_file) - set(points_by_file)))
        print("  Только в аудите горизонтов:", sorted(set(points_by_file) - set(labels_by_file)))

    episodes = []
    for file, points in sorted(points_by_file.items()):
        if file not in labels_by_file:
            continue
        lab = labels_by_file[file]
        starts = {point["transient_start"] for point in points}
        if len(starts) != 1:
            raise ValueError(f"Файл {file} содержит несколько переходных эпизодов; "
                             "их нужно рассматривать отдельно")
        counts = Counter(read_target(point["target"]) for point in points)
        event_times = {point["event_time"] for point in points if point["event_time"]}
        if len(event_times) > 1:
            raise ValueError(f"В файле {file} обнаружено несколько моментов события")
        event_observed = bool(event_times)
        if event_observed != (int(lab["event_rows"]) > 0):
            raise ValueError(f"Несогласованность меток события в файле {file}")
        if event_observed and counts["unknown"]:
            raise ValueError(f"Неожиданные неизвестные цели в завершённом файле {file}")
        if not event_observed and counts["positive"]:
            raise ValueError(f"Положительные цели в незавершённом файле {file}")
        episodes.append({
            "file": file,
            "well_id": str(lab["well_id"]).zfill(5),
            "event_observed": int(event_observed),
            "transient_hours": round(int(lab["transient_rows"]) / 3600, 3),
            "normal_hours": round(int(lab["normal_rows"]) / 3600, 3),
            "windows_total": len(points),
            "positive_windows": counts["positive"],
            "negative_windows": counts["negative"],
            "unknown_windows": counts["unknown"],
            "transient_start": next(iter(starts)),
            "event_time": next(iter(event_times), ""),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)

    complete = [e for e in episodes if e["event_observed"]]
    censored = [e for e in episodes if not e["event_observed"]]
    print(f"Горизонт: {HORIZON_HOURS} ч")
    print(f"Эпизодов: {len(episodes)}; скважин: {len(set(e['well_id'] for e in episodes))}")
    print(f"С зарегистрированным событием: {len(complete)}")
    print(f"Без зарегистрированного события до конца записи (правоцензурированные): {len(censored)}")
    print("Скважины с цензурированием:", ", ".join(sorted(set(e["well_id"] for e in censored))) or "нет")
    if complete:
        durations = [e["transient_hours"] for e in complete]
        print(f"Продолжительность завершённых переходов: медиана {median(durations):.2f} ч; "
              f"минимум {min(durations):.2f} ч; максимум {max(durations):.2f} ч")
    print("Окон с метками: target=1:", sum(e["positive_windows"] for e in episodes),
          "; target=0:", sum(e["negative_windows"] for e in episodes),
          "; неизвестно:", sum(e["unknown_windows"] for e in episodes))

    print("\nПравоцензурированные эпизоды (отсутствие события не означает, что его не будет позже):")
    for e in censored:
        print(f"  {e['file']}: {e['transient_hours']:.2f} ч, "
              f"negative={e['negative_windows']}, unknown={e['unknown_windows']}")
    print("\nОтчёт:", OUTPUT_PATH)
    print("Примечание: окна внутри одного эпизода не являются независимыми событиями.")


if __name__ == "__main__":
    main()
