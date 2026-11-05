"""CPS-PredDiag: аудит горизонтов прогноза для реальных эпизодов класса 2.

Задача: во время состояния 102 предсказать появление класса 2 в ближайшие
15, 30 или 60 минут. Это аудит меток, а не обучение или проверка модели.

Запуск из корня проекта:
    python experiments/audit_class2_horizons.py

Читаются только метки class и индекс времени; исходные Parquet не изменяются.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inventory"

NORMAL = 0
TRANSIENT = 102
EVENT = 2
HISTORY_SECONDS = 5 * 60  # Только данные 102, уже доступные к моменту решения
STEP_SECONDS = 5 * 60
HORIZONS_MINUTES = (15, 30, 60)
SECOND_NS = 1_000_000_000


def load_runs(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int, int]]]:
    df = pd.read_parquet(path, columns=["class"])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ожидался DatetimeIndex; проверьте структуру Parquet")
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("пустой файл, повторяющееся или неупорядоченное время")

    labels = (
        pd.to_numeric(df["class"], errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    time_ns = df.index.asi8
    if len(df) > 1:
        breaks = np.flatnonzero(
            (labels[1:] != labels[:-1])
            | (np.diff(time_ns) != SECOND_NS)
        ) + 1
    else:
        breaks = np.array([], dtype=np.int64)
    bounds = np.r_[0, breaks, len(df)]
    runs = [
        (int(a), int(b), int(labels[a]))
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    return df, time_ns, runs


def audit_file(path: Path) -> tuple[list[dict], list[dict], list[str]]:
    df, times, runs = load_runs(path)
    match = re.match(r"WELL-(\d+)_", path.name)
    if not match:
        raise ValueError("имя файла не содержит идентификатор WELL-xxxxx")
    well_id = match.group(1)

    points = []
    episodes = []
    messages = []
    for run_i, (start, end, label) in enumerate(runs):
        if label != TRANSIENT:
            continue

        # Не считаем неизвестный префикс нормой: нужен явный переход 0 -> 102.
        if run_i == 0 or runs[run_i - 1][2] != NORMAL:
            messages.append(f"{path.name}: участок 102 без предшествующей метки 0 — пропущен")
            continue
        if times[start] - times[start - 1] != SECOND_NS:
            messages.append(f"{path.name}: разрыв времени на границе 0 -> 102 — пропущен")
            continue

        following_label = runs[run_i + 1][2] if run_i + 1 < len(runs) else None
        event_time_ns = None
        if following_label == EVENT:
            next_start = runs[run_i + 1][0]
            if times[next_start] - times[end - 1] == SECOND_NS:
                event_time_ns = int(times[next_start])

        episode_key = f"{path.stem}__run{run_i}"
        status = "COMPLETED" if event_time_ns is not None else "NOT_OBSERVED"
        episode_record = {
            "episode_id": episode_key,
            "file": path.name,
            "well_id": well_id,
            "status": status,
            "transient_start": df.index[start],
            "transient_last": df.index[end - 1],
            "event_time": (pd.Timestamp(event_time_ns)
                           if event_time_ns is not None else pd.NaT),
            "transient_duration_min": round((end - start) / 60, 3),
            "following_label": following_label,
            "decision_points": 0,
        }

        # В конце каждого пятиминутного окна — точка прогноза.
        # Начало истории никогда не выходит за участок 102.
        for decision_idx in range(start + HISTORY_SECONDS - 1, end, STEP_SECONDS):
            decision_ns = int(times[decision_idx])
            for horizon_min in HORIZONS_MINUTES:
                target = None
                horizon_ns = horizon_min * 60 * SECOND_NS
                if event_time_ns is not None:
                    # До события 102 непрерывен, значит будущий горизонт проверяем.
                    target = int(event_time_ns <= decision_ns + horizon_ns)
                elif decision_ns + horizon_ns <= int(times[end - 1]):
                    # Только наблюдаемое будущее; не предполагаем норму после конца файла.
                    target = 0

                points.append({
                    "episode_id": episode_key,
                    "file": path.name,
                    "well_id": well_id,
                    "decision_time": df.index[decision_idx],
                    "horizon_minutes": horizon_min,
                    "target": target,
                    "transient_start": df.index[start],
                    "event_time": episode_record["event_time"],
                    "minutes_since_transient_start": round(
                        (decision_ns - int(times[start])) / (60 * SECOND_NS), 3
                    ),
                    "minutes_until_event": (
                        round((event_time_ns - decision_ns) / (60 * SECOND_NS), 3)
                        if event_time_ns is not None else None
                    ),
                })
            episode_record["decision_points"] += 1

        episodes.append(episode_record)
        if following_label not in (None, EVENT):
            messages.append(
                f"{path.name}: после 102 стоит метка {following_label}; "
                "непроверенное будущее помечено как неизвестное"
            )

    return points, episodes, messages


def main() -> None:
    files = sorted(DATA_DIR.glob("WELL-*.parquet"))
    if not files:
        raise SystemExit(f"Не найдены WELL-*.parquet в {DATA_DIR}")

    all_points, all_episodes, warnings, errors = [], [], [], []
    print(f"Класс 2: найдено {len(files)} реальных файлов")

    for path in files:
        try:
            points, episodes, notes = audit_file(path)
            all_points.extend(points)
            all_episodes.extend(episodes)
            warnings.extend(notes)
            print(f"  {path.name}: эпизодов={len(episodes)}, "
                  f"точек прогноза={sum(e['decision_points'] for e in episodes)}")
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"  ОШИБКА {errors[-1]}")

    if errors:
        raise SystemExit("Ошибки чтения: отчёты не сохранены, проверьте файлы выше.")
    if not all_points:
        raise SystemExit("Нет подходящих переходов 0 -> 102 с 5 минутами истории.")

    points_df = pd.DataFrame(all_points)
    episodes_df = pd.DataFrame(all_episodes)
    summary = []
    for horizon in HORIZONS_MINUTES:
        sub = points_df.loc[points_df["horizon_minutes"] == horizon]
        positive = sub.loc[sub["target"] == 1]
        negative = sub.loc[sub["target"] == 0]
        unknown = sub.loc[sub["target"].isna()]
        summary.append({
            "horizon_min": horizon,
            "positive_windows": len(positive),
            "negative_windows": len(negative),
            "unknown_windows": len(unknown),
            "positive_episodes": positive["episode_id"].nunique(),
            "positive_wells": positive["well_id"].nunique(),
            "negative_episodes": negative["episode_id"].nunique(),
            "negative_wells": negative["well_id"].nunique(),
        })

    summary_df = pd.DataFrame(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points_path = OUTPUT_DIR / "class2_horizon_points.csv"
    episodes_path = OUTPUT_DIR / "class2_episodes.csv"
    summary_path = OUTPUT_DIR / "class2_horizon_summary.csv"
    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    episodes_df.to_csv(episodes_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nПоследовательности (после 0 -> 102):")
    print(episodes_df["status"].value_counts().to_string())
    print(f"Уникальных скважин: {episodes_df['well_id'].nunique()}")
    print("\nАудит горизонтов (окна НЕ независимые события):")
    print(summary_df.to_string(index=False))
    print("\nПримечания о неоднозначных участках:")
    if warnings:
        for warning in warnings[:15]:
            print("  -", warning)
        if len(warnings) > 15:
            print(f"  ... и ещё {len(warnings) - 15}")
    else:
        print("  Не обнаружено")
    print("\nОтчёты:")
    print(" ", summary_path)
    print(" ", episodes_path)
    print(" ", points_path)
    print("\nИнтерпретация: target=0 — отсутствие события в полностью "
          "наблюдаемом горизонте, target=1 — событие в пределах горизонта, "
          "пустая метка — неизвестное будущее.")
    print("Пяти-минутные окна и время с начала 102 не означают "
          "независимых событий и не доказывают прогнозную способность.")


if __name__ == "__main__":
    main()
