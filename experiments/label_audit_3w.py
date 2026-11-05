"""Аудит разметки реальных временных рядов 3W, имеющих переходные коды.
Запуск из корня CPS-PredDiag:
    python experiments/label_audit_3w.py
Читает только столбец class и индекс времени, не данные датчиков.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inventory"
EVENT_CLASSES = (1, 2, 5, 6, 7, 8, 9)
ONE_SECOND_NS = 1_000_000_000


def adjacent_count(labels: np.ndarray, continuous: np.ndarray, first: int, second: int) -> int:
    if len(labels) < 2:
        return 0
    mask = (labels[:-1] == first) & (labels[1:] == second) & continuous
    return int(np.count_nonzero(mask))


def audit_file(file_path: Path, class_id: int) -> dict:
    df = pd.read_parquet(file_path, columns=["class"])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Не найден DatetimeIndex")

    labels = df["class"].fillna(-1).astype("int16").to_numpy()
    continuous = np.diff(df.index.asi8) == ONE_SECOND_NS
    transition_label = 100 + class_id

    known_positions = np.flatnonzero(labels != -1)
    unknown_prefix = int(known_positions[0]) if len(known_positions) else len(labels)

    match = re.match(r"WELL-(\d+)_", file_path.name)
    well_id = match.group(1) if match else ""

    n_0_to_t = adjacent_count(labels, continuous, 0, transition_label)
    n_t_to_e = adjacent_count(labels, continuous, transition_label, class_id)
    n_0_to_e = adjacent_count(labels, continuous, 0, class_id)

    return {
        "file": str(file_path.relative_to(DATASET_DIR)),
        "well_id": well_id,
        "event_class": class_id,
        "rows": len(labels),
        "normal_rows": int(np.count_nonzero(labels == 0)),
        "transient_rows": int(np.count_nonzero(labels == transition_label)),
        "event_rows": int(np.count_nonzero(labels == class_id)),
        "unknown_rows": int(np.count_nonzero(labels == -1)),
        "unknown_prefix_rows": unknown_prefix,
        "direct_0_to_transient": n_0_to_t,
        "direct_transient_to_event": n_t_to_e,
        "direct_0_to_event": n_0_to_e,
    }


def main() -> None:
    records = []
    errors = []

    for class_id in EVENT_CLASSES:
        folder = DATASET_DIR / str(class_id)
        files = sorted(folder.glob("WELL-*.parquet"))
        print(f"Класс {class_id}: проверяем {len(files)} реальных файлов")
        for file_path in files:
            try:
                records.append(audit_file(file_path, class_id))
            except Exception as exc:
                errors.append((str(file_path), str(exc)))

    if not records:
        print("Реальные файлы с переходными кодами не найдены")
        return

    result = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "real_label_audit.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")

    result["has_normal"] = result["normal_rows"] > 0
    result["has_transient"] = result["transient_rows"] > 0
    result["has_event"] = result["event_rows"] > 0
    result["has_unknown"] = result["unknown_rows"] > 0
    result["has_0_to_t"] = result["direct_0_to_transient"] > 0
    result["has_t_to_e"] = result["direct_transient_to_event"] > 0
    result["has_full_chain"] = result["has_0_to_t"] & result["has_t_to_e"]

    summary = result.groupby("event_class").agg(
        files=("file", "size"),
        wells=("well_id", "nunique"),
        with_normal=("has_normal", "sum"),
        with_transient=("has_transient", "sum"),
        with_event=("has_event", "sum"),
        with_unknown=("has_unknown", "sum"),
        direct_0_to_transient=("has_0_to_t", "sum"),
        direct_transient_to_event=("has_t_to_e", "sum"),
        both_transitions=("has_full_chain", "sum"),
    )

    print("\nРезультаты аудита разметки (количество файлов):")
    print(summary.to_string())
    print(f"\nОбработано: {len(records)} файлов; ошибок: {len(errors)}")
    print("Отчёт:", out)
    for path, error in errors[:10]:
        print("ОШИБКА:", path, error)


if __name__ == "__main__":
    main()
