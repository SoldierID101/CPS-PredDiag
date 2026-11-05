"""Сравнение структуры реальных эпизодов 3W по классам.

Вход: outputs/inventory/real_label_audit.csv (создан ранее).
Выход:
  outputs/inventory/3w_candidate_classes.csv
  outputs/inventory/3w_candidate_wells.csv

Только описательный анализ меток; без чтения Parquet, обучения и утечки в ML.
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "inventory" / "real_label_audit.csv"
OUT = ROOT / "outputs" / "inventory"

REQUIRED = {
    "file", "well_id", "event_class", "normal_rows", "transient_rows",
    "event_rows", "direct_0_to_transient", "direct_transient_to_event",
}


def classify(row):
    """Категория определяется наличием меток и наблюдаемых переходов."""
    if (
        row["event_rows"] > 0
        and row["transient_rows"] > 0
        and row["direct_0_to_transient"] == 1
        and row["direct_transient_to_event"] == 1
    ):
        return "completed_sequence"
    if row["transient_rows"] > 0 and row["event_rows"] == 0:
        return "transient_censored"
    if row["normal_rows"] > 0 and row["transient_rows"] == 0 and row["event_rows"] == 0:
        return "normal_only"
    return "other_review_needed"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Отсутствует исходный отчёт: {INPUT}")

    df = pd.read_csv(INPUT, dtype={"well_id": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Необходимые столбцы отсутствуют: {sorted(missing)}")

    df["well_id"] = df["well_id"].str.zfill(5)
    df["episode_kind"] = df.apply(classify, axis=1)

    class_rows = []
    well_rows = []
    for event_class, group in df.groupby("event_class", sort=True):
        completed = group[group.episode_kind.eq("completed_sequence")]
        censored = group[group.episode_kind.eq("transient_censored")]
        normal = group[group.episode_kind.eq("normal_only")]
        other = group[group.episode_kind.eq("other_review_needed")]
        completed_wells = set(completed.well_id)
        censored_wells = set(censored.well_id)
        class_rows.append({
            "event_class": event_class,
            "files": len(group),
            "wells": group.well_id.nunique(),
            "completed_files": len(completed),
            "censored_transient_files": len(censored),
            "normal_only_files": len(normal),
            "other_files": len(other),
            "completed_wells": len(completed_wells),
            "censored_wells": len(censored_wells),
            "wells_with_both": len(completed_wells & censored_wells),
        })
        for well_id, wells_group in group.groupby("well_id"):
            counts = wells_group.episode_kind.value_counts()
            well_rows.append({
                "event_class": event_class,
                "well_id": well_id,
                "files": len(wells_group),
                "completed_files": int(counts.get("completed_sequence", 0)),
                "censored_transient_files": int(counts.get("transient_censored", 0)),
                "normal_only_files": int(counts.get("normal_only", 0)),
                "other_files": int(counts.get("other_review_needed", 0)),
            })

    summary = pd.DataFrame(class_rows)
    by_well = pd.DataFrame(well_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "3w_candidate_classes.csv", index=False, encoding="utf-8-sig")
    by_well.to_csv(OUT / "3w_candidate_wells.csv", index=False, encoding="utf-8-sig")

    print("\nСравнение классов по реальным файлам:")
    print(summary.to_string(index=False))
    print("\nКласс 2 по скважинам:")
    print(by_well[by_well.event_class.eq(2)].to_string(index=False))
    print("\nВАЖНО: censored означает отсутствие события до конца записи,\n"
          "а normal_only — отсутствие размеченного перехода и события.\n"
          "completed_sequence подтверждает наличие обеих смен меток,\n"
          "но не гарантирует единственную непрерывную последовательность.")
    print(f"\nОтчёты: {OUT / '3w_candidate_classes.csv'} и {OUT / '3w_candidate_wells.csv'}")


if __name__ == "__main__":
    main()
