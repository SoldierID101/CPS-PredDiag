from pathlib import Path
from collections import Counter
import csv
import re
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inventory"

CLASSES = [str(i) for i in range(10)]


def detect_source(filename):
    if filename.startswith("WELL-"):
        return "REAL"
    if filename.startswith("SIMULATED_"):
        return "SIMULATED"
    if filename.startswith("DRAWN_"):
        return "DRAWN"
    return "UNKNOWN"


def extract_well_id(filename):
    match = re.match(r"WELL-(\d+)_", filename)
    return match.group(1) if match else None


def main():
    records = []
    errors = []

    for class_id in CLASSES:
        folder = DATASET_DIR / class_id

        if not folder.exists():
            print(f"Папка {class_id} отсутствует")
            continue

        files = sorted(folder.rglob("*.parquet"))

        for file_path in files:
            try:
                metadata = pq.read_metadata(file_path)

                records.append({
                    "class_folder": class_id,
                    "filename": file_path.name,
                    "source": detect_source(file_path.name),
                    "well_id": extract_well_id(file_path.name),
                    "rows": metadata.num_rows,
                    "columns": metadata.num_columns,
                    "size_mb": round(
                        file_path.stat().st_size / 1024**2,
                        3
                    ),
                })

            except Exception as error:
                errors.append(
                    (str(file_path), str(error))
                )

    if not records:
        print("Подходящие файлы не найдены.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "dataset_inventory.csv"

    with output_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nРаспределение файлов по классам:")
    print(
        f"{'Class':<8}{'REAL':>10}"
        f"{'SIMULATED':>13}{'DRAWN':>10}"
        f"{'TOTAL':>10}"
    )

    for class_id in CLASSES:
        subset = [
            r for r in records
            if r["class_folder"] == class_id
        ]

        counts = Counter(r["source"] for r in subset)

        print(
            f"{class_id:<8}"
            f"{counts['REAL']:>10}"
            f"{counts['SIMULATED']:>13}"
            f"{counts['DRAWN']:>10}"
            f"{len(subset):>10}"
        )

    real_files = [
        r for r in records
        if r["source"] == "REAL"
    ]

    well_ids = {
        r["well_id"]
        for r in real_files
        if r["well_id"] is not None
    }

    print("\nОбщая статистика:")
    print("Всего файлов:", len(records))
    print("Реальных файлов:", len(real_files))
    print("Уникальных скважин:", len(well_ids))
    print(
        "Всего строк (включая все источники):",
        sum(r["rows"] for r in records)
    )
    print(
        "Размер Parquet (МБ):",
        round(sum(r["size_mb"] for r in records), 2)
    )
    print("Ошибок чтения:", len(errors))

    for path, error in errors[:10]:
        print("Ошибка:", path, error)

    print("\nОтчёт сохранён:", output_file)


if __name__ == "__main__":
    main()
