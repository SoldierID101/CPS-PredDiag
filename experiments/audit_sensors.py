from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"


def audit_file(file_path: Path) -> None:
    df = pd.read_parquet(file_path)

    # Не используем неизвестные классы
    labeled = df[df["class"].notna()].copy()

    groups = {
        "empty": [],
        "constant": [],
        "discrete": [],
        "variable": [],
    }

    sensors = [
        col for col in df.columns
        if col not in ("class", "state")
    ]

    for sensor in sensors:
        values = labeled[sensor].dropna()
        unique_count = values.nunique()

        if unique_count == 0:
            groups["empty"].append(sensor)

        elif unique_count == 1:
            groups["constant"].append(
                (sensor, values.iloc[0])
            )

        elif unique_count <= 3:
            groups["discrete"].append(
                (sensor, unique_count)
            )

        else:
            groups["variable"].append(
                (sensor, unique_count)
            )

    print("\n" + "=" * 65)
    print(file_path.name)
    print("=" * 65)
    print("Измерений с известным классом:", len(labeled))

    for category, items in groups.items():
        print(f"\n{category.upper()} ({len(items)}):")
        for item in items:
            print(" ", item)


def main() -> None:
    files = sorted(DATASET_DIR.rglob("*.parquet"))

    for file_path in files:
        audit_file(file_path)


if __name__ == "__main__":
    main()
