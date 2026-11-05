from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "features"

# 300 измерений при частоте 1 Гц = 5 минут
WINDOW_SIZE = 300

SENSORS = [
    "P-ANULAR",
    "P-JUS-CKGL",
    "P-MON-CKP",
    "P-TPT",
    "T-JUS-CKP",
    "T-TPT",
]


def extract_features(file_path: Path) -> list:
    df = pd.read_parquet(file_path)

    # Неизвестные классы не используем
    labeled = df.dropna(subset=["class"]).copy()

    if labeled.empty:
        return []

    # Разделяем участки при смене класса
    # или разрыве временных меток
    class_change = (
        labeled["class"]
        .ne(labeled["class"].shift())
        .fillna(True)
    )

    time_gap = (
        labeled.index.to_series()
        .diff()
        .gt(pd.Timedelta(seconds=1))
    )

    segment_ids = (class_change | time_gap).cumsum()

    results = []

    for _, segment in labeled.groupby(segment_ids):

        # Непересекающиеся полные окна
        for start in range(
            0,
            len(segment) - WINDOW_SIZE + 1,
            WINDOW_SIZE
        ):
            window = segment.iloc[
                start:start + WINDOW_SIZE
            ]

            features = {
                "file": file_path.name,
                "start": window.index[0],
                "end": window.index[-1],
                "class": int(window["class"].iloc[0]),
            }

            for sensor in SENSORS:
                if sensor not in window.columns:
                    raise ValueError(
                        f"Нет датчика {sensor} в {file_path.name}"
                    )

                values = window[sensor].dropna()

                if len(values) != WINDOW_SIZE:
                    features[f"{sensor}_mean"] = None
                    features[f"{sensor}_std"] = None
                    features[f"{sensor}_range"] = None
                    continue

                features[f"{sensor}_mean"] = values.mean()
                features[f"{sensor}_std"] = values.std()
                features[f"{sensor}_range"] = (
                    values.max() - values.min()
                )

            results.append(features)

    return results


def main():
    all_features = []

    for file_path in sorted(
        DATASET_DIR.rglob("*.parquet")
    ):
        features = extract_features(file_path)
        all_features.extend(features)

        print(
            f"{file_path.name}: "
            f"{len(features)} окон"
        )

    if not all_features:
        print("Нет подходящих окон.")
        return

    result = pd.DataFrame(all_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "window_features.csv"

    result.to_csv(output_file, index=False)

    print("\nРаспределение окон по классам:")
    print(result["class"].value_counts().sort_index())

    print("\nМедианные значения STD по классам:")

    std_columns = [
        f"{sensor}_std"
        for sensor in SENSORS
    ]

    summary = result.groupby("class")[
        std_columns
    ].median()

    print(summary.round(3).to_string())

    print(f"\nСохранено: {output_file}")


if __name__ == "__main__":
    main()
