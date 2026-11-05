from pathlib import Path
import pandas as pd


# Папка проекта CPS-PredDiag
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"


def inspect_file(file_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"Файл: {file_path.name}")
    print("=" * 70)

    # 1. Чтение Parquet
    df = pd.read_parquet(file_path)

    print(f"\nРазмер: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # 2. Структура данных
    print("\nСтолбцы и типы:")
    print(df.dtypes.to_string())

    # 3. Временные метки
    print("\nТип индекса:", type(df.index).__name__)

    if isinstance(df.index, pd.DatetimeIndex):
        print("Начало:", df.index.min())
        print("Конец:", df.index.max())
        print("Упорядочен:", df.index.is_monotonic_increasing)
        print("Дубликаты времени:", df.index.duplicated().sum())

        intervals = df.index.to_series().diff().dropna()

        print("\nНаиболее частые интервалы между измерениями:")
        print(intervals.value_counts().head(5))

    # 4. Диагностические метки
    for column in ("class", "state"):
        if column in df.columns:
            print(f"\nРаспределение {column}:")
            print(df[column].value_counts(dropna=False).to_string())
        else:
            print(f"\nСтолбец {column} отсутствует")

    # 5. Признаки датчиков
    sensors = [
        col for col in df.columns
        if col not in ("class", "state")
    ]

    print("\nКоличество признаков:", len(sensors))

    if sensors:
        missing = (
            df[sensors].isna().mean() * 100
        ).sort_values(ascending=False)

        print("\nДоля пропусков по датчикам (%):")
        print(missing.round(2).to_string())

    # 6. Переходы между диагностическими классами
    if "class" in df.columns and not df.empty:
        labels = df["class"].astype("string").fillna("NA")
        changes = labels.ne(labels.shift()).fillna(True)

        print("\nКоличество участков с постоянным классом:",
              int(changes.sum()))

        print("\nПервые переходы между классами:")
        print(df.loc[changes, ["class"]].head(15).to_string())

    # 7. Первые строки
    display_columns = sensors[:4] + [
        col for col in ("class", "state")
        if col in df.columns
    ]

    print("\nПервые три записи:")
    print(df[display_columns].head(3).to_string())


def main() -> None:
    files = sorted(DATASET_DIR.rglob("*.parquet"))

    if not files:
        print("Parquet-файлы не найдены!")
        return

    print(f"Найдено файлов: {len(files)}")

    for file_path in files:
        try:
            inspect_file(file_path)
        except Exception as error:
            print(f"Ошибка при обработке {file_path.name}: {error}")


if __name__ == "__main__":
    main()
