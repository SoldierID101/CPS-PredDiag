from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"

SENSORS = {
    "P-PDG": "Downhole pressure (Pa)",
    "P-TPT": "Tree pressure (Pa)",
    "P-MON-CKP": "Upstream choke pressure (Pa)",
    "T-TPT": "Tree temperature (°C)",
    "QGL": "Gas lift flow rate (m³/s)",
}


def analyze_file(file_path: Path) -> None:
    df = pd.read_parquet(file_path)

    # Используем только реально доступные датчики.
    available = [
        name for name in SENSORS
        if name in df.columns and df[name].notna().any()
    ]

    if not available:
        print(f"Нет доступных датчиков: {file_path.name}")
        return

    print(f"\nФайл: {file_path.name}")

    for sensor in available:
        values = df[sensor].dropna()

        print(
            f"{sensor}: "
            f"min={values.min():.3f}, "
            f"max={values.max():.3f}, "
            f"std={values.std():.3f}, "
            f"unique={values.nunique()}"
        )

    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(14, 11),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    # Это граница появления известной метки,
    # а НЕ доказанное время начала неисправности.
    first_labeled = (
        df["class"].first_valid_index()
        if "class" in df.columns else None
    )

    for ax, sensor in zip(axes.flat, available):
        ax.plot(
            df.index,
            df[sensor],
            linewidth=0.9,
        )

        ax.set_ylabel(SENSORS[sensor])
        ax.grid(True, alpha=0.3)

        # Отмечаем неразмеченный первый участок.
        if first_labeled is not None:
            if first_labeled > df.index.min():
                ax.axvspan(
                    df.index.min(),
                    first_labeled,
                    color="gray",
                    alpha=0.18,
                )

            ax.axvline(
                first_labeled,
                linestyle="--",
                color="black",
                alpha=0.6,
            )

    axes[-1, 0].set_xlabel("Timestamp")

    labels = (
        df["class"].dropna().unique().tolist()
        if "class" in df.columns else []
    )

    fig.suptitle(
        f"{file_path.name}\nKnown classes: {labels}",
        fontsize=12,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{file_path.stem}.png"

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"График сохранён: {output_file}")


def main() -> None:
    files = sorted(DATASET_DIR.rglob("*.parquet"))

    for file_path in files:
        analyze_file(file_path)


if __name__ == "__main__":
    main()
