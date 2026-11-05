"""CPS-PredDiag: exploratory plots of annotated transitions for real 3W events.
Uses two short (within each class) completed examples from the real_label_audit.csv.
No model training; plotted resampling is for visualization only.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "transitions"

# Confirmed completed episodes from the earlier label audit (both transitions present).
EXAMPLES = {
    2: "2/WELL-00009_20170313150804.parquet",
    8: "8/WELL-00032_20110830235321.parquet",
}

SENSORS = {
    "P-PDG": "Забойное давление P-PDG, Па",
    "P-TPT": "Давление P-TPT, Па",
    "P-MON-CKP": "Давление перед штуцером P-MON-CKP, Па",
    "T-TPT": "Температура T-TPT, °C",
}

STAGE_COLORS = {
    "UNKNOWN": "#9CA3AF",
    "0": "#38A169",
    "TRANSIENT": "#E9B44C",
    "EVENT": "#E76F51",
}


def get_segments(df: pd.DataFrame) -> list[dict]:
    """Return consecutively labeled intervals, keeping missing labels distinct."""
    labels = df["class"].astype("string").fillna("UNKNOWN")
    segment_ids = (
        labels.ne(labels.shift())
        .fillna(True)
        .astype("int64")
        .cumsum()
    )
    segments = []
    for _, part in df.groupby(segment_ids, sort=False):
        label = str(labels.loc[part.index[0]])
        segments.append({
            "label": label,
            "start": part.index[0],
            "end": part.index[-1],
            "rows": len(part),
        })
    return segments


def analyze(event_class: int, relative_path: str) -> None:
    path = DATASET_DIR / Path(relative_path)
    if not path.is_file():
        print(f"Не найден файл: {path}")
        return

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex) or "class" not in df.columns:
        print(f"Нет временного индекса или class: {path.name}")
        return

    df = df.sort_index()
    segments = get_segments(df)
    transient_label = str(100 + event_class)
    event_label = str(event_class)

    print(f"\n{'=' * 65}\n{path.name} (папка {event_class})")
    print("Последовательные размеченные участки:")
    for seg in segments:
        hours = (seg["end"] - seg["start"]).total_seconds() / 3600
        print(
            f"  {seg['label']:>7} | {seg['start']} → {seg['end']}"
            f" | {seg['rows']} строк | {hours:.2f} ч"
        )

    first_transient = next(
        (s["start"] for s in segments if s["label"] == transient_label), None
    )
    first_event = next(
        (s["start"] for s in segments if s["label"] == event_label), None
    )
    if first_transient is not None and first_event is not None:
        print(
            "От первой метки переходного состояния до первой метки события: "
            f"{(first_event - first_transient).total_seconds() / 3600:.2f} ч"
        )

    available = [
        col for col in SENSORS
        if col in df and df[col].notna().any() and df[col].nunique(dropna=True) > 1
    ]
    if not available:
        print("Нет изменяющихся датчиков из выбранного списка.")
        return

    # Only for viewing; do not use the resampled measurements as ML training data.
    frequency = "15s" if event_class == 2 else "1min"
    display = df[available].resample(frequency).median()

    fig, axes = plt.subplots(
        len(available), 1, figsize=(15, 3 * len(available)),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for ax, sensor in zip(axes.flat, available):
        for seg in segments:
            kind = (
                "TRANSIENT" if seg["label"] == transient_label else
                "EVENT" if seg["label"] == event_label else seg["label"]
            )
            ax.axvspan(
                seg["start"], seg["end"],
                color=STAGE_COLORS.get(kind, "#B0B0B0"), alpha=0.16,
                linewidth=0,
            )
        ax.plot(display.index, display[sensor], linewidth=0.85)
        ax.set_ylabel(SENSORS[sensor], fontsize=9)
        ax.grid(alpha=0.25)
        if first_transient is not None:
            ax.axvline(first_transient, linestyle="--", color="#B8860B", linewidth=1)
        if first_event is not None:
            ax.axvline(first_event, linestyle="--", color="#B22222", linewidth=1)

    fig.suptitle(
        f"3W — класс {event_class}: {path.name}\n"
        f"Границы классов точные; графики датчиков агрегированы ({frequency})",
        fontsize=11,
    )
    fig.legend(
        handles=[
            Patch(color=STAGE_COLORS["UNKNOWN"], alpha=0.4, label="Неизвестно"),
            Patch(color=STAGE_COLORS["0"], alpha=0.4, label="Норма (0)"),
            Patch(color=STAGE_COLORS["TRANSIENT"], alpha=0.4,
                  label=f"Переход ({transient_label})"),
            Patch(color=STAGE_COLORS["EVENT"], alpha=0.4,
                  label=f"Событие ({event_label})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=9,
    )
    axes[-1, 0].set_xlabel("Дата и время")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"timeline_{event_class}_{path.stem}.png"
    fig.savefig(output_file, dpi=130)
    plt.close(fig)
    print(f"График: {output_file}")


def main() -> None:
    for event_class, relative_path in EXAMPLES.items():
        analyze(event_class, relative_path)


if __name__ == "__main__":
    main()
