"""Проверка времени предупреждений фиксированной модели class 8 (оффлайн).

Не обучает модель и не подбирает порог на тестовых скважинах.
Использует фактическое время события ТОЛЬКО для оценки и рисунков.

Из корня CPS-PredDiag:
    python experiments/inspect_class8_alert_timeline.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_DIR / "outputs/models/class8_baseline/oof_predictions.csv"
HORIZONS = PROJECT_DIR / "outputs/inventory/class8_horizon_audit.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs/models/class8_baseline/alert_timeline"

FILES = [
    "WELL-00027_20230918170106.parquet",
    "WELL-00032_20110830235321.parquet",
]
PROBABILITY_COLUMN = "prob_rf_all_12"
HORIZON_HOURS = 6
THRESHOLD = 0.5


def load_data(predictions_path: Path, horizon_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path, parse_dates=["decision_time"])
    truth = pd.read_csv(
        horizon_path, parse_dates=["decision_time", "event_time"]
    )
    truth = truth.loc[
        (truth["horizon_hours"] == HORIZON_HOURS) & truth["target"].notna(),
        ["file", "decision_time", "target", "event_time"],
    ]

    result = preds.merge(
        truth,
        on=["file", "decision_time"],
        how="left",
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    if result["target_reference"].isna().any():
        raise ValueError("Для некоторых точек не найдены метки в horizon_audit")
    if not (result["target"] == result["target_reference"]).all():
        raise ValueError("Метки target в CSV не совпадают")
    return result


def analyze_file(df: pd.DataFrame, filename: str, output_dir: Path) -> None:
    part = df.loc[df["file"] == filename].sort_values("decision_time").copy()
    if part.empty:
        raise ValueError(f"Файл не найден в прогнозах: {filename}")

    observed_events = part["event_time"].dropna().unique()
    if len(observed_events) != 1:
        raise ValueError(f"Нет единственного времени события: {filename}")
    event_time = pd.Timestamp(observed_events[0])
    horizon_start = event_time - pd.Timedelta(hours=HORIZON_HOURS)

    probs = part[PROBABILITY_COLUMN]
    alarm = probs.ge(THRESHOLD)
    early = alarm & part["target"].eq(0)
    in_time = alarm & part["target"].eq(1)
    first_alarm = part.loc[alarm, "decision_time"].min()
    first_tp = part.loc[in_time, "decision_time"].min()

    print(f"\n{filename}")
    print(f"  Момент события: {event_time}")
    print(f"  Всего точек: {len(part)}; FP-окон: {early.sum()}; TP-окон: {in_time.sum()}")
    if pd.notna(first_alarm):
        print(f"  Первое любое предупреждение: {first_alarm} "
              f"({(event_time - first_alarm).total_seconds()/3600:.2f} ч до события)")
    else:
        print("  Предупреждений при пороге 0,5 нет")
    if pd.notna(first_tp):
        print(f"  Первое правильное для H=6ч: {first_tp} "
              f"({(event_time-first_tp).total_seconds()/3600:.2f} ч до события)")
    else:
        print("  Правильных предупреждений за 6 часов до события нет")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(part["decision_time"], probs, linewidth=1.0, label="RF: прогнозируемый риск")
    ax.axhline(THRESHOLD, linestyle="--", linewidth=1.2, label="Порог 0,5")
    ax.axvspan(horizon_start, event_time, alpha=0.14, label="Последние 6 часов")
    ax.axvline(event_time, linestyle=":", linewidth=1.3, label="Начало класса 8")
    if early.any():
        ax.scatter(part.loc[early, "decision_time"], probs.loc[early],
                   marker="x", s=18, label="FP: >6 ч до события", zorder=3)
    if in_time.any():
        ax.scatter(part.loc[in_time, "decision_time"], probs.loc[in_time],
                   marker="o", s=18, label="TP: событие в пределах 6 ч", zorder=3)
    ax.set(title=f"Временная динамика предупреждений: {filename}",
           xlabel="Время", ylabel="Выход predict_proba для класса 1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    well_id = str(int(part["well_id"].iloc[0])).zfill(5)
    output_file = output_dir / f"well_{well_id}_alert_timeline.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"  График: {output_file}")


def main() -> None:
    df = load_data(PREDICTIONS, HORIZONS)
    for filename in FILES:
        analyze_file(df, filename, OUTPUT_DIR)
    print("\nЭто описательная проверка неизменённой модели; порог по тесту не настраиваем.")


if __name__ == "__main__":
    main()
