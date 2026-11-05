"""Проверить загрузку ML Agent на одном историческом окне класса 2.

Используется истинная метка 102 только для ИМИТАЦИИ внешнего подтверждения
переходного состояния. Это демонстрация интерфейса, а НЕ независимая оценка.

Запуск: python experiments/smoke_test_class2_ml_agent.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_agent import MLAgent, SENSORS, WINDOW_SECONDS  # noqa: E402


def main() -> None:
    path = ROOT / "outputs/inventory/class2_horizon_points.csv"
    points = pd.read_csv(path, parse_dates=["decision_time"])
    points = points.loc[
        (points.horizon_minutes == 15) & (points.target == 1)
    ].sort_values(["file", "decision_time"])
    if points.empty:
        raise ValueError("Нет окон с положительной меткой")
    point = points.iloc[0]
    parquet = ROOT / "dataset/2" / point["file"]
    df = pd.read_parquet(parquet)
    for sensor in SENSORS:
        if sensor not in df.columns:
            df[sensor] = np.nan
    moment = pd.Timestamp(point.decision_time)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Ожидался DatetimeIndex")
    if moment not in df.index:
        raise ValueError("Точка прогноза отсутствует в Parquet")
    history = df.loc[:moment].tail(WINDOW_SECONDS)
    labels = pd.to_numeric(history["class"], errors="coerce")
    if not bool(labels.eq(102).all()) or len(history) != WINDOW_SECONDS:
        raise ValueError("Тестовая точка не находится внутри полного окна 102")

    agent = MLAgent(ROOT / "outputs/models/class2_prototype")
    unavailable = agent.predict(history, transition_102_confirmed=None,
                                decision_time=moment)
    if unavailable["status"] != "not_applicable":
        raise RuntimeError("Агент должен отказать при неподтверждённом 102")
    result = agent.predict(history, transition_102_confirmed=True,
                           decision_time=moment)
    print("=== Тест работы ML Agent ===")
    print("Файл:", point["file"])
    print("Время прогноза:", moment)
    print("Внешнее подтверждение class=102: ИСТИННАЯ РАЗМЕТКА (oracle)")
    print("Без подтверждения:", unavailable["status"])
    print("С подтверждением:", result["status"])
    print("Доступность датчиков:", result.get("sensor_coverage"))
    print("Оценка модели:", result.get("model_score"))
    print("Выше порога 0.5:", result.get("above_research_threshold"))
    print("ВАЖНО: модель обучена на всей выборке; демонстрация НЕ является тестом качества.")
    if result["status"] != "experimental_prediction":
        raise RuntimeError(f"Проверка не прошла: {result}")


if __name__ == "__main__":
    main()
