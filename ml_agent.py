"""Исследовательский ML Agent для Petrobras 3W, класса 2.

Оценивает риск события class=2 за следующие 15 минут *только при внешне
подтверждённом текущем состоянии 102*. Не определяет состояние 102 и не
выдаёт управляющих команд. Значение predict_proba не калибровано.

Загружайте только собственные доверенные .joblib-файлы: joblib использует pickle.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd


SENSORS = ("P-MON-CKP", "P-TPT", "T-TPT")
STATISTICS = ("mean", "std", "delta", "slope_per_hour")
FEATURES = [f"{s}_{stat}" for s in SENSORS for stat in STATISTICS]
WINDOW_SECONDS = 300
MIN_COVERAGE = 0.9


def features_from_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Признаки по 300 последним секундам, включая момент принятия решения.

    Формулы соответствуют describe_sensor() из train_class2_baseline.py.
    Метки class/state и будущие строки НЕ используются.
    """
    if not isinstance(history, pd.DataFrame) or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history должен быть DataFrame с DatetimeIndex")
    if len(history) < WINDOW_SECONDS:
        raise ValueError(f"Нужно минимум {WINDOW_SECONDS} последовательных измерений")
    window = history.iloc[-WINDOW_SECONDS:]
    if window.index.has_duplicates or not window.index.is_monotonic_increasing:
        raise ValueError("В истории есть повторяющиеся или неупорядоченные метки времени")
    if not np.all(np.diff(window.index.asi8) == 1_000_000_000):
        raise ValueError("Ожидались последовательные измерения с интервалом 1 секунда")

    features: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for sensor in SENSORS:
        if sensor in window.columns:
            y_all = pd.to_numeric(window[sensor], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
        else:
            y_all = np.full(WINDOW_SECONDS, np.nan, dtype=float)
        good = np.isfinite(y_all)
        coverage[sensor] = round(float(good.mean()), 4)
        if int(good.sum()) < int(WINDOW_SECONDS * MIN_COVERAGE):
            values = {stat: np.nan for stat in STATISTICS}
        else:
            x = np.flatnonzero(good).astype(np.float64)
            y = y_all[good]
            values = {
                "mean": float(y.mean()),
                "std": float(y.std(ddof=1)),
                "delta": float(y[-1] - y[0]),
                "slope_per_hour": float(np.polyfit(x, y, 1)[0] * 3600),
            }
        features.update({f"{sensor}_{stat}": value for stat, value in values.items()})
    return pd.DataFrame([features], columns=FEATURES), coverage


class MLAgent:
    """Независимый исследовательский компонент, пока без связи с Coordinator."""

    def __init__(self, artifact_dir: str | Path):
        folder = Path(artifact_dir)
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Не найден manifest.json: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("features") != FEATURES:
            raise ValueError("Схема признаков модели не совпадает со схемой ML Agent")
        if self.manifest.get("window_seconds") != WINDOW_SECONDS:
            raise ValueError("Неподдерживаемая длина окна")
        if self.manifest.get("horizon_minutes") != 15 or self.manifest.get("event_class") != 2:
            raise ValueError("Этот ML Agent поддерживает только класс 2 / горизонт 15 минут")
        model_file = self.manifest.get("model_file")
        if not isinstance(model_file, str) or Path(model_file).name != model_file:
            raise ValueError("Некорректное имя файла модели")
        # Загружать только созданную нами модель, никогда непроверенный файл из сети.
        self.model = joblib.load(folder / model_file)
        if list(getattr(self.model, "classes_", [])) != [0, 1]:
            raise ValueError("Ожидалась бинарная модель с классами [0, 1]")

    def predict(self, history: pd.DataFrame, *, transition_102_confirmed: bool | None,
                decision_time: pd.Timestamp | None = None) -> dict[str, Any]:
        """Возвращает диагностическую оценку, а не команду оборудованию.

        transition_102_confirmed должна прийти от независимого диагностического
        компонента. Если статус неизвестен, прогноз запрещён. Во время offline
        replay его можно взять из истинной разметки, явно пометив это как oracle.
        """
        base = {
            "agent": "ml",
            "model_version": self.manifest["model_version"],
            "event_class": 2,
            "horizon_minutes": 15,
            "threshold": 0.5,
            "uncalibrated": True,
            "requires_confirmed_state_102": True,
        }
        if transition_102_confirmed is not True:
            return {**base, "status": "not_applicable", "reason":
                    "Состояние 102 не подтверждено независимым диагностическим компонентом"}
        try:
            features, coverage = features_from_history(history)
        except ValueError as exc:
            return {**base, "status": "insufficient_data", "reason": str(exc)}
        timestamp = history.index[-1]
        if decision_time is not None and pd.Timestamp(decision_time) != timestamp:
            return {**base, "status": "invalid_time", "reason":
                    "Последняя запись истории не совпадает с моментом прогноза"}
        usable = sum(coverage[s] >= MIN_COVERAGE for s in SENSORS)
        if usable < 2:
            return {**base, "status": "insufficient_data", "reason":
                    "Доступно меньше двух из трёх датчиков", "sensor_coverage": coverage}
        score = float(self.model.predict_proba(features)[0, 1])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Некорректная оценка модели")
        return {
            **base,
            "status": "experimental_prediction",
            "decision_time": timestamp.isoformat(),
            "model_score": score,
            "above_research_threshold": score >= 0.5,
            "sensor_coverage": coverage,
            "note": "Не калиброванная оценка модели; не использовать для управления оборудованием",
        }
