"""CPS-PredDiag: Prognostic Agent v0.1 (только исследование, без БД).

Линейная экстраполяция тренда измеряемого параметра до заданного
ДОКУМЕНТИРОВАННОГО порога. Время до порога НЕ равно времени до отказа.
По умолчанию пороги НЕ заданы, поэтому агент ничего не предсказывает.

Вход: DataFrame, индекс времени с шагом 1 секунда; используются только
последние 300 секунд, включая момент прогноза. Поля class/state игнорируются.
Выход: список словарей для coordinator.coordinate_decision(predictions=...).

Важное ограничение: простая экстраполяция тренда не валидирована на скважинах;
не используется для управления промышленным оборудованием.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd

WINDOW_SECONDS = 300
MIN_COVERAGE = 0.9
SENSORS = frozenset({"P-PDG", "P-TPT", "P-MON-CKP", "T-TPT"})
_SEVERITIES = frozenset({"warning", "critical"})
_DIRECTIONS = frozenset({"above", "below"})


def _to_finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class PrognosticAgent:
    """Исследовательский прогноз пересечения обоснованных порогов.

    thresholds: [{"id": ..., "sensor": "P-TPT", "threshold": number,
                  "direction": "above" | "below", "severity": "warning" | "critical",
                  "unit": "Pa", "source": "источник конкретного порога"}]

    `source` должен подтверждать именно ЧИСЛЕННЫЙ порог, а не только
    факт роста/снижения давления из публикации.
    """

    def __init__(self, thresholds: Sequence[Mapping[str, Any]] | None = None, *,
                 horizon_minutes: float = 15.0):
        horizon = _to_finite(horizon_minutes)
        if horizon is None or horizon <= 0:
            raise ValueError("Горизонт прогноза должен быть положительным конечным числом минут")
        self.horizon_minutes = horizon
        registered_ids: set[str] = set()
        self.thresholds: list[dict[str, Any]] = []
        for raw in thresholds or []:
            if not isinstance(raw, Mapping):
                raise ValueError("Описание порога должно быть словарём")
            item = dict(raw)
            ident = item.get("id")
            sensor = item.get("sensor")
            if not isinstance(ident, str) or not ident.strip() or ident in registered_ids:
                raise ValueError("Порог должен иметь уникальный непустой id")
            if sensor not in SENSORS:
                raise ValueError(f"Неподдерживаемый сигнал {sensor!r}")
            if item.get("direction") not in _DIRECTIONS:
                raise ValueError("direction должен быть 'above' или 'below'")
            if item.get("severity") not in _SEVERITIES:
                raise ValueError("severity должен быть 'warning' или 'critical'")
            if not isinstance(item.get("source"), str) or not item["source"].strip():
                raise ValueError("Укажите проверяемый источник ЧИСЛЕННОГО порога")
            if not isinstance(item.get("unit"), str) or not item["unit"].strip():
                raise ValueError("Укажите единицу измерения порога")
            threshold = _to_finite(item.get("threshold"))
            if threshold is None:
                raise ValueError("Порог должен быть конечным числом")
            item["threshold"] = threshold
            registered_ids.add(ident)
            self.thresholds.append(item)

    def predict(self, history: pd.DataFrame, *,
                decision_time: pd.Timestamp | None = None) -> list[dict[str, Any]]:
        """Предсказания по документированным порогам; no_forecast != безопасность."""
        if not self.thresholds:
            return []

        issue: str | None = None
        if not isinstance(history, pd.DataFrame) or not isinstance(history.index, pd.DatetimeIndex):
            issue = "Ожидался DataFrame с DatetimeIndex"
        elif len(history) < WINDOW_SECONDS:
            issue = f"Недостаточно истории: требуется {WINDOW_SECONDS} измерений"
        else:
            window = history.iloc[-WINDOW_SECONDS:]
            ticks = window.index.as_unit("ns").asi8
            if np.any(np.diff(ticks) != 1_000_000_000):
                issue = "Не соблюдён шаг измерений 1 секунда"
            elif decision_time is not None and pd.Timestamp(decision_time) != window.index[-1]:
                issue = "Последний момент измерений отличается от decision_time"

        if issue:
            return [self._result(cfg, "insufficient_data", reason=issue)
                    for cfg in self.thresholds]

        outputs: list[dict[str, Any]] = []
        for config in self.thresholds:
            sensor = config["sensor"]
            if sensor not in window.columns:
                outputs.append(self._result(config, "insufficient_data", reason="Датчик отсутствует", sensor_coverage=0.0))
                continue
            numeric = pd.to_numeric(window[sensor], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan)
            good = np.isfinite(numeric)
            coverage = float(good.mean())
            if coverage < MIN_COVERAGE or not good[-1]:
                outputs.append(self._result(
                    config, "insufficient_data", reason="Недостаточное покрытие или отсутствует текущее показание",
                    sensor_coverage=round(coverage, 4)))
                continue

            x = np.flatnonzero(good).astype(np.float64)  # секунды от начала окна
            y = numeric[good]
            xc = x - x.mean()
            yc = y - y.mean()
            denominator = float(xc @ xc)
            if denominator <= 0:
                outputs.append(self._result(config, "insufficient_data", reason="Невозможно вычислить тренд"))
                continue
            slope_per_sec = float((xc @ yc) / denominator)
            intercept = float(y.mean() - slope_per_sec * x.mean())
            yhat = slope_per_sec * x + intercept
            sst = float(yc @ yc)
            residual = float(np.sum((y - yhat) ** 2))
            trend_r2 = float(max(0.0, 1.0 - residual / sst)) if sst > 0 else None
            current = float(y[-1])
            threshold = config["threshold"]
            direction = config["direction"]
            base_extra = {
                "sensor_coverage": round(coverage, 4),
                "current_value": current,
                "trend_slope_per_hour": slope_per_sec * 3600,
                "trend_r2": trend_r2,
                "decision_time": window.index[-1].isoformat(),
            }
            already = current >= threshold if direction == "above" else current <= threshold
            if already:
                outputs.append(self._result(config, "already_exceeded", time_to_threshold_h=0.0,
                                            **base_extra))
                continue
            toward = slope_per_sec > 0 if direction == "above" else slope_per_sec < 0
            if not toward:
                outputs.append(self._result(config, "no_forecast", **base_extra))
                continue
            seconds_to_cross = float((threshold - current) / slope_per_sec)
            if 0 < seconds_to_cross <= self.horizon_minutes * 60:
                outputs.append(self._result(config, "predicted_crossing",
                                            time_to_threshold_h=seconds_to_cross / 3600,
                                            **base_extra))
            else:
                outputs.append(self._result(config, "no_forecast", **base_extra))
        return outputs

    def _result(self, cfg: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
        return {
            "agent": "prognostic", "forecast_status": status, "rule_id": cfg["id"],
            "sensor": cfg["sensor"], "severity": cfg["severity"],
            "threshold": cfg["threshold"], "direction": cfg["direction"],
            "unit": cfg["unit"], "source": cfg["source"],
            "horizon_minutes": self.horizon_minutes,
            "time_to_threshold_h": None,
            "recommended_action_by": None,
            "research_only": True,
            "note": "Только экстраполяция тренда до порога; НЕ время до отказа",
            **extra,
        }
