"""Только исследовательский shadow ML Agent (класс 2, 30 минут).

Не подключен к Coordinator или Streamlit. Не управляет оборудованием.
Модель передается явно и должна быть загружена из доверенного источника;
никакой автоматической загрузки pickle/joblib в этом модуле нет.

Точный смысл выходного `model_score`: НЕКАЛИБРОВАННАЯ оценка модели.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_agent import FEATURES, SENSORS, MIN_COVERAGE, features_from_history

class InvalidHistoryTime(ValueError):
    """Временная ось не соответствует моменту принятия решения."""


REQUIRED_HISTORY_SECONDS = 900
STRIDE_SECONDS = 300
MODEL_EVENT_CLASS = 2
HORIZON_MINUTES = 30


@dataclass(frozen=True)
class ShadowConfig:
    """Настройки только для исследовательской проверки, не для эксплуатации."""

    model_version: str
    threshold: float
    confirmations: int
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("Нужна непустая версия модели")
        if not isinstance(self.threshold, (float, int)) or not np.isfinite(self.threshold):
            raise ValueError("Порог должен быть конечным числом")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("Порог должен лежать в диапазоне (0, 1]")
        if self.confirmations not in (1, 2):
            raise ValueError("Поддерживаются 1 или 2 последовательных подтверждения")
        if self.artifact_sha256 is not None:
            s = self.artifact_sha256
            if not isinstance(s, str) or len(s) != 64 or any(c not in "0123456789abcdefABCDEF" for c in s):
                raise ValueError("artifact_sha256 должен быть SHA256 в hex; модуль его не проверяет")


@dataclass
class _WellState:
    last_decision: pd.Timestamp | None = None
    consecutive: int = 0
    active: bool = False
    last_result: dict[str, Any] | None = None


class ShadowMLAgentV02:
    """Отдельная исследовательская модель; состояние предупреждений изолировано по WELL.

    `model` предоставляет `classes_ == [0, 1]` и `predict_proba(pd.DataFrame)`.
    Чтобы не объявлять произвольный вариант производственным, настройки политики
    всегда должны быть переданы явно.
    """

    def __init__(self, *, model: Any | None, config: ShadowConfig):
        self.model = model
        self.config = config
        self._wells: dict[str, _WellState] = {}

    def reset(self, well_id: str | None = None) -> None:
        """Начать новый независимый replay; не переносить тревоги между файлами."""
        if well_id is None:
            self._wells.clear()
        else:
            self._wells.pop(well_id, None)

    def _base(self, well_id: str, time: pd.Timestamp) -> dict[str, Any]:
        return {
            "agent": "ml",
            "protocol_version": "0.2",
            "model_version": self.config.model_version,
            "artifact_sha256": self.config.artifact_sha256,
            "well_id": well_id,
            "decision_time": time.isoformat(),
            "event_class": MODEL_EVENT_CLASS,
            "horizon_minutes": HORIZON_MINUTES,
            "model_score": None,
            "uncalibrated": True,
            "sensor_coverage": None,
            "threshold": float(self.config.threshold),
            "confirmations": self.config.confirmations,
            "alarm_active": None,
            "alarm_new_start": None,
            "reason": None,
            "shadow_mode": True,
        }

    @staticmethod
    def _history_for_t(history: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
        if not isinstance(history, pd.DataFrame) or not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("Нужен DataFrame с DatetimeIndex")
        index = history.index
        if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            raise InvalidHistoryTime("Некорректные временные метки")
        if index[-1] != t:
            raise InvalidHistoryTime("История должна заканчиваться в decision_time (без будущего)")
        if len(history) < REQUIRED_HISTORY_SECONDS:
            raise ValueError("Для допуска нужны 900 последовательных секунд")
        past = history.iloc[-REQUIRED_HISTORY_SECONDS:]
        if not np.all(np.diff(past.index.as_unit("ns").asi8) == 1_000_000_000):
            raise InvalidHistoryTime("Разрыв во входной 900-секундной истории")
        # Поля class/state/future и идентификаторы скважин сюда не проходят.
        return past.reindex(columns=SENSORS)

    def _record_failure(self, base: dict[str, Any], st: _WellState,
                        t: pd.Timestamp, status: str, reason: str) -> dict[str, Any]:
        st.last_decision = t
        st.consecutive = 0
        st.active = False
        result = {**base, "status": status, "reason": reason}
        st.last_result = deepcopy(result)
        return result

    def predict(self, history: pd.DataFrame, *, well_id: str,
                decision_time: pd.Timestamp) -> dict[str, Any]:
        if not isinstance(well_id, str) or not well_id.strip():
            raise ValueError("well_id обязателен для изоляции состояния")
        try:
            t = pd.Timestamp(decision_time)
        except Exception as exc:
            raise ValueError("Некорректное decision_time") from exc
        if pd.isna(t):
            raise ValueError("decision_time не может быть NaT")
        st = self._wells.setdefault(well_id, _WellState())
        base = self._base(well_id, t)

        if st.last_decision is not None:
            if t < st.last_decision:
                # Старое решение не должно повреждать текущее состояние.
                return {**base, "status": "invalid_time", "reason": "Время решения раньше уже обработанного"}
            if t == st.last_decision:
                return deepcopy(st.last_result)

        try:
            past = self._history_for_t(history, t)
            # Legacy features_from_history() checks DatetimeIndex.asi8 against
            # nanoseconds. Under pandas 3.x, a DatetimeIndex may use
            # microsecond storage (datetime64[us]); its .asi8 then represents
            # microseconds rather than nanoseconds. Normalize only the local
            # 300-row feature window, keeping the caller's history untouched.
            sensor_window = past.iloc[-300:].copy()
            sensor_window.index = sensor_window.index.as_unit("ns")
            features, coverage = features_from_history(sensor_window)
        except InvalidHistoryTime as exc:
            return self._record_failure(base, st, t, "invalid_time", str(exc))
        except (ValueError, TypeError, IndexError) as exc:
            return self._record_failure(base, st, t, "insufficient_data", str(exc))
        if list(features.columns) != FEATURES:
            return self._record_failure(base, st, t, "model_error", "Схема признаков не совпадает")
        base["sensor_coverage"] = coverage
        if sum(coverage[s] >= MIN_COVERAGE for s in SENSORS) < 2:
            return self._record_failure(base, st, t, "insufficient_data", "Доступно менее двух сенсоров")
        if self.model is None:
            return self._record_failure(base, st, t, "model_unavailable", "Модель не предоставлена")
        try:
            if list(getattr(self.model, "classes_", [])) != [0, 1]:
                raise ValueError("Неверные классы модели")
            probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
            if probabilities.shape != (1, 2):
                raise ValueError("Некорректная форма predict_proba")
            score = float(probabilities[0, 1])
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Некорректный model_score")
        except Exception:
            return self._record_failure(base, st, t, "model_error", "Не удалось получить допустимую оценку модели")

        if st.last_decision is not None and (t - st.last_decision) != pd.Timedelta(seconds=STRIDE_SECONDS):
            st.consecutive = 0
            st.active = False
        was_active = st.active
        if score >= self.config.threshold:
            st.consecutive += 1
            st.active = st.consecutive >= self.config.confirmations
        else:
            st.consecutive = 0
            st.active = False

        result = {
            **base, "status": "experimental_prediction", "model_score": score,
            "alarm_active": st.active, "alarm_new_start": st.active and not was_active,
        }
        st.last_decision = t
        st.last_result = deepcopy(result)
        return result
