"""Единый исследовательский запуск CPS-PredDiag без БД и управляющих команд.

Не изменяет существующие классы агентов. В режиме blind разметка Petrobras
никогда не подтверждает transition=102. В режиме oracle_replay разрешается
использовать только ПРОШЛУЮ разметку для демонстрации подключения ML Agent.
Это не независимая оценка качества и не реальная онлайн-диагностика.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from coordinator import coordinate_decision
from explanation import generate_explanation
from rule_based_agent import RuleBasedAgent
from prognostic_agent import PrognosticAgent

Mode = Literal["blind", "oracle_replay"]
# Эти поля размечают событие или результаты эксперимента и НЕ являются телеметрией.
_LABEL_FIELDS = frozenset({"class", "state", "target", "target_h15", "event_class", "label"})


class DiagnosticPipeline:
    """Оркестрирует агентов. ml_agent должен предоставлять predict(...).

    Для исторического воспроизведения передаётся весь префикс до момента
    решения или больший DataFrame с явным decision_time. Будущие строки
    отсекаются ДО вызовов любого агента.
    """

    def __init__(self, *, rule_agent: RuleBasedAgent,
                 prognostic_agent: PrognosticAgent,
                 ml_agent: Any) -> None:
        if not isinstance(rule_agent, RuleBasedAgent):
            raise TypeError("rule_agent должен быть RuleBasedAgent")
        if not isinstance(prognostic_agent, PrognosticAgent):
            raise TypeError("prognostic_agent должен быть PrognosticAgent")
        if ml_agent is None or not callable(getattr(ml_agent, "predict", None)):
            raise TypeError("ml_agent должен иметь метод predict")
        self.rule_agent = rule_agent
        self.prognostic_agent = prognostic_agent
        self.ml_agent = ml_agent

    def run(self, history: pd.DataFrame, *,
            decision_time: pd.Timestamp | None = None,
            mode: Mode = "blind") -> dict[str, Any]:
        """Проводит диагностику одной точки времени.

        oracle_replay использует ТОЛЬКО историю class==102 в текущих 300
        секундах; передача истинной разметки внутрь агентов запрещена.
        При пустых правилах и порогах Coordinator возвращает unknown.
        """
        if mode not in {"blind", "oracle_replay"}:
            raise ValueError("mode должен быть 'blind' или 'oracle_replay'")
        if not isinstance(history, pd.DataFrame) or not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("Ожидается DataFrame с DatetimeIndex")
        if history.empty:
            raise ValueError("История пуста")
        if history.index.has_duplicates or not history.index.is_monotonic_increasing:
            raise ValueError("Метки времени должны быть уникальными и возрастающими")
        if history.index.hasnans:
            raise ValueError("Метки времени не могут содержать NaT")

        moment = pd.Timestamp(decision_time) if decision_time is not None else history.index[-1]
        if moment not in history.index:
            raise ValueError("decision_time должен присутствовать в истории")
        past = history.loc[:moment]
        # В pandas 3.x встречается datetime64[us]. Приведение к ns обеспечивает
        # совместимость старого MLAgent.features_from_history с шагом 1 секунда.
        past = past.copy()
        past.index = past.index.as_unit("ns")
        moment = past.index[-1]
        # Передаём агентам только историю, завершённую текущим моментом.
        observed = past.drop(columns=[col for col in past if str(col).lower() in _LABEL_FIELDS])
        snapshot = observed.iloc[-1].to_dict()

        # Oracle-флаг разрешён только явно и при непрерывных 300 метках класса 102.
        oracle_confirmed = False
        if mode == "oracle_replay":
            if "class" not in past.columns:
                raise ValueError("Для oracle_replay требуется столбец class")
            tail = past.iloc[-300:]
            if len(tail) == 300:
                tick = tail.index.asi8
                contiguous = bool(np.all(np.diff(tick) == 1_000_000_000))
                labels = pd.to_numeric(tail["class"], errors="coerce")
                # Series.all() skips missing values by default, which could
                # falsely confirm an incomplete 102 segment.
                oracle_confirmed = contiguous and bool(labels.eq(102).fillna(False).all())

        rules = self.rule_agent.evaluate(snapshot)
        forecasts = self.prognostic_agent.predict(observed, decision_time=moment)
        ml = self.ml_agent.predict(
            observed,
            transition_102_confirmed=oracle_confirmed if mode == "oracle_replay" else None,
            decision_time=moment,
        )
        decision = coordinate_decision(rule_summary=rules, predictions=forecasts,
                                       ml_result=ml)
        explanation = generate_explanation(rule_summary=rules, predictions=forecasts,
                                           decision=decision, ml_result=ml)
        return {
            "pipeline_version": "0.1",
            "research_only": True,
            "mode": mode,
            "decision_time": moment.isoformat(),
            "transition_evidence": {
                "source": "oracle_past_labels" if mode == "oracle_replay" else "unavailable",
                "confirmed_102": oracle_confirmed if mode == "oracle_replay" else False,
                "note": ("Разметка использована только для офлайн-демонстрации, "
                         "не для независимой оценки качества") if mode == "oracle_replay"
                        else "Независимый детектор 102 пока не разработан",
            },
            "agents": {"rule_based": rules, "prognostic": forecasts, "ml": ml},
            "decision": decision,
            "explanation": explanation,
        }
