"""Coordinator CPS-PredDiag: независимый от БД, совместимый с ML Agent v0.1.

Различает диагностические решения проверяемых агентов и исследовательский
сигнал ML. Не объединяет непроверенные оценки в 'вероятность отказа'.
"""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence


ACTION_CONTINUE = "continue_operation"
ACTION_MONITOR = "enhanced_monitoring"
ACTION_PLAN_MAINTENANCE = "plan_maintenance"
ACTION_IMMEDIATE = "immediate_attention"
ACTION_UNDETERMINED = "no_automatic_action"

VALID_RULE_STATES = {"normal", "warning", "critical"}


def _valid_crossing(prediction: Mapping[str, Any]) -> float | None:
    try:
        hours = float(prediction.get("time_to_threshold_h"))
    except (TypeError, ValueError):
        return None
    return hours if math.isfinite(hours) and hours >= 0 else None


def coordinate_decision(
    rule_summary: Mapping[str, Any] | None = None,
    predictions: Sequence[Mapping[str, Any]] | None = None,
    *,
    ml_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Объединяет заключения, не выполняя I/O и не отправляя управляющих команд.

    rule_summary: результат Rule-Based Agent либо None, если агент ещё недоступен.
    predictions: прогнозы пересечения физических порогов от Prognostic Agent.
    ml_result: результат MLAgent.predict (отдельный исследовательский сигнал).

    При отсутствии достоверного заключения итог UNKNOWN, а НЕ NORMAL.
    Оценка ML никогда сама не повышает уровень до WARNING/CRITICAL.
    """
    rules = rule_summary or {}
    state = rules.get("status")
    current_status = state if state in VALID_RULE_STATES else "unknown"
    rule_available = current_status != "unknown"

    prognosis = [p for p in (predictions or []) if isinstance(p, Mapping)]
    exceeded = [p for p in prognosis if p.get("forecast_status") == "already_exceeded"]
    critical_exceeded = [p for p in exceeded if p.get("severity") == "critical"]
    warning_exceeded = [p for p in exceeded if p.get("severity") == "warning"]
    future = [
        (p, h)
        for p in prognosis
        if p.get("forecast_status") == "predicted_crossing"
        if (h := _valid_crossing(p)) is not None
    ]

    ml = dict(ml_result) if isinstance(ml_result, Mapping) else None
    ml_status = ml.get("status", "unavailable") if ml is not None else "unavailable"
    ml_signal = (
        ml_status == "experimental_prediction"
        and ml is not None
        and ml.get("above_research_threshold") is True
    )
    reasons: list[str] = []
    nearest_sensor = None
    nearest_crossing_h = None
    recommended_action_by = None

    if current_status == "critical" or critical_exceeded:
        level, action = "critical", ACTION_IMMEDIATE
        forecast_status = "threshold_exceeded" if critical_exceeded else "no_forecast"
        if current_status == "critical":
            reasons.append("Экспертный агент зарегистрировал критическое состояние.")
        if critical_exceeded:
            reasons.append(f"Превышено критических диагностических порогов: {len(critical_exceeded)}.")
    elif future:
        nearest, nearest_crossing_h = min(future, key=lambda item: item[1])
        nearest_sensor = nearest.get("sensor")
        recommended_action_by = nearest.get("recommended_action_by")
        severity = nearest.get("severity", "warning")
        level = "warning"
        action = ACTION_PLAN_MAINTENANCE if severity == "critical" else ACTION_MONITOR
        forecast_status = "predicted_crossing"
        reasons.append(
            f"Прогнозируется достижение порога ({severity}) по {nearest_sensor} "
            f"через {nearest_crossing_h:.2f} ч; это НЕ прогноз времени отказа."
        )
        if current_status == "warning":
            reasons.append("Экспертный агент также зарегистрировал предупреждение.")
    elif current_status == "warning" or warning_exceeded:
        level, action = "warning", ACTION_MONITOR
        forecast_status = "threshold_exceeded" if warning_exceeded else "no_forecast"
        if current_status == "warning":
            reasons.append("Экспертный агент зарегистрировал предупреждающее отклонение.")
        if warning_exceeded:
            reasons.append(f"Превышено предупреждающих порогов: {len(warning_exceeded)}.")
    elif current_status == "normal":
        level, action, forecast_status = "normal", ACTION_CONTINUE, "no_forecast"
        reasons.append("По имеющимся экспертным правилам текущее состояние нормально.")
        if not prognosis:
            reasons.append("Прогностический агент не предоставил результата; отсутствие прогноза не гарантирует безопасность.")
    else:
        level, action, forecast_status = "unknown", ACTION_UNDETERMINED, "unavailable"
        reasons.append("Недостаточно проверенных диагностических данных для оценки текущего состояния.")

    if ml_signal:
        reasons.append(
            "ML Agent выдал исследовательский сигнал выше порога 0.5; "
            "оценка не калибрована и не меняет автоматически решение об эксплуатации."
        )
    elif ml_status == "experimental_prediction":
        reasons.append("ML Agent выполнил экспериментальный прогноз ниже порога 0.5.")
    elif ml_status == "not_applicable":
        reasons.append("ML Agent не применён: переходное состояние 102 не подтверждено.")
    elif ml_status in {"insufficient_data", "invalid_time"}:
        reasons.append(f"ML Agent не смог выполнить прогноз: {ml_status}.")

    return {
        "decision_level": level,
        "action": action,
        "current_status": current_status,
        "rule_agent_available": rule_available,
        "forecast_status": forecast_status,
        "nearest_sensor": nearest_sensor,
        "nearest_crossing_h": nearest_crossing_h,
        "recommended_action_by": recommended_action_by,
        "ml_status": ml_status,
        "ml_research_alert": bool(ml_signal),
        "ml_result": ml,
        "reasons": reasons,
        "research_only": True,
    }
