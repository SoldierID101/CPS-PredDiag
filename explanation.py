"""Explanation CPS-PredDiag: текстовое объяснение без БД и генеративного ИИ."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence


ACTION_RU = {
    "continue_operation": "продолжение эксплуатации по имеющимся диагностическим данным",
    "enhanced_monitoring": "усиленный мониторинг",
    "plan_maintenance": "планирование технического обслуживания",
    "immediate_attention": "немедленная проверка оборудования",
    "no_automatic_action": "автоматическая рекомендация не сформирована",
}

LEVEL_RU = {
    "normal": "По доступным экспертным данным состояние оценивается как нормальное.",
    "warning": "Выявлены признаки, требующие внимания.",
    "critical": "Выявлено критическое диагностическое состояние.",
    "unknown": "Текущее состояние не установлено: данных действующих диагностических агентов недостаточно.",
}


def _number(value: Any) -> str | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return f"{result:.3f}" if math.isfinite(result) else None


def generate_explanation(
    rule_summary: Mapping[str, Any] | None = None,
    predictions: Sequence[Mapping[str, Any]] | None = None,
    decision: Mapping[str, Any] | None = None,
    *,
    ml_result: Mapping[str, Any] | None = None,
) -> str:
    """Сохраняет совместимость со старой сигнатурой и поясняет ML-результат."""
    rules = rule_summary or {}
    conclusion = decision or {}
    parts: list[str] = [LEVEL_RU.get(conclusion.get("decision_level"), LEVEL_RU["unknown"])]

    rule_available = conclusion.get("rule_agent_available", rules.get("status") in {"normal", "warning", "critical"})
    if rule_available:
        critical = int(rules.get("critical_count") or 0)
        warning = int(rules.get("warning_count") or 0)
        parts.append(f"Экспертные правила: критических отклонений — {critical}, предупреждений — {warning}.")
        for result in rules.get("triggered_rules", []) or []:
            if not isinstance(result, Mapping):
                continue
            description = result.get("message")
            if description:
                parts.append(f"Правило: {description}.")
    else:
        parts.append("Rule-Based Agent пока не предоставил подтверждённого заключения.")

    prognosis = [p for p in (predictions or []) if isinstance(p, Mapping)]
    if conclusion.get("forecast_status") == "predicted_crossing":
        sensor = conclusion.get("nearest_sensor", "параметру")
        hours = _number(conclusion.get("nearest_crossing_h"))
        parts.append(
            f"Prognostic Agent: порог по {sensor} может быть достигнут через {hours} ч "
            "при сохранении текущего тренда. Это не оценка времени до отказа."
            if hours is not None else
            "Prognostic Agent: есть прогноз достижения порога, но время не определено."
        )
    elif conclusion.get("forecast_status") == "threshold_exceeded":
        parts.append("По доступной диагностике зафиксировано превышение порога.")
    elif not prognosis:
        parts.append("Prognostic Agent не предоставил результата; это не означает отсутствия будущих событий.")

    ml = ml_result if ml_result is not None else conclusion.get("ml_result")
    if isinstance(ml, Mapping):
        status = ml.get("status")
        if status == "experimental_prediction":
            score = _number(ml.get("model_score"))
            horizon = ml.get("horizon_minutes", 15)
            score_text = score if score is not None else "не определена"
            parts.append(
                f"ML Agent: исследовательская оценка = {score_text}, горизонт {horizon} мин, "
                "для случая подтверждённого переходного состояния 102. "
                "Оценка не калибрована и не является вероятностью физического отказа."
            )
            if conclusion.get("ml_research_alert"):
                parts.append("Оценка превышает экспериментальный порог; необходима проверка человеком.")
        elif status == "not_applicable":
            parts.append("ML Agent не применяется, пока состояние 102 не подтверждено независимым способом.")
        elif status in {"insufficient_data", "invalid_time"}:
            parts.append(f"ML Agent не выполнил прогноз: {ml.get('reason', status)}.")
    else:
        parts.append("Результат ML Agent не предоставлен.")

    action = conclusion.get("action", "no_automatic_action")
    parts.append(f"Решение Coordinator: {ACTION_RU.get(action, action)}.")
    parts.append("Исследовательский прототип; управляющие команды оборудованию не формируются.")
    return "\n\n".join(parts)
