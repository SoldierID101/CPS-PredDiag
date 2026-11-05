"""CPS-PredDiag: Rule-Based Agent v0.1 (без БД).

Только правила с внешне обоснованными порогами для текущих значений датчиков.
В проект НЕ включены произвольные пороги Petrobras 3W.

Контракт совместим с coordinator.coordinate_decision(rule_summary=...).
НЕ определяет метку 102, НЕ читает class/state, НЕ отправляет команды.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


_OPERATORS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}
_SEVERITIES = {"warning", "critical"}
_FORBIDDEN_SENSORS = {"class", "state", "timestamp", "well_id", "file", "decision_time"}


class RuleBasedAgent:
    """Принимает только явно заданные правила с указанным источником.

    Каждый элемент rules имеет структуру::

        {"id": "...", "sensor": "P-TPT", "condition": ">",
         "threshold": ..., "severity": "warning", "unit": "Pa",
         "source": "проверенный источник порога", "message": "..."}

    Значение threshold необходимо обосновать до использования на реальных
    данных. Пустой список правил возвращает статус 'unknown'.
    """

    def __init__(self, rules: Sequence[Mapping[str, Any]] | None = None):
        prepared: list[dict[str, Any]] = []
        ids: set[str] = set()

        for raw in (rules or []):
            if not isinstance(raw, Mapping):
                raise ValueError("Каждое правило должно быть словарём")
            item = dict(raw)
            rid = item.get("id")
            sensor = item.get("sensor")
            severity = item.get("severity")
            condition = item.get("condition")
            unit = item.get("unit")
            source = item.get("source")
            message = item.get("message")
            if not isinstance(rid, str) or not rid.strip() or rid in ids:
                raise ValueError("Правила должны иметь уникальный непустой id")
            if not isinstance(sensor, str) or not sensor.strip() or sensor in _FORBIDDEN_SENSORS:
                raise ValueError(f"Недопустимое имя измеряемого датчика: {sensor!r}")
            if severity not in _SEVERITIES or condition not in _OPERATORS:
                raise ValueError(f"Недопустимое условие/критичность в правиле {rid}")
            if not all(isinstance(v, str) and v.strip() for v in (unit, source, message)):
                raise ValueError(f"Укажите единицу измерения, источник и описание правила {rid}")
            threshold = item.get("threshold")
            if isinstance(threshold, bool):
                raise ValueError(f"Порог правила {rid} должен быть числом")
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Порог правила {rid} должен быть числом") from exc
            if not math.isfinite(threshold):
                raise ValueError(f"Порог правила {rid} должен быть конечным числом")
            item["threshold"] = threshold
            ids.add(rid)
            prepared.append(item)

        self.rules = tuple(prepared)

    def evaluate(self, snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
        """Оценка только показаний на текущий момент, без будущей разметки.

        При отсутствии/неполноте измерений запрещено выдавать NORMAL.
        Найденное превышение можно отразить, даже если другие датчики отсутствуют.
        """
        if snapshot is not None and not isinstance(snapshot, Mapping):
            raise TypeError("snapshot должен быть словарём значений датчиков")
        data = snapshot or {}
        triggered: list[dict[str, Any]] = []
        unavailable: list[str] = []
        checked = 0

        for rule in self.rules:
            sensor = rule["sensor"]
            raw = data.get(sensor)
            value: float | None = None
            if not isinstance(raw, bool):
                try:
                    candidate = float(raw)
                    if math.isfinite(candidate):
                        value = candidate
                except (TypeError, ValueError):
                    pass
            if value is None:
                unavailable.append(rule["id"])
                continue

            checked += 1
            if _OPERATORS[rule["condition"]](value, rule["threshold"]):
                triggered.append({
                    "rule_id": rule["id"],
                    "sensor": sensor,
                    "sensor_value": value,
                    "threshold": rule["threshold"],
                    "condition": rule["condition"],
                    "severity": rule["severity"],
                    "unit": rule["unit"],
                    "source": rule["source"],
                    "message": rule["message"],
                })

        critical_count = sum(x["severity"] == "critical" for x in triggered)
        warning_count = sum(x["severity"] == "warning" for x in triggered)
        if critical_count:
            status = "critical"
        elif warning_count:
            status = "warning"
        elif not self.rules or unavailable:
            status = "unknown"
        else:
            status = "normal"

        return {
            "agent": "rule_based",
            "status": status,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "triggered_rules": triggered,
            "configured_rules": len(self.rules),
            "checked_rules": checked,
            "unavailable_rule_ids": unavailable,
            "complete": bool(self.rules) and not unavailable,
            "note": "Пороги требуют подтверждённого физического обоснования для Petrobras 3W",
        }
