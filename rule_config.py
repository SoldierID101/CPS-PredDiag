"""Загрузка утверждённых JSON-конфигураций для существующего RuleBasedAgent.

Не обучает правила, не устанавливает пороги и не использует разметку событий.
Флаг approval_status отражает решение исследователя, а не автоматическую
научную проверку. Поддерживаются только пороги одиночных измерений.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rule_based_agent import RuleBasedAgent


SCHEMA_VERSION = 1


def load_rule_config(path: str | Path, *, expected_event_class: int | None = None) -> dict[str, Any]:
    """Прочитать и проверить конфигурацию; пустой набор правил допустим."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл правил: {path}")

    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Некорректный JSON правил: {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Конфигурация правил должна быть JSON-объектом")

    allowed_keys = {
        "schema_version", "dataset", "event_class", "approval_status",
        "description", "rules",
    }
    extra_keys = set(config) - allowed_keys
    if extra_keys:
        raise ValueError(f"Неизвестные поля конфигурации: {sorted(extra_keys)}")

    if type(config.get("schema_version")) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Поддерживается только schema_version={SCHEMA_VERSION}")
    if not isinstance(config.get("dataset"), str) or not config["dataset"].strip():
        raise ValueError("Необходимо непустое поле dataset")
    event_class = config.get("event_class")
    if type(event_class) is not int or event_class < 1:
        raise ValueError("event_class должен быть положительным целым числом")
    if expected_event_class is not None and event_class != expected_event_class:
        raise ValueError(f"Ожидался event_class={expected_event_class}, получен {event_class}")

    status = config.get("approval_status")
    if not isinstance(status, str) or status not in {"draft", "approved"}:
        raise ValueError("approval_status должен быть 'draft' или 'approved'")
    rules = config.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules должен быть JSON-массивом")
    if rules and status != "approved":
        raise ValueError("Нельзя загружать исполняемые правила из неутверждённой конфигурации")

    # Используем уже существующую проверку порогов, источников и запрета class/state.
    RuleBasedAgent(rules)
    return config


def load_rule_agent(path: str | Path, *, expected_event_class: int | None = None) -> RuleBasedAgent:
    """Создать RuleBasedAgent без изменения его текущего интерфейса."""
    config = load_rule_config(path, expected_event_class=expected_event_class)
    return RuleBasedAgent(config["rules"])
