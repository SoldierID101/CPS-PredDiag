"""Запуск CPS-PredDiag на одном историческом срезе Parquet Petrobras.

Из корня проекта:
    python experiments/run_pipeline_once.py --file dataset/2/WELL-00002_20131104004101.parquet
    python experiments/run_pipeline_once.py --file dataset/2/WELL-00002_20131104004101.parquet --mode oracle_replay

Требуется установленный pyarrow (или fastparquet), обученный доверенный
артефакт outputs/models/class2_prototype, rules/class2_dhsv.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic_pipeline import DiagnosticPipeline  # noqa: E402
from ml_agent import MLAgent  # noqa: E402
from prognostic_agent import PrognosticAgent  # noqa: E402
from rule_config import load_rule_agent  # noqa: E402


def first_complete_102_time(frame: pd.DataFrame) -> pd.Timestamp:
    """Первый момент с 300 непрерывными прошлыми секундами class=102."""
    if "class" not in frame:
        raise ValueError("В Parquet отсутствует разметка class")
    index = frame.index.as_unit("ns")
    # Nullable integer/Arrow-backed columns can contain pd.NA. Missing labels
    # cannot confirm 102; normalize to a real bool array before iterating.
    labels = (pd.to_numeric(frame["class"], errors="coerce")
              .eq(102).fillna(False).to_numpy(dtype=bool, na_value=False))
    ticks = index.asi8
    # Каждый True подтверждает, что со времени предыдущего измерения прошла 1с.
    step_ok = np.r_[False, np.diff(ticks) == 1_000_000_000]
    run_length = 0
    for i, label_102 in enumerate(labels):
        if label_102:
            run_length = (run_length + 1) if (i > 0 and step_ok[i]) else 1
            if run_length >= 300:
                return index[i]
        else:
            run_length = 0
    raise ValueError("В файле нет непрерывного 300-секундного окна перехода 102")


def main() -> None:
    parser = argparse.ArgumentParser(description="Один исследовательский запуск CPS-PredDiag на Petrobras")
    parser.add_argument("--file", required=True, type=Path, help="Путь к Parquet эпизода класса 2")
    parser.add_argument("--mode", choices=["blind", "oracle_replay"], default="blind")
    parser.add_argument("--decision-time", default=None, help="Точное время измерения; иначе последнее, либо первое окно 102 для oracle")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs/models/class2_prototype")
    parser.add_argument("--rules", type=Path, default=ROOT / "rules/class2_dhsv.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pipeline_runs/one_window.json")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"Parquet не найден: {args.file}")
    try:
        frame = pd.read_parquet(args.file)
    except (ImportError, OSError) as exc:
        parser.error(f"Не удалось прочитать Parquet: {exc}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        parser.error("Ожидается DatetimeIndex; структура файла не соответствует ожидаемой")
    if args.decision_time:
        moment = pd.Timestamp(args.decision_time)
    elif args.mode == "oracle_replay":
        moment = first_complete_102_time(frame)
    else:
        moment = frame.index[-1]

    # Намеренно НЕ загружаем и НЕ настраиваем неутверждённые правила/пороги.
    rule_agent = load_rule_agent(args.rules, expected_event_class=2)
    prognostic_agent = PrognosticAgent()
    ml_agent = MLAgent(args.model_dir)  # только доверенные локальные .joblib
    pipeline = DiagnosticPipeline(rule_agent=rule_agent,
                                  prognostic_agent=prognostic_agent,
                                  ml_agent=ml_agent)
    result = pipeline.run(frame, decision_time=moment, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    print(f"Файл: {args.file.name}")
    print(f"Момент: {result['decision_time']}")
    print(f"Режим: {result['mode']}")
    print(f"Oracle 102: {result['transition_evidence']['confirmed_102']}")
    print(f"Rule-Based: {result['agents']['rule_based']['status']}")
    print(f"Prognostic: {len(result['agents']['prognostic'])} результатов")
    print(f"ML: {result['agents']['ml']['status']}")
    print(f"Coordinator: {result['decision']['decision_level']}")
    print(f"Результат: {args.output}")
    print("ВНИМАНИЕ: демонстрация связности системы, не оценка точности и не система управления.")


if __name__ == "__main__":
    main()
