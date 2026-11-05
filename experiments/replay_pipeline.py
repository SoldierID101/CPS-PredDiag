"""Последовательное воспроизведение одного файла Petrobras 3W.

Задача: функциональная проверка DiagnosticPipeline на реальной истории,
а НЕ независимая оценка диагностического качества или управление скважиной.

Примеры (из корня CPS-PredDiag):
    python experiments/replay_pipeline.py --file dataset/2/WELL-00002_20131104004101.parquet
    python experiments/replay_pipeline.py --file dataset/2/WELL-00002_20131104004101.parquet --mode oracle_replay

Даты и метки в будущем запрещены для каждого вызова pipeline.run.
Промежутки времени (> 1 с) сбрасывают набор непрерывного окна.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic_pipeline import DiagnosticPipeline  # noqa: E402
from ml_agent import MLAgent  # noqa: E402
from prognostic_agent import PrognosticAgent  # noqa: E402
from rule_config import load_rule_agent  # noqa: E402

WINDOW_SECONDS = 300
DEFAULT_STRIDE_SECONDS = 300


def iter_decision_positions(
    frame: pd.DataFrame, *, stride_seconds: int = DEFAULT_STRIDE_SECONDS,
) -> Iterator[int]:
    """Индексы решений после 300 непрерывных секунд; без обращения к class.

    Первое решение происходит на 300-м измерении непрерывного сегмента.
    Следующие — не ранее чем через stride_seconds от предыдущего решения.
    Если ряд прерывается, окно начинается заново после разрыва.
    """
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Нужен DataFrame с DatetimeIndex")
    if type(stride_seconds) is not int or stride_seconds < 1:
        raise ValueError("stride_seconds должен быть положительным целым числом")
    index = frame.index
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Временной индекс должен быть уникальным, без NaT и возрастающим")
    ticks = index.as_unit("ns").asi8
    contiguous_run = 0
    last_decision_tick: int | None = None
    for i, tick in enumerate(ticks):
        if i == 0 or tick - ticks[i - 1] != 1_000_000_000:
            contiguous_run = 1
        else:
            contiguous_run += 1
        if contiguous_run < WINDOW_SECONDS:
            continue
        if last_decision_tick is None or tick - last_decision_tick >= stride_seconds * 1_000_000_000:
            yield i
            last_decision_tick = int(tick)


def replay_pipeline(
    frame: pd.DataFrame, *, pipeline: Any, mode: str = "blind",
    stride_seconds: int = DEFAULT_STRIDE_SECONDS, max_points: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Возвращает результаты последовательных вызовов Pipeline.

    Передаём ТОЛЬКО прошлый префикс, не весь файл с будущими данными.
    В blind-режиме класс не влияет ни на выбор времени, ни на подтверждение 102.
    """
    if mode not in {"blind", "oracle_replay"}:
        raise ValueError("mode должен быть blind или oracle_replay")
    if max_points is not None and (type(max_points) is not int or max_points < 1):
        raise ValueError("max_points должен быть положительным целым числом")
    if mode == "oracle_replay" and "class" not in frame.columns:
        raise ValueError("Для oracle_replay необходим столбец class")
    count = 0
    for pos in iter_decision_positions(frame, stride_seconds=stride_seconds):
        # Принципиально передаём срез ДО решения включительно.
        yield pipeline.run(frame.iloc[:pos + 1], mode=mode)
        count += 1
        if max_points is not None and count >= max_points:
            break


def timeline_row(result: dict[str, Any]) -> dict[str, Any]:
    agents = result["agents"]
    ml = agents["ml"]
    decision = result["decision"]
    coverage = ml.get("sensor_coverage") or {}
    return {
        "decision_time": result["decision_time"],
        "mode": result["mode"],
        "oracle_confirmed_102": bool(result["transition_evidence"]["confirmed_102"]),
        "rule_status": agents["rule_based"].get("status", "unknown"),
        "prognostic_results": len(agents["prognostic"]),
        "ml_status": ml.get("status"),
        "ml_score_uncalibrated": ml.get("model_score"),
        "ml_above_research_threshold": ml.get("above_research_threshold"),
        "coverage_P_MON_CKP": coverage.get("P-MON-CKP"),
        "coverage_P_TPT": coverage.get("P-TPT"),
        "coverage_T_TPT": coverage.get("T-TPT"),
        "decision_level": decision.get("decision_level"),
        "action": decision.get("action"),
    }


def write_replay_outputs(
    results: Iterator[dict[str, Any]], output_dir: Path | str, *,
    source_filename: str, mode: str, stride_seconds: int,
) -> dict[str, Any]:
    """Пишет JSONL (полные заключения), CSV (временная шкала) и summary JSON.

    Не вычисляет Precision/Recall по oracle: выводы не являются валидацией.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    jsonl = target / "decisions.jsonl"
    csv_path = target / "timeline.csv"
    summary_path = target / "summary.json"
    statuses: Counter[str] = Counter()
    ml_statuses: Counter[str] = Counter()
    oracle_confirmed = 0
    research_alerts = 0
    count = 0
    fields: list[str] | None = None
    with jsonl.open("w", encoding="utf-8") as stream, csv_path.open("w", newline="", encoding="utf-8-sig") as table_file:
        writer = None
        for result in results:
            # JSONL позволяет читать / дополнять последовательные решения без БД.
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            row = timeline_row(result)
            if writer is None:
                fields = list(row)
                writer = csv.DictWriter(table_file, fieldnames=fields)
                writer.writeheader()
            writer.writerow(row)
            count += 1
            statuses[str(row["decision_level"])] += 1
            ml_statuses[str(row["ml_status"])] += 1
            oracle_confirmed += int(row["oracle_confirmed_102"])
            research_alerts += int(row["ml_above_research_threshold"] is True)
    if count == 0:
        jsonl.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise ValueError("В файле нет ни одного полноценного 300-секундного участка")
    summary: dict[str, Any] = {
        "pipeline_replay_version": "0.1",
        "research_only": True,
        "source_filename": source_filename,
        "mode": mode,
        "decision_stride_seconds": stride_seconds,
        "window_seconds": WINDOW_SECONDS,
        "decisions_count": count,
        "decision_levels": dict(statuses),
        "ml_statuses": dict(ml_statuses),
        "oracle_confirmed_102_count": oracle_confirmed,
        "ml_research_alerts_count": research_alerts,
        "warning": (
            "Это демонстрация связности на последовательных исторических данных, "
            "НЕ независимое измерение точности диагностики. "
            "ML оценки не калиброваны; oracle использует истинную разметку."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Историческое воспроизведение CPS-PredDiag (одна скважина)")
    p.add_argument("--file", type=Path, required=True, help="Parquet-файл события класса 2")
    p.add_argument("--mode", choices=["blind", "oracle_replay"], default="blind")
    p.add_argument("--stride-seconds", type=int, default=DEFAULT_STRIDE_SECONDS)
    p.add_argument("--max-points", type=int, default=None, help="Ограничение точек для быстрой проверки")
    p.add_argument("--model-dir", type=Path, default=ROOT / "outputs/models/class2_prototype")
    p.add_argument("--rules", type=Path, default=ROOT / "rules/class2_dhsv.json")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.file.is_file():
        p.error(f"Parquet не найден: {args.file}")
    if type(args.stride_seconds) is not int or args.stride_seconds < 1:
        p.error("--stride-seconds должен быть положительным числом")
    if args.max_points is not None and args.max_points < 1:
        p.error("--max-points должен быть положительным числом")
    output_dir = args.output_dir or ROOT / "outputs/pipeline_replay" / args.mode
    try:
        frame = pd.read_parquet(args.file)
        pipeline = DiagnosticPipeline(
            rule_agent=load_rule_agent(args.rules, expected_event_class=2),
            prognostic_agent=PrognosticAgent(),
            ml_agent=MLAgent(args.model_dir),
        )
        summary = write_replay_outputs(
            replay_pipeline(frame, pipeline=pipeline, mode=args.mode,
                            stride_seconds=args.stride_seconds, max_points=args.max_points),
            output_dir,
            source_filename=args.file.name, mode=args.mode,
            stride_seconds=args.stride_seconds,
        )
    except (ValueError, OSError, ImportError, FileNotFoundError) as exc:
        p.error(str(exc))
    print("Файл:", summary["source_filename"])
    print("Режим:", summary["mode"])
    print("Точек диагностики:", summary["decisions_count"])
    print("Результаты Coordinator:", summary["decision_levels"])
    print("Статусы ML:", summary["ml_statuses"])
    print("Экспериментальных ML-предупреждений:", summary["ml_research_alerts_count"])
    print("Oracle-подтверждений 102:", summary["oracle_confirmed_102_count"])
    print("Отчёты:", output_dir)
    print("ВНИМАНИЕ: replay НЕ является независимой валидацией качества диагностики.")


if __name__ == "__main__":
    main()
