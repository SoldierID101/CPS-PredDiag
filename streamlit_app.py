"""CPS-PredDiag — исследовательский Streamlit-интерфейс v0.1.

Запуск из корня проекта: streamlit run streamlit_app.py
Зависит от уже существующих диагностических модулей; старой БД нет.
Это просмотр исторического воспроизведения, не система мониторинга/управления.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnostic_pipeline import DiagnosticPipeline
from experiments.replay_pipeline import replay_pipeline, timeline_row, write_replay_outputs
from ml_agent import MLAgent
from prognostic_agent import PrognosticAgent
from rule_config import load_rule_agent

ROOT = Path(__file__).resolve().parent
DATASET_CLASS2_DIR = ROOT / "dataset" / "2"
MODEL_DIR = ROOT / "outputs" / "models" / "class2_prototype"
RULES_FILE = ROOT / "rules" / "class2_dhsv.json"
OUTPUT_ROOT = ROOT / "outputs" / "pipeline_replay"
SENSORS = ("P-PDG", "P-TPT", "P-MON-CKP", "T-TPT")


def list_class2_files(dataset_dir: Path = DATASET_CLASS2_DIR) -> list[Path]:
    """Только файлы реальных скважин для класса 2; без SIMULATED/DRAWN."""
    return sorted(path for path in Path(dataset_dir).glob("WELL-*.parquet") if path.is_file())


def build_ui_timeline(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Формирует историю для интерфейса из результатов существующего Replay.

    Нет предсказания => NaN, а НЕ нулевая оценка риска.
    """
    rows = [timeline_row(result) for result in results]
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table["decision_time"] = pd.to_datetime(table["decision_time"])
    table["ml_score_uncalibrated"] = pd.to_numeric(
        table["ml_score_uncalibrated"], errors="coerce"
    )
    return table


def ml_chart_data(timeline: pd.DataFrame) -> pd.DataFrame:
    """Порог 0,5 показываем только там, где ML действительно оценивался."""
    if timeline.empty:
        return pd.DataFrame(columns=["ML-оценка (некалибр.)", "Порог 0,5"])
    scores = pd.to_numeric(timeline["ml_score_uncalibrated"], errors="coerce")
    out = pd.DataFrame({
        "ML-оценка (некалибр.)": scores.to_numpy(dtype=float),
        "Порог 0,5": np.where(scores.notna(), 0.5, np.nan),
    }, index=pd.DatetimeIndex(timeline["decision_time"]))
    out.index.name = "Момент диагностики"
    return out


def sensor_chart_data(frame: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Для визуализации: поминутная медиана, НЕ используется агентами.

    Показания с пропусками остаются пропусками, не заполняются нулями.
    """
    if sensor not in SENSORS:
        raise ValueError(f"Неизвестный датчик: {sensor}")
    if sensor not in frame or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    series = pd.to_numeric(frame[sensor], errors="coerce")
    series = series.astype("float64").replace([np.inf, -np.inf], np.nan)
    sampled = series.resample("60s").median()
    return sampled.to_frame(sensor)


def output_directory(file_path: Path, mode: str, output_root: Path = OUTPUT_ROOT) -> Path:
    """Раздельные каталоги по файлу и режиму, без перезаписи других скважин."""
    if mode not in {"blind", "oracle_replay"}:
        raise ValueError("Недопустимый режим")
    if not file_path.name.startswith("WELL-") or file_path.suffix.lower() != ".parquet":
        raise ValueError("Для интерфейса разрешён только WELL-*.parquet")
    return Path(output_root) / mode / file_path.stem


def run_selected_file(
    file_path: Path, *, mode: str, stride_seconds: int = 300,
    max_points: int | None = None, model_dir: Path = MODEL_DIR,
    rules_file: Path = RULES_FILE, output_root: Path = OUTPUT_ROOT,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any], Path]:
    """Один запуск на выбранной скважине через ИМЕЮЩИЕСЯ модули Replay.

    Запись локальных JSONL/CSV/JSON для истории диагностики, без БД.
    """
    if mode not in {"blind", "oracle_replay"}:
        raise ValueError("Режим должен быть blind или oracle_replay")
    if type(stride_seconds) is not int or stride_seconds < 1:
        raise ValueError("Шаг должен быть положительным целым числом")
    if max_points is not None and (type(max_points) is not int or max_points < 1):
        raise ValueError("Максимум точек должен быть положительным целым числом")
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл скважины не найден: {file_path}")
    if not Path(model_dir, "manifest.json").is_file():
        raise FileNotFoundError(f"Не найдены артефакты ML-модели: {model_dir}")
    if not Path(rules_file).is_file():
        raise FileNotFoundError(f"Не найден файл конфигурации правил: {rules_file}")

    frame = pd.read_parquet(file_path)
    pipeline = DiagnosticPipeline(
        rule_agent=load_rule_agent(rules_file, expected_event_class=2),
        prognostic_agent=PrognosticAgent(),  # пороги не определены — не выдумываем
        ml_agent=MLAgent(model_dir),
    )
    results = list(replay_pipeline(
        frame, pipeline=pipeline, mode=mode, stride_seconds=stride_seconds,
        max_points=max_points,
    ))
    destination = output_directory(file_path, mode, output_root)
    summary = write_replay_outputs(
        iter(results), destination, source_filename=file_path.name,
        mode=mode, stride_seconds=stride_seconds,
    )
    return frame, results, build_ui_timeline(results), summary, destination


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Не установлен Streamlit. Установите: python -m pip install -r requirements-ui.txt"
        ) from exc

    st.set_page_config(page_title="CPS-PredDiag · Research", page_icon="📈", layout="wide")
    st.title("CPS-PredDiag")
    st.caption("Исследовательская диагностика нефтяных скважин · Petrobras 3W Dataset v2.0.0")
    st.warning(
        "Исследовательский прототип. Не предназначен для промышленного управления. "
        "Оценки ML не калиброваны; режим oracle_replay использует истинную разметку."
    )

    files = list_class2_files()
    if not files:
        st.error(
            f"Не найдены реальные файлы класса 2 в {DATASET_CLASS2_DIR}. "
            "Ожидаются WELL-*.parquet."
        )
        st.stop()

    with st.sidebar:
        st.header("Параметры воспроизведения")
        selected = st.selectbox(
            "Реальный эпизод класса 2", files, format_func=lambda path: path.name,
        )
        mode = st.radio(
            "Режим", ["blind", "oracle_replay"],
            format_func=lambda value: (
                "Blind — без истинных меток" if value == "blind"
                else "Oracle — с исторической разметкой 102"
            ),
        )
        stride_seconds = int(st.number_input(
            "Шаг диагностики (сек.)", min_value=60, max_value=1800,
            value=300, step=60,
        ))
        limit = int(st.number_input(
            "Макс. число точек (0 = все)", min_value=0, max_value=1000,
            value=0, step=1,
        ))
        run_clicked = st.button("Запустить диагностику", type="primary", use_container_width=True)

    options = (str(selected), mode, stride_seconds, limit)
    if run_clicked:
        try:
            with st.spinner("Воспроизводим историю и формируем отчёты..."):
                data = run_selected_file(
                    selected, mode=mode, stride_seconds=stride_seconds,
                    max_points=limit or None,
                )
        except (ValueError, FileNotFoundError, OSError, ImportError, KeyError, TypeError) as exc:
            st.error(f"Не удалось выполнить воспроизведение: {exc}")
            st.stop()
        st.session_state["cps_replay"] = {"options": options, "data": data}

    saved = st.session_state.get("cps_replay")
    if not saved or saved["options"] != options:
        st.info("Выберите параметры слева и нажмите «Запустить диагностику».")
        st.stop()

    frame, results, timeline, summary, output_dir = saved["data"]
    if mode == "oracle_replay":
        st.info("Oracle-режим: метка 102 взята из исторических данных. Это НЕ независимая проверка модели.")
    else:
        st.info("Blind-режим: истинные метки исключены из входа агентов. Детектор 102 пока отсутствует.")

    st.subheader(f"Эпизод {selected.name}")
    n_scores = int(timeline["ml_score_uncalibrated"].notna().sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Точек диагностики", summary["decisions_count"])
    col2.metric("Оценок ML", n_scores)
    col3.metric("ML-сигналов > 0,5", summary["ml_research_alerts_count"])
    col4.metric("Неопределённых решений", summary["decision_levels"].get("unknown", 0))

    st.subheader("Оценка ML во времени")
    scores_chart = ml_chart_data(timeline)
    if n_scores:
        st.line_chart(scores_chart, height=280)
    else:
        st.info("В этом режиме ML-прогнозы не сформированы. Отсутствие оценки не означает нулевой риск.")

    st.subheader("Показания датчиков")
    st.caption("На графиках — поминутные медианы для удобства просмотра. Агенты работают с исходными секундными данными.")
    pressure_tab, temperature_tab = st.tabs(["Давление (Па)", "Температура"])
    with pressure_tab:
        for sensor in ("P-PDG", "P-TPT", "P-MON-CKP"):
            chart = sensor_chart_data(frame, sensor)
            if chart.empty or chart[sensor].notna().sum() == 0:
                st.caption(f"{sensor}: показания недоступны")
            else:
                st.caption(sensor)
                st.line_chart(chart, height=170)
    with temperature_tab:
        chart = sensor_chart_data(frame, "T-TPT")
        if chart.empty or chart["T-TPT"].notna().sum() == 0:
            st.info("T-TPT: показания недоступны")
        else:
            st.line_chart(chart, height=240)

    st.subheader("История диагностических решений")
    st.dataframe(timeline, use_container_width=True, hide_index=True)
    selected_idx = st.selectbox(
        "Посмотреть объяснение решения", list(range(len(results))),
        format_func=lambda i: (
            f"{results[i]['decision_time']} · "
            f"{results[i]['decision'].get('decision_level', 'unknown')} · "
            f"ML: {results[i]['agents']['ml'].get('status', 'unknown')}"
        ),
        index=len(results) - 1,
    )
    current = results[selected_idx]
    st.markdown(current["explanation"].replace("\n", "\n\n"))
    with st.expander("Полный результат выбранной точки (JSON)"):
        st.code(json.dumps(current, ensure_ascii=False, indent=2), language="json")

    st.caption(f"Локальные отчёты: {output_dir}")
    st.caption("Сохранены decisions.jsonl, timeline.csv, summary.json. База данных не используется.")
    st.download_button(
        "Скачать CSV истории диагностики", data=timeline.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{selected.stem}_{mode}_timeline.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()
