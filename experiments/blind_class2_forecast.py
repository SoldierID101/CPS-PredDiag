"""Исследовательский ПРОГНОЗ класса 2 за 15 мин БЕЗ oracle-условия 102.

Запуск из корня CPS-PredDiag:
  python experiments/blind_class2_forecast.py

Протокол:
- Используются только реальные файлы dataset/2/WELL-*.parquet.
- Все моменты диагностики задаются РЕГУЛЯРНО по времени (шаг 300 с), не по метке 102.
- Для признаков доступны только прошлые показания датчиков.
- Истинные class используются ТОЛЬКО для формирования обучающей цели,
  ретроспективного определения предсобытийного интервала и отчётности.
- Отрицательный ответ допустим лишь при ПОЛНОСТЬЮ наблюдавшихся
  15 мин будущего без события класса 2; неизвестное будущее = цензура.
- В сравнении фиксированные гиперпараметры и порог 0.5, без подбора
  по контрольным скважинам. LOOW проверяет новые скважины.
- Задача: прогноз начала класса 2, НЕ детекция класса 102.

Никаких изменений текущего ML Agent и работающего приложения.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_agent import FEATURES, SENSORS, STATISTICS, features_from_history

SEED = 42
PAST_SECONDS = 900
HORIZON_SECONDS = 900
STRIDE_SECONDS = 300
NSEC = 1_000_000_000
EXTENDED_SENSORS = (*SENSORS, "P-PDG")
SCALES = (60, 300, 900)
BASELINE_FEATURES = list(FEATURES)  # те же 12, что у RF/ML Agent v0.1
EXTENDED_FEATURES = [
    f"{sensor}_{seconds}s_{stat}"
    for sensor in EXTENDED_SENSORS
    for seconds in SCALES
    for stat in (*STATISTICS, "coverage")
]
MODELS = ("rf_12", "catboost_12", "catboost_multiscale")
FIXED_THRESHOLD = 0.5
MIN_SENSOR_COVERAGE = 0.9


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Без ресэмплинга, сортировки и искусственного заполнения пропусков во времени."""
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(frame.index.dtype):
            raise ValueError("В Parquet отсутствует ось времени; числовой индекс не подходит")
        try:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index, errors="raise")
        except Exception as exc:
            raise ValueError("Некорректная ось времени") from exc
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Метки времени повторяются или идут не по возрастанию")
    frame = frame.copy()
    frame.index = frame.index.as_unit("ns")
    if frame.index.hasnans:
        raise ValueError("Временные метки содержат NaT")
    if "class" not in frame.columns:
        raise ValueError("Отсутствует поле class: невозможно определить цель обучения")
    return frame


def contiguous_segments(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Полуоткрытые интервалы [start, end) непрерывных секундных записей."""
    if not isinstance(index, pd.DatetimeIndex) or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Нужен строго возрастающий DatetimeIndex")
    if index.hasnans:
        raise ValueError("NaT в оси времени")
    ns = index.as_unit("ns").asi8
    if len(ns) == 0:
        return []
    cut = np.flatnonzero(np.diff(ns) != NSEC) + 1
    limits = [0, *cut.tolist(), len(ns)]
    return list(zip(limits[:-1], limits[1:]))


def classify_future(labels: np.ndarray, i: int, end: int,
                    horizon_seconds: int = HORIZON_SECONDS) -> tuple[float, int | None, str]:
    """Метка будущего для текущего известного 0/102.

    1: событие class 2 впервые наблюдено в интервале (t,t+H], и все
       предшествующие будущие метки известны.
    0: наблюдаются все H секунд, все метки 0/102, события нет.
    nan: цензура / разрыв / неоднозначная разметка.
    """
    if i < 0 or i >= end or end > len(labels):
        raise ValueError("Некорректные границы будущего")
    if not np.isin(labels[i], [0, 102]):
        raise ValueError("Текущая точка должна относиться к известной предсобытийной фазе")
    stop = min(i + horizon_seconds + 1, end)
    future = labels[i + 1:stop]
    event_pos = np.flatnonzero(future == 2)
    if len(event_pos):
        p = int(event_pos[0])
        if np.isin(future[:p], [0, 102]).all():
            return 1.0, i + 1 + p, "positive"
        return np.nan, None, "unknown_before_event"
    if len(future) == horizon_seconds and np.isin(future, [0, 102]).all():
        return 0.0, None, "negative"
    return np.nan, None, "censored_or_unknown"


def feature_stats(window: pd.DataFrame, sensor: str) -> dict[str, float]:
    """Статистики с пропусками, без использования label/state и будущего."""
    n = len(window)
    if sensor in window:
        y_all = pd.to_numeric(window[sensor], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)
    else:
        y_all = np.full(n, np.nan, dtype=float)
    good = np.isfinite(y_all)
    cov = float(good.mean()) if n else 0.0
    vals = {"coverage": cov}
    if good.sum() < np.ceil(MIN_SENSOR_COVERAGE * n):
        vals.update({k: np.nan for k in STATISTICS})
        return vals
    x = np.flatnonzero(good).astype(np.float64)
    y = y_all[good]
    vals.update({
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)),
        "delta": float(y[-1] - y[0]),
        "slope_per_hour": float(np.polyfit(x, y, 1)[0] * 3600),
    })
    return vals


def past_only_features(frame: pd.DataFrame, i: int) -> tuple[dict[str, float], int]:
    """Вычисляет две схемы признаков исключительно из [i-899, i].

    Датчики передаются отдельным DataFrame. class, state, time_from_event,
    future и идентификатор скважины в признаки не попадают.
    """
    if i < PAST_SECONDS - 1:
        raise ValueError("Недостаточно истории")
    past = frame.iloc[i - PAST_SECONDS + 1:i + 1].reindex(columns=EXTENDED_SENSORS).copy()
    # pandas 3.x может создавать DatetimeIndex с точностью us вместо ns.
    # as_unit("ns") нормализует единицы без изменения реальных моментов времени.
    # Копия защищает исходный frame от изменения индекса в этой функции.
    if not isinstance(past.index, pd.DatetimeIndex):
        raise ValueError("Для признаков требуется DatetimeIndex")
    past.index = past.index.as_unit("ns")
    if len(past) != PAST_SECONDS or not np.all(np.diff(past.index.asi8) == NSEC):
        raise ValueError("Внутри истории есть временные разрывы")
    base, coverage = features_from_history(past.iloc[-300:].reindex(columns=SENSORS))
    data = {key: float(base.iloc[0][key]) for key in BASELINE_FEATURES}
    for sensor in EXTENDED_SENSORS:
        for seconds in SCALES:
            values = feature_stats(past.iloc[-seconds:], sensor)
            for stat in (*STATISTICS, "coverage"):
                data[f"{sensor}_{seconds}s_{stat}"] = float(values[stat])
    usable = sum(coverage[s] >= MIN_SENSOR_COVERAGE for s in SENSORS)
    return data, usable


def select_decision_points(frame: pd.DataFrame, stride_seconds: int = STRIDE_SECONDS,
                           horizon_seconds: int = HORIZON_SECONDS) -> tuple[pd.DataFrame, dict]:
    """Расписание зависит ТОЛЬКО от меток времени, но не от class.

    Для оценки/обучения определяем предсобытийную risk-set по меткам.
    В будущем онлайн-модуле такого фильтра по class быть не должно.
    """
    if stride_seconds < 1 or horizon_seconds < 1:
        raise ValueError("Шаг и горизонт должны быть положительными")
    if "class" not in frame:
        raise ValueError("Для определения цели нужен class")
    labels = pd.to_numeric(frame["class"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    segments = contiguous_segments(frame.index)
    first_event_positions = np.flatnonzero(labels == 2)
    first_event_index = int(first_event_positions[0]) if len(first_event_positions) else len(labels)
    known_before_event = np.r_[False, np.isin(labels[:-1], [0, 102])]
    report = {
        "segments": len(segments), "candidate_times": 0,
        "observed_class2_onsets": int(((labels == 2) & known_before_event).sum()),
        "excluded_unknown_current": 0, "excluded_already_event": 0,
        "excluded_after_first_event": 0,
        "censored_or_unknown_future": 0, "insufficient_sensors": 0,
        "eligible_negative": 0, "eligible_positive": 0,
    }
    rows: list[dict] = []
    for begin, end in segments:
        # Время принятия решений не зависит от исхода события.
        for i in range(begin + PAST_SECONDS - 1, end, stride_seconds):
            report["candidate_times"] += 1
            # Вычисление доступных онлайн-признаков НЕ зависит от class.
            feat, usable = past_only_features(frame, i)
            # Для ОЦЕНКИ первого наступления (событие, уже случившееся в файле,
            # не является будущим). Это ретроспективное ограничение когорты,
            # не правило запуска онлайн-модели.
            if i >= first_event_index:
                report["excluded_after_first_event"] += 1
                continue
            if not np.isfinite(labels[i]) or labels[i] not in (0, 102, 2):
                report["excluded_unknown_current"] += 1
                continue
            if labels[i] == 2:
                report["excluded_already_event"] += 1
                continue
            y, event_i, outcome = classify_future(labels, i, end, horizon_seconds)
            if not np.isfinite(y):
                report["censored_or_unknown_future"] += 1
                continue
            if usable < 2:
                report["insufficient_sensors"] += 1
                continue
            report["eligible_positive" if y == 1 else "eligible_negative"] += 1
            rows.append({
                "decision_time": frame.index[i], "target": int(y),
                "phase_for_audit_only": int(labels[i]),
                "event_time": frame.index[event_i] if event_i is not None else pd.NaT,
                "future_status": outcome,
                **feat,
            })
    return pd.DataFrame(rows), report


def read_cohort(dataset_dir: Path, max_files: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    files = sorted(dataset_dir.glob("WELL-*.parquet"))
    if max_files is not None:
        if max_files < 1:
            raise ValueError("max_files должен быть >=1")
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Не найдены WELL-*.parquet в {dataset_dir}")
    all_rows, audits, provenance = [], [], []
    for path in files:
        match = re.fullmatch(r"WELL-(\d{5})_\d+\.parquet", path.name)
        if not match:
            raise ValueError(f"Неожиданное имя WELL-файла: {path.name}")
        well = int(match.group(1))
        frame = prepare_frame(pd.read_parquet(path))
        samples, info = select_decision_points(frame)
        if len(samples):
            samples.insert(0, "file", path.name)
            samples.insert(1, "well_id", well)
            all_rows.append(samples)
        audits.append({"file": path.name, "well_id": well, "rows": len(frame), **info})
        provenance.append({"file": path.name, "bytes": path.stat().st_size})
        print(f"{path.name}: {len(frame)} строк, pre-event окон: {len(samples)} "
              f"(pos={info['eligible_positive']}, neg={info['eligible_negative']}, "
              f"цензура={info['censored_or_unknown_future']})")
    cohort = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not cohort.empty:
        if cohort.duplicated(["file", "decision_time"]).any():
            raise RuntimeError("Дубли выборки")
        cohort = cohort.sort_values(["file", "decision_time"], kind="stable").reset_index(drop=True)
    return cohort, pd.DataFrame(audits), provenance


def train_oof(cohort: pd.DataFrame, cb_iterations: int = 300) -> pd.DataFrame:
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    needed = {"file", "well_id", "decision_time", "target", "event_time", *BASELINE_FEATURES, *EXTENDED_FEATURES}
    if not needed.issubset(cohort.columns):
        raise ValueError("В когорте не хватает обязательных столбцов")
    if cohort.empty or cohort.well_id.nunique() < 3 or cohort.target.nunique() < 2:
        raise ValueError("Для LOOW нужны минимум 3 скважины и оба целевых класса")
    if cb_iterations < 1:
        raise ValueError("iterations должен быть >=1")
    y = cohort.target.to_numpy(dtype=int)
    wells = cohort.well_id.to_numpy(dtype=int)
    x12 = cohort[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    xmulti = cohort[EXTENDED_FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    predictions = cohort[["file", "well_id", "decision_time", "target", "event_time", "phase_for_audit_only"]].copy()
    for model in MODELS:
        predictions[model] = np.nan
    for well in sorted(np.unique(wells)):
        train, test = wells != well, wells == well
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"Невозможно обучить на остальных скважинах при исключении {well}")
        rf = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=70, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        )
        rf.fit(x12.loc[train], y[train])
        predictions.loc[test, "rf_12"] = rf.predict_proba(x12.loc[test])[:, 1]
        config = dict(iterations=cb_iterations, depth=4, learning_rate=0.03,
                      l2_leaf_reg=5, loss_function="Logloss", auto_class_weights="Balanced",
                      nan_mode="Min", random_seed=SEED, thread_count=2,
                      verbose=False, allow_writing_files=False)
        cb12 = CatBoostClassifier(**config)
        cb12.fit(x12.loc[train], y[train])
        predictions.loc[test, "catboost_12"] = cb12.predict_proba(x12.loc[test])[:, 1]
        cbmulti = CatBoostClassifier(**config)
        cbmulti.fit(xmulti.loc[train], y[train])
        predictions.loc[test, "catboost_multiscale"] = cbmulti.predict_proba(xmulti.loc[test])[:, 1]
        print(f"  LOOW WELL-{well:05d}: тест {int(test.sum())}, обуч. {int(train.sum())}, "
              f"положительных в обучении {int(y[train].sum())}")
    if predictions[list(MODELS)].isna().any().any():
        raise RuntimeError("Не все тестовые примеры получили OOF-прогноз")
    return predictions


def summarize_binary(data: pd.DataFrame, score_column: str, threshold: float = FIXED_THRESHOLD) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    y = data.target.to_numpy(dtype=int)
    s = data[score_column].to_numpy(dtype=float)
    alerts = s >= threshold
    fp = int(((y == 0) & alerts).sum())
    neg = int((y == 0).sum())
    positive = int((y == 1).sum())
    return {
        "n": int(len(y)), "positive": positive, "prevalence": float(positive / len(y)),
        "ap": float(average_precision_score(y, s)) if positive else None,
        "roc_auc": float(roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
        "precision_05": float(precision_score(y, alerts, zero_division=0)),
        "recall_05": float(recall_score(y, alerts, zero_division=0)),
        "f1_05": float(f1_score(y, alerts, zero_division=0)),
        "tp_windows": int(((y == 1) & alerts).sum()),
        "fp_windows": fp,
        "fn_windows": int(((y == 1) & ~alerts).sum()),
        "tn_windows": int(((y == 0) & ~alerts).sum()),
        # Окна отбора по 5 минут, не общая продолжительность всех записей.
        "eligible_negative_hours": float(neg * STRIDE_SECONDS / 3600),
        "fp_windows_per_eligible_negative_hour": float(fp / (neg * STRIDE_SECONDS / 3600)) if neg else None,
    }


def event_summary(oof: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Эпизоды с хотя бы одним положительным пригодным окном (не все события в наборе)."""
    positives = oof.loc[oof.target.eq(1) & oof.event_time.notna()].copy()
    if positives.empty:
        return pd.DataFrame(columns=["model", "file", "well_id", "event_time", "positive_windows", "detected", "first_alert_time", "lead_minutes"])
    rows = []
    for (file, well, event), group in positives.groupby(["file", "well_id", "event_time"], sort=True):
        hits = group.loc[group[score_column] >= FIXED_THRESHOLD].sort_values("decision_time")
        first = hits.decision_time.iloc[0] if len(hits) else pd.NaT
        rows.append({"model": score_column, "file": file, "well_id": int(well),
                     "event_time": event, "positive_windows": int(len(group)),
                     "detected": bool(len(hits)), "first_alert_time": first,
                     "lead_minutes": float((event-first).total_seconds()/60) if len(hits) else np.nan})
    return pd.DataFrame(rows)


def count_false_alert_clusters(oof: pd.DataFrame, score_column: str, threshold: float = FIXED_THRESHOLD) -> int:
    """Подсчёт серий FP-окон на каждом файле (исследовательская приближенность)."""
    count = 0
    false = oof.loc[(oof.target.eq(0)) & (oof[score_column] >= threshold)]
    for _, sub in false.groupby("file"):
        moments = pd.to_datetime(sub.decision_time).sort_values()
        previous = None
        for moment in moments:
            if previous is None or (moment - previous).total_seconds() > STRIDE_SECONDS:
                count += 1
            previous = moment
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "2")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "blind_class2")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--max-files", type=int, default=None,
                    help="Для технической отладки; неполные метрики не использовать в научных выводах")
    args = ap.parse_args()
    cohort, audit, provenance = read_cohort(args.dataset_dir, args.max_files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    if cohort.empty:
        raise ValueError("Нет подходящих размеченных предсобытийных окон, см. cohort_audit.csv")
    cohort.to_csv(args.output_dir / "cohort_features.csv", index=False)
    print(f"\nBLIND COHORT: {len(cohort)} окон, положительных {int(cohort.target.sum())}, "
          f"скважин {cohort.well_id.nunique()}, файлов {cohort.file.nunique()}, "
          f"наблюдаемых переходов в класс 2: {int(audit.observed_class2_onsets.sum())}")
    print("Целевые фазы в выборке (только для аудита):",
          cohort.groupby(["phase_for_audit_only", "target"]).size().to_dict())
    oof = train_oof(cohort, args.iterations)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    comparisons, by_well, by_event = [], [], []
    for model in MODELS:
        results = summarize_binary(oof, model)
        events = event_summary(oof, model)
        count = count_false_alert_clusters(oof, model)
        results.update({"model": model,
                        "eligible_completed_events": int(len(events)),
                        "detected_events": int(events.detected.sum()) if len(events) else 0,
                        "fp_alert_clusters_approx": int(count),
                        "median_lead_detected_min": float(events.loc[events.detected, "lead_minutes"].median())
                        if len(events) and events.detected.any() else None})
        comparisons.append(results)
        by_event.append(events)
        for well, part in oof.groupby("well_id"):
            by_well.append({"model": model, "well_id": int(well), **summarize_binary(part, model),
                            "fp_alert_clusters_approx": count_false_alert_clusters(part, model)})
    pd.DataFrame(comparisons).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(by_well).to_csv(args.output_dir / "by_well.csv", index=False)
    pd.concat(by_event, ignore_index=True).to_csv(args.output_dir / "by_event.csv", index=False)
    import catboost, sklearn
    manifest = {
        "task": "blind_class2_onset_within_next_15m", "dataset": "Petrobras 3W v2.0.0 real WELL class 2 files",
        "study_type": "exploratory; previously studied wells", "oracle_gate": False,
        "labels_only_for_training_targets_and_retrospective_evaluation": True,
        "risk_set": "class 0/102 at decision time; restriction only for retrospective supervised evaluation, NOT an online gate",
        "decision_schedule": "every 300s after 900s continuous observations; independent of labels",
        "past_seconds": PAST_SECONDS, "horizon_seconds": HORIZON_SECONDS,
        "censoring": "negative requires full known future; positive requires observed onset and no unknown labels before onset",
        "event_metric": "events with >=1 eligible positive window, not all originally completed events",
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_calibrated": False, "param_tuning": "none; predeclared values",
        "random_seed": SEED, "catboost_iterations": args.iterations,
        "model_columns": {"rf_12": BASELINE_FEATURES, "catboost_12": BASELINE_FEATURES,
                          "catboost_multiscale": EXTENDED_FEATURES},
        "feature_selection_notes": "previously examined 3W signals; results exploratory, not external validation",
        "scored_source_files": provenance,
        "n_wells": int(cohort.well_id.nunique()), "n_rows": int(len(cohort)),
        "observed_class2_onsets": int(audit.observed_class2_onsets.sum()),
        "positive": int(cohort.target.sum()), "n_files": int(cohort.file.nunique()),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "catboost": catboost.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "Only class2 event-enriched WELL files, not representative general operation",
            "Only 7 wells and 11 completed events in historical audit",
            "Only first class2 onset per file; post-event periods excluded retrospectively and not usable as runtime gate",
            "Adjacent 5-minute windows strongly correlated",
            "Prior exploration of same wells means no untouched final test cohort",
            "FP per eligible negative hour is NOT all-field-operating-hours FAR",
            "Neither AP nor predicted score establishes calibrated physical failure probability",
            "Do not directly compare AP to 0.3243 oracle-gated on different cohort",
        ],
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for model in MODELS:
            p, r, _ = precision_recall_curve(oof.target, oof[model])
            ax.plot(r, p, label=model)
        ax.axhline(oof.target.mean(), linestyle=":", label="Positive prevalence")
        ax.set(xlabel="Recall", ylabel="Precision", title="Class 2 onset <=15 min, LOOW (blind features)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "pr_curves.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib отсутствует: PR-график не построен")
    summary = pd.DataFrame(comparisons)
    print("\n=== Результат LOOW (фиксированный порог 0.5) ===")
    print(summary[["model", "n", "positive", "ap", "precision_05", "recall_05", "f1_05",
                   "fp_windows", "detected_events", "eligible_completed_events"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Отчёты: {args.output_dir}")
    print("ВАЖНО: это exploratory blind-feature эксперименты на class2 event-enriched записях; "
          "не готовая промышленная система. Метрики не сопоставлять напрямую с oracle-выборкой.")


if __name__ == "__main__":
    main()
