"""Исследовательское сравнение CatBoost-12: горизонты 15, 30, 60 минут.

Запуск из корня CPS-PredDiag:
    python experiments/compare_class2_horizons.py

Требует существующий experiments/blind_class2_forecast.py (исправленная
версия после 114 успешных тестов) и ml_agent.py. Не меняет боевые агенты.

Важно: признаки формируются исключительно из прошлого; целевые метки и
ретроспективная risk-set требуют истинной разметки. Это exploratory LOOW
на enriched class-2 файлах, НЕ независимая проверка промышленного риска.
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments import blind_class2_forecast as baseline
DEFAULT_HORIZONS = (15, 30, 60)
RESEARCH_THRESHOLD = 0.5
MODEL_NAME = "catboost_12"


def check_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    """Горизонты заранее определены; запрет дублей предотвращает перезапись выводов."""
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Горизонты должны быть положительными целыми минутами")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Горизонты не должны повторяться")
    return tuple(sorted(horizons))


def build_cohorts(dataset_dir: Path, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                  max_files: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Один список файлов и одна регулярная сетка времени для всех горизонтов.

    Различаются ТОЛЬКО метки будущего и пригодность окна по цензуре.
    Признаки идентичны в совпадающих моментах диагностики.
    """
    horizons = check_horizons(horizons)
    if max_files is not None and max_files < 1:
        raise ValueError("max_files должен быть >=1")
    files = sorted(Path(dataset_dir).glob("WELL-*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Нет WELL-*.parquet в {dataset_dir}")

    cohorts, audit_rows, provenance = [], [], []
    for path in files:
        match = re.fullmatch(r"WELL-(\d{5})_\d+\.parquet", path.name)
        if not match:
            raise ValueError(f"Некорректное имя файла: {path.name}")
        well = int(match.group(1))
        frame = baseline.prepare_frame(pd.read_parquet(path))
        provenance.append({"filename": path.name, "bytes": path.stat().st_size})
        counts = []
        for horizon in horizons:
            # Существующий код проверяет секунды без разрывов, NA-метки,
            # статус до первого события и использует только прошлую телеметрию.
            samples, audit = baseline.select_decision_points(
                frame, stride_seconds=baseline.STRIDE_SECONDS,
                horizon_seconds=horizon * 60)
            if not samples.empty:
                samples.insert(0, "file", path.name)
                samples.insert(1, "well_id", well)
                samples.insert(2, "horizon_minutes", horizon)
                cohorts.append(samples)
            audit_rows.append({"file": path.name, "well_id": well,
                               "horizon_minutes": horizon, "source_rows": len(frame), **audit})
            counts.append(f"{horizon}м:{len(samples)}/{int(samples.target.sum()) if len(samples) else 0}")
        print(f"{path.name}: {len(frame)} строк; пригодных/положительных окон "+", ".join(counts))

    data = pd.concat(cohorts, ignore_index=True) if cohorts else pd.DataFrame()
    if not data.empty:
        key = ["horizon_minutes", "file", "decision_time"]
        if data.duplicated(key).any():
            raise ValueError("Дубли диагностических моментов")
        data.sort_values(key, kind="stable", inplace=True)
        data.reset_index(drop=True, inplace=True)
        verify_same_past_features(data)
    return data, pd.DataFrame(audit_rows), provenance


def verify_same_past_features(cohorts: pd.DataFrame) -> None:
    """Строгая проверка: признаки в общем моменте НЕ зависят от горизонта."""
    if cohorts.empty:
        return
    keys = ["file", "decision_time"]
    columns = baseline.BASELINE_FEATURES
    for _, group in cohorts.groupby(keys, sort=False):
        if len(group) <= 1:
            continue
        ref = group.iloc[0][columns].to_numpy(dtype=float)
        for i in range(1, len(group)):
            current = group.iloc[i][columns].to_numpy(dtype=float)
            if not np.allclose(ref, current, rtol=0, atol=1e-8, equal_nan=True):
                raise ValueError("Признаки меняются с горизонтом: подозрение на утечку данных")


def train_catboost_oof(cohort: pd.DataFrame, iterations: int = 300) -> pd.DataFrame:
    """Точная конфигурация CatBoost-12 из blind_class2_forecast.py.

    Каждая скважина исключается из обучения целиком. Порог не подбирается.
    """
    from catboost import CatBoostClassifier

    required = {"horizon_minutes", "file", "well_id", "decision_time", "target",
                "event_time", "phase_for_audit_only", *baseline.BASELINE_FEATURES}
    if cohort.empty or not required.issubset(cohort.columns):
        raise ValueError("Нет данных или не хватает полей для обучения")
    if iterations < 1:
        raise ValueError("iterations должен быть >=1")
    if cohort.well_id.nunique() < 3 or cohort.target.nunique() < 2:
        raise ValueError("Для LOOW требуются >=3 скважины и оба класса")
    x = cohort[baseline.BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    y = cohort.target.to_numpy(dtype=int)
    wells = cohort.well_id.to_numpy(dtype=int)
    scores = np.full(len(cohort), np.nan)
    config = dict(iterations=iterations, depth=4, learning_rate=0.03,
                  l2_leaf_reg=5, loss_function="Logloss", auto_class_weights="Balanced",
                  nan_mode="Min", random_seed=baseline.SEED, thread_count=2,
                  verbose=False, allow_writing_files=False)
    for well in sorted(np.unique(wells)):
        train = wells != well
        test = ~train
        if np.unique(y[train]).size < 2:
            raise ValueError(f"Обучение без WELL-{well:05d} содержит один класс")
        model = CatBoostClassifier(**config)
        model.fit(x.iloc[np.flatnonzero(train)], y[train])
        scores[test] = model.predict_proba(x.iloc[np.flatnonzero(test)])[:, 1]
        print(f"    контроль WELL-{well:05d}: {int(test.sum())} окон; "
              f"обучение {int(train.sum())} (положительных {int(y[train].sum())})")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Есть окна без OOF-прогноза")
    if np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("Оценки вне [0, 1]")
    preds = cohort[["horizon_minutes", "file", "well_id", "decision_time", "target",
                    "event_time", "phase_for_audit_only"]].copy()
    preds["score"] = scores
    return preds


def model_summary(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = baseline.summarize_binary(pred, "score", RESEARCH_THRESHOLD)
    events = baseline.event_summary(pred, "score")
    false_clusters = baseline.count_false_alert_clusters(pred, "score", RESEARCH_THRESHOLD)
    summary.update({
        "detected_events": int(events.detected.sum()) if len(events) else 0,
        "eligible_completed_events": len(events),
        "event_recall": float(events.detected.mean()) if len(events) else None,
        "median_lead_detected_min": float(events.loc[events.detected, "lead_minutes"].median())
        if len(events) and events.detected.any() else None,
        "fp_alert_clusters_approx": false_clusters,
        "ap_lift_vs_prevalence": float(summary["ap"] / summary["prevalence"])
        if summary["ap"] is not None and summary["prevalence"] else None,
    })
    return summary, events


def common_decision_subset(oof: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Одинаковые моменты оценки для доп. анализа; метки всё равно различаются."""
    horizons = check_horizons(horizons)
    key = ["file", "decision_time"]
    valid = oof.groupby(key, dropna=False)["horizon_minutes"].nunique()
    common = valid.loc[valid.eq(len(horizons))].index
    idx = pd.MultiIndex.from_frame(oof[key])
    subset = oof.loc[idx.isin(common)].copy()
    counts = subset.groupby("horizon_minutes").size().to_dict()
    if subset.empty or any(counts.get(h, 0) != len(common) for h in horizons):
        raise ValueError("Недостаточно общих моментов для сопоставления горизонтов")
    return subset


def create_report(cohort: pd.DataFrame, audit: pd.DataFrame,
                  horizons: tuple[int, ...], iterations: int,
                  output_dir: Path, provenance: list[dict]) -> None:
    import catboost
    import sklearn
    from sklearn.metrics import precision_recall_curve

    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "cohort_audit.csv", index=False)
    cohort.to_csv(output_dir / "cohort_features.csv", index=False)
    oof_groups = []
    for h in horizons:
        subset = cohort.loc[cohort.horizon_minutes.eq(h)].copy()
        print(f"\nГОРИЗОНТ {h} мин: {len(subset)} пригодных окон; "
              f"положительных {int(subset.target.sum())}; "
              f"скважин {subset.well_id.nunique()}")
        oof_groups.append(train_catboost_oof(subset, iterations))
    oof = pd.concat(oof_groups, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    summaries, by_well, by_event = [], [], []
    for h, part in oof.groupby("horizon_minutes", sort=True):
        metrics, events = model_summary(part)
        summaries.append({"horizon_minutes": int(h), **metrics})
        if not events.empty:
            events.insert(0, "horizon_minutes", int(h))
            by_event.append(events)
        for well, well_part in part.groupby("well_id"):
            well_metrics, well_events = model_summary(well_part)
            by_well.append({"horizon_minutes": int(h), "well_id": int(well),
                            **well_metrics})
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(by_well).to_csv(output_dir / "by_well.csv", index=False)
    (pd.concat(by_event, ignore_index=True) if by_event else pd.DataFrame()).to_csv(
        output_dir / "by_event.csv", index=False)

    common = common_decision_subset(oof, horizons)
    common.to_csv(output_dir / "common_oof_predictions.csv", index=False)
    common_metrics = []
    for h, part in common.groupby("horizon_minutes"):
        m, _ = model_summary(part)
        common_metrics.append({"horizon_minutes": int(h), **m})
    pd.DataFrame(common_metrics).to_csv(output_dir / "comparison_common_times.csv", index=False)

    sources = [ROOT / "experiments" / "blind_class2_forecast.py",
               ROOT / "experiments" / "compare_class2_horizons.py",
               ROOT / "ml_agent.py"]
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources if p.is_file()
    }
    manifest = {
        "name": "class2_catboost12_horizons_15_30_60",
        "dataset": "Petrobras 3W v2.0.0, real WELL, class 2, event-enriched",
        "study_type": "exploratory, NOT independent final validation",
        "horizons_minutes": list(horizons), "past_seconds": baseline.PAST_SECONDS,
        "decision_stride_seconds": baseline.STRIDE_SECONDS,
        "features": baseline.BASELINE_FEATURES,
        "model": "CatBoostClassifier", "iterations": iterations,
        "hyperparameters": {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5,
                            "auto_class_weights": "Balanced", "nan_mode": "Min", "random_seed": baseline.SEED},
        "threshold": RESEARCH_THRESHOLD, "threshold_selection": "predeclared, not tuned",
        "oracle_gate": False, "uses_labels_for_target_and_retrospective_risk_set": True,
        "future_unknown": "censored; never labelled as normal",
        "split": "Leave-One-Well-Out, 7 folds when all files present",
        "test_well_used_for_tuning": False,
        "source_files": provenance,
        "source_sha256": source_hashes,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "catboost": catboost.__version__,
                     "scikit_learn": sklearn.__version__},
        "limitations": [
            "Risk set excludes current/after first event retrospectively; not deployable as online gate.",
            "Only class2 event-enriched wells: FP/h is NOT field operating false alarm rate.",
            "Changing horizon changes positive prevalence and eligible windows: AP/F1 not directly comparable.",
            "Common-times output reduces cohort difference but targets still differ.",
            "11 recorded onsets / 7 wells; windows overlap; previously examined wells.",
            "Fixed threshold 0.5 is not calibrated or optimized.",
            "Do not select winning horizon on these held-out wells and claim unbiased final quality.",
        ],
    }
    (output_dir / "experiment.json").write_text(json.dumps(manifest, indent=2,
                                                             ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for h, part in oof.groupby("horizon_minutes"):
            p, r, _ = precision_recall_curve(part.target, part.score)
            ax.plot(r, p, label=f"{h} min")
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Different horizon targets (interpret with care)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curves.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.bar([str(int(i)) for i in comparison.horizon_minutes],
               comparison.event_recall.fillna(0) * 100)
        ax.set(xlabel="Horizon (min)", ylabel="Detected events (%)",
               ylim=(0, 105), title="LOOW event detection, threshold 0.5")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "event_detection.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib отсутствует — график не создан")

    print("\n=== CATBOOST-12: сравнение горизонтов, LOOW, порог 0.5 ===")
    cols = ["horizon_minutes", "n", "positive", "prevalence", "ap",
            "precision_05", "recall_05", "f1_05", "fp_windows",
            "detected_events", "eligible_completed_events", "median_lead_detected_min"]
    print(comparison[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nОбщих оцениваемых моментов:", int(len(common) / len(horizons)))
    print("Отчёты:", output_dir)
    print("ПРЕДУПРЕЖДЕНИЕ: проценты между горизонтами не являются прямым доказательством улучшения: "
          "изменяются исходы, число положительных окон и состав оцениваемой когорты.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "class2_horizons")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Техническая проверка; частичную выборку нельзя использовать в научном выводе")
    args = parser.parse_args()
    horizons = check_horizons(DEFAULT_HORIZONS)
    cohort, audit, provenance = build_cohorts(args.dataset_dir, horizons, args.max_files)
    if cohort.empty:
        raise ValueError("Нет пригодных окон для эксперимента")
    create_report(cohort, audit, horizons, args.iterations, args.output_dir, provenance)


if __name__ == "__main__":
    main()
