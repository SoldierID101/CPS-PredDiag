"""Эксперимент hard negatives для CatBoost class2, горизонт 30 минут.

Сравнивает CatBoost-12, обученный только на class2-эпизодах, с CatBoost-12,
дополнительно обученным на проверенных отрицательных окнах классов 0,1,3..9.
УТЕЧКА ЗАПРЕЩЕНА: контрольные скважины для финального теста не участвуют
в обучении; class2 тестовая скважина удалена из обеих обучающих групп.

Команды из корня CPS-PredDiag:
  python experiments/evaluate_class2_hard_negatives.py
  python -m unittest discover -s tests -v

Метрики предварительные, так как исходные скважины и горизонты исследовали ранее.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import evaluate_class2_controls as ctl
from experiments.evaluate_class2_alert_policy import catboost_model, trigger_policy

FEATURES = list(ctl.FEATURES)
STRIDE_SECONDS = ctl.STRIDE_SECONDS
HORIZON_MINUTES = 30
SPLIT_SALT = "CPS-PredDiag hardneg holdout v0.1"
EXTERNAL_HOLDOUT_FRACTION = 0.30


def assign_external_holdout(control_wells: set[int], class2_wells: set[int],
                            holdout_fraction: float = EXTERNAL_HOLDOUT_FRACTION,
                            salt: str = SPLIT_SALT) -> tuple[set[int], set[int]]:
    """Детерминированное разделение только по ID скважины, без меток или прогнозов.

    Скважины класса2 ВСЕГДА проходят через внешнюю LOOW-проверку отдельно.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("Доля удержанных скважин должна лежать между 0 и 1")
    others = set(map(int, control_wells)) - set(map(int, class2_wells))
    if len(others) < 2:
        raise ValueError("Недостаточно независимых контрольных скважин")
    heldout = {well for well in others if int(hashlib.sha256(
        f"{salt}:{well}".encode("utf-8")).hexdigest()[:12], 16) / (16 ** 12)
        < holdout_fraction}
    development = others - heldout
    if not heldout or not development:
        raise ValueError("Разделение дало пустую обучающую/контрольную группу")
    assert not (heldout & set(class2_wells))
    return development, heldout


def load_selected(path: Path, dataset_dir: Path) -> pd.DataFrame:
    """Оригинальная фиксированная выборка из эксперимента специфичности.

    Пользователь не выбирает файлы по числу ложных тревог новой модели.
    """
    data = pd.read_csv(path)
    required = {"file", "well_id", "event_class_folder"}
    if not required.issubset(data):
        raise ValueError("selected_files.csv: отсутствуют обязательные поля")
    if data.file.duplicated().any() or data.empty:
        raise ValueError("Дубли/пустой selected_files.csv")
    root = dataset_dir.resolve()
    paths = []
    for row in data.itertuples(index=False):
        folder = int(row.event_class_folder)
        well = int(row.well_id)
        rel = Path(str(row.file))
        if folder == 2 or folder not in ctl.CLASS_FOLDERS:
            raise ValueError("Недопустимый класс в контроле")
        if len(rel.parts) != 2 or rel.parts[0] != str(folder) or ctl.well_id_from_name(rel.name) != well:
            raise ValueError("Несоответствие путей/скважин в selected_files.csv")
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise FileNotFoundError(f"Отсутствует контрольный файл {p}")
        paths.append(str(p))
    data["path"] = paths
    return data


def validate_controls(controls: pd.DataFrame, class2: pd.DataFrame) -> None:
    fields = {"well_id", "file", "decision_time", "event_class_folder", *FEATURES}
    if controls.empty or not fields.issubset(controls):
        raise ValueError("Контрольная выборка пуста или не содержит признаков")
    if controls.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли окон в контроле")
    if controls.file.astype(str).str.split("/").str[-1].isin(class2.file.astype(str)).any():
        raise ValueError("Один и тот же файл присутствует в обеих выборках")
    if (controls.event_class_folder == 2).any():
        raise ValueError("Класс2 в отрицательных контролях")
    if not set(FEATURES).issubset(class2):
        raise ValueError("Отсутствуют признаки класса2")


def sample_train_controls(controls: pd.DataFrame, allowed_wells: set[int],
                          per_file: int = 12, per_well_class: int = 24) -> pd.DataFrame:
    """Ограничиваем коррелированные окна на один файл и скважину/класс.

    Порядок времени для отбора окон фиксирован, score/target не используются.
    """
    if per_file < 1 or per_well_class < 1:
        raise ValueError("Неверный лимит окон")
    allowed = controls.loc[controls.well_id.isin(allowed_wells)].copy()
    portions = []
    for _, sub in allowed.groupby("file", sort=True):
        sub = sub.sort_values("decision_time", kind="stable")
        k = min(per_file, len(sub))
        positions = np.floor(np.arange(k, dtype=float) * len(sub) / k).astype(int)
        portions.append(sub.iloc[positions])
    if not portions:
        raise ValueError("Нет доступных контрольных окон для обучения")
    sampled = pd.concat(portions)
    parts = []
    for _, sub in sampled.groupby(["well_id", "event_class_folder"], sort=True):
        sub = sub.sort_values(["file", "decision_time"], kind="stable")
        k = min(per_well_class, len(sub))
        positions = np.floor(np.arange(k, dtype=float) * len(sub) / k).astype(int)
        parts.append(sub.iloc[positions])
    return pd.concat(parts, ignore_index=True).sort_values(
        ["well_id", "event_class_folder", "file", "decision_time"], kind="stable").reset_index(drop=True)


def fit_catboost(train: pd.DataFrame, extra_negatives: pd.DataFrame | None,
                 iterations: int = 300):
    """Обучаем одну и ту же архитектуру с/без hard negatives."""
    frame = train[["well_id", "target", *FEATURES]].copy()
    if extra_negatives is not None:
        extra = extra_negatives[["well_id", *FEATURES]].copy()
        extra["target"] = 0
        frame = pd.concat([frame, extra[frame.columns]], ignore_index=True)
    y = frame.target.to_numpy(dtype=int)
    if len(set(y)) != 2:
        raise ValueError("В обучении нужны оба класса")
    model = catboost_model(iterations)
    model.fit(frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float), y)
    return model


def score_model(model, df: pd.DataFrame) -> np.ndarray:
    values = model.predict_proba(df[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float))[:, 1]
    if not np.isfinite(values).all():
        raise RuntimeError("Модель выдала нечисловые оценки")
    return values


def make_outer_predictions(training: pd.DataFrame, controls: pd.DataFrame,
                           development: set[int], external: set[int],
                           iterations: int = 300, per_file: int = 12,
                           per_well_class: int = 24):
    """Внешняя LOOW по 7 скважинам, плюс тест на полностью новых контролях.

    Из обоих вариантов модели исключена ВСЯ тестовая скважина.
    Зарезервированные внешние контроли НИКОГДА не участвуют в fit.
    """
    validate_controls(controls, training)
    class2_wells = set(map(int, training.well_id.unique()))
    if len(class2_wells) < 3 or (class2_wells & external) or (development & external):
        raise ValueError("Невалидные группы скважин")
    # Из 33+ контрольных скважин лишь development поступают в обучение.
    all_allowed = class2_wells | development
    test_control_wells = class2_wells & set(map(int, controls.well_id.unique()))
    reports, class2_scores, known_scores = [], [], []
    chosen_controls = sample_train_controls(controls, all_allowed, per_file, per_well_class)
    for test_well in sorted(class2_wells):
        train = training.loc[training.well_id.ne(test_well)].copy()
        test = training.loc[training.well_id.eq(test_well)].copy()
        additional = chosen_controls.loc[chosen_controls.well_id.ne(test_well)].copy()
        if test.empty or additional.empty or test_well in set(train.well_id) | set(additional.well_id):
            raise AssertionError("Утечка или пустое разбиение LOOW")
        test_controls = controls.loc[controls.well_id.eq(test_well)].copy()
        for variant in ("baseline", "hard_negative"):
            model = fit_catboost(train, additional if variant == "hard_negative" else None, iterations)
            p = test[["file", "well_id", "decision_time", "target", "event_time"]].copy()
            p["score"] = score_model(model, test)
            p["variant"] = variant
            class2_scores.append(p)
            if not test_controls.empty:
                c = test_controls[["file", "well_id", "decision_time", "event_class_folder", "control_phase"]].copy()
                c["score"] = score_model(model, test_controls)
                c["variant"] = variant
                c["cohort"] = "loow_class2_wells"
                known_scores.append(c)
        reports.append({"outer_well": test_well, "train_class2_rows": len(train),
                        "test_class2_rows": len(test),
                        "train_control_rows": len(additional),
                        "test_control_rows": len(test_controls),
                        "train_control_wells": additional.well_id.nunique()})
        print(f"LOOW WELL-{test_well:05d}: class2 train/test {len(train)}/{len(test)}, "
              f"hard negatives train {len(additional)}, control test {len(test_controls)}")
    # Отдельная проверка на совсем других скважинах, без DHSV-событий.
    external_controls = controls.loc[controls.well_id.isin(external)].copy()
    external_scores = []
    if not external_controls.empty:
        for variant in ("baseline", "hard_negative"):
            model = fit_catboost(training, chosen_controls if variant == "hard_negative" else None, iterations)
            c = external_controls[["file", "well_id", "decision_time", "event_class_folder", "control_phase"]].copy()
            c["score"] = score_model(model, external_controls)
            c["variant"] = variant
            c["cohort"] = "heldout_control_only_wells"
            external_scores.append(c)
    all_controls = known_scores + external_scores
    return (pd.concat(class2_scores, ignore_index=True),
            pd.concat(all_controls, ignore_index=True) if all_controls else pd.DataFrame(),
            chosen_controls, pd.DataFrame(reports))


def apply_fixed_policy(df: pd.DataFrame, confirmations: int) -> pd.DataFrame:
    """Одинаковая политика у обеих моделей, без подбора на тесте."""
    out = []
    for variant, part in df.groupby("variant", sort=True):
        applied = trigger_policy(part, threshold=0.5, confirmations=confirmations)
        out.append(applied)
    return pd.concat(out).sort_values(["variant", "file", "decision_time"], kind="stable")


def score_event_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import average_precision_score
    rows, well_rows = [], []
    for variant, part in scores.groupby("variant", sort=True):
        one = apply_fixed_policy(part, confirmations=1)
        two = apply_fixed_policy(part, confirmations=2)
        for policy, dataset in (("fixed05_1", one), ("fixed05_2", two)):
            y = dataset.target.to_numpy(dtype=int)
            alerts = dataset.alert_active.to_numpy(dtype=bool)
            true_events = dataset.loc[dataset.target.eq(1), ["file", "event_time"]].drop_duplicates()
            detected = dataset.loc[dataset.target.eq(1) & dataset.alert_active,
                                   ["file", "event_time"]].drop_duplicates()
            rows.append({"variant": variant, "policy": policy, "n": len(dataset),
                         "positive": int(y.sum()),
                         "ap": float(average_precision_score(y, dataset.score)),
                         "precision": float(precision_score(y, alerts, zero_division=0)),
                         "recall": float(recall_score(y, alerts, zero_division=0)),
                         "f1": float(f1_score(y, alerts, zero_division=0)),
                         "fp_windows": int((alerts & (y == 0)).sum()),
                         "negative_active_fraction": float(alerts[y == 0].mean()) if (y == 0).any() else None,
                         "alarm_starts": int(dataset.alert_start.sum()),
                         "out_of_horizon_alarm_starts": int((dataset.alert_start & dataset.target.eq(0)).sum()),
                         "events_detected_active": len(detected),
                         "events_total": len(true_events)})
            for well, sub in dataset.groupby("well_id", sort=True):
                pos = sub.loc[sub.target.eq(1)]
                ev = pos[["file", "event_time"]].drop_duplicates()
                found = pos.loc[pos.alert_active, ["file", "event_time"]].drop_duplicates()
                well_rows.append({"variant": variant, "policy": policy, "well_id": int(well),
                                  "events_total": len(ev), "events_detected_active": len(found),
                                  "n": len(sub), "fp_windows": int((sub.target.eq(0) & sub.alert_active).sum())})
    return pd.DataFrame(rows), pd.DataFrame(well_rows)


def score_control_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        raise ValueError("Нет удержанных контрольных оценок")
    output, details, per_well = [], [], []
    for policy, conf in (("fixed05_1", 1), ("fixed05_2", 2)):
        applied = apply_fixed_policy(scores, conf)
        for (variant, cohort), part in applied.groupby(["variant", "cohort"], sort=True):
            output.append({"variant": variant, "cohort": cohort, "policy": policy,
                           "wells": part.well_id.nunique(), "files": part.file.nunique(),
                           "windows": len(part), "eligible_negative_hours": len(part)*STRIDE_SECONDS/3600,
                           "alarm_starts": int(part.alert_start.sum()),
                           "false_alarm_starts_per_eligible_hour": int(part.alert_start.sum()) /
                               (len(part)*STRIDE_SECONDS/3600),
                           "active_windows": int(part.alert_active.sum()),
                           "active_fraction": float(part.alert_active.mean())})
        for (variant, cohort, event_class), part in applied.groupby(
                ["variant", "cohort", "event_class_folder"], sort=True):
            details.append({"variant": variant, "cohort": cohort, "event_class_folder": int(event_class),
                            "policy": policy, "windows": len(part),
                            "alarm_starts": int(part.alert_start.sum()),
                            "active_fraction": float(part.alert_active.mean())})
        for (variant, cohort, well), part in applied.groupby(["variant", "cohort", "well_id"], sort=True):
            per_well.append({"variant": variant, "cohort": cohort, "well_id": int(well),
                             "policy": policy, "windows": len(part),
                             "alarm_starts": int(part.alert_start.sum()),
                             "active_fraction": float(part.alert_active.mean())})
    return pd.DataFrame(output), pd.DataFrame(details), pd.DataFrame(per_well)


def run(training_path: Path, selected_path: Path, dataset_dir: Path, output_dir: Path,
        iterations: int = 300, per_file: int = 12, per_well_class: int = 24):
    import catboost
    training = ctl.load_training(training_path)
    if "event_time" not in training.columns:
        raise ValueError("В class2-когорте отсутствует столбец event_time")
    selected = load_selected(selected_path, dataset_dir)
    development, external = assign_external_holdout(
        set(map(int, selected.well_id)), set(map(int, training.well_id)))
    print(f"Контрольные скважины только для проверки: {sorted(external)}")
    print(f"Контрольные скважины для обучения: {sorted(development)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    controls, audit = ctl.build_controls(selected)
    # При ошибках сохраняем аудит, чтобы невозможно было принять пропущенные
    # файлы за отсутствие тревог.
    audit.to_csv(output_dir / "control_file_audit.csv", index=False)
    if audit.get("error", pd.Series(dtype=object)).notna().sum():
        raise RuntimeError(f"Есть ошибки чтения контрольных файлов; аудит: {output_dir / 'control_file_audit.csv'}")
    validate_controls(controls, training)
    class2, control_scores, chosen, folds = make_outer_predictions(
        training, controls, development, external, iterations, per_file, per_well_class)
    event_metrics, event_by_well = score_event_metrics(class2)
    control_metrics, control_by_class, control_by_well = score_control_metrics(control_scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.drop(columns=["path"]).to_csv(output_dir / "selected_files.csv", index=False)
    audit.to_csv(output_dir / "control_file_audit.csv", index=False)
    folds.to_csv(output_dir / "outer_folds.csv", index=False)
    chosen[["file", "well_id", "decision_time", "event_class_folder", "control_phase"]].to_csv(
        output_dir / "training_controls_sample.csv", index=False)
    class2.to_csv(output_dir / "class2_oof_scores.csv", index=False)
    control_scores.to_csv(output_dir / "control_heldout_scores.csv", index=False)
    event_metrics.to_csv(output_dir / "event_comparison.csv", index=False)
    event_by_well.to_csv(output_dir / "event_by_well.csv", index=False)
    control_metrics.to_csv(output_dir / "control_comparison.csv", index=False)
    control_by_class.to_csv(output_dir / "control_by_class.csv", index=False)
    control_by_well.to_csv(output_dir / "control_by_well.csv", index=False)
    evidence = {
        "experiment": "class2_hard_negative_catboost_30m_v0.1",
        "dataset": "Petrobras 3W v2.0.0, real WELL",
        "training": "class2 30m risk set + sampled verified negative controls",
        "control_file_selection": "reused controls v0.1 selected_files.csv; NO new manual selection",
        "control_only_split": "SHA256 on well ID only; outside-class2 wells separated by group",
        "salt": SPLIT_SALT,
        "external_control_only_test_wells": sorted(external),
        "development_control_only_wells": sorted(development),
        "class2_loow_wells": sorted(map(int, training.well_id.unique())),
        "per_file_cap": per_file, "per_well_class_cap": per_well_class,
        "training_controls_used": len(chosen),
        "class2_windows": len(training), "control_windows_available": len(controls),
        "model_iterations": iterations,
        "model_params": "unchanged from catboost_model in evaluate_class2_alert_policy.py",
        "fixed_policy": ["0.5 threshold / 1 evaluation", "0.5 threshold / 2 consecutive evaluations"],
        "no_oracle_gate": True,
        "labels_in_inference": False,
        "training_csv_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "selected_csv_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "catboost": catboost.__version__},
        "limitations": [
            "All wells/controls have been reviewed in preceding research; not a new untouched validation.",
            "External controls are held out by WELL for this experiment; class2 LOOW is separate.",
            "Control-only holdout contains no class2 positive events; cannot measure recall on new control wells.",
            "Class2 retrospective pre-event cohort selection still uses true labels for evaluation.",
            "Event-enriched class2 cohorts and sampled controls do not define production prevalence.",
            "A non-class2 hazard is negative ONLY for specificity of class2, not a safe state.",
            "Fixed threshold 0.5 is uncalibrated; comparison isolates training-set change.",
            "5-minute evaluation and censoring affect alert episode and exposure statistics.",
            "No final model selected, no deployment, no Streamlit/ML Agent changes.",
        ],
    }
    (output_dir / "experiment.json").write_text(json.dumps(evidence, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
    print("\n=== ДЕТЕКЦИЯ DHSV НА LOOW-СКВАЖИНАХ (30 мин) ===")
    print(event_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== ЛОЖНЫЕ ТРЕВОГИ НА УДЕРЖАННЫХ КОНТРОЛЯХ ===")
    print(control_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nРезультаты: {output_dir.resolve()}")
    print("ВНИМАНИЕ: предварительный эксперимент; НЕ промышленная валидация")
    return event_metrics, control_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training", type=Path,
                    default=ROOT / "outputs/class2_horizons/cohort_features.csv")
    ap.add_argument("--selected", type=Path,
                    default=ROOT / "outputs/class2_specificity/selected_files.csv")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/class2_hard_negatives")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--per-file", type=int, default=12)
    ap.add_argument("--per-well-class", type=int, default=24)
    args = ap.parse_args()
    run(args.training, args.selected, args.dataset, args.output,
        args.iterations, args.per_file, args.per_well_class)


if __name__ == "__main__":
    main()
