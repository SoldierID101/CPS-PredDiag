"""Проверка специфичности CatBoost-12/30 мин на файлах БЕЗ класса 2.

Запуск из корня CPS-PredDiag:
    python experiments/evaluate_class2_controls.py

Это исследовательский стресс-тест на выборке class 0 и других классов,
НЕ итоговая оценка частоты ложных тревог в эксплуатации.

Обучение: прежняя 30-минутная class2-когорта, только TRAIN скважины.
Контроли: ранее не входившие в class2-когорту реальные WELL-файлы.
Для знакомых well_id используется соответствующая LOOW-модель (без этой
скважины в обучении). Для остальных — отдельная модель на всех 7 скважинах;
эти две группы НЕЛЬЗЯ объединять для утверждений о качестве.
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

from experiments import blind_class2_forecast as blind
from experiments.evaluate_class2_alert_policy import trigger_policy

HORIZON_MIN = 30
HORIZON_SECONDS = HORIZON_MIN * 60
PAST_SECONDS = blind.PAST_SECONDS
STRIDE_SECONDS = blind.STRIDE_SECONDS
FEATURES = tuple(blind.BASELINE_FEATURES)
CLASS_FOLDERS = (0, 1, 3, 4, 5, 6, 7, 8, 9)
WELL_PATTERN = re.compile(r"WELL-(\d{5})_\d+\.parquet$")


def well_id_from_name(name: str) -> int:
    match = WELL_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Некорректное имя файла WELL: {name}")
    return int(match.group(1))


def choose_control_files(dataset_root: Path, classes: tuple[int, ...] = CLASS_FOLDERS,
                         per_class_well: int = 2) -> pd.DataFrame:
    """Детерминированный отбор: до N файлов на (класс, скважина).

    Приоритет по SHA256 пути внутри dataset: не выбираем вручную «удачные»
    эпизоды и не используем данные/разметку внутри файла для отбора.
    """
    if per_class_well < 1 or not classes or len(set(classes)) != len(classes):
        raise ValueError("Неверные параметры выборки")
    if any(c == 2 or c < 0 or c > 9 for c in classes):
        raise ValueError("Контрольная выборка не должна содержать класс 2")
    selected = []
    for event_class in classes:
        folder = Path(dataset_root) / str(event_class)
        if not folder.is_dir():
            raise FileNotFoundError(f"Отсутствует папка датасета: {folder}")
        buckets = {}
        for path in folder.glob("WELL-*.parquet"):
            well = well_id_from_name(path.name)
            buckets.setdefault(well, []).append(path)
        for well in sorted(buckets):
            options = sorted(buckets[well], key=lambda p: (
                hashlib.sha256(f"{event_class}/{p.name}".encode()).hexdigest(), p.name))
            for path in options[:per_class_well]:
                selected.append({
                    "event_class_folder": event_class,
                    "well_id": well,
                    "file": f"{event_class}/{path.name}",
                    "path": str(path),
                    "source_bytes": path.stat().st_size,
                    "group_available_files": len(options),
                })
    if not selected:
        raise FileNotFoundError("Реальные WELL-файлы для контроля не найдены")
    return pd.DataFrame(selected).sort_values(
        ["event_class_folder", "well_id", "file"], kind="stable").reset_index(drop=True)


def future_is_verifiable_negative(labels: np.ndarray, at: int, segment_end: int,
                                  class_folder: int) -> str:
    """Использует метки ТОЛЬКО для оценки полноты исхода, не для признаков.

    Для отрицательного исхода должно наблюдаться всё будущее в 30 минут,
    без class2, class102, неизвестных и некорректных меток.
    """
    future = labels[at + 1: min(at + 1 + HORIZON_SECONDS, segment_end)]
    if len(future) != HORIZON_SECONDS:
        return "right_censored"
    entire = np.r_[labels[at], future]
    if not np.isfinite(entire).all() or np.any(entire < 0) or np.any(entire != np.floor(entire)):
        return "unknown_labels"
    if np.isin(entire, [2, 102]).any():
        return "contains_class2_or_102"
    if class_folder == 0 and not np.all(entire == 0):
        return "class0_file_not_all_normal"
    return "known_no_class2"


def current_phase(class_folder: int, label: float) -> str:
    if class_folder == 0:
        return "normal_only_file"
    if label == 0:
        return "pre_other_event"
    if 100 <= label < 200:
        return "other_transition"
    return "other_event"


def extract_negative_controls(frame: pd.DataFrame, class_folder: int,
                              file_id: str, well_id: int) -> tuple[pd.DataFrame, dict]:
    """Сетка зависит только от времени. Для метрик отсеивается неизвестный исход.

    Никаких подтверждений класса 102 и никакого доступа модели к class/state.
    Признаки считаются исходной функцией blind.past_only_features.
    """
    prepared = blind.prepare_frame(frame)
    labels = pd.to_numeric(prepared["class"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    audit = {"file": file_id, "well_id": well_id,
             "event_class_folder": class_folder, "source_rows": len(prepared),
             "scheduled_windows": 0, "known_negative_windows": 0,
             "unusable_sensors": 0, "right_censored": 0,
             "unknown_labels": 0, "contains_class2_or_102": 0,
             "class0_file_not_all_normal": 0}
    rows = []
    for begin, end in blind.contiguous_segments(prepared.index):
        for i in range(begin + PAST_SECONDS - 1, end, STRIDE_SECONDS):
            audit["scheduled_windows"] += 1
            state = future_is_verifiable_negative(labels, i, end, class_folder)
            if state != "known_no_class2":
                audit[state] += 1
                continue
            features, usable = blind.past_only_features(prepared, i)
            if usable < 2:
                audit["unusable_sensors"] += 1
                continue
            audit["known_negative_windows"] += 1
            rows.append({"file": file_id, "well_id": well_id,
                         "event_class_folder": class_folder,
                         "control_phase": current_phase(class_folder, labels[i]),
                         "decision_time": prepared.index[i],
                         **{k: features[k] for k in FEATURES}})
    return pd.DataFrame(rows), audit


def build_controls(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty or not {"path", "file", "well_id", "event_class_folder"}.issubset(selected):
        raise ValueError("Отсутствуют выбранные файлы контроля")
    rows, audits = [], []
    for item in selected.itertuples(index=False):
        try:
            frame = pd.read_parquet(item.path)
            windows, audit = extract_negative_controls(
                frame, int(item.event_class_folder), str(item.file), int(item.well_id))
        except Exception as exc:
            # Нет ложного вывода «нулевая частота тревог» при ошибках чтения.
            audits.append({"file": item.file, "well_id": item.well_id,
                           "event_class_folder": item.event_class_folder,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"ОШИБКА {item.file}: {type(exc).__name__}: {exc}")
            continue
        audits.append(audit)
        if not windows.empty:
            rows.append(windows)
        print(f"{item.file}: проверено {audit['known_negative_windows']}/"
              f"{audit['scheduled_windows']} окон, пропуски датчиков {audit['unusable_sensors']}")
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return data, pd.DataFrame(audits)


def load_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"horizon_minutes", "file", "well_id", "decision_time", "target", *FEATURES}
    if not required.issubset(df):
        raise ValueError("Недостаточно столбцов в обучающей выборке горизонтов")
    df = df.loc[df.horizon_minutes.eq(HORIZON_MIN)].copy()
    if df.empty or df.well_id.nunique() < 4 or df.target.nunique() != 2:
        raise ValueError("Нужны данные горизонта 30 мин, >=4 скважин и двух классов")
    if not df.target.isin([0, 1]).all() or df.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли или некорректные целевые метки")
    if not np.isfinite(df["well_id"].to_numpy(dtype=float)).all():
        raise ValueError("Некорректные идентификаторы скважин")
    return df


def load_policy(path: Path, training_wells: set[int]) -> dict[int, tuple[float, int]]:
    df = pd.read_csv(path)
    needed = {"outer_test_well", "threshold", "confirmations"}
    if not needed.issubset(df):
        raise ValueError("Не хватает полей в policy_by_well.csv")
    if df.outer_test_well.duplicated().any():
        raise ValueError("Дубли outer_test_well")
    result = {}
    for item in df.itertuples(index=False):
        well = int(item.outer_test_well)
        threshold = float(item.threshold)
        conf = int(item.confirmations)
        if not (0 < threshold < 1 and conf in (1, 2)):
            raise ValueError("Недопустимая настройка политики")
        result[well] = threshold, conf
    if not training_wells.issubset(result):
        raise ValueError("Не для всех обучающих скважин найдено вложенное правило")
    return result


def train_and_score(training: pd.DataFrame, controls: pd.DataFrame,
                    iterations: int = 300) -> pd.DataFrame:
    """Для известных скважин — train != test well. Для новых — train all.

    Модель НИКОГДА не обучается на файлах контроля или их метках.
    """
    from experiments.evaluate_class2_alert_policy import catboost_model
    if controls.empty or iterations < 1:
        raise ValueError("Нет пригодных контрольных окон / неверное iterations")
    if controls.duplicated(["file", "decision_time"]).any():
        raise ValueError("Дубли контрольных моментов")
    x = training[list(FEATURES)].replace([np.inf, -np.inf], np.nan).astype(float)
    y = training.target.to_numpy(dtype=int)
    wells = training.well_id.to_numpy(dtype=int)
    train_wells = set(wells.tolist())
    control_wells = controls.well_id.to_numpy(dtype=int)
    # Гарантируем, что ни один исходный файл из обучения не входит в контроли.
    base_names = controls.file.astype(str).str.split("/").str[-1]
    if base_names.isin(training.file.astype(str)).any():
        raise ValueError("Файл контроля присутствует в обучающей выборке")
    pred = controls.copy()
    pred["score"] = np.nan
    pred["training_mode"] = ""
    for well in sorted(np.unique(control_wells)):
        known = well in train_wells
        fit = wells != well if known else np.ones(len(training), dtype=bool)
        if np.unique(y[fit]).size != 2:
            raise ValueError(f"В обучении модели для well={well} остался один класс")
        model = catboost_model(iterations)
        model.fit(x.iloc[np.flatnonzero(fit)], y[fit])
        selected = np.flatnonzero(control_wells == well)
        scores = model.predict_proba(
            pred.iloc[selected][list(FEATURES)].replace([np.inf, -np.inf], np.nan).astype(float))[:, 1]
        pred.loc[pred.index[selected], "score"] = scores
        pred.loc[pred.index[selected], "training_mode"] = (
            "loow_heldout_well" if known else "new_well_fullfit")
        print(f"  WELL-{well:05d}: {len(selected)} контрольных окон; "
              f"обучено на {int(fit.sum())} class2-окнах; "
              f"режим {'LOOW' if known else 'unseen/full-fit'}")
    if not pred.score.between(0, 1).all() or pred.score.isna().any():
        raise RuntimeError("Не удалось получить все оценки")
    return pred


def apply_policies(scored: pd.DataFrame,
                   rules: dict[int, tuple[float, int]]) -> pd.DataFrame:
    """Сохраняем три правила; для новых скважин nested-порог НЕ переносим.

    Старты тревог считаются лишь на последовательных пригодных окнах,
    поэтому частота ниже/выше реальной может оказаться смещённой.
    """
    if scored.empty or not {"well_id", "file", "decision_time", "score", "training_mode"}.issubset(scored):
        raise ValueError("Нет оценок для формирования тревог")
    all_parts = []
    for well, subset in scored.groupby("well_id", sort=True):
        f1 = trigger_policy(subset, threshold=0.5, confirmations=1)
        f2 = trigger_policy(subset, threshold=0.5, confirmations=2)
        if (subset.training_mode == "loow_heldout_well").all():
            threshold, conf = rules[int(well)]
            source = "inner_train_only"
        else:
            threshold, conf = (0.5, 2)
            source = "fixed_fallback_unseen"
        nested = trigger_policy(subset, threshold, conf)
        part = subset.copy()
        for name, outcome in (("fixed05_1", f1), ("fixed05_2", f2), ("nested", nested)):
            part[f"{name}_active"] = outcome.loc[part.index, "alert_active"].astype(bool)
            part[f"{name}_start"] = outcome.loc[part.index, "alert_start"].astype(bool)
        part["nested_threshold"] = threshold
        part["nested_confirmations"] = conf
        part["nested_source"] = source
        all_parts.append(part)
    return pd.concat(all_parts).sort_values(["file", "decision_time"], kind="stable").reset_index(drop=True)


def summarize_controls(pred: pd.DataFrame) -> pd.DataFrame:
    """ТОЛЬКО ложные class2-тревоги на проверенных отрицательных окнах.

    Никаких TP/Recall/Precision: контроли заведомо не содержат class2.
    """
    rows = []
    for policy in ("fixed05_1", "fixed05_2", "nested"):
        for (cohort, cls, phase), subset in pred.groupby(
                ["training_mode", "event_class_folder", "control_phase"], sort=True):
            n = len(subset)
            starts = int(subset[f"{policy}_start"].sum())
            active = int(subset[f"{policy}_active"].sum())
            eligible_hours = n * STRIDE_SECONDS / 3600
            rows.append({
                "policy": policy, "training_mode": cohort,
                "event_class_folder": int(cls), "control_phase": phase,
                "windows": n, "files": subset.file.nunique(),
                "wells": subset.well_id.nunique(),
                "eligible_negative_hours": eligible_hours,
                "false_alarm_starts": starts,
                "alert_active_windows": active,
                "starts_per_eligible_negative_hour": starts / eligible_hours,
                "active_window_fraction": active / n,
            })
    return pd.DataFrame(rows)


def summarize_controls_by_well(pred: pd.DataFrame) -> pd.DataFrame:
    """Детализация по скважине для поиска источников ложных тревог."""
    rows = []
    for policy in ("fixed05_1", "fixed05_2", "nested"):
        keys = ["training_mode", "well_id", "event_class_folder", "control_phase"]
        for (mode, well, cls, phase), group in pred.groupby(keys, sort=True):
            starts = int(group[f"{policy}_start"].sum())
            hours = len(group) * STRIDE_SECONDS / 3600
            rows.append({"policy": policy, "training_mode": mode,
                         "well_id": int(well), "event_class_folder": int(cls),
                         "control_phase": phase, "windows": len(group),
                         "files": group.file.nunique(),
                         "eligible_negative_hours": hours,
                         "false_alarm_starts": starts,
                         "alert_active_windows": int(group[f"{policy}_active"].sum()),
                         "starts_per_eligible_negative_hour": starts/hours})
    return pd.DataFrame(rows)


def overall_by_training_mode(detail: pd.DataFrame) -> pd.DataFrame:
    """НЕ объединяет LOOW и unseen/fullfit в одну оценку."""
    rolled = detail.groupby(["policy", "training_mode"], as_index=False)[
        ["windows", "eligible_negative_hours", "false_alarm_starts",
         "alert_active_windows"]].sum()
    rolled["starts_per_eligible_negative_hour"] = (
        rolled.false_alarm_starts / rolled.eligible_negative_hours)
    return rolled


def run(root: Path, training_path: Path, policy_path: Path, out_dir: Path,
        classes: tuple[int, ...], per_class_well: int, iterations: int):
    import catboost
    import sklearn
    training = load_training(training_path)
    train_wells = set(training.well_id.astype(int).unique())
    rules = load_policy(policy_path, train_wells)
    selection = choose_control_files(root, classes=classes, per_class_well=per_class_well)
    print(f"Выбрано контрольных файлов: {len(selection)}; классы: {sorted(selection.event_class_folder.unique())}")
    controls, audit = build_controls(selection)
    if controls.empty:
        raise ValueError("Нет пригодных контрольных окон; см. ошибки/цензуру")
    print(f"Пригодно {len(controls)} отрицательных окон на {controls.well_id.nunique()} скважинах")
    scored = train_and_score(training, controls, iterations=iterations)
    result = apply_policies(scored, rules)
    summary = summarize_controls(result)
    per_well = summarize_controls_by_well(result)
    overall = overall_by_training_mode(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection.drop(columns="path").to_csv(out_dir / "selected_files.csv", index=False)
    audit.to_csv(out_dir / "file_audit.csv", index=False)
    result.drop(columns=list(FEATURES)).to_csv(out_dir / "control_predictions.csv", index=False)
    summary.to_csv(out_dir / "by_class_and_phase.csv", index=False)
    per_well.to_csv(out_dir / "by_well.csv", index=False)
    overall.to_csv(out_dir / "overall_by_training_mode.csv", index=False)
    for cohort, part in overall.groupby("training_mode"):
        print(f"\n{cohort}: ")
        for row in part.itertuples(index=False):
            print(f"  {row.policy}: {row.false_alarm_starts} начал тревог / "
                  f"{row.eligible_negative_hours:.1f} ч пригодных отрицательных окон "
                  f"= {row.starts_per_eligible_negative_hour:.3f} /ч")
    errors = int(audit.error.notna().sum()) if "error" in audit else 0
    provenance = {
        "experiment": "class2_30min_specificity_controls_v0.1",
        "training": "class2 30-minute event-enriched cohort_features.csv",
        "control_source": "real WELL files in selected classes excluding class2",
        "selected_classes": list(classes), "per_class_well": per_class_well,
        "selection": "SHA256-order per (class folder, well); no label-dependent selection",
        "num_files_selected": len(selection), "num_files_failed": errors,
        "num_control_windows": len(result), "num_training_windows": len(training),
        "training_csv_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "policy_csv_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "training_wells": sorted(map(int, train_wells)),
        "features": list(FEATURES), "model_iterations": iterations,
        "inference_gate": "none; future labels used exclusively to verify evaluation negatives",
        "policy": "previous nested thresholds for LOOW wells; fixed 0.5,2 for unseen wells",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "limitations": [
            "Class 0 files may not cover all operational regimes; 2 files per class/well is a sample.",
            "Other-class signals are not physical negatives for every general hazard, only class-2 specificity controls.",
            "Models trained on class-2 event-enriched episodes, not representative operations.",
            "Known well controls use LOOW; unseen well controls use a fullfit model. Do not pool estimates.",
            "Previous experiments studied the same well cohort and chose 30m; not untouched validation.",
            "Missing/unknown labels and unreadable files are excluded but separately audited.",
            "Alarm starts and eligible-hour estimates depend on sparse 5-minute decision schedule.",
            "No measurement of class-2 recall here; no production FAR inference.",
        ],
    }
    (out_dir / "experiment.json").write_text(json.dumps(provenance, ensure_ascii=False,
                    indent=2), encoding="utf-8")
    print(f"Отчёты: {out_dir.resolve()}")
    if errors:
        print(f"ВНИМАНИЕ: не удалось прочесть {errors} файлов; не трактуйте пропуски как нулевые тревоги")
    print("НАУЧНОЕ ОГРАНИЧЕНИЕ: эти контроли проверяют специфичность к классу 2, а не полевую FAR")
    return result, audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--training", type=Path,
                        default=ROOT / "outputs/class2_horizons/cohort_features.csv")
    parser.add_argument("--policy", type=Path,
                        default=ROOT / "outputs/class2_alert_policy/policy_by_well.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/class2_specificity")
    parser.add_argument("--classes", type=str, default=",".join(map(str, CLASS_FOLDERS)))
    parser.add_argument("--per-class-well", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    try:
        classes = tuple(int(x.strip()) for x in args.classes.split(","))
    except ValueError as exc:
        parser.error("--classes: нужны номера классов через запятую")
        raise exc
    run(args.dataset, args.training, args.policy, args.output,
        classes, args.per_class_well, args.iterations)


if __name__ == "__main__":
    main()
