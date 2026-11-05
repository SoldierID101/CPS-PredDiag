"""Исследование компромисса hard negatives и обнаружения DHSV, 30 мин.

Запуск из корня CPS-PredDiag:
    python experiments/evaluate_class2_hard_negative_balance.py

НЕ меняет текущий ML Agent и не выполняет новую итоговую валидацию.
Разделение по WELL ID и правила оценки описаны в docs/hard_negative_balance.md.
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import evaluate_class2_hard_negatives as hn
from experiments import evaluate_class2_controls as ctl
from experiments.evaluate_class2_alert_policy import trigger_policy

# Заданы до вычисления тестовых результатов. Не подбирать по внешним скважинам.
RATIOS = (0.0, 0.25, 0.50, 1.0)
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CONFIRMATIONS = (1, 2)
INNER_EVENT_TARGET = 0.80
TUNE_SALT = "CPS-PredDiag hardneg tuning controls v0.1"
STRIDE_SECONDS = hn.STRIDE_SECONDS
FEATURES = hn.FEATURES


def _hash_unit(salt: str, well_id: int) -> float:
    return int(hashlib.sha256(f"{salt}:{int(well_id)}".encode()).hexdigest()[:12], 16) / (16**12)


def partition_development_controls(wells: set[int], fraction: float = 0.30,
                                   salt: str = TUNE_SALT) -> tuple[set[int], set[int]]:
    """Не используем метки, scores или оценки тестовых скважин для разделения."""
    if not 0 < fraction < 1 or len(wells) < 3:
        raise ValueError("Для разделения контролей нужны >=3 скважины и корректная доля")
    tune = {int(w) for w in wells if _hash_unit(salt, int(w)) < fraction}
    fit = set(map(int, wells)) - tune
    if not fit or not tune:
        raise ValueError("Не удалось получить непустые fit/tuning группы")
    return fit, tune


def ensure_tuning_class_coverage(fit_wells: set[int], tune_wells: set[int],
                                 selected: pd.DataFrame, required: tuple[int, ...] = (0, 8)
                                 ) -> tuple[set[int], set[int]]:
    """Сохраняем class0 и class8 в обоих пулах, не пользуясь scores/результатами.

    Папка события — известная характеристика источника для стратификации.
    Переносим минимальное количество WELL ID из fit в tune; tie — SHA256.
    """
    needed = {'well_id', 'event_class_folder'}
    if not needed.issubset(selected):
        raise ValueError('Нет полей для проверки охвата классов')
    fit, tune = set(fit_wells), set(tune_wells)
    folder_sets = {int(w): set(map(int, chunk.event_class_folder))
                   for w, chunk in selected.groupby('well_id')}
    for cls in required:
        if not any(cls in folder_sets.get(w, set()) for w in tune):
            candidates = [w for w in fit if cls in folder_sets.get(w, set())
                          and sum(cls2 in folder_sets.get(w, set()) for cls2 in required) >= 1]
            # Предпочтение WELL, одновременно покрывающей больше недостающих классов.
            missing = {c for c in required if not any(c in folder_sets.get(w, set()) for w in tune)}
            ranked = sorted(candidates, key=lambda w: (
                -len(folder_sets[w] & missing), _hash_unit(TUNE_SALT, w), w))
            chosen = next((w for w in ranked if
                           all(sum(other_cls in folder_sets.get(v, set()) for v in fit if v != w) >= 1
                               for other_cls in required)), None)
            if chosen is None:
                raise ValueError(f'Недостаточно control-only WELL для обеих групп класса {cls}')
            fit.remove(chosen)
            tune.add(chosen)
    for cls in required:
        if not any(cls in folder_sets.get(w, set()) for w in fit) or not any(
                cls in folder_sets.get(w, set()) for w in tune):
            raise ValueError(f'Класс {cls} не представлен в fit/tune отдельно')
    if fit & tune:
        raise AssertionError('Пересекающиеся WELL группы')
    return fit, tune


def fraction_of_controls(sampled: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Вложенные детерминированные подвыборки без просмотра target/score.

    Для 25% берём каждый четвёртый элемент группы, для 50% каждый второй.
    Одна строка от малой группы сохраняется, если ratio > 0.
    """
    if ratio not in RATIOS:
        raise ValueError("Неизвестный размер hard-negative набора")
    if ratio == 0:
        return sampled.iloc[0:0].copy()
    if sampled.empty:
        raise ValueError("Нет контрольных окон для ненулевого отношения")
    groups = []
    for _, part in sampled.groupby(["well_id", "event_class_folder", "file"], sort=True):
        ordered = part.sort_values("decision_time", kind="stable")
        count = min(len(ordered), max(1, int(np.ceil(len(ordered) * ratio))))
        # Внутри файла индексы упорядочены по времени; наборы 25% и 50% вложены.
        if ratio == 1:
            positions = np.arange(len(ordered))
        elif ratio == 0.5:
            positions = np.arange(0, len(ordered), 2)[:count]
        else:
            positions = np.arange(0, len(ordered), 4)[:count]
        groups.append(ordered.iloc[positions])
    return pd.concat(groups, ignore_index=True)


def _model(train: pd.DataFrame, negatives: pd.DataFrame, iterations: int):
    return hn.fit_catboost(train, negatives if not negatives.empty else None, iterations)


def _score_class2(model, data: pd.DataFrame) -> pd.DataFrame:
    out = data[["file", "well_id", "decision_time", "target", "event_time"]].copy()
    out["score"] = hn.score_model(model, data)
    return out


def _score_controls(model, data: pd.DataFrame) -> pd.DataFrame:
    out = data[["file", "well_id", "decision_time", "event_class_folder", "control_phase"]].copy()
    out["score"] = hn.score_model(model, data)
    return out


def inner_predictions(train: pd.DataFrame, pool: pd.DataFrame, ratio: float,
                      iterations: int) -> pd.DataFrame:
    """Внутренние LOOW оценки. Ни тестовая class2 WELL, ни её контроли не учатся."""
    wells = sorted(map(int, train.well_id.unique()))
    if len(wells) < 3:
        raise ValueError("Внутреннему LOOW нужны >=3 скважины")
    chunks = []
    for well in wells:
        fit = train.loc[train.well_id.ne(well)]
        test = train.loc[train.well_id.eq(well)]
        extra = fraction_of_controls(pool.loc[pool.well_id.ne(well)], ratio)
        if well in set(fit.well_id) or well in set(extra.well_id):
            raise AssertionError("Утечка внутренней контрольной скважины")
        chunks.append(_score_class2(_model(fit, extra, iterations), test))
    result = pd.concat(chunks, ignore_index=True)
    if len(result) != len(train) or result.duplicated(["file", "decision_time"]).any():
        raise RuntimeError("Нарушена полнота внутреннего OOF")
    return result


def policy_grid(event_scores: pd.DataFrame, tune_controls: pd.DataFrame,
                event_target: float = INNER_EVENT_TARGET) -> tuple[dict, pd.DataFrame]:
    """Выбор по inner OOF событиям и отдельным tune-only контрольным скважинам.

    Цель: >=80% обнаруженных эпизодов. Среди достижимых — минимальная
    средняя доля ложных активных окон в двух доменах: class2-negative и
    tune-only controls. Если цель недостижима — максимальный event recall.
    """
    if (event_scores.empty or tune_controls.empty or not 0 < event_target <= 1
            or "event_class_folder" not in tune_controls.columns):
        raise ValueError("Недостаточные входные данные для выбора правила")
    if set(event_scores.well_id) & set(tune_controls.well_id):
        raise ValueError("Tuning-контроли не должны совпадать с class2 WELL")
    assert event_scores.target.isin([0, 1]).all()
    events = event_scores.loc[event_scores.target.eq(1), ["file", "event_time"]].drop_duplicates()
    if events.empty or not (event_scores.target == 0).any():
        raise ValueError("Для выбора правила нужны события и отрицательные окна")
    rows = []
    for conf in CONFIRMATIONS:
        for thr in THRESHOLDS:
            ev = trigger_policy(event_scores, thr, conf)
            ct = trigger_policy(tune_controls, thr, conf)
            found = ev.loc[ev.target.eq(1) & ev.alert_active,
                           ["file", "event_time"]].drop_duplicates()
            rec = len(found) / len(events)
            neg = ev.target.eq(0)
            event_neg_active = float(ev.loc[neg, "alert_active"].mean())
            control_active = float(ct.alert_active.mean())
            # Macro по известным типам контроля: длинный класс8 не поглощает класс0.
            macro_control_active = float(ct.groupby("event_class_folder").alert_active.mean().mean())
            # Два домена имеют равный вес: class2-негативы и macro-контроли.
            burden = (event_neg_active + macro_control_active) / 2
            alarm_rate = (float(ct.alert_start.sum()) /
                          (len(ct) * STRIDE_SECONDS / 3600))
            rows.append({"threshold": thr, "confirmations": conf,
                         "inner_event_recall": rec, "inner_events": len(events),
                         "inner_event_negative_active": event_neg_active,
                         "tuning_control_active": control_active,
                         "tuning_control_active_macro": macro_control_active,
                         "tuning_alarm_starts_per_eligible_hour": alarm_rate,
                         "combined_burden": burden})
    grid = pd.DataFrame(rows)
    acceptable = grid.loc[grid.inner_event_recall >= event_target - 1e-12]
    if len(acceptable):
        ranked = acceptable.sort_values(
            ["combined_burden", "inner_event_recall", "tuning_alarm_starts_per_eligible_hour",
             "threshold", "confirmations"],
            ascending=[True, False, True, False, False], kind="stable")
        target_reached = True
    else:
        ranked = grid.sort_values(
            ["inner_event_recall", "combined_burden", "tuning_alarm_starts_per_eligible_hour",
             "threshold", "confirmations"],
            ascending=[False, True, True, False, False], kind="stable")
        target_reached = False
    winning = ranked.iloc[0].to_dict()
    return {"threshold": float(winning["threshold"]),
            "confirmations": int(winning["confirmations"]),
            "inner_target_reached": bool(target_reached),
            "inner_event_recall": float(winning["inner_event_recall"]),
            "inner_control_active": float(winning["tuning_control_active"]),
            "inner_control_active_macro": float(winning["tuning_control_active_macro"]),
            "inner_combined_burden": float(winning["combined_burden"])}, grid


def choose_nested_variant(selection: pd.DataFrame) -> dict:
    """Выбор ДОЛИ и порога исключительно по inner-статистике, до чтения outer-меток."""
    needed = {"ratio", "threshold", "confirmations", "inner_target_reached",
              "inner_event_recall", "inner_combined_burden"}
    if selection.empty or not needed.issubset(selection):
        raise ValueError("Нет достаточных внутренних результатов для выбора модели")
    good = selection.loc[selection.inner_target_reached.astype(bool)]
    if len(good):
        rank = good.sort_values(["inner_combined_burden", "inner_event_recall", "ratio",
                                 "threshold", "confirmations"],
                                ascending=[True, False, True, False, False], kind="stable")
    else:
        rank = selection.sort_values(["inner_event_recall", "inner_combined_burden",
                                      "ratio", "threshold", "confirmations"],
                                     ascending=[False, True, True, False, False], kind="stable")
    return rank.iloc[0].to_dict()


def apply_rule(scores: pd.DataFrame, policy: dict) -> pd.DataFrame:
    # Метки/класс/фаза отсутствуют на входе правила.
    scores = scores.copy()
    scores["decision_time"] = pd.to_datetime(scores["decision_time"])
    pure = scores[["file", "decision_time", "score"]]
    pred = trigger_policy(pure, float(policy["threshold"]), int(policy["confirmations"]))
    out = scores.merge(pred[["file", "decision_time", "alert_active", "alert_start"]],
                       on=["file", "decision_time"], validate="one_to_one")
    if len(out) != len(scores):
        raise RuntimeError("Потеря решений в правиле")
    return out


def event_row(scores: pd.DataFrame) -> dict:
    if scores.empty:
        raise ValueError("Пустые DHSV оценки")
    y = scores.target.to_numpy(dtype=int)
    active = scores.alert_active.to_numpy(dtype=bool)
    events = scores.loc[scores.target.eq(1), ["file", "event_time"]].drop_duplicates()
    found = scores.loc[scores.target.eq(1) & scores.alert_active,
                       ["file", "event_time"]].drop_duplicates()
    if not len(events):
        raise ValueError("Нет размеченных DHSV событий")
    neg = y == 0
    return {"n": len(scores), "positive": int(y.sum()),
            "ap": float(average_precision_score(y, scores.score)),
            "precision": float(precision_score(y, active, zero_division=0)),
            "recall": float(recall_score(y, active, zero_division=0)),
            "f1": float(f1_score(y, active, zero_division=0)),
            "fp_windows": int(np.sum(active & neg)),
            "class2_negative_active": float(active[neg].mean()) if neg.any() else None,
            "events_detected": len(found), "events_total": len(events),
            "event_recall": len(found) / len(events),
            "alarm_starts": int(scores.alert_start.sum()),
            "out_of_horizon_alarm_starts": int((scores.alert_start & scores.target.eq(0)).sum())}


def control_row(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {"wells": 0, "files": 0, "windows": 0,
                "eligible_negative_hours": 0.0, "active_fraction": None,
                "alarm_starts": 0, "alarm_starts_per_eligible_hour": None}
    hours = len(scores) * STRIDE_SECONDS / 3600
    return {"wells": int(scores.well_id.nunique()), "files": int(scores.file.nunique()),
            "windows": len(scores), "eligible_negative_hours": hours,
            "active_fraction": float(scores.alert_active.mean()),
            "alarm_starts": int(scores.alert_start.sum()),
            "alarm_starts_per_eligible_hour": int(scores.alert_start.sum()) / hours}


def _evaluate_methods(scores: pd.DataFrame, cohort: str, outer_well: int | None,
                      ratio: float, chosen: dict, save_scores: bool = False):
    """Три правила для одинаковых scores: 0.5/1, 0.5/2, inner-selected."""
    rows, predictions = [], []
    for name, policy in (("fixed05_1", {"threshold": 0.5, "confirmations": 1}),
                         ("fixed05_2", {"threshold": 0.5, "confirmations": 2}),
                         ("nested_selected", chosen)):
        applied = apply_rule(scores, policy)
        data = event_row(applied) if cohort == "class2_loow" else control_row(applied)
        rows.append({"cohort": cohort, "outer_well": outer_well, "ratio": ratio,
                     "policy": name, "threshold": float(policy["threshold"]),
                     "confirmations": int(policy["confirmations"]), **data})
        if save_scores and name == "nested_selected":
            applied["ratio"] = ratio
            applied["outer_well"] = outer_well
            applied["policy"] = name
            applied["cohort"] = cohort
            predictions.append(applied)
    return rows, predictions


def run(training_path: Path, selected_path: Path, dataset_dir: Path, output_dir: Path,
        iterations: int = 300, per_file: int = 12, per_well_class: int = 24):
    import catboost
    training = ctl.load_training(training_path)
    if "event_time" not in training.columns:
        raise ValueError("Нет момента наступления события в class2-когорте")
    selected = hn.load_selected(selected_path, dataset_dir)
    dev, external = hn.assign_external_holdout(
        set(map(int, selected.well_id)), set(map(int, training.well_id)))
    fit_dev, tune_dev = partition_development_controls(dev)
    fit_dev, tune_dev = ensure_tuning_class_coverage(fit_dev, tune_dev, selected)
    if (fit_dev & tune_dev) or (fit_dev & external) or (tune_dev & external):
        raise AssertionError("Пересечение контрольных групп")
    print(f"Контрольные WELL только для обучения: {sorted(fit_dev)}")
    print(f"Контрольные WELL только для выбора порога: {sorted(tune_dev)}")
    print(f"Контрольные WELL только для внешней проверки: {sorted(external)}")
    controls, audit = ctl.build_controls(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "control_file_audit.csv", index=False)
    if audit.get("error", pd.Series(dtype=str)).notna().any():
        raise RuntimeError("Не все контрольные файлы считаны; см. control_file_audit.csv")
    hn.validate_controls(controls, training)
    class2_wells = set(map(int, training.well_id.unique()))
    allowed_train_wells = class2_wells | fit_dev
    sampled = hn.sample_train_controls(controls, allowed_train_wells, per_file, per_well_class)
    tune_controls = controls.loc[controls.well_id.isin(tune_dev)].copy()
    external_controls = controls.loc[controls.well_id.isin(external)].copy()
    if tune_controls.empty or external_controls.empty:
        raise ValueError("Нет пригодных tune-only или внешних контрольных окон")
    if not {0, 8}.issubset(set(map(int, tune_controls.event_class_folder))):
        raise ValueError("Нет валидных контрольных окон классов 0 и 8 для настройки")
    if set(sampled.well_id) & set(tune_controls.well_id) or set(sampled.well_id) & external:
        raise AssertionError("Тестовые контроли случайно попали в обучение")
    print(f"Class2 окон: {len(training)}; базовая выборка контролей: {len(sampled)}; "
          f"tune контролей: {len(tune_controls)}; external контролей: {len(external_controls)}")

    summaries, grids, selections, event_preds, control_preds = [], [], [], [], []
    for ratio in RATIOS:
        base_extra = fraction_of_controls(sampled, ratio)
        print(f"\n=== Доля hard negatives: {ratio:.0%}; обучающих контрольных окон {len(base_extra)} ===")
        outer_event_parts = []
        outer_known_control_parts = []
        for well in sorted(class2_wells):
            outer_train = training.loc[training.well_id.ne(well)].copy()
            outer_test = training.loc[training.well_id.eq(well)].copy()
            outer_extra = base_extra.loc[base_extra.well_id.ne(well)].copy()
            assert well not in set(outer_train.well_id) | set(outer_extra.well_id)
            outer_model = _model(outer_train, outer_extra, iterations)
            # Для настройки: inner-LOOW событий + отдельные tune-only controls.
            inner = inner_predictions(outer_train, sampled.loc[sampled.well_id.ne(well)],
                                      ratio, iterations)
            tune_score = _score_controls(outer_model, tune_controls)
            selected_policy, grid = policy_grid(inner, tune_score)
            grid["outer_well"] = well
            grid["ratio"] = ratio
            grids.append(grid)
            selections.append({"outer_well": well, "ratio": ratio,
                               "train_controls": len(outer_extra), **selected_policy})
            test_pred = _score_class2(outer_model, outer_test)
            test_pred["outer_well"] = well
            known = controls.loc[controls.well_id.eq(well)].copy()
            if not known.empty:
                known_pred = _score_controls(outer_model, known)
                known_pred["outer_well"] = well
                outer_known_control_parts.append(known_pred)
            outer_event_parts.append(test_pred)
            for policy_name, policy in (("fixed05_1", {"threshold": 0.5, "confirmations": 1}),
                                        ("fixed05_2", {"threshold": 0.5, "confirmations": 2}),
                                        ("nested_selected", selected_policy)):
                annotated = apply_rule(test_pred, policy)
                annotated["ratio"] = ratio
                annotated["outer_well"] = well
                annotated["policy"] = policy_name
                event_preds.append(annotated)
                if not known.empty:
                    kpred = apply_rule(known_pred, policy)
                    kpred["ratio"] = ratio
                    kpred["outer_well"] = well
                    kpred["policy"] = policy_name
                    kpred["cohort"] = "loow_class2_wells"
                    control_preds.append(kpred)
            print(f"  LOOW WELL-{well:05d}: порог={selected_policy['threshold']:.2f}, "
                  f"подтверждений={selected_policy['confirmations']}, "
                  f"inner event recall={selected_policy['inner_event_recall']:.3f}")
        # Внешние контрольные WELL только для проверки: ни один из них не в обучении.
        full_model = _model(training, base_extra, iterations)
        full_controls = _score_controls(full_model, external_controls)
        # Внешнее правило выбираем без контроля external и без их score/labels.
        full_inner = inner_predictions(training, sampled, ratio, iterations)
        full_tune = _score_controls(full_model, tune_controls)
        external_policy, full_grid = policy_grid(full_inner, full_tune)
        full_grid["outer_well"] = "external_controls"
        full_grid["ratio"] = ratio
        grids.append(full_grid)
        selections.append({"outer_well": "external_controls", "ratio": ratio,
                           "train_controls": len(base_extra), **external_policy})
        for name, policy in (("fixed05_1", {"threshold": 0.5, "confirmations": 1}),
                             ("fixed05_2", {"threshold": 0.5, "confirmations": 2}),
                             ("nested_selected", external_policy)):
            applied = apply_rule(full_controls, policy)
            applied["ratio"] = ratio
            applied["outer_well"] = "external_controls"
            applied["policy"] = name
            applied["cohort"] = "heldout_control_only_wells"
            control_preds.append(applied)
        print(f"  Внешняя контрольная политика: {external_policy['threshold']:.2f}, "
              f"подтверждений={external_policy['confirmations']}")

    event_all = pd.concat(event_preds, ignore_index=True)
    controls_all = pd.concat(control_preds, ignore_index=True)
    # Контроль целостности: для каждой настройки одна оценка каждого DHSV окна.
    count = len(training)
    for (ratio, policy), chunk in event_all.groupby(["ratio", "policy"]):
        if len(chunk) != count or chunk.duplicated(["file", "decision_time"]).any():
            raise RuntimeError(f"Пропущены или задублированы DHSV-окна: {ratio}, {policy}")
    # Aggregated DHSV scores (по фиксированному ratio/policy, события и метки не объединяем с контролями).
    event_summary = []
    for (ratio, policy), sub in event_all.groupby(["ratio", "policy"], sort=True):
        event_summary.append({"ratio": ratio, "policy": policy, **event_row(sub)})
    control_summary, control_by_class = [], []
    for (ratio, policy, cohort), sub in controls_all.groupby(["ratio", "policy", "cohort"], sort=True):
        control_summary.append({"ratio": ratio, "policy": policy, "cohort": cohort,
                                **control_row(sub)})
        for cls, csub in sub.groupby("event_class_folder"):
            control_by_class.append({"ratio": ratio, "policy": policy, "cohort": cohort,
                                     "event_class_folder": int(cls), **control_row(csub)})
    events_by_well = []
    for (ratio, policy, well), sub in event_all.groupby(["ratio", "policy", "well_id"], sort=True):
        events_by_well.append({"ratio": ratio, "policy": policy, "well_id": int(well),
                               **event_row(sub)})
    # Строго вложенный отбор не только threshold, но и ratio: внешние оценки
    # не участвуют в выборе. Это основной результат исследования баланса.
    selected_folds = []
    chosen_event_parts, chosen_control_parts = [], []
    all_choices = pd.DataFrame(selections)
    fold_ids = [*sorted(class2_wells), "external_controls"]
    for fold in fold_ids:
        options = all_choices.loc[all_choices.outer_well.astype(str).eq(str(fold))]
        best = choose_nested_variant(options)
        selected_folds.append(best)
        is_outer = event_all.outer_well.astype(str).eq(str(fold))
        if fold != "external_controls":
            ev = event_all.loc[is_outer & event_all.ratio.eq(best["ratio"]) &
                               event_all.policy.eq("nested_selected")].copy()
            if ev.empty:
                raise RuntimeError(f"Нет DHSV оценок для выбранного fold {fold}")
            chosen_event_parts.append(ev)
        is_control = controls_all.outer_well.astype(str).eq(str(fold))
        co = controls_all.loc[is_control & controls_all.ratio.eq(best["ratio"]) &
                              controls_all.policy.eq("nested_selected")].copy()
        if not co.empty:
            chosen_control_parts.append(co)
    selected_event = pd.concat(chosen_event_parts, ignore_index=True)
    selected_controls = pd.concat(chosen_control_parts, ignore_index=True)
    if len(selected_event) != len(training) or selected_event.duplicated(
            ["file", "decision_time"]).any():
        raise RuntimeError("Вложенный выбор потерял DHSV окна")
    selected_summary = event_row(selected_event)
    selected_control_summary = []
    for cohort, sub in selected_controls.groupby("cohort", sort=True):
        selected_control_summary.append({"cohort": cohort, **control_row(sub)})
    pd.DataFrame(selected_folds).to_csv(output_dir / "nested_choice_by_outer_well.csv", index=False)
    pd.DataFrame([selected_summary]).to_csv(output_dir / "nested_choice_event_comparison.csv", index=False)
    pd.DataFrame(selected_control_summary).to_csv(
        output_dir / "nested_choice_control_comparison.csv", index=False)
    selected_event.to_csv(output_dir / "nested_choice_class2_alerts.csv", index=False)
    selected_controls.to_csv(output_dir / "nested_choice_control_alerts.csv", index=False)

    pd.DataFrame(event_summary).to_csv(output_dir / "event_comparison.csv", index=False)
    pd.DataFrame(control_summary).to_csv(output_dir / "control_comparison.csv", index=False)
    pd.DataFrame(control_by_class).to_csv(output_dir / "control_by_class.csv", index=False)
    pd.DataFrame(events_by_well).to_csv(output_dir / "event_by_well.csv", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "selected_policy_by_well.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(output_dir / "inner_selection_grid.csv", index=False)
    event_all.to_csv(output_dir / "class2_scores_and_alerts.csv", index=False)
    controls_all.to_csv(output_dir / "control_scores_and_alerts.csv", index=False)
    pd.DataFrame({"train_sample_wells": sorted(set(sampled.well_id))}).to_csv(
        output_dir / "training_control_wells.csv", index=False)
    audit.to_csv(output_dir / "control_file_audit.csv", index=False)
    provenance = {
        "experiment": "class2_hard_negative_balance_v0.1", "horizon_minutes": 30,
        "ratios": RATIOS, "thresholds": THRESHOLDS, "confirmations": CONFIRMATIONS,
        "target_inner_event_recall": INNER_EVENT_TARGET,
        "selection": "min mean of inner class2-negative active fraction and MACRO tuning-control active fraction, subject to event recall >=0.8; if impossible maximize recall",
        "nested_ratio_selection": "ratio selected with threshold inside training-only wells and tuning controls; separate outer metrics, no outer labels in selection",
        "tuning_stratification": "class0 and class8 represented in training and tuning control-only WELL; based on folder metadata only",
        "training_controls_base_count": len(sampled), "class2_count": len(training),
        "control_tune_rows": len(tune_controls), "control_external_rows": len(external_controls),
        "fit_dev_wells": sorted(fit_dev), "tune_dev_wells": sorted(tune_dev),
        "external_control_wells": sorted(external), "class2_wells": sorted(class2_wells),
        "sha256_training": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "sha256_selected": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "iterations": iterations, "per_file": per_file, "per_well_class": per_well_class,
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "catboost": catboost.__version__},
        "limitations": [
            "Previously explored 7 class2 wells, 11 onsets. Not independent final validation.",
            "30min horizon previously chosen after observing same data.",
            "Ratio and threshold comparisons exploratory. Do not choose winner by outer test results and claim unbiased quality.",
            "Labels used retrospectively to define class2 risk set and verifiable negative controls, not for model input.",
            "Tuning controls separate by well from training controls and external controls.",
            "External control-only wells have no class2 events; cannot estimate detection recall on them.",
            "Class2 events and control-only wells are different populations; never combine denominators.",
            "Eligible negative hours based on sparse 5min decisions, not production false alarm rate.",
            "Pooled windows highly correlated; only 11 positive event episodes.",
            "No calibrated probability and no model deployment.",
        ]}
    (output_dir / "experiment.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== DHSV обнаружение (7 LOOW WELL) ===")
    print(pd.DataFrame(event_summary).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== ВЛОЖЕННЫЙ ВЫБОР ratio + threshold ===")
    print(pd.DataFrame(selected_folds)[["outer_well", "ratio", "threshold",
          "confirmations", "inner_event_recall", "inner_combined_burden"]].to_string(
          index=False, float_format=lambda x: f"{x:.3f}"))
    print("DHSV (вложенно выбранный):", selected_summary)
    print("Контроли (вложенно выбранный):", selected_control_summary)
    print("\n=== Контрольные ложные тревоги ===")
    print(pd.DataFrame(control_summary).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nРезультаты: {output_dir.resolve()}")
    print("ОГРАНИЧЕНИЕ: результаты exploratory, не полевая частота тревог")
    return pd.DataFrame(event_summary), pd.DataFrame(control_summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training", type=Path,
                    default=ROOT / "outputs/class2_horizons/cohort_features.csv")
    ap.add_argument("--selected", type=Path,
                    default=ROOT / "outputs/class2_specificity/selected_files.csv")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/class2_hard_negative_balance")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--per-file", type=int, default=12)
    ap.add_argument("--per-well-class", type=int, default=24)
    args = ap.parse_args()
    run(args.training, args.selected, args.dataset, args.output,
        iterations=args.iterations, per_file=args.per_file, per_well_class=args.per_well_class)


if __name__ == "__main__":
    main()
