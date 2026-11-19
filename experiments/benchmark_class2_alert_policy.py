"""Exploratory nested selection of warning policies for the existing DHSV model benchmark.

Re-trains INNER models within outer-training wells; outer scores come from the
previously saved LOOW benchmark. No modifications to the running app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_class2_models as bench

THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
CONFIRMATIONS = (1, 2)
TARGET_EVENT_RECALL = 0.80
STEP = pd.Timedelta(minutes=5)
KEYS = ("file", "well_id", "decision_time")
SCHEMA = ("file", "well_id", "horizon_minutes", "decision_time", "target", "event_time", "score", "model", "test_well")


def load_oof(path: Path, cohort: pd.DataFrame, model_names: tuple[str, ...]) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [x for x in SCHEMA if x not in data.columns]
    if missing:
        raise ValueError(f"В OOF нет столбцов: {missing}")
    data = data[list(SCHEMA)].copy()
    data = data.loc[data.model.isin(model_names)].copy()
    data["decision_time"] = pd.to_datetime(data["decision_time"], errors="raise")
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    for c in ("well_id", "test_well", "horizon_minutes", "target"):
        data[c] = pd.to_numeric(data[c], errors="raise").astype(int)
    data["score"] = pd.to_numeric(data["score"], errors="raise")
    if data.duplicated(["model", *KEYS]).any():
        raise ValueError("Повторные OOF-решения одной модели")
    if data["score"].isna().any() or not data["score"].between(0, 1).all():
        raise ValueError("OOF score должен быть числом в [0,1]")
    if data.model.isna().any() or set(data.model) != set(model_names):
        raise ValueError("Список моделей в OOF отличается от ожидаемого")
    if not data.well_id.eq(data.test_well).all() or not data.horizon_minutes.eq(30).all():
        raise ValueError("OOF содержит неверную test_well или горизонт")
    truth = cohort[[*KEYS, "target", "event_time"]].copy()
    if truth.duplicated(KEYS).any():
        raise ValueError("Дублирующиеся решения когорты")
    for model in model_names:
        subset = data.loc[data.model.eq(model)]
        both = truth.merge(subset, on=list(KEYS), how="outer", suffixes=("_expected", "_got"), indicator=True, validate="one_to_one")
        if len(both) != len(truth) or not both._merge.eq("both").all():
            raise ValueError(f"OOF {model} не совпадает с когортой по решениям")
        if not both.target_expected.eq(both.target_got).all():
            raise ValueError(f"OOF {model} имеет другую разметку")
        expected_event = pd.to_datetime(both.event_time_expected).astype("string").fillna("<NA>")
        actual_event = pd.to_datetime(both.event_time_got).astype("string").fillna("<NA>")
        if not expected_event.equals(actual_event):
            raise ValueError(f"OOF {model} имеет другую дату события")
    return data


def apply_policy(predictions: pd.DataFrame, threshold: float, confirmations: int) -> pd.DataFrame:
    if not 0 < threshold <= 1 or confirmations not in CONFIRMATIONS:
        raise ValueError("Некорректная политика")
    required = {"file", "well_id", "decision_time", "score"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Недостаточно полей: {required - set(predictions.columns)}")
    frame = predictions.sort_values(["well_id", "file", "decision_time"], kind="stable").copy()
    if frame.duplicated(KEYS).any():
        raise ValueError("Дубли решений в политике")
    if not frame.score.between(0, 1).all():
        raise ValueError("Некорректные оценки риска")
    frame.decision_time = pd.to_datetime(frame.decision_time)
    active = np.zeros(len(frame), dtype=bool)
    starts = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby(["well_id", "file"], sort=False):
        streak, previous_time, was_active = 0, None, False
        indices = frame.index.get_indexer(group.index)
        if np.any(indices < 0):
            raise AssertionError("Индекс кадров нарушен")
        for i, (t, score) in zip(indices, zip(group.decision_time, group.score)):
            if previous_time is None or t - previous_time != STEP:
                streak, was_active = 0, False
            if float(score) >= threshold:
                streak += 1
            else:
                streak = 0
            now = streak >= confirmations
            active[i] = now
            starts[i] = bool(now and not was_active)
            was_active, previous_time = now, t
    frame["alert_active"] = active
    frame["alarm_start"] = starts
    return frame


def measure(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Пустая выборка")
    for c in ("target", "event_time", "alert_active", "alarm_start"):
        if c not in df:
            raise ValueError(f"Нет столбца {c}")
    positive = df.target.eq(1)
    negative = ~positive
    pred = df.alert_active.astype(bool)
    event_keys = ["file", "event_time"]
    events = df.loc[positive, event_keys].drop_duplicates()
    detected = df.loc[positive & pred, event_keys].drop_duplicates()
    starts_inside = df.loc[positive & df.alarm_start.astype(bool), event_keys].drop_duplicates()
    n_events = len(events)
    tp = int((positive & pred).sum())
    fp = int((negative & pred).sum())
    fn = int((positive & ~pred).sum())
    return {
        "n": int(len(df)), "positive": int(positive.sum()), "negative": int(negative.sum()),
        "precision": tp/(tp+fp) if tp+fp else 0.0,
        "recall": tp/(tp+fn) if tp+fn else 0.0,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        "fp_windows": fp, "negative_active_fraction": fp/int(negative.sum()) if negative.any() else 0.0,
        "alarm_starts": int(df.alarm_start.sum()),
        "out_of_horizon_alarm_starts": int((negative & df.alarm_start).sum()),
        "events_detected": int(len(detected)), "events_total": n_events,
        "event_recall": len(detected)/n_events if n_events else 0.0,
        "events_with_new_start_in_horizon": int(len(starts_inside)),
    }


def choose_policy(inner_oof: pd.DataFrame, *, target_event_recall: float = TARGET_EVENT_RECALL) -> tuple[dict, pd.DataFrame]:
    """Choose using ONLY inner-model OOF predictions and inner labels."""
    grid = []
    for threshold in THRESHOLDS:
        for conf in CONFIRMATIONS:
            tested = apply_policy(inner_oof, threshold, conf)
            grid.append({"threshold": threshold, "confirmations": conf, **measure(tested)})
    all_rows = pd.DataFrame(grid)
    valid = all_rows.loc[all_rows.event_recall.ge(target_event_recall - 1e-12)]
    if valid.empty:
        max_recall = float(all_rows.event_recall.max())
        valid = all_rows.loc[all_rows.event_recall.ge(max_recall - 1e-12)]
        reached = False
    else:
        reached = True
    # First protect target recall; among policies satisfying it, reduce negative-window
    # burden, out-of-horizon starts, all starts. Prefer higher thresholds for ties.
    chosen = valid.sort_values(
        ["negative_active_fraction", "out_of_horizon_alarm_starts", "alarm_starts", "threshold", "confirmations"],
        ascending=[True, True, True, False, False], kind="stable",
    ).iloc[0]
    return {"threshold": float(chosen.threshold), "confirmations": int(chosen.confirmations),
            "inner_event_recall": float(chosen.event_recall), "inner_goal_reached": bool(reached)}, all_rows


def make_outer_predictions(cohort: pd.DataFrame, frozen_oof: pd.DataFrame, models: tuple[str, ...]):
    """The frozen OOF outer scores are not used for policy selection."""
    for model in models:
        for outer_well in sorted(cohort.well_id.unique()):
            test = frozen_oof.loc[frozen_oof.model.eq(model) & frozen_oof.well_id.eq(outer_well)].copy()
            train = cohort.loc[cohort.well_id.ne(outer_well)].reset_index(drop=True)
            if test.empty or set(train.well_id).intersection({outer_well}):
                raise ValueError("Недопустимое внешнее разбиение")
            # Required: re-train inner models on only five of the six training wells.
            inner_oof = bench.predict_oof(train, model)
            selected, grid = choose_policy(inner_oof)
            evaluated = apply_policy(test, selected["threshold"], selected["confirmations"])
            evaluated["threshold"] = selected["threshold"]
            evaluated["confirmations"] = selected["confirmations"]
            yield model, int(outer_well), selected, grid, evaluated


def run(cohort_path: Path, oof_path: Path, out: Path, models: tuple[str, ...] = (
    "logreg", "random_forest", "catboost", "lightgbm")) -> dict:
    if not cohort_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError("Нужны cohort_features.csv и oof_predictions.csv предыдущего benchmark")
    cohort = bench.load_cohort(cohort_path)
    oof = load_oof(oof_path, cohort, models)
    out.mkdir(parents=True, exist_ok=True)
    records, policies, grids = [], [], []
    for model, w, selected, grid, scored in make_outer_predictions(cohort, oof, models):
        policies.append({"model": model, "outer_well": w, **selected})
        grid["model"] = model
        grid["outer_well"] = w
        grids.append(grid)
        records.append(scored)
        print(f"{model:>13s} WELL-{w:05d}: threshold={selected['threshold']:.2f}; "
              f"confirms={selected['confirmations']}; inner recall={selected['inner_event_recall']:.2f}", flush=True)
    evaluated = pd.concat(records, ignore_index=True)
    policies_df = pd.DataFrame(policies)
    overview, per_well, episodes = [], [], []
    for model in models:
        subs = evaluated.loc[evaluated.model.eq(model)]
        overview.append({"model": model, **measure(subs)})
        for w, chunk in subs.groupby("well_id", sort=True):
            per_well.append({"model": model, "well_id": int(w), **measure(chunk)})
        for (f, et), chunk in subs.loc[subs.target.eq(1)].groupby(["file", "event_time"], sort=True):
            active = chunk.loc[chunk.alert_active]
            starts_inside = chunk.loc[chunk.alarm_start]
            episodes.append({"model": model, "file": f, "event_time": str(et),
                             "detected": bool(len(active)), "new_start_within_horizon": bool(len(starts_inside)),
                             "first_active_time": str(active.decision_time.min()) if len(active) else None,
                             "lead_minutes_first_active": float((pd.Timestamp(et)-active.decision_time.min()).total_seconds()/60)
                             if len(active) else None})
    evaluated.to_csv(out / "outer_alerts.csv", index=False)
    policies_df.to_csv(out / "policy_by_well.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(out / "inner_policy_grid.csv", index=False)
    pd.DataFrame(overview).to_csv(out / "policy_comparison.csv", index=False)
    pd.DataFrame(per_well).to_csv(out / "by_well.csv", index=False)
    pd.DataFrame(episodes).to_csv(out / "event_details.csv", index=False)
    manifest = {"experiment": "class2_nested_alert_policy_v0.1_exploratory", "models": list(models),
                "source_cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                "source_oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
                "n_windows_per_model": len(cohort), "n_wells": int(cohort.well_id.nunique()),
                "num_events": int(cohort.loc[cohort.target.eq(1), ["file", "event_time"]].drop_duplicates().shape[0]),
                "threshold_grid": list(THRESHOLDS), "confirmations_grid": list(CONFIRMATIONS),
                "inner_target_event_recall": TARGET_EVENT_RECALL,
                "inner_models_retrained": True,
                "outer_models_retrained": False,
                "outer_scores_frozen_from_benchmark": True,
                "controls_evaluated": False,
                "limitations": ["Exploratory: wells already repeatedly examined",
                                "No control specificity in these outputs; requires identical control features for ALL models",
                                "Event recall counts active alert within 30m, not necessarily a new alarm start",
                                "Policy selection minimizes negative-window activity only on inner LOOW data",
                                "No confidence intervals or independent industrial FAR"]}
    (out / "policy_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(overview).round(4).to_string(index=False))
    print(f"\nОтчёты: {out}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("outputs/class2_horizons/cohort_features.csv"))
    parser.add_argument("--oof", type=Path, default=Path("outputs/class2_model_benchmark/oof_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/class2_nested_alert_benchmark"))
    parser.add_argument("--models", nargs="+", choices=bench.MODEL_NAMES, default=["logreg","random_forest","catboost","lightgbm"])
    args = parser.parse_args()
    run(args.cohort, args.oof, args.out, tuple(dict.fromkeys(args.models)))


if __name__ == "__main__":
    main()
