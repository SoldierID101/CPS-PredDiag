"""Описательный аудит двухдатчиковых направлений давления класса 2.

Источник: outputs/rule_audit/class2_rule_sign_windows.csv, полученный
экспериментом audit_class2_rule_signals.py. Разметка class применяется ТОЛЬКО
для ретроспективной группировки. Никаких порогов/моделей не подбираем.

Запуск из корня CPS-PredDiag:
    python experiments/analyze_class2_rule_pairs.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS = {
    "PDG_up_TPT_down": (("pdg", 1), ("tpt", -1)),
    "PDG_up_MON_down": (("pdg", 1), ("mon_ckp", -1)),
    "TPT_down_MON_down": (("tpt", -1), ("mon_ckp", -1)),
}
REQUIRED = {"file", "well", "phase", "decision_time"} | {
    f"{sensor}_slope_pa_h" for sensor in ("pdg", "tpt", "mon_ckp")
}


def evaluate_pairs(windows: pd.DataFrame) -> pd.DataFrame:
    """Возвращает 3 записи на каждое исходное окно без использования метки в формуле."""
    absent = REQUIRED - set(windows.columns)
    if absent:
        raise ValueError(f"В CSV отсутствуют поля: {sorted(absent)}")
    records = []
    for pair_name, ((sensor_a, direction_a), (sensor_b, direction_b)) in PAIRS.items():
        a = pd.to_numeric(windows[f"{sensor_a}_slope_pa_h"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(windows[f"{sensor_b}_slope_pa_h"], errors="coerce").to_numpy(dtype=float)
        eligible = np.isfinite(a) & np.isfinite(b)
        match = eligible & (direction_a * a > 0) & (direction_b * b > 0)
        part = windows[["file", "well", "phase", "decision_time"]].copy()
        part["pair"] = pair_name
        part["eligible"] = eligible
        # Не путать отсутствие двух датчиков с несрабатыванием правила!
        part["sign_match"] = pd.Series(match, index=windows.index).where(eligible, other=pd.NA).astype("boolean")
        records.append(part)
    return pd.concat(records, ignore_index=True)


def summarize_pairs(detailed: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame(columns=group_cols + ["all_windows", "eligible_windows", "matches", "coverage", "fraction_given_eligible", "fraction_among_all"])
    rows = []
    for key, group in detailed.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        eligible = int(group["eligible"].sum())
        total = len(group)
        matches = int(group["sign_match"].fillna(False).sum())
        rows.append({
            **dict(zip(group_cols, key)),
            "all_windows": total,
            "eligible_windows": eligible,
            "matches": matches,
            "coverage": round(eligible / total, 4),
            "fraction_given_eligible": round(matches / eligible, 4) if eligible else float("nan"),
            "fraction_among_all": round(matches / total, 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Аудит пар датчиков без настройки порогов")
    ap.add_argument("--input", type=Path, default=Path("outputs/rule_audit/class2_rule_sign_windows.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/rule_audit"))
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Не найден отчёт: {args.input}. Сначала запустите audit_class2_rule_signals.py")
    windows = pd.read_csv(args.input)
    if windows.empty:
        raise SystemExit("CSV пуст: нет окон для анализа")
    detailed = evaluate_pairs(windows)
    by_phase = summarize_pairs(detailed, ["pair", "phase"])
    by_well = summarize_pairs(detailed, ["pair", "well", "phase"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_phase.to_csv(args.output_dir / "class2_rule_pairs_by_phase.csv", index=False, encoding="utf-8-sig")
    by_well.to_csv(args.output_dir / "class2_rule_pairs_by_well.csv", index=False, encoding="utf-8-sig")
    print(f"Окон: {len(windows)}; скважин: {windows['well'].nunique()}; пар: {len(PAIRS)}")
    print("\nОписательные доли по фазам (не Recall/Precision):")
    print(by_phase.to_string(index=False))
    print("\nФаза 102 по скважинам (прочерк = пара датчиков недоступна):")
    print(by_well[by_well["phase"] == "transition_102"].to_string(index=False))
    print(f"\nОтчёты: {args.output_dir.resolve()}")
    print("ОГРАНИЧЕНИЕ: сравнение с истинными фазами ретроспективное; это НЕ независимая оценка правила.")


if __name__ == "__main__":
    main()
