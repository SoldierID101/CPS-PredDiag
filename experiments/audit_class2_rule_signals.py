"""Аудит физической гипотезы RULE-DHSV-001 на реальных эпизодах Petrobras 3W.

НЕ обучает модель, НЕ настраивает пороги, НЕ реализует диагностическое правило.
Применяет только прошлое 5-минутное окно измерений. Разметка class нужна
исключительно для последующего описательного сравнения фаз 0, 102 и 2.

Запуск из корня CPS-PredDiag:
    python experiments/audit_class2_rule_signals.py

Зависимости: pandas, numpy, pyarrow.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WINDOW_SECONDS = 300
STRIDE_SECONDS = 300
MIN_COVERAGE = 0.9
SENSORS = {"pdg": "P-PDG", "tpt": "P-TPT", "mon_ckp": "P-MON-CKP"}
PHASES = {0: "normal_0", 102: "transition_102", 2: "event_2"}


def slope_pa_per_hour(values: np.ndarray, min_coverage: float = MIN_COVERAGE) -> tuple[float, float]:
    """(наклон Па/ч, доля конечных измерений); на основе одной прошлой выборки."""
    numeric = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(numeric)
    coverage = float(valid.mean()) if numeric.size else 0.0
    if numeric.size < 2 or coverage < min_coverage or valid.sum() < 2:
        return float("nan"), coverage
    x = np.flatnonzero(valid).astype(np.float64)  # секунды, если шаг данных = 1 с
    y = numeric[valid]
    # МНК без квадратичной сложности; пересчёт в Па/час.
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.dot(xc, xc))
    if denominator <= 0:
        return float("nan"), coverage
    return float(np.dot(xc, yc) / denominator * 3600.0), coverage


def audit_window(window: pd.DataFrame) -> dict[str, Any]:
    """Для каждого датчика считает тренд и доступность. class здесь не используется."""
    if len(window) != WINDOW_SECONDS or not isinstance(window.index, pd.DatetimeIndex):
        raise ValueError("Ожидалось окно из 300 измерений с DatetimeIndex")
    # pandas/Parquet могут представлять время с разрешением s, ms, us или ns.
    # .asi8 использует единицы самого индекса; приводим к ns перед проверкой.
    ticks_ns = window.index.as_unit("ns").asi8
    if np.any(np.diff(ticks_ns) != 1_000_000_000):
        raise ValueError("В окне есть разрывы или дубли во времени (ожидался шаг 1 с)")

    result: dict[str, Any] = {}
    for key, column in SENSORS.items():
        if column in window:
            values = pd.to_numeric(window[column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
        else:
            values = np.full(WINDOW_SECONDS, np.nan, dtype=np.float64)
        slope, coverage = slope_pa_per_hour(values)
        result[f"{key}_coverage"] = round(coverage, 4)
        result[f"{key}_slope_pa_h"] = slope

    pdg, tpt, mon = (result[f"{key}_slope_pa_h"] for key in SENSORS)
    eligible = bool(np.isfinite([pdg, tpt, mon]).all())
    result["three_sensors_usable"] = eligible
    # Проверяются только НАПРАВЛЕНИЯ, без произвольной величины порога.
    result["sign_pattern_matches"] = bool(pdg > 0 and tpt < 0 and mon < 0) if eligible else None
    return result


def parse_well(file: Path) -> str:
    match = re.match(r"^(WELL-\d+)_", file.name)
    return match.group(1) if match else "unknown"


def summarize(windows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    records = []
    for group, frame in windows.groupby(columns, dropna=False, sort=True):
        # pandas возвращает кортеж ключей и при группировке по одному столбцу.
        keys = group if isinstance(group, tuple) else (group,)
        eligible = frame[frame["three_sensors_usable"] == True]  # noqa: E712
        matches = int(eligible["sign_pattern_matches"].fillna(False).astype(bool).sum())
        records.append({
            **dict(zip(columns, keys)),
            "windows": len(frame),
            "three_sensors_usable_windows": len(eligible),
            "sign_pattern_matches": matches,
            "coverage_of_three": round(len(eligible) / len(frame), 4),
            "pattern_fraction_given_three": (
                round(matches / len(eligible), 4) if len(eligible) else float("nan")
            ),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит направлений давления класса 2, без настройки порогов")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rule_audit"))
    args = parser.parse_args()

    event_dir = args.dataset_root / "2"
    files = sorted(event_dir.glob("WELL-*.parquet"))
    if not files:
        raise SystemExit(f"Не найдены реальные файлы WELL-*.parquet в {event_dir.resolve()}")

    rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for file in files:
        try:
            df = pd.read_parquet(file)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Индекс Parquet не является DatetimeIndex")
            if "class" not in df.columns:
                raise ValueError("Нет столбца class")
            if not df.index.is_monotonic_increasing or df.index.has_duplicates:
                raise ValueError("Временной индекс неупорядочен или содержит дубликаты")

            well = parse_well(file)
            labels = pd.to_numeric(df["class"], errors="coerce")
            counts = {int(value): int((labels == value).sum()) for value in PHASES}
            file_summary: dict[str, Any] = {
                "file": file.name, "well": well, "rows": len(df),
                "has_steady_event_2": bool(counts[2]),
                "normal_rows": counts[0], "transition_rows": counts[102],
                "event_rows": counts[2],
            }
            for key, sensor in SENSORS.items():
                if sensor in df.columns:
                    numeric = pd.to_numeric(df[sensor], errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    )
                    file_summary[f"{key}_available_fraction"] = round(
                        float(np.isfinite(numeric).mean()), 4
                    ) if len(numeric) else 0.0
                else:
                    file_summary[f"{key}_available_fraction"] = 0.0
            file_rows.append(file_summary)

            valid_windows = 0
            for end in range(WINDOW_SECONDS - 1, len(df), STRIDE_SECONDS):
                start = end - WINDOW_SECONDS + 1
                window = df.iloc[start : end + 1]
                try:
                    result = audit_window(window)
                except ValueError:
                    continue  # окно с пропусками по времени исключается из аудита

                label = labels.iloc[end]
                if pd.isna(label) or int(label) not in PHASES:
                    continue
                rows.append({
                    "file": file.name, "well": well,
                    "decision_time": df.index[end].isoformat(),
                    "observed_class_for_analysis_only": int(label),
                    "phase": PHASES[int(label)], **result,
                })
                valid_windows += 1
            print(f"{file.name}: {len(df):,} строк; окон для аудита: {valid_windows}")
        except (OSError, ValueError, TypeError, ImportError, KeyError) as exc:
            errors.append({"file": file.name, "error": str(exc)})
            print(f"[ОШИБКА] {file.name}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    availability = pd.DataFrame(file_rows)
    windows = pd.DataFrame(rows)
    availability.to_csv(args.output_dir / "class2_file_availability.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(args.output_dir / "class2_rule_sign_windows.csv", index=False, encoding="utf-8-sig")
    per_phase = summarize(windows, ["phase"])
    per_well = summarize(windows, ["well", "phase"])
    per_phase.to_csv(args.output_dir / "class2_rule_sign_by_phase.csv", index=False, encoding="utf-8-sig")
    per_well.to_csv(args.output_dir / "class2_rule_sign_by_well.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(
        args.output_dir / "class2_rule_audit_errors.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== ИТОГ АУДИТА RULE-DHSV-001 (описательно, НЕ валидация) ===")
    print(f"Реальных файлов найдено: {len(files)}; прочитано: {len(availability)}")
    print(f"Скважин: {availability['well'].nunique() if not availability.empty else 0}")
    print(f"Окон с известной разметкой и шагом 1 с: {len(windows)}")
    if not per_phase.empty:
        print(per_phase.to_string(index=False))
    if errors:
        print(f"Ошибок чтения/формата: {len(errors)}; см. class2_rule_audit_errors.csv")
    if not len(windows):
        raise SystemExit("Нет пригодных окон. Проверьте формат временного индекса.")
    print(f"\nОтчёты: {args.output_dir.resolve()}")
    print("Важно: 'sign_pattern_matches' не является доказательством обнаружения неисправности.")


if __name__ == "__main__":
    main()
