"""All-well offline parity audit for the research ML Agent v0.2.

Replays the original 30-minute CatBoost-12 baseline (hard-negative ratio 0,
threshold 0.5, 1 confirmation) across ALL class-2 files in the archived
cohort, with one held-out WELL per fitted model. No Coordinator or UI changes.

The archived cohort is used exclusively to define training and comparable
moments, not as a runtime input into the agent. This is parity, not a newly
independent evaluation of predictive accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_agent import FEATURES
from experiments.offline_replay_ml_v02 import (
    DEFAULT_COHORT, DEFAULT_REFERENCE, _sha256, load_cohort_for_well,
    load_raw, load_reference, parse_well_id, replay, train_baseline,
)

DEFAULT_DATASET_DIR = ROOT / 'dataset' / '2'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs' / 'ml_agent_v02_replay_all_wells'


def file_inventory(cohort_path: Path) -> tuple[list[tuple[int, str]], int]:
    """Inventory obtained only from the archived cohort, not from labels during replay."""
    if not cohort_path.is_file():
        raise FileNotFoundError(f'Нет файла когорты: {cohort_path}')
    data = pd.read_csv(cohort_path)
    mandatory = {'file', 'well_id', 'horizon_minutes', 'decision_time', 'target', *FEATURES}
    if not mandatory.issubset(data):
        raise ValueError(f'Недостающие столбцы: {sorted(mandatory - set(data.columns))}')
    data = data.loc[data['horizon_minutes'].eq(30)].copy()
    if data.empty:
        raise ValueError('30-минутная когорта пуста')
    if data[['file', 'decision_time']].isna().any().any():
        raise ValueError('В архиве пропущены file/decision_time')
    if data.duplicated(['file', 'decision_time']).any():
        raise ValueError('В архиве дубликаты file/decision_time')
    pairs: list[tuple[int, str]] = []
    for (wid, name), grp in data.groupby(['well_id', 'file'], sort=True):
        name = str(name)
        wid = int(wid)
        if Path(name).name != name or parse_well_id(name) != wid or not name.endswith('.parquet'):
            raise ValueError(f'Небезопасное имя или WELL не совпадает с файлом: {name} / {wid}')
        pairs.append((wid, name))
    return pairs, len(data)


def _verify_preflight(dataset_dir: Path, inventory: list[tuple[int, str]], reference_path: Path):
    """Fail before training if even one source file is absent."""
    if not reference_path.is_file():
        raise FileNotFoundError(f'Нет архива прогнозов: {reference_path}')
    missing = [name for _, name in inventory if not (dataset_dir / name).is_file()]
    if missing:
        examples = ', '.join(missing[:5])
        raise FileNotFoundError(f'Не хватает {len(missing)} Parquet-файлов в {dataset_dir}: {examples}')


def _score_summary(rows: list[dict], expected_files: int, expected_wells: int,
                   expected_eligible: int, actual_files: int) -> dict:
    # A useful pure helper to make aggregate validation easily testable.
    checked = [r for r in rows if r.get('status') == 'checked']
    failures = [r for r in rows if r.get('status') != 'checked' or not r.get('parity_pass', False)]
    total = sum(int(r.get('archived_eligible_decisions', 0)) for r in checked)
    all_dec = sum(int(r.get('all_blind_replay_decisions', 0)) for r in checked)
    extra = sum(int(r.get('non_comparable_decisions', 0)) for r in checked)
    wells = len({int(r['well_id']) for r in checked})
    return {
        'files_expected': expected_files,
        'files_found': actual_files,
        'files_checked': len(checked),
        'wells_expected': expected_wells,
        'wells_checked': wells,
        'archived_eligible_expected': expected_eligible,
        'archived_eligible_checked': total,
        'all_blind_replay_decisions_checked': all_dec,
        'non_comparable_decisions_checked': extra,
        'files_failed': len(failures),
        'parity_pass': bool(not failures and len(checked) == expected_files and
                            actual_files == expected_files and wells == expected_wells and
                            total == expected_eligible),
        'important': 'Technical parity only; not an independent metric of failure prediction. '
                     'Decisions outside the retrospective risk set remain UNLABELED.'
    }


def run_all(dataset_dir: Path = DEFAULT_DATASET_DIR, cohort_path: Path = DEFAULT_COHORT,
            reference_path: Path = DEFAULT_REFERENCE, output_dir: Path = DEFAULT_OUTPUT_DIR,
            tolerance: float = 1e-6, *, fail_on_mismatch: bool = True) -> dict:
    """Fit one LOOW model per WELL; reset shadow agent between Parquet files.

    Preflight is strict. Per-file comparison errors are saved and do NOT stop
    auditing remaining files; after writing the report strict mode raises.
    """
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tolerance должна быть конечным положительным числом')
    dataset_dir, cohort_path, reference_path, output_dir = map(
        Path, (dataset_dir, cohort_path, reference_path, output_dir))
    inventory, eligible_expected = file_inventory(cohort_path)
    _verify_preflight(dataset_dir, inventory, reference_path)
    wells = sorted({w for w, _ in inventory})
    per_file: list[dict] = []
    fingerprints: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for well in wells:
        train, heldout = load_cohort_for_well(cohort_path, well)
        # A single fitted model for the whole held-out WELL prevents refitting
        # on file-specific labels, and matches historical LOOW training.
        model = train_baseline(train)
        well_files = [name for w, name in inventory if w == well]
        if set(heldout['file'].unique()) != set(well_files):
            raise AssertionError(f'Неполный список файлов WELL-{well:05d}')
        for file in well_files:
            source = dataset_dir / file
            row = {'file': file, 'well_id': well, 'status': 'error', 'parity_pass': False,
                   'error': '', 'train_rows': len(train), 'train_wells': ','.join(
                       map(str, sorted(map(int, train['well_id'].unique()))))}
            try:
                raw = load_raw(source)
                archived = load_reference(reference_path, well, file)
                scores, sm = replay(raw, well=well, file=file, model=model,
                                    test_cohort=heldout, reference=archived, tolerance=tolerance)
                valid = bool(sm['eligible_feature_match'] and sm['eligible_score_match'] is True
                             and sm['eligible_alarm_match'] is True)
                row.update(sm)
                row['status'] = 'checked'
                row['parity_pass'] = valid
                row['error'] = '' if valid else 'Несовпадение признаков, оценок или тревог'
                # Generated score rows are a technical trace, not a labeled
                # quality evaluation. Intentionally no target column.
                scores.to_csv(output_dir / f'{Path(file).stem}_replay.csv', index=False)
                fingerprints[file] = _sha256(source)
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            per_file.append(row)
            print(f'WELL-{well:05d} {file}: '
                  f'{"PASS" if row["parity_pass"] else "FAIL"}; '
                  f'archived={row.get("archived_eligible_decisions", "—")}; '
                  f'{row["error"]}')

    files_df = pd.DataFrame(per_file)
    files_df.to_csv(output_dir / 'file_parity.csv', index=False)
    by_well = []
    for w, grp in files_df.groupby('well_id', sort=True):
        by_well.append({'well_id': int(w), 'files': len(grp),
            'archived_eligible_decisions': int(pd.to_numeric(
                grp.get('archived_eligible_decisions', pd.Series(dtype=float)),
                errors='coerce').fillna(0).sum()),
            'files_failed': int((~grp['parity_pass'].astype(bool)).sum()),
            'parity_pass': bool(grp['parity_pass'].astype(bool).all())})
    pd.DataFrame(by_well).to_csv(output_dir / 'well_parity.csv', index=False)
    summary = _score_summary(per_file, len(inventory), len(wells), eligible_expected, len(fingerprints))
    summary.update({
        'experiment': 'ml_agent_v02_offline_replay_all_wells_v0.1',
        'baseline': 'CatBoost-12, horizon=30m, hard_negative_ratio=0, threshold=0.5, confirmations=1',
        'loow_train_split': 'Exclude entire test WELL, fit once per test WELL',
        'cohort_sha256': _sha256(cohort_path),
        'reference_sha256': _sha256(reference_path),
        'telemetry_sha256_by_file': fingerprints,
        'tolerance': tolerance,
        'software': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'catboost': __import__('catboost').__version__},
    })
    (output_dir / 'all_wells_replay_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in (
        'files_expected', 'files_checked', 'wells_expected', 'wells_checked',
        'archived_eligible_expected', 'archived_eligible_checked',
        'all_blind_replay_decisions_checked', 'non_comparable_decisions_checked',
        'files_failed', 'parity_pass')}, ensure_ascii=False, indent=2))
    print(f'Отчёты: {output_dir}')
    if fail_on_mismatch and not summary['parity_pass']:
        raise RuntimeError('Multi-WELL replay не прошёл: см. file_parity.csv и all_wells_replay_summary.json')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    ap.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--tolerance', type=float, default=1e-6)
    args = ap.parse_args()
    run_all(args.dataset_dir, args.cohort, args.reference, args.output, args.tolerance)


if __name__ == '__main__':
    main()
