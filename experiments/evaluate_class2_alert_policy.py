"""Вложенный подбор правила предупреждения для CatBoost-12, горизонт 30 минут.

Запуск из корня CPS-PredDiag:
    python experiments/evaluate_class2_alert_policy.py

Использует сохранённые *прошлые* признаки и OOF внешних скважин из
outputs/class2_horizons/. Вложенные модели обучаются только на шести
скважинах внешней обучающей части. Не меняет ML Agent и Streamlit.

Важные ограничения: event-enriched записи, ретроспективное исключение
послесобытийного периода, до этого исследованные скважины; НЕ независимая
итоговая валидация и НЕ реальная частота ложных тревог эксплуатации.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [f'{sensor}_{stat}' for sensor in ('P-MON-CKP', 'P-TPT', 'T-TPT')
            for stat in ('mean', 'std', 'delta', 'slope_per_hour')]
KEYS = ('horizon_minutes', 'file', 'well_id', 'decision_time')
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CONFIRMATIONS = (1, 2)
HORIZON_MINUTES = 30
STRIDE_SECONDS = 300
SEED = 42


def load_and_match(feature_path: Path, oof_path: Path,
                   horizon: int = HORIZON_MINUTES) -> pd.DataFrame:
    """Запрещает скрытое несовпадение окон и смешивание разных постановок."""
    features = pd.read_csv(feature_path)
    scores = pd.read_csv(oof_path)
    required_features = set(KEYS) | {'target', 'event_time', *FEATURES}
    required_scores = set(KEYS) | {'target', 'event_time', 'score'}
    if not required_features.issubset(features.columns) or not required_scores.issubset(scores.columns):
        raise ValueError('Не хватает обязательных столбцов cohort_features / oof_predictions')
    features = features.loc[features.horizon_minutes.eq(horizon)].copy()
    scores = scores.loc[scores.horizon_minutes.eq(horizon)].copy()
    if features.empty or scores.empty or len(features) != len(scores):
        raise ValueError('Нет оценок или различается количество окон выбранного горизонта')
    for df in (features, scores):
        if df.duplicated(list(KEYS)).any():
            raise ValueError('Найдены дубли окон')
        df['decision_time'] = pd.to_datetime(df['decision_time'], errors='raise')
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        if df.decision_time.isna().any() or df.well_id.isna().any():
            raise ValueError('Неверные идентификаторы/временные метки')
    out = features.merge(scores[list(KEYS) + ['target', 'event_time', 'score']],
                         on=list(KEYS), how='outer', validate='one_to_one',
                         indicator=True, suffixes=('_features', '_score'))
    if not out['_merge'].eq('both').all():
        raise ValueError('Наборы временных окон не совпадают')
    if not np.array_equal(out['target_features'].to_numpy(), out['target_score'].to_numpy()):
        raise ValueError('Несовпадение целевых меток в двух файлах')
    if not (out['event_time_features'].fillna(pd.Timestamp('1970-01-01')) ==
            out['event_time_score'].fillna(pd.Timestamp('1970-01-01'))).all():
        raise ValueError('Несовпадение времени событий в двух файлах')
    out = out.rename(columns={'target_features': 'target',
                              'event_time_features': 'event_time'})
    out = out.drop(columns=['_merge', 'target_score', 'event_time_score'])
    if not out.target.isin([0, 1]).all() or out['score'].isna().any():
        raise ValueError('Неверная разметка или пустые OOF-оценки')
    if not np.isfinite(out.score.to_numpy(dtype=float)).all() or not out.score.between(0, 1).all():
        raise ValueError('Оценки вне [0,1]')
    if (out.loc[out.target.eq(1), 'event_time'].isna()).any():
        raise ValueError('Положительным окнам нужен известный момент события')
    if (out.loc[out.target.eq(0), 'event_time'].notna()).any():
        raise ValueError('У отрицательных окон не должно быть события внутри горизонта')
    if out.well_id.nunique() < 4:
        raise ValueError('Для внешней/внутренней LOOW нужно не менее четырёх скважин')
    out = out.sort_values(['well_id', 'file', 'decision_time'], kind='stable').reset_index(drop=True)
    return out


def catboost_model(iterations: int = 300):
    from catboost import CatBoostClassifier
    if iterations < 1:
        raise ValueError('iterations должен быть >= 1')
    return CatBoostClassifier(iterations=iterations, depth=4, learning_rate=0.03,
                              l2_leaf_reg=5, loss_function='Logloss',
                              auto_class_weights='Balanced', nan_mode='Min',
                              random_seed=SEED, thread_count=2,
                              verbose=False, allow_writing_files=False)


def inner_oof(outer_train: pd.DataFrame, iterations: int = 300) -> pd.DataFrame:
    """Внутренний OOF — НИ ОДНОЙ строки внешнего test well."""
    df = outer_train.copy().reset_index(drop=True)
    if df.well_id.nunique() < 3:
        raise ValueError('Внутреннему LOOW нужны >= 3 обучающие скважины')
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    y = df.target.to_numpy(dtype=int)
    wells = df.well_id.to_numpy(dtype=int)
    pred = np.full(len(df), np.nan)
    for well in sorted(np.unique(wells)):
        fit = wells != well
        valid = ~fit
        if np.unique(y[fit]).size != 2:
            raise ValueError(f'Нет обоих классов без внутренней WELL-{well:05d}')
        model = catboost_model(iterations)
        model.fit(x.iloc[np.flatnonzero(fit)], y[fit])
        pred[valid] = model.predict_proba(x.iloc[np.flatnonzero(valid)])[:, 1]
    if not np.isfinite(pred).all():
        raise RuntimeError('Не все внутренние окна получили OOF-прогноз')
    out = df[['file', 'well_id', 'decision_time', 'target', 'event_time']].copy()
    out['score'] = pred
    return out


def trigger_policy(df: pd.DataFrame, threshold: float, confirmations: int,
                   stride_seconds: int = STRIDE_SECONDS) -> pd.DataFrame:
    """Правило использует только текущую и предыдущие оценки, не метки class/target.

    alert_active может продолжаться после возникновения сигнала; alert_start
    отмечает только начало непрерывного предупреждения. Разрывы > шага
    и переход в другой файл сбрасывают накопленную последовательность.
    """
    if not 0 < threshold < 1 or confirmations not in (1, 2) or stride_seconds < 1:
        raise ValueError('Недопустимые параметры правила')
    needed = {'file', 'decision_time', 'score'}
    if not needed.issubset(df.columns):
        raise ValueError('Для правила нужны file/decision_time/score')
    result = df.copy()
    result['decision_time'] = pd.to_datetime(result['decision_time'])
    if result.duplicated(['file', 'decision_time']).any():
        raise ValueError('Повторные диагностические моменты')
    result = result.sort_values(['file', 'decision_time'], kind='stable')
    if not np.isfinite(result.score.to_numpy(dtype=float)).all():
        raise ValueError('Оценки должны быть конечными')
    active = pd.Series(False, index=result.index, dtype=bool)
    starts = pd.Series(False, index=result.index, dtype=bool)
    for file_name, part in result.groupby('file', sort=False):
        prev = None
        streak = 0
        was_active = False
        for ix, row in part.iterrows():
            moment = row['decision_time']
            continuous = prev is not None and (moment - prev).total_seconds() == stride_seconds
            if not continuous:
                streak = 0
                was_active = False
            if float(row['score']) >= threshold:
                streak += 1
            else:
                streak = 0
            is_active = streak >= confirmations
            active.loc[ix] = is_active
            starts.loc[ix] = bool(is_active and not was_active)
            prev, was_active = moment, is_active
    result['alert_active'] = active
    result['alert_start'] = starts
    return result.sort_index()


def policy_metrics(pred: pd.DataFrame) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score
    if pred.empty or not {'target', 'event_time', 'alert_active', 'alert_start'}.issubset(pred):
        raise ValueError('Не хватает событий и предупреждений')
    y = pred.target.to_numpy(dtype=int)
    a = pred.alert_active.to_numpy(dtype=bool)
    starts = pred.alert_start.to_numpy(dtype=bool)
    positive = pred.loc[pred.target.eq(1)].copy()
    events = positive[['file', 'event_time']].drop_duplicates()
    detected_active = positive.loc[positive.alert_active, ['file', 'event_time']].drop_duplicates()
    detected_new = positive.loc[positive.alert_start, ['file', 'event_time']].drop_duplicates()
    n_events = len(events)
    # 5 мин — интервал между оцениваемыми окнами; НЕ полноценный знаменатель FAR.
    neg_hours = float((y == 0).sum() * STRIDE_SECONDS / 3600)
    out_of_horizon_starts = int(((y == 0) & starts).sum())
    detected_leads = positive.loc[positive.alert_active].copy()
    if not detected_leads.empty:
        detected_leads['lead_minutes'] = ((detected_leads.event_time - detected_leads.decision_time)
                                          .dt.total_seconds() / 60)
        lead = float(detected_leads.groupby(['file', 'event_time']).lead_minutes.max().median())
    else:
        lead = None
    return {
        'n': len(pred), 'positive_windows': int(y.sum()),
        'precision': float(precision_score(y, a, zero_division=0)),
        'recall': float(recall_score(y, a, zero_division=0)),
        'f1': float(f1_score(y, a, zero_division=0)),
        'tp_windows': int((a & (y == 1)).sum()),
        'fp_windows': int((a & (y == 0)).sum()),
        'alarm_starts': int(starts.sum()),
        'out_of_horizon_alarm_starts': out_of_horizon_starts,
        'out_of_horizon_alarm_starts_per_eligible_negative_hour':
            out_of_horizon_starts / neg_hours if neg_hours > 0 else None,
        'eligible_negative_hours': neg_hours,
        'events_total': n_events,
        'events_detected_active': len(detected_active),
        'events_detected_new_alarm': len(detected_new),
        'event_recall_active': float(len(detected_active) / n_events) if n_events else None,
        'event_recall_new_alarm': float(len(detected_new) / n_events) if n_events else None,
        'median_lead_active_min': lead,
    }


def choose_policy(inner: pd.DataFrame, min_event_recall: float = 0.8,
                  thresholds: tuple[float, ...] = THRESHOLDS,
                  confirmations: tuple[int, ...] = CONFIRMATIONS) -> tuple[dict, pd.DataFrame]:
    """Выбор ТОЛЬКО на inner OOF.

    1. Внутренний event recall (активный сигнал) >= цели 0.8.
    2. Среди отвечающих цели минимизируем новые тревоги вне горизонта / ч.
    3. Если цель недостижима — максимизируем event recall, затем уменьшаем
       частоту вне-горизонтных тревог. Параметры заданы до внешней оценки.
    """
    if not 0 < min_event_recall <= 1:
        raise ValueError('min_event_recall должен быть в (0,1]')
    rows = []
    for conf in confirmations:
        for threshold in thresholds:
            pred = trigger_policy(inner, float(threshold), int(conf))
            stats = policy_metrics(pred)
            rows.append({'threshold': float(threshold), 'confirmations': int(conf), **stats})
    table = pd.DataFrame(rows)
    viable = table.loc[table.event_recall_active >= min_event_recall - 1e-12]
    if not viable.empty:
        # Сначала минимизируем частоту нежелательных начал; при равенстве
        # максимизируем обнаружение, prefer higher threshold for conservatism.
        rank = viable.sort_values(
            ['out_of_horizon_alarm_starts_per_eligible_negative_hour', 'event_recall_active',
             'threshold', 'confirmations'],
            ascending=[True, False, False, False], kind='stable')
        achieved = True
    else:
        rank = table.sort_values(
            ['event_recall_active', 'out_of_horizon_alarm_starts_per_eligible_negative_hour',
             'threshold', 'confirmations'],
            ascending=[False, True, False, False], kind='stable')
        achieved = False
    row = rank.iloc[0]
    return {'threshold': float(row['threshold']), 'confirmations': int(row['confirmations']),
            'inner_target_recall_reached': achieved,
            'inner_event_recall_active': float(row['event_recall_active']),
            'inner_out_of_horizon_alarm_starts_per_negative_hour':
                float(row['out_of_horizon_alarm_starts_per_eligible_negative_hour'])}, table


def nested_evaluation(data: pd.DataFrame, iterations: int = 300,
                      min_event_recall: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_parts, policies, grids = [], [], []
    wells = sorted(data.well_id.unique())
    if len(wells) < 4:
        raise ValueError('Требуются минимум 4 скважины')
    for outer_well in wells:
        train = data.loc[data.well_id.ne(outer_well)].copy()
        test = data.loc[data.well_id.eq(outer_well)].copy()
        # В train доступны метки и признаки; test передаётся в политику БЕЗ меток.
        inner = inner_oof(train, iterations)
        choice, grid = choose_policy(inner, min_event_recall)
        policy_input = test[['file', 'decision_time', 'score']].copy()
        active = trigger_policy(policy_input, choice['threshold'], choice['confirmations'])
        joined = test.drop(columns=['alert_active', 'alert_start'], errors='ignore').merge(
            active[['file', 'decision_time', 'alert_active', 'alert_start']],
            on=['file', 'decision_time'], validate='one_to_one')
        if len(joined) != len(test):
            raise RuntimeError('Часть оценок потеряна на внешней скважине')
        joined['outer_test_well'] = int(outer_well)
        joined['selected_threshold'] = choice['threshold']
        joined['selected_confirmations'] = choice['confirmations']
        out_parts.append(joined)
        policies.append({'outer_test_well': int(outer_well), 'inner_rows': len(train),
                         'outer_test_rows': len(test), **choice})
        grid.insert(0, 'outer_test_well', int(outer_well))
        grids.append(grid)
        print(f"OUTER WELL-{int(outer_well):05d}: threshold={choice['threshold']:.2f}, "
              f"confirmations={choice['confirmations']}, "
              f"inner_event_recall={choice['inner_event_recall_active']:.3f}, "
              f"inner_goal_reached={choice['inner_target_recall_reached']}")
    combined = pd.concat(out_parts, ignore_index=True)
    if len(combined) != len(data) or combined.duplicated(['file', 'decision_time']).any():
        raise RuntimeError('Вложенный протокол потерял или дублировал окна')
    return combined, pd.DataFrame(policies), pd.concat(grids, ignore_index=True)


def event_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Времена тревог считаются постфактум; НЕ передаются в правило сигнала."""
    rows = []
    for (file, event_time), sub in pred.loc[pred.target.eq(1)].groupby(['file', 'event_time']):
        all_file = pred.loc[pred.file.eq(file)].sort_values('decision_time')
        found = sub.loc[sub.alert_active].sort_values('decision_time')
        fresh = sub.loc[sub.alert_start].sort_values('decision_time')
        pre_event = all_file.loc[all_file.alert_start & all_file.decision_time.lt(event_time)]
        earliest_alarm = pre_event.decision_time.iloc[0] if len(pre_event) else pd.NaT
        rows.append({
            'file': file, 'well_id': int(sub.well_id.iloc[0]), 'event_time': event_time,
            'detected_active': not found.empty, 'detected_new_alarm': not fresh.empty,
            'earliest_active_time': str(found.decision_time.iloc[0]) if len(found) else '',
            'lead_minutes_active': float((event_time - found.decision_time.iloc[0]).total_seconds() / 60)
            if len(found) else None,
            'first_new_alarm_within_horizon': str(fresh.decision_time.iloc[0]) if len(fresh) else '',
            'first_alarm_before_event': str(earliest_alarm) if pd.notna(earliest_alarm) else '',
            'lead_first_alarm_before_event_minutes':
                float((event_time - earliest_alarm).total_seconds() / 60)
                if pd.notna(earliest_alarm) else None,
            'had_any_pre_event_alarm': bool(len(pre_event)),
        })
    return pd.DataFrame(rows)


def report(data: pd.DataFrame, pred: pd.DataFrame, selection: pd.DataFrame,
           grid: pd.DataFrame, output_dir: Path, args) -> None:
    from sklearn.metrics import average_precision_score
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = trigger_policy(data[['file', 'decision_time', 'score', 'target', 'event_time']], .5, 1)
    baseline['well_id'] = data.set_index(['file', 'decision_time']).loc[
        pd.MultiIndex.from_frame(baseline[['file', 'decision_time']]), 'well_id'].to_numpy()
    m0 = policy_metrics(baseline)
    m1 = policy_metrics(pred)
    ap = float(average_precision_score(data.target, data.score))
    overview = pd.DataFrame([{'policy': 'fixed_threshold_0.5_one', 'ap_unchanged': ap, **m0},
                             {'policy': 'nested_train_only', 'ap_unchanged': ap, **m1}])
    overview.to_csv(output_dir / 'comparison.csv', index=False)
    pred.sort_values(['file', 'decision_time']).to_csv(output_dir / 'outer_oof_alerts.csv', index=False)
    selection.to_csv(output_dir / 'policy_by_well.csv', index=False)
    grid.to_csv(output_dir / 'inner_selection_grid.csv', index=False)
    event_table(pred).to_csv(output_dir / 'by_event.csv', index=False)
    by_well = []
    for well, sub in pred.groupby('well_id'):
        by_well.append({'well_id': int(well), **policy_metrics(sub)})
    pd.DataFrame(by_well).to_csv(output_dir / 'by_well.csv', index=False)
    manifest = {
        'experiment': 'catboost12_30min_nested_warning_policy',
        'source': 'Petrobras 3W Dataset v2.0.0, real WELL class 2 event-enriched files',
        'horizon_minutes': 30, 'time_step_seconds': STRIDE_SECONDS,
        'features': FEATURES, 'model_iterations_inner': args.iterations,
        'outer_predictions': 'precomputed LOOW scores from compare_class2_horizons.py',
        'inner_training': '6 wells only, 6-way LOOW',
        'threshold_grid': list(THRESHOLDS), 'confirmations_grid': list(CONFIRMATIONS),
        'selection_target_event_recall': args.target_recall,
        'selection': 'minimize out-of-horizon alarm starts/h if inner event recall target reached',
        'test_labels_used_to_select_policy': False,
        'scores_calibrated': False,
        'software': {'python': platform.python_version(), 'pandas': pd.__version__},
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.features, args.oof)},
        'caveats': [
            'All wells were previously studied: estimates exploratory, not independent final validation.',
            '30-min horizon selected after exploring 15/30/60 on same wells.',
            'Targets and retrospective pre-event risk set use true labels; not a full online test.',
            'Missing or censored decisions excluded: alarms on those periods not evaluated.',
            'Event-enriched episodes, not general normal operation; event-level sample only 11.',
            'Out-of-horizon alert is not necessarily a physically false alarm.',
            'A persistent warning that begins earlier than 30 min may be active during positive window.',
            'Scores are not calibrated failure probabilities.',
            'No changes to deployed ML Agent/Coordinator/Streamlit.',
        ],
    }
    (output_dir / 'experiment.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== Сравнение политик для 30 мин (внешние LOOW-прогнозы) ===')
    cols = ['policy', 'precision', 'recall', 'f1', 'fp_windows', 'alarm_starts',
            'out_of_horizon_alarm_starts', 'events_detected_active',
            'events_detected_new_alarm', 'events_total']
    print(overview[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('AP исходных моделей (не меняется политикой):', f'{ap:.4f}')
    print('Файлы:', output_dir)
    print('ОГРАНИЧЕНИЕ: предварительный анализ на enriched событиях, а не промышленная валидация.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT / 'outputs' / 'class2_horizons' / 'cohort_features.csv')
    parser.add_argument('--oof', type=Path, default=ROOT / 'outputs' / 'class2_horizons' / 'oof_predictions.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'class2_alert_policy')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--target-recall', type=float, default=0.80)
    args = parser.parse_args()
    source = load_and_match(args.features, args.oof)
    print('Горизонт 30 минут; окон:', len(source), 'положительных:', int(source.target.sum()),
          'скважин:', source.well_id.nunique())
    alerts, selected, grid = nested_evaluation(source, args.iterations, args.target_recall)
    report(source, alerts, selected, grid, args.output_dir, args)


if __name__ == '__main__':
    main()
