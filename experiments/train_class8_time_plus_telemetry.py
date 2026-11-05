"""CPS-PredDiag: preliminary test of incremental telemetry value beyond elapsed time.

Compare existing LOOW OOF predictions (Random Forest, time-only LogReg)
with Random Forest trained on 12 sensor features + oracle elapsed time.

Run from project root:
    python experiments/train_class8_time_plus_telemetry.py

Research restriction: transient start is taken from offline dataset labels.
All models are evaluated conditional on a known state 108. This is NOT an
independently confirmed improvement or an operational online forecast.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / 'outputs' / 'features' / 'class8_6h_ml_features.csv'
TIME_OOF_PATH = ROOT / 'outputs' / 'models' / 'class8_time_control' / 'oof_predictions.csv'
OUTPUT = ROOT / 'outputs' / 'models' / 'class8_time_plus_telemetry'
SENSORS = ('P-TPT', 'P-MON-CKP', 'T-TPT')
FEATURE_COLS = [f'{s}_{stat}' for s in SENSORS
                for stat in ('mean', 'std', 'delta', 'slope_per_hour')]
KEY = ['file', 'well_id', 'decision_time', 'target']


def load_and_check():
    features = pd.read_csv(FEATURE_PATH, dtype={'well_id': 'string'})
    baseline = pd.read_csv(TIME_OOF_PATH, dtype={'well_id': 'string'})
    expected_base = {'elapsed_hours', 'prob_rf_old12', 'prob_elapsed_only', *KEY}
    expected_features = {*KEY, *FEATURE_COLS}
    for frame, required, name in ((features, expected_features, 'features'),
                                  (baseline, expected_base, 'baseline')):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: отсутствуют столбцы {sorted(missing)}')
        frame['decision_time'] = pd.to_datetime(frame['decision_time'], errors='raise')
        if frame.duplicated(['file', 'decision_time']).any():
            raise ValueError(f'{name}: повторяются file + decision_time')
        if frame[KEY].isna().any().any():
            raise ValueError(f'{name}: пропуски в идентификаторах/целях')

    joined = features.merge(baseline[KEY + [
        'elapsed_hours', 'prob_rf_old12', 'prob_elapsed_only'
    ]], on=KEY, how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Не все строки feature CSV совпали с контрольными прогнозами')
    joined = joined.drop(columns=['_merge']).sort_values(
        ['file', 'decision_time']).reset_index(drop=True)
    if len(joined) != 4006 or joined['well_id'].nunique() != 9:
        raise ValueError('Неожиданное число окон или групп скважин')
    if set(joined['target'].unique()) != {0, 1}:
        raise ValueError('Ожидались только target=0 и target=1')
    numeric = joined[FEATURE_COLS + ['elapsed_hours']].apply(
        pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    if numeric['elapsed_hours'].isna().any() or (numeric['elapsed_hours'] < 0).any():
        raise ValueError('Время от начала 108 имеет пропуски или отрицательные значения')
    # Same transformation as time-only control, but learned only from history.
    numeric['log_elapsed_hours'] = np.log1p(numeric.pop('elapsed_hours'))
    for col in ('prob_rf_old12', 'prob_elapsed_only'):
        if not np.isfinite(joined[col]).all() or not joined[col].between(0, 1).all():
            raise ValueError(f'Некорректные вероятности: {col}')
    return joined, numeric


def metrics(y, p):
    pred = p >= .5
    return {
        'ap': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
        'precision_at_05': float(precision_score(y, pred, zero_division=0)),
        'recall_at_05': float(recall_score(y, pred, zero_division=0)),
        'f1_at_05': float(f1_score(y, pred, zero_division=0)),
        'fp_windows_at_05': int(((y == 0) & pred).sum()),
        'tp_windows_at_05': int(((y == 1) & pred).sum()),
    }


def main():
    d, x = load_and_check()
    y = d['target'].to_numpy(dtype=int)
    groups = d['well_id'].to_numpy()
    x = x[FEATURE_COLS + ['log_elapsed_hours']]
    def fit_grouped_rf(frame):
        predictions = np.full(len(d), np.nan)
        for train, test in LeaveOneGroupOut().split(frame, y, groups):
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('rf', RandomForestClassifier(
                    n_estimators=70, min_samples_leaf=5,
                    class_weight='balanced_subsample', random_state=42, n_jobs=1
                )),
            ])
            model.fit(frame.iloc[train], y[train])
            predictions[test] = model.predict_proba(frame.iloc[test])[:, 1]
        if not np.isfinite(predictions).all():
            raise RuntimeError('Не все точки получили прогноз')
        return predictions

    d['prob_rf_elapsed_only'] = fit_grouped_rf(x[['log_elapsed_hours']])
    d['prob_rf_time_plus_12'] = fit_grouped_rf(x)

    columns = {
        'rf_old12': 'prob_rf_old12',
        'logreg_elapsed_only': 'prob_elapsed_only',
        'rf_elapsed_only': 'prob_rf_elapsed_only',
        'rf_time_plus_12': 'prob_rf_time_plus_12',
    }
    summary = []
    per_well = []
    per_file = []
    for name, col in columns.items():
        p = d[col].to_numpy()
        detection_count = sum(bool(((g['target'] == 1) & (g[col] >= .5)).any())
                              for _, g in d.groupby('file') if (g['target'] == 1).any())
        summary.append({'model': name, **metrics(y, p), 'detected_events': detection_count})
        for well, part in d.groupby('well_id'):
            yy = part['target'].to_numpy(dtype=int)
            pp = part[col].to_numpy()
            per_well.append({'model': name, 'well_id': well, 'n': len(part),
                             'positives': int(yy.sum()), **metrics(yy, pp)})
        for file, part in d.groupby('file'):
            yy = part['target'].to_numpy(dtype=int)
            pp = part[col].to_numpy()
            per_file.append({'model': name, 'file': file, 'well_id': part['well_id'].iloc[0],
                             'has_event': bool((yy == 1).any()),
                             'detected_event': bool(((yy == 1) & (pp >= .5)).any()),
                             'fp_windows': int(((yy == 0) & (pp >= .5)).sum()),
                             'tp_windows': int(((yy == 1) & (pp >= .5)).sum())})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUTPUT / 'summary.csv', index=False)
    pd.DataFrame(per_well).to_csv(OUTPUT / 'by_well.csv', index=False)
    pd.DataFrame(per_file).to_csv(OUTPUT / 'by_file.csv', index=False)
    d[KEY + ['elapsed_hours', *columns.values()]].to_csv(
        OUTPUT / 'oof_predictions.csv', index=False)
    print('\nLOOW: одна скважина вне обучения, всего 9 групп')
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print('\nAP по скважинам:')
    print(pd.DataFrame(per_well).pivot(index='well_id', columns='model', values='ap').round(3).to_string())
    print('\nРезультаты:', OUTPUT)
    print('ОГРАНИЧЕНИЯ: начало 108 получено из истинной офлайн-разметки; тестовые скважины уже изучались.')
    print('Порог 0.5 не настраивался; оценки Random Forest не калиброваны.')


if __name__ == '__main__':
    main()
