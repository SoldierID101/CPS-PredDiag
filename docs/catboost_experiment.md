# Эксперимент: CatBoost против Random Forest (Petrobras 3W, класс 2)

**Статус:** эксперимент подготовлен, результаты CatBoost ещё не получены.  
**Цель:** сравнить CatBoost с прежней RF на **одних и тех же размеченных окнах**.

## Входные данные

- `experiments/inputs/class2_horizon_points.csv` — **исходные 858 записей** прошлого исследования горизонтов; при 15 мин остаются **253 окна с известным исходом**, из них 33 положительных. Остальные 33 окна с неизвестным исходом не объявляются отрицательными.
- `experiments/inputs/class2_rf_baseline_oof.csv` — исходные **253 out-of-fold** прогноза прежней RF (`prob_rf_all12`); не является full-fit моделью.
- `dataset/2/WELL-*.parquet` — локальная копия реальных файлов из Petrobras 3W Dataset v2.0.0.

## Методика

1. Только горизонт **15 минут** и **условие oracle**: окна были отобраны по настоящей размеченной переходной фазе `102` предыдущим исследованием.
2. Из истории каждой точки берутся **ровно предшествующие 300 секунд**, 12 признаков по `P-MON-CKP`, `P-TPT`, `T-TPT` (mean, std, delta, slope_per_hour), согласно `ml_agent.features_from_history`.
3. В признаки не попадают `class`, `state`, момент события, идентификатор скважины или будущие измерения.
4. **Leave-One-Well-Out**, семь тестовых скважин. В каждом fold обе модели обучаются на остальных шести и оцениваются на удержанной.
5. Параметры моделей зафиксированы **до** сравнения: RF: 70 деревьев, `min_samples_leaf=5`, `balanced_subsample`; CatBoost: 300 итераций, `depth=4`, `learning_rate=0.03`, `l2_leaf_reg=5`, `auto_class_weights=Balanced`.
6. Импутация медианой для RF обучается **только на тренировочных скважинах**; CatBoost обрабатывает числовые NaN собственным механизмом.
7. Сравниваются **исторические out-of-fold оценки RF** + новая RF, переобученная тем же скриптом, + CatBoost. Основная метрика — **Average Precision (AP / PR-AUC-аппроксимация)**. Дополнительные: ROC-AUC, Precision/Recall/F1 при фиксированном исследовательском пороге 0.5, число ложноположительных окон и доля обнаруженных *доступных для обнаружения* эпизодов.

## Интерпретация

- Это **не сравнение в режиме `blind`**: сам набор точек прогнозирования выбран по истинной размеченной фазе `102`. Ни одна модель пока не детектирует `102` самостоятельно.
- Первоначальные 12 признаков и постановка задачи уже изучались на тех же 7 скважинах; сравнение **исследовательское**, а не финальная независимая проверка новой гипотезы.
- Значения `predict_proba` не откалиброваны. Порог 0.5 — только для сопоставимости, **не** промышленный порог тревоги.
- 253 соседних окна **не** являются 253 независимыми событиями. Для выводов важны результаты по скважинам и независимым эпизодам.
- Если CatBoost даст лучшие AP/F1, это **не докажет** превосходство при отсутствии `102` в режиме `blind`; для этого потребуется новый протокол подготовки окон.
- Перенос в `ml_agent.py`, Coordinator и Streamlit до оценки не выполняем.

## Запуск (PowerShell, из корня CPS-PredDiag)

```powershell
python -m pip install catboost
python -m unittest discover -s tests -v
python experiments/compare_class2_catboost.py
```

**Отчёты:** `outputs/catboost_compare/` — `comparison.csv`, `by_well.csv`, `by_event.csv`, `oof_predictions.csv`, `features_audit.csv`, `experiment.json`, `pr_curves.png` (если установлен matplotlib).

**После запуска** нужно прислать `comparison.csv`, `by_well.csv` и `by_event.csv`. По ним выберем, стоит ли развивать CatBoost или менять постановку задачи.

## Источники

- Petrobras 3W Dataset: https://github.com/petrobras/3W
- CatBoost, обработка пропусков: https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing
- CatBoost, установка: https://catboost.ai/docs/en/installation/python-installation-method-pip-install
