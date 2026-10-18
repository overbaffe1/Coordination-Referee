# Coordination-Referee — participant solution

Диагностика сбоев координации в мультиагентных прогонах: `label` (7 классов),
`success` (0/1), `fault_turn` (int, -1 для clean).

## Локальный запуск

```bash
python3 solution.py --train data/train.csv --test data/test.csv --output predictions.csv
```

Зависимости: `pip3 install --break-system-packages pandas numpy scikit-learn lightgbm scipy torch`

Время на 2-ядерном CPU: ~5.5 мин (10k train / 4k test). На платформе (8 vCPU, опц. A100):
NN автоматически переключается на GPU и бэггирует 3 сида × 12 эпох (~3–4 мин); на CPU — 1 сид × 6 эпох. Детерминировано (сид 42 везде), без сети.

## Архитектура

| Файл | Что делает |
|---|---|
| `solution.py` | точка входа: фичи → LGBM(label) + BiGRU NN → logit-blend 0.7/0.3 + смещения классов → LGBM(success, + предсказанный label one-hot) → правила локализации fault_turn |
| `engine.py` | шаблонный парсер сообщений → 13 типов событий (assign/deliver/work/wait/loop/drift/conflict_msg/…) |
| `features4.py` | 100 структурных признаков: учёт handoff (назначено/занято/произведено), взаимные ожидания, конфликты ключей state, дрейф-подмены, серии циклов, топология, n-граммы типов |
| `nn_model.py` | BiGRU 160×2 + attention над событийной последовательностью + табличная ветка |

## Ожидаемые метрики (OOF/holdout 2000)

- MacroF1 ≈ 0.84–0.85 (бленд + пороги), Robustness-срез ≈ 0.85
- SuccessF1 ≈ 0.92–0.94, fault_turn hit@2 ≈ 0.70
- Композит ≈ 0.83

## Данные

`data/train.csv`, `data/test.csv` — получаются из `competition-ref/scripts/prepare_data.sh`
(в git не кладутся: 150 MB). В smoke-режиме (усечённый датафрейм) решение пишет
константу clean/1/-1 и выходит с кодом 0.
