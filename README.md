# Coordination-Referee — participant solution

Диагностика сбоев координации в мультиагентных прогонах: `label` (7 классов),
`success` (0/1), `fault_turn` (int, -1 для clean).

## Быстрый старт

```bash
# 1. Обучение офлайн (на боевом train.csv из раздела «Данные» платформы)
python3 pretrain.py --train data/train_battle.csv --out weights

# 2. Сборка сабмита (плоский zip, solution.py в корне)
python3 make_submission.py --weights weights --out submission.zip

# 3. Локальная проверка контракта
python3 solution.py --train data/train_battle.csv --test data/test.csv --output predictions.csv
```

Зависимости: `pip3 install --break-system-packages pandas numpy scikit-learn lightgbm scipy torch`
(GPU используется автоматически, если видна CUDA).

## Три режима solution.py (выбираются автоматически)

1. **weights/ в архиве** — только инференс предобученных моделей (~1 мин на 4k ранов).
2. **нет weights/** — быстрое обучение LightGBM-голов в самом job'е (~2–3 мин на 8 vCPU).
3. **smoke-тест** (в test.csv только колонки сабмита) — константный predictions.csv, exit 0.

## Архитектура

| Файл | Что делает |
|---|---|
| `pretrain.py` | офлайн-обучение: LGBM(label)+LGBM(success)+BiGRU-бэггинг, тюнинг порогов на OOF, всё в `weights/` |
| `solution.py` | точка входа платформы: фичи → бленд LGBM+NN → success → правила fault_turn |
| `sol/engine.py` | шаблонный парсер сообщений → 13 типов событий (опирается на поле `type`, устойчив к смене домена) |
| `sol/features4.py` | 108 структурных признаков + словарь-независимые прокси для сдвинутого среза |
| `sol/nn_model.py` | BiGRU 160×2 + attention над событийной последовательностью + табличная ветка |
| `sol/localize.py` | правила локализации fault_turn (hit@2: conflict .94, dup .90, deadlock .78, …) |

## Ожидаемое качество

OOF MacroF1 ≈ 0.85 (бленд), SuccessF1 ≈ 0.94, fault_turn hit@2 ≈ 0.70.
Композит на holdout ≈ 0.83.

## Воспроизводимость (п. 6 правил)

Все веса получаются одной командой `pretrain.py` (сид 42 везде). Скрипт лежит
в корне архива сабмита.

