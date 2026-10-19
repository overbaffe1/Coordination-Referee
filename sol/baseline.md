# Reference baseline

```
Run → Feature Extraction → Graph Features → Tabular Features → LightGBM → Prediction
```

Файлы:

* [`baseline/features.py`](../baseline/features.py) — 122 признака из одного рана;
* [`baseline/localize.py`](../baseline/localize.py) — правила для `fault_turn`;
* [`baseline/solution.py`](../baseline/solution.py) — точка входа под контракт DSWorks.

## Признаки (122)

| Группа | Примеры | Против какого класса работает |
|---|---|---|
| Объём | `n_messages`, `n_agents`, `msgs_per_agent`, `arts_per_msg` | длина как прокси зацикливания |
| Типы сообщений | доли `handoff / inform / status / state_update / final / system` | `deadlock` (status), `clean` (final) |
| Отправители | энтропия, максимальная доля, доля молчащих агентов | `dropped_handoff` |
| Повторы | точные и нормализованные дубликаты, максимальная серия | `runaway_loop` |
| Учёт передач | `unanswered_ratio`, `undelivered_ratio`, `coverage`, задержка ответа | `dropped_handoff` |
| Дублирование | `max_assign_per_intent`, `intents_with_multiple_owners`, `dup_hash_pairs` | `duplicated_work` |
| Общее состояние | перезаписи, число значений на ключ, «мигания» значения | `conflict` |
| Цель | пересечение лексики с `goal` в начале/конце, дельта | `goal_drift` |
| Граф | плотность, реципрокность, 2-циклы, треугольники, размер SCC, ping-pong | `deadlock`, `runaway_loop` |
| Топология | one-hot типа, число рёбер | контекст |
| Хвост рана | доли типов в последних 20 % ходов | `success` |

Граф считается вручную (Tarjan SCC ~30 строк), без `networkx` — меньше
зависимостей в образе.

## Модели

* `label`: LightGBM multiclass, 600 деревьев, lr 0.05, 63 листа;
* `success`: LightGBM binary, 500 деревьев;
* `fault_turn`: правила, специфичные для предсказанного класса
  (первое недоставленное назначение, второе назначение той же подзадачи,
  начало серии статусов, первая перезапись «горячего» ключа, первый разрыв
  связи с целью, начало самой длинной серии повторов).

Fallback: при отсутствии `lightgbm` используется
`HistGradientBoostingClassifier` из scikit-learn.

## Результаты

| Метрика | public | private |
|---|---|---|
| Macro F1 | 0.741 | 0.759 |
| Robustness F1 | 0.706 | 0.679 |
| Success F1 | 0.853 | 0.849 |
| fault_turn_hit@2 | 0.361 | 0.372 |
| **Score** | **0.712** | **0.713** |

F1 по классам (весь тест): `conflict 0.88`, `goal_drift 0.85`, `clean 0.82`,
`duplicated_work 0.76`, `runaway_loop 0.70`, `deadlock 0.64`,
`dropped_handoff 0.60`.

Ресурсы: ~1.5 минуты и < 2 GB RAM на 4 vCPU (10 000 train + 4 000 test),
то есть в 20 раз меньше лимита участника.

## Где baseline слаб (намеренно)

1. `dropped_handoff` (0.60) — путается с `deadlock` и с «поздней доставкой».
2. `deadlock` (0.64) — страдает на длинных сдвинутых ранах.
3. `runaway_loop` (0.70) — граница с итеративной работой в `clean`.
4. `fault_turn_hit@2` (0.36) — чистые правила, никакого обучения.
5. Текст используется только через счётчики ключевых слов.
