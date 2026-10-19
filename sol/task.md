# Задание участнику

## TL;DR

Вам даны логи запусков мультиагентных AI-систем, решающих бизнес-задачи
(закупки, поддержка, продажи, юридическая проверка, онбординг, комплаенс).
Часть запусков прошла нормально, часть сломалась из-за **сбоя координации**
между агентами.

Постройте модель, которая по логу запуска отвечает на три вопроса:

1. **Какой сбой координации произошёл?** — 7 классов
   (`clean`, `dropped_handoff`, `duplicated_work`, `deadlock`, `conflict`,
   `goal_drift`, `runaway_loop`);
2. **Достигнута ли бизнес-цель?** — `success ∈ {0, 1}`;
3. **На каком ходу начался сбой?** — `fault_turn` (бонусная метрика).

Решение — файл `solution.py`, который на платформе запускается так:

```bash
python solution.py --train train.csv --test test.csv --output predictions.csv
```

То есть **обучение и инференс происходят внутри одного запуска**, за один
проход, без интернета, максимум 30 минут.

---

## 1. Что лежит в архиве участника

```
train.csv               10 000 ранов с разметкой (label, success, fault_turn)
test.csv                 4 000 ранов без разметки
sample_submission.csv    пример корректного формата ответа
docs/ontology.md         точные определения 7 классов  <- читать обязательно
docs/data_schema.md      описание полей и JSON-схем
docs/metrics.md          формула скора
baseline/                рабочий baseline (LightGBM), запускается как есть
participant/             шаблон solution.py и локальный прогон
```

## 2. Формат ответа

`predictions.csv` — ровно одна строка на каждый `run_id` из `test.csv`:

```csv
run_id,label,success,fault_turn
r_e8cbd1c0ed74,clean,1,-1
r_611b6210f9db,deadlock,0,61
r_9a41cb70f22d,dropped_handoff,1,14
```

Требования:

* колонки строго `run_id,label,success,fault_turn`, порядок колонок любой;
* `label` — одна из 7 строк ровно в таком написании;
* `success` — целое `0` или `1`;
* `fault_turn` — целое; `-1`, если сбоя нет или локализация не делается;
* без дубликатов `run_id`, покрыты все 4 000 ранов.

Проверить формат локально:

```bash
python scripts/validate_submission.py --pred predictions.csv --test data/test.csv
```

## 3. Как оценивается

```
score = 0.50·MacroF1 + 0.25·RobustnessF1 + 0.15·SuccessF1 + 0.10·hit@2
```

Подробности и референсные значения baseline — в [`docs/metrics.md`](metrics.md).
Baseline даёт **0.712** на public. Всё, что выше, — ваш вклад.

## 4. Ограничения запуска

| | Участник | Baseline |
|---|---|---|
| CPU | 8 vCPU | 4 vCPU |
| RAM | 32 GB | 16 GB |
| GPU | до 1× A100 40 GB | нет |
| Диск | 10 GB | — |
| Время | 30 минут на весь `solution.py` | ~1.5 мин |
| Интернет | нет | нет |

Из этого следуют практические выводы:

* **образ собирать не нужно** — вы присылаете только код (`solution.py` плюс,
  при необходимости, модули рядом с ним), окружение готово на стороне платформы;
* **никаких загрузок внутри job** — сети нет, поэтому ни `pip install`, ни
  `from_pretrained` с хаба не сработают. Предобученных моделей в образе нет:
  либо обучайте с нуля на `train.csv`, либо кладите свои веса прямо в архив
  сабмита и загружайте по относительному пути от `solution.py` (лимит размера
  сабмита — 5 ГБ);
* **GPU есть, но не обязателен** — A100 40 GB доступна, при этом baseline
  целиком CPU-шный и укладывается в 1.5 минуты. Проверяйте наличие устройства
  через `torch.cuda.is_available()` и предусматривайте CPU-ветку;
* **30 минут — это обучение и инференс вместе** на 10 000 train + 4 000 test.
  Дообучение крупного энкодера в лимит не влезет;
* решение должно быть детерминированным: фиксируйте `seed`, а для torch —
  `torch.manual_seed` и `torch.backends.cudnn.deterministic = True`.

## 5. С чего начать (30 минут работы)

```bash
git clone <repo>
cd coordination-referee
bash scripts/prepare_data.sh          # распаковать данные
pip install -r requirements.txt

# запустить baseline как есть
python baseline/solution.py --train data/train.csv --test data/test.csv \
       --output predictions.csv

# посмотреть свой скор на public (ground truth есть только у организаторов;
# для локальной валидации разбейте train сами — см. пример ниже)
```

Локальная валидация без ground truth теста:

```bash
python scripts/local_cv.py --train data/train.csv --folds 3
```

## 6. Примеры: как выглядит сигнал

### Пример A — `dropped_handoff`

```
t=12  a1 -> a4  handoff  intent=verify_compliance  refs=[]
      "a4, please take care of verify_compliance for network switches."
t=13  a4 -> a1  inform   intent=verify_compliance
      "Acknowledged, starting verify_compliance."
...   a4 больше не отправляет ни одного handoff с refs
      в artifacts нет записи с subtask=verify_compliance
```
Ответ: `label=dropped_handoff`, `fault_turn=12`.
Ключевое: назначение есть, доставки нет. Подтверждение (`ack`) не считается
доставкой.

### Пример B — `deadlock` (важно не путать с A)

```
t=31  a3 -> a5  status  "Waiting for a5 to deliver draft_redlines ..."
t=32  a5 -> a3  status  "Cannot proceed until a3 finishes check_liability."
t=33  a3 -> a5  status  "Blocked on a5: I need draft_redlines first."
t=34  a5 -> a3  status  "My step depends on a3 closing check_liability."
```
Ответ: `label=deadlock`, `fault_turn=31`.
Ключевое: **взаимная** ссылка друг на друга и отсутствие артефактов после.

### Пример C — `duplicated_work`

```
t=8   a1 -> a2  handoff  intent=build_pricing  refs=[]
t=11  a2 -> a1  handoff  intent=build_pricing  refs=[art_03]
t=12  a1 -> a6  handoff  intent=build_pricing  refs=[]      <-- второй раз
t=15  a6 -> a1  handoff  intent=build_pricing  refs=[art_04]
artifacts: art_03.hash == art_04.hash
```
Ответ: `label=duplicated_work`, `fault_turn=12`.
Ловушка: в ~45 % случаев хеши **не** совпадают (парафраз) — опирайтесь ещё и
на «две доставки по одной подзадаче от разных агентов».

### Пример D — `conflict`

```
shared_state: unit_price=820 (a3, set)
              unit_price=910 (a4, override)
              unit_price=820 (a3, override)
              unit_price=910 (a4, override)
messages:     "I disagree: unit_price must be 910, not 820."
```
Ответ: `label=conflict`, `fault_turn` = ход первой перезаписи-возражения.
Ловушка: одна перезапись — это норма (уточнение), не конфликт.

### Пример E — `goal_drift`

```
goal: "Resolve escalated ticket INC-4821 for a Q3 enterprise customer"
t=19  a2 -> a1  inform  intent=the_office_relocation
      "Side note: the office relocation looks more promising, switching focus."
t=20+ подзадачи из другого домена, финальный артефакт типа equipment_manifest
```
Ответ: `label=goal_drift`, `fault_turn=19`.
Ключевое: пересечение лексики сообщений с `goal` падает и не восстанавливается.

### Пример F — `runaway_loop` vs `clean`

```
runaway_loop:  9 подряд сообщений вида
               "Re-checking check_budget once more with db_query."
               (отличаются только числами), артефакта так и нет

clean (шум):   3 таких сообщения, затем handoff с refs=[art_05]
```
Граница между «итеративной работой» и «зацикливанием» намеренно размыта —
именно здесь теряется больше всего F1.

## 7. Идеи, которые дают прирост над baseline

Baseline — это 122 табличных признака + LightGBM + правила для `fault_turn`.
Что он **не** делает (и что стоит попробовать):

1. **Текст.** TF-IDF / char-n-граммы по сообщениям, эмбеддинги.
   Baseline использует только счётчики ключевых слов.
2. **Последовательность.** Раны — это последовательности событий: HMM,
   1D-CNN, transformer над (тип, отправитель, получатель, intent).
3. **Граф.** GNN над графом «агент → агент» или «агент → артефакт»;
   baseline берёт только агрегаты (плотность, SCC, реципрокность).
4. **Локализация как отдельная задача.** Обучите per-turn ранжировщик:
   для каждого хода — вероятность быть `fault_turn`. Баллы `hit@2` у baseline
   всего 0.36, здесь самый дешёвый прирост.
5. **Устойчивость к сдвигу.** Аугментация (удаление `intent`/`refs`,
   удлинение ранов), нормировка признаков на длину, групповая валидация по
   топологии и домену.
6. **`success` как отдельная голова**, а не производная от класса: смотрите
   на покрытие плана артефактами и на наличие восстановления.
7. **Иерархия классов.** Сначала `clean` vs `fault`, потом тип сбоя — часто
   лучше, чем плоская 7-классовая задача.

## 8. Частые ошибки

* Считать, что `run_status = completed` в `shared_state` означает `success = 1`.
  Это самооценка команды агентов, она врёт.
* Считать `label = clean` эквивалентом `success = 1`.
* Игнорировать перемешанность сообщений: подзадачи идут параллельно,
  соседние по `t` сообщения часто относятся к разным подзадачам.
* Опираться на конкретные фразы — в сдвинутой части теста формулировки другие,
  и `Robustness F1` (четверть скора) просядет.
* Забыть про раны с пустыми `intent` — их ~12 %, и признаки, построенные
  только на `intent`, там дают нули.
