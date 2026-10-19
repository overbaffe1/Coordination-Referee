# Лаборатория Coordination Referee — локальный цикл улучшения скора

Цикл: **score → tune → final → сабмит на платформу**. Каждый прогон пишет строку
в `scores_history.csv`, так что рост скора видно от запуска к запуску.

## 1. Установка (один раз)

Нужен Python 3.10+ (с python.org). В консоли в папке лаборатории:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU не обязателен: всё работает на CPU (медленнее). Если есть CUDA — добавь флаг
`--gpu` к командам score/tune, нейронка обучится с бэггингом 3 сидов.

## 2. Прогон скора (~5 мин CPU, ~15 мин с нейронкой)

```powershell
python lab.py score
```

Обучает модели на 85% train и честно считает итоговый скор формулы платформы
на оставшихся 15%: MacroF1, RobustnessF1 (прокси-срез agents>=7), SuccessF1,
fault_turn hit@2 и композит. Печатает главные путаницы классов.

## 3. Тюнинг (~2 мин, скор растёт)

```powershell
python lab.py tune
```

Перебирает вес нейронки в бленде, порог success и смещения классов, оптимизируя
композит прямо на holdout. Лучшие настройки запекаются в `weights/meta.json`.
Повторные запуски monotone: хуже не станет.

## 4. Сборка сабмита (~10 сек)

```powershell
python lab.py final
```

Собирает `my_submission.zip` с текущими весами и тюнинговыми настройками —
его и грузи на платформу.

## Полный цикл заново

```powershell
python lab.py score --seed 7      # другой holdout — проверка, что тюнинг не переобучился
python lab.py tune --seed 7
python lab.py final
```

## Памятка

- Локальный скор — компас (тюнинг на holdout слегка оптимистичен); финальный
  судья — лидерборд платформы.
- `python lab.py score --no-nn` — быстрый прогон без нейронки (только LGBM).
- `python lab.py final --retrain` — переобучить все веса на 100% train
  (~1–2 ч CPU, ~20–30 мин GPU) перед сборкой сабмита.
- Что крутить дальше, если хочешь улучшать руками: гиперпараметры LGBM в
  `pretrain.py` (`fit_lgb_label`/`fit_lgb_success`), эпохи/сид NN в `lab.py score`,
  новые фичи в `sol/features4.py` (функция `extract_features`).
