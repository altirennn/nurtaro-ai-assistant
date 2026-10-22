# Эксперименты и подтверждённые результаты

## Reference evaluation

`evaluation/cases.jsonl` содержит 32 фиксированных вопроса: 16 RU, 16 KK, два вопроса без ответа в синтетической базе. `python -m evaluation.run_eval` сохраняет сводку и per-case данные в JSON/CSV.

Прогон по умолчанию использует локальные hash/ngram vectors и extractive ответы, не вызывает внешнюю модель и не создаёт затрат на inference. В отчёте отражены попадание в ожидаемый документ, MRR, совпадение expected evidence фраз, корректность отказа, покрытие answerable вопросов, проверка стабильности, groundedness proxy и latency. Groundedness/evidence — технические прокси, а не человеческая оценка. Опция `--with-configured-model` сравнивает тот же генератор с контекстом и без него и может сделать до 62 вызовов, поэтому включается отдельно.

| Поле | Результат |
|---|---|
| Последний прогон | `20261022T161904Z` (UTC), Python 3.13, hashing-ngram, extractive, top-k 4, порог 0.08 |
| Вопросы | 32 (16 RU, 16 KK) |
| Expected document hit / MRR | 1.000 / 0.9833 на 30 answerable кейсах |
| Expected evidence phrase recall proxy | 1.000 на тех же 30 кейсах |
| Language accuracy / abstention accuracy | 1.000 / 1.000 на этом фиксированном наборе |
| Groundedness proxy / stability | 1.000 / 1.000 |
| Без контекста: answer coverage | 0.000 для extractive baseline; он намеренно воздерживается без источника |
| Средняя / p95 latency | 7.81 / 10.17 ms на локальном CPU, 32 in-process запроса; не production benchmark |
| Измерения | Retrieval, evidence phrase recall proxy, abstention, стабильность и latency |
| Unit/integration tests | 15 passed на Python 3.13 |
| LoRA/QLoRA | Не запускался; обучающих метрик и адаптера нет |
| Среда для обучения | RTX 3050 Laptop GPU (4 GiB) обнаружена, но доступный PyTorch сообщает `cuda_available=False`; Transformers, PEFT и Datasets не установлены. Обучение не запускалось. |
| Ручная педагогическая проверка | Не выполнена |

## LoRA / QLoRA experiment status

`finetuning/examples.jsonl` содержит 28 коротких синтетических RU/KK примеров: 16 train, 4 validation, 8 test. `python -m finetuning.prepare_data` проверяет схему, split и повторяющиеся prompts. Тренировочный скрипт обучается только на train и сохраняет validation loss и trainer logs. `python -m finetuning.evaluate_adapter` сравнивает base/adapter на test по exact expected-term recall и abstention markers. Эти метрики — автоматические proxies; они не оценивают полезность ответа носителем языка.

Фактический LoRA/QLoRA запуск в текущей среде **не выполнен**. Поэтому adapter, validation loss и base-vs-adapter результаты не созданы; не публикуйте численные claims до запуска команд из README на машине с установленными зависимостями/весами и фиксированным окружением. Синтетический датасет нужен только для проверки формы пайплайна и не заменяет экспертную разметку.

Команда подготовки успешно проверила все 28 строк и разделение 16/4/8. Исполнение модели, training и evaluation adapter не выполнялись.

## Video pipeline status

Optional endpoint `/vision/analyze` использует OpenCV и Ultralytics YOLO для выборочного inference на коротких роликах. Субтитры VTT/SRT/TXT индексируются отдельно в RAG. Размеченный набор и детекторные метрики отсутствуют: precision/recall, mAP, runtime и VRAM нельзя указывать как измеренные в этом проекте.

Не представляйте автоматические retrieval metrics как клиническую, педагогическую или пользовательскую оценку качества. Для ручной рубрики см. [MANUAL_REVIEW.md](evaluation/MANUAL_REVIEW.md).

## Повторить

```bash
python -m evaluation.run_eval
python -m pytest tests -q -p no:cacheprovider -p no:tmpdir
```

## Наблюдение для следующего этапа

Сначала нужен честный bilingual corpus с лицензиями и expert-approved expected passages. Затем следует сравнить default hash/ngram retrieval с `intfloat/multilingual-e5-small` на фиксированном наборе, учитывать latency и ошибки отдельно по RU/KK, провести слепую ручную оценку и сохранять commit/model/config identifiers вместе с каждым результатом. До такой проверки не выбирается победитель.
