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
| LoRA/QLoRA | Выполнен QLoRA запуск с native chat template; validation loss 2.3581; адаптер и holdout отчёт сохранены в `finetuning/results/qwen-qlora/` |
| Среда для обучения | Windows, Python 3.13.0, PyTorch 2.12.0+cu130, Transformers 4.57.6, PEFT 0.21.2, Datasets 4.8.5, Accelerate 1.15.0, bitsandbytes 0.50.2; RTX 3050 Laptop GPU, 4 GiB |
| Ручная педагогическая проверка | Не выполнена |

## LoRA / QLoRA experiment

`finetuning/examples.jsonl` содержит 28 коротких синтетических RU/KK примеров: 16 train, 4 validation, 8 test. `python -m finetuning.prepare_data` проверяет схему, split и повторяющиеся prompts. Тренировочный скрипт обучается только на train и сохраняет validation loss и trainer logs. `python -m finetuning.evaluate_adapter` сравнивает base/adapter на test по exact expected-term recall и abstention markers. Эти метрики — автоматические proxies; они не оценивают полезность ответа носителем языка.

Запуск от 2026-10-22: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), revision `7ae557604adf67be50417f59c2c2f167def9a775`, QLoRA 4-bit NF4, LoRA rank 8 / alpha 16 / dropout 0.05, родной chat template токенизатора, sequence length 256, batch size 1, gradient accumulation 8, learning rate 2e-4, 1 epoch и 2 optimizer steps. Trainer занял 11.87 секунды без загрузки модели; validation loss — 2.3581. Адаптер занимает 17,640,136 байт и приложен в репозитории. Точный конфиг и логи лежат в `finetuning/results/qwen-qlora/training_metadata.json`.

| Holdout proxy, 8 синтетических примеров, max 96 новых токенов | Base | QLoRA |
|---|---:|---:|
| Exact expected-term recall | 0.0833 | 0.0833 |
| Abstention behavior accuracy | 0.7500 | 0.7500 |

Этот запуск **не улучшил** выбранные прокси. Base выдал неподтверждённые рекомендации по снижению температуры; adapter на этом кейсе ответил только «Обратитесь к врачу», но не прошёл заданную проверку отказа. Несколько казахских ответов обеих версий бессвязны или не по теме. Не используйте этот адаптер для родительских ответов, медицинских запросов или демонстрации как готового качества. Нужна экспертная bilingual разметка, больше данных и ручная слепая оценка.

`finetuning/results/qwen-qlora/adapter-evaluation.json` сохраняет сырые base/adapter ответы. `adapter_config.json` указывает на открытый базовый model ID; базовые веса не включены. Проведённый run использовал локальный кэш Hugging Face, а не абсолютный путь из машины разработчика.

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
