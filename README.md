# Nurtaro AI — Bilingual Parenting Assistant

**Локальный демонстрационный проект для портфолио ML Engineer.** Он показывает end-to-end поток: загрузка материалов, индексация, RU/KK retrieval, профили, API, Telegram-интерфейс, воспроизводимая оценка и отдельный LoRA/QLoRA pipeline.

> Это независимый проект кандидата. Он не принадлежит Nurtaro и не использует внутренние материалы компании. Все демонстрационные памятки и fine-tuning примеры синтетические. Не загружайте реальные персональные данные ребёнка.

## Зачем

Nurtaro публично описывает экосистему раннего развития 0–3 лет и RU/KZ AI-помощника для родителей ([сайт Nurtaro](https://nurtaro.kz/)). В этом прототипе задача сведена к проверяемому MVP: отвечать на вопросы по предоставленным материалам, возвращать источники, учитывать явные предпочтения семьи и воздерживаться при отсутствии опоры в источниках.

## Что реализовано

- FastAPI: `/health`, `/chat`, загрузка и индексация документов, список документов, создание и чтение профиля, `/evaluate`; OpenAPI доступен на `/docs`.
- RAG: PDF/TXT/Markdown, ограничение размера и страниц PDF, удаление очевидных prompt-injection строк, разбиение на фрагменты, векторы и cosine retrieval в SQLite, порог отказа, ссылки `document_id#chunk-N`.
- RU/KK: определение казахского по характерным буквам, русский fallback, короткий ответ на языке вопроса, 32 кейса для двух языков.
- Персонализация: возраст, язык, интересы, материалы и пожелания; профиль используется в prompt и слегка влияет на ранжирование источников.
- Генерация: автономный extractive режим по умолчанию, поддержка Qwen через Ollama, а также OpenAI-compatible chat API. Настраиваются температура, лимит токенов и таймаут.
- Telegram bot: `/start`, `/profile`, `/ask`, `/sources`, `/help`; запускается только с токеном в окружении.
- Видео: защищённый `/vision/analyze` выборочно анализирует короткий ролик YOLO; оригинал хранится только во временном файле и удаляется после обработки. `/videos/transcripts/upload` индексирует русские/казахские TXT/VTT/SRT-субтитры с временными метками в тот же RAG.
- Fine-tuning: 28 синтетических RU/KK примеров с train/validation/test разделением, LoRA/QLoRA pipeline на PyTorch/Transformers/PEFT, загрузка адаптера, validation loss и воспроизводимое base-vs-adapter сравнение на holdout.
- Production основа: структурированные JSON-логи без текстов вопросов, API-key principals, owner scoping, Docker Compose, health check, CI, тесты.

## Быстрый локальный запуск

Требуются Python 3.12+ и pip. В PowerShell из корня проекта:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Откройте `.env` и замените оба API-секрета на случайные длинные значения. `BOT_API_KEY` должен совпадать с секретом principal `telegram-bot`. Затем запустите API:

```powershell
python -m uvicorn app.main:app --reload
```

Откройте [http://localhost:8000/docs](http://localhost:8000/docs). Локальный режим по умолчанию не скачивает модельные веса и не вызывает внешние API.

### Запустить Qwen через Ollama

Установите Ollama на выбранной машине и скачайте настроенную 4-bit модель `qwen2.5:3b-instruct-q4_K_M`:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

Размер опубликованного тега — около 1.9 GB. Для линейки Qwen2.5 3B применяется лицензия Qwen; до коммерческого использования проверьте лицензионные условия.

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b-instruct-q4_K_M
```

Перезапустите API. Для Docker Compose адрес Ollama на хосте обычно задан как `http://host.docker.internal:11434`; при другом окружении поправьте `OLLAMA_BASE_URL`. Если модель недоступна или истёк таймаут, API использует extractive fallback и не теряет найденные ссылки.

### Улучшить embeddings

Текущий default — лёгкие детерминированные hashed word/character n-gram vectors с небольшим RU/KK concept bridge. Он не эквивалентен pretrained semantic embedding и нужен, чтобы MVP запускался офлайн.

```powershell
python -m pip install -r requirements-semantic.txt
$env:EMBEDDING_BACKEND='sentence-transformers'
$env:SENTENCE_TRANSFORMER_MODEL='intfloat/multilingual-e5-small'
python -m uvicorn app.main:app --reload
```

В этом режиме первый запуск скачает веса модели. После смены embedding backend переиндексируйте документы через `POST /documents/index` с телом `{"force":true}`: векторы разных моделей несовместимы.

### Видео и RAG по видеоурокам

Для CV установите optional-зависимости. На GPU сначала установите PyTorch wheel, подходящий вашей версии CUDA:

```powershell
python -m pip install -r requirements-vision.txt
```

`POST /vision/analyze` принимает MP4/MOV/MKV/WEBM/AVI до 50 MiB и 30 секунд по умолчанию, выбирает кадр в секунду и возвращает названия классов, confidence, координаты рамок и latency. При первом запросе Ultralytics скачает модель `yolo11n.pt` в свой локальный cache; вес можно заменить настройкой `VISION_MODEL` на локальный `.pt` файл.

```powershell
curl.exe -H "X-API-Key: your-demo-key" -F "file=@C:\temp\licensed-demo.mp4" http://localhost:8000/vision/analyze
```

Чтобы включить содержание видеоурока в RAG, загрузите заранее подготовленные UTF-8 субтитры TXT, VTT или SRT:

```powershell
curl.exe -H "X-API-Key: your-demo-key" -F "file=@C:\temp\lesson.vtt" "http://localhost:8000/videos/transcripts/upload?language=ru"
```

После этого обычный `POST /chat` ищет по транскрипту и возвращает ссылку на фрагмент. ASR не встроен: субтитры должен подготовить внешний/локальный ASR или человек. YOLO-демо показывает переносимый CV/video inference, но не угадывает скрытый продукт студии и не анализирует развитие, эмоции или безопасность ребёнка. Качество YOLO по целевым классам пока **не измерялось** на лицензированном размеченном наборе.

## API пример

Задайте в PowerShell тот же ключ, который указан у principal `demo` в `.env`:

```powershell
$headers = @{ 'X-API-Key' = 'your-demo-key' }
$profile = Invoke-RestMethod -Method Post -Uri http://localhost:8000/profiles `
  -Headers $headers -ContentType 'application/json' `
  -Body '{"age_months":18,"preferred_language":"kk","interests":["кітап"],"available_materials":["кітап","кубик"],"parent_preferences":"Қысқа ойындар"}'

Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat `
  -Headers $headers -ContentType 'application/json' `
  -Body (@{ question = 'Текшемен ойынды қалай дайындауға болады?'; profile_id = $profile.id } | ConvertTo-Json)
```

Ответ содержит `answer`, `language`, `sources`, `abstained`, `generation_mode`, latency и использованные поля персонализации. Нерелевантный вопрос вроде «Какой сегодня курс евро?» должен вернуть пустой список источников и сообщение о недостатке данных.

### Загрузить и проиндексировать материал

```powershell
curl.exe -H "X-API-Key: your-demo-key" -F "file=@C:\temp\guide.md" http://localhost:8000/documents/upload
curl.exe -X POST -H "X-API-Key: your-demo-key" -H "Content-Type: application/json" -d "{}" http://localhost:8000/documents/index
```

Поддерживаются `.pdf`, `.txt`, `.md`, `.markdown`; текущие default лимиты: 5 MiB и 50 страниц PDF. Сервер сохраняет извлечённый очищенный текст в локальной SQLite базе, а оригинальный файл не архивирует.

## Telegram

В `.env` установите `TELEGRAM_BOT_TOKEN`, `BOT_API_KEY` (тот же секрет, что у `telegram-bot` в `API_KEYS`) и `API_BASE_URL=http://localhost:8000`. В отдельном терминале:

```powershell
python -m app.bot.telegram_bot
```

Бот отвечает только в личном чате. Его состояние с последними источниками хранится в памяти процесса; вопросы/ответы в SQLite не пишутся.

## Docker Compose

Скопируйте `.env.example` в `.env`, задайте секреты и запустите API:

```bash
docker compose up --build api
```

С Telegram:

```bash
docker compose --profile telegram up --build
```

SQLite хранится в именованном volume `nurtaro-data`. Порт API — `8000`. Перед production-развёртыванием нужен TLS reverse proxy, секрет-хранилище, политика retention, мониторинг и отдельное решение для зашифрованного хранения персональных данных.

## Проверки и evaluation

```powershell
python -m compileall -q app evaluation finetuning tests
python -m pytest tests -q -p no:cacheprovider -p no:tmpdir
python -m evaluation.run_eval
```

Evaluation содержит 32 вопроса: 16 русских и 16 казахских, включая два вопроса без ожидаемого источника. Метрики и per-case CSV сохраняются в `evaluation/results.json` и `evaluation/results.csv`. Automated metrics — retrieval/evidence proxies; они не являются human quality score. Слепой baseline без контекста воздерживается и не цитирует источник. Перед заявлением о качестве требуется ручной просмотр по [evaluation/MANUAL_REVIEW.md](evaluation/MANUAL_REVIEW.md).

Текущий локальный reference run `20261022T161904Z` на Python 3.13: expected document hit 1.00, MRR 0.9833, evidence phrase recall proxy 1.00, language/abstention accuracy 1.00, стабильность 1.00; средняя latency 7.81 ms и p95 10.17 ms. Это маленький синтетический набор без оценки носителем языка и не прогноз production качества.

QLoRA фактически обучен на RTX 3050 Laptop GPU; результаты и ограничения опубликованы в [EXPERIMENTS.md](EXPERIMENTS.md), адаптер сохранён в `finetuning/results/qwen-qlora/`. На 8 синтетических holdout-кейсах exact-term recall proxy остался 0.0833 после обучения, abstention proxy — 0.75. Это отрицательный результат маленького учебного запуска, а не доказательство качества модели. CV-код есть, но inference на лицензированном целевом видеонаборе и его метрики пока не проверялись. Evaluation не вызывает внешнюю LLM по умолчанию. Для сравнения того же настроенного генератора с retrieved context и без него используйте `python -m evaluation.run_eval --with-configured-model`; это до 62 вызовов для набора из 32 кейсов и может расходовать квоту endpoint.

## LoRA / QLoRA pipeline

Набор синтетический и мал, он демонстрирует форму pipeline, а не пригоден для обучения продуктовой модели.

```bash
python -m pip install -r finetuning/requirements-finetuning.txt
python -m finetuning.prepare_data
python -m finetuning.train_lora --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 1 --max-length 128
# QLoRA требует совместимую CUDA GPU и bitsandbytes:
# Сначала установите CUDA-сборку PyTorch для своего драйвера/ОС, затем зависимости QLoRA.
python -m pip install -r finetuning/requirements-qlora.txt
python -m finetuning.train_lora --base-model Qwen/Qwen2.5-0.5B-Instruct --model-id Qwen/Qwen2.5-0.5B-Instruct --epochs 1 --max-length 128 --qlora --output-dir finetuning/artifacts/qwen-qlora
python -m finetuning.load_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter finetuning/artifacts/qwen-qlora --prompt "Предложи игру с книгой."
python -m finetuning.compare_base_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter finetuning/artifacts/qwen-qlora
python -m finetuning.evaluate_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter finetuning/artifacts/qwen-qlora --max-new-tokens 96
```

Запуск использует только `train`, считает validation loss, а `evaluate_adapter` сравнивает ответы base/adapter на 8 отложенных RU/KK примерах по точному совпадению заданных понятий и маркерам отказа. Это маленькие автоматические прокси, а не ручная оценка ответа. Датасет синтетический и создан для демонстрации pipeline; это не экспертный корпус и не показатель качества для production. Основные временные файлы находятся в игнорируемой `finetuning/artifacts/`; компактный adapter и отчёты эксперимента сохранены в `finetuning/results/qwen-qlora/`.

## Документация

- [Архитектура и Mermaid схема](ARCHITECTURE.md)
- [Эксперименты и их реальные ограничения](EXPERIMENTS.md)
- [Вопросы и ответы к собеседованию (45+)](INTERVIEW_QA.md)
- [Двухминутный сценарий демонстрации](docs/DEMO_SCRIPT.md)
- [Видео pipeline и протокол его оценки](docs/VIDEO_PIPELINE.md)
- [Рубрика ручной проверки](evaluation/MANUAL_REVIEW.md)

## Ограничения и следующий шаг

Это локальный MVP: SQLite cosine scan не рассчитан на большой корпус, hashed embeddings имеют низкую семантическую полноту, language detection эвристическая, фильтр injection не является полной защитой, а extractive ответы не заменяют пользовательскую оценку. Видео использует общую предобученную COCO-модель, без адаптации под целевую задачу и без трекинга объектов. Для production нужны очередь/worker, rate limiting, мониторинг, TLS и изолированная обработка недоверенных медиафайлов. Распознавание речи не входит в приложение, а транскрипты загружаются отдельно. Следующие шаги — лицензированный размеченный набор видео, ручная bilingual оценка RAG и отдельный воспроизводимый запуск LoRA на GPU.

## Технические ссылки

- [FastAPI: загрузка файлов](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Ollama: Chat API](https://docs.ollama.com/api/chat)
- [Hugging Face PEFT: quantization / QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Ollama library: Qwen2.5 model tags and license](https://ollama.com/library/qwen2.5/tags)
- [Ultralytics: video prediction](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics: license options](https://docs.ultralytics.com/platform/account/settings/)
