# qwen3-tts-server

Локальный сервер синтеза речи на базе [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) с OpenAI-совместимым API (`/v1/audio/speech`). Предназначен для подключения к Open WebUI и другим клиентам, которые умеют работать с OpenAI TTS.

- Модель: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (Hugging Face), 9 встроенных голосов, образец голоса не нужен.
- Язык синтеза: русский (задан в коде).
- Выполнение на GPU (`cuda:0`, `bfloat16`, внимание через SDPA, без FlashAttention).

> Среди встроенных голосов нет носителей русского языка, поэтому возможен акцент. Сравните голоса на слух (например, `Serena` и `Ryan`).

## Требования

| Компонент | Версия / примечание |
|---|---|
| ОС | Ubuntu (проверено на 26.04) |
| GPU | NVIDIA с поддержкой CUDA (проверено на RTX 5070 Laptop) |
| Python | 3.12 (рекомендация разработчиков Qwen3-TTS; системный Python 3.14 не трогаем) |
| Системные пакеты | `ffmpeg`, `libsndfile1`, `build-essential` (GCC нужен Triton) |
| Диск | несколько ГБ под PyTorch и веса модели |

Проверенные версии пакетов: `qwen-tts 0.1.1`, `torch 2.14.1`, `transformers 4.57.3`, `fastapi 0.146.0`, `uvicorn 0.54.0`, `soundfile 0.14.0`.

## Установка

### 1. Системные зависимости

```bash
sudo apt update
sudo apt install -y pipx ffmpeg libsndfile1 build-essential
pipx install uv
```

Проверьте, что компилятор установлен:

```bash
gcc --version
```

### 2. Клонирование и окружение Python 3.12

```bash
git clone git@github.com:kolandor/qwen3-tts-server.git ~/ai/qwen3-tts
cd ~/ai/qwen3-tts

~/.local/bin/uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3. Python-пакеты

```bash
~/.local/bin/uv pip install -U qwen-tts fastapi uvicorn soundfile
```

### 4. Проверка GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Недоступен')"
```

Ожидаемый вывод: `CUDA: True` и название вашей видеокарты. Если выводится `CUDA: False`, сначала разберитесь с установкой PyTorch и драйвером NVIDIA.

## Запуск сервера

```bash
cd ~/ai/qwen3-tts
source .venv/bin/activate
export CC=$(command -v gcc)

uvicorn tts_server:app --host 127.0.0.1 --port 8002
```

- При первом запуске веса модели скачиваются с Hugging Face в `~/.cache/huggingface`. Дальше они берутся из кеша.
- Модель загружается на GPU при старте, в логе появится `Loading Qwen3-TTS on GPU...`.
- Если сервер должен быть доступен из Docker-контейнера (например, Open WebUI) или с другой машины, укажите `--host 0.0.0.0`.
- Если VRAM мало, можно выгрузить модель Ollama, например `ollama stop qwen3.5:4b`. Сама модель при этом не удаляется.

Видеопамять удобно отслеживать во втором терминале: `watch -n 1 nvidia-smi`.

## API

### `GET /health`

```bash
curl http://127.0.0.1:8002/health
# {"status":"ok"}
```

### `GET /v1/voices`

Список встроенных голосов модели.

```bash
curl http://127.0.0.1:8002/v1/voices
```

### `POST /v1/audio/speech`

OpenAI-совместимый синтез речи. Возвращает `audio/wav` (PCM 16 бит).

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `input` | string | — | Текст, от 1 до 1500 символов (обязательное поле) |
| `voice` | string | `Serena` | Имя голоса из `/v1/voices`. `default` и `alloy` заменяются на `Serena` |
| `model` | string | `tts-1` | Принимается для совместимости, не используется |
| `response_format` | string | `wav` | Поддерживается только `wav` |
| `speed` | float | `1.0` | Принимается для совместимости, не используется |

```bash
curl -X POST http://127.0.0.1:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Привет! Как же здорово, что мы наконец-то можем поговорить.", "voice": "Serena"}' \
  -o speech.wav
```

Запросы обрабатываются по одному (глобальная блокировка), поэтому параллельные запросы встают в очередь.

## Подключение к Open WebUI

**Admin Panel → Settings → Audio → Text-to-Speech:**

| Параметр | Значение |
|---|---|
| Text-to-Speech Engine | OpenAI |
| API Base URL | `http://127.0.0.1:8002/v1`; если Open WebUI работает в Docker: `http://host.docker.internal:8002/v1` |
| API Key | любая непустая строка, например `local` |
| TTS Voice | `Serena` (или другой голос из `/v1/voices`) |
| TTS Model | `tts-1` |

## Демо-интерфейс (опционально)

Чтобы послушать голоса через официальный веб-интерфейс Gradio без API:

```bash
export CC=$(command -v gcc)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --device cuda:0 --dtype bfloat16 --no-flash-attn \
  --ip 127.0.0.1 --port 7860
```

Затем откройте http://127.0.0.1:7860, выберите Language: Russian, голос и нажмите **Generate**.

## Решение проблем

**`RuntimeError: Failed to find C compiler. Please specify via CC environment variable...`**
Triton не нашёл компилятор C. Установите `build-essential` и перед запуском выполните `export CC=$(command -v gcc)`.

**`CUDA: False`**
PyTorch не видит GPU. Проверьте драйвер (`nvidia-smi`) и что пакеты установлены внутри активированного `.venv`.

**Ошибка 400 при синтезе**
Неизвестное имя голоса (сверьтесь со списком из `/v1/voices`) или формат, отличный от `wav`.

**Нехватка видеопамяти (CUDA out of memory)**
Освободите VRAM: остановите другие модели (`ollama stop <model>`) или закройте запущенный `qwen-tts-demo`.
