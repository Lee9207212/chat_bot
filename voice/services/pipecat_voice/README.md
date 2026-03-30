# Pipecat Voice Service

語音對話服務，基於 [Pipecat AI](https://github.com/pipecat-ai/pipecat) 框架，透過 SmallWebRTC 提供瀏覽器端即時語音對話。

## 架構

```
Browser ←─ SmallWebRTC ─→ Pipecat Voice Service
                              │
                  ┌───────────┼───────────┐
                  ↓           ↓           ↓
             STT Service  LLM (API)  TTS Service
             (HTTP/Local) (openai_agent) (VibeVoice)
```

## 快速開始

### 1. 前置需求

- Python 3.11+
- 遠端 STT 服務 (`services/stt-service/`) 已啟動
- 遠端 TTS 服務 (`services/tts-service/`) 已啟動
- openai_agent LLM 服務已啟動

### 2. 安裝

```bash
cd services/pipecat_voice
pip install -r requirements.txt
cp .env.sample .env
# 編輯 .env 設定 STT/TTS/LLM 的 URL
```

### 3. 設定環境變數

```bash
# MacBook 開發（連遠端 GPU server）
export STT_BASE_URL=http://gpu-server:8100/v1
export TTS_BASE_URL=http://gpu-server:8200/v1
export LLM_BASE_URL=http://gpu-server:3082/v1

# 或 GPU server 上（本地 STT）
export STT_MODE=local
export TTS_BASE_URL=http://tts:8200/v1
export LLM_BASE_URL=http://aiassistant:3082/v1
```

### 4. 啟動

```bash
python server.py
# → http://localhost:8300
```

### 5. 測試

```bash
curl http://localhost:8300/health
```

## Browser Demo UI (Minimal)

This service now includes a minimal web UI route:

- `GET /` -> chat + TTS demo page
- `POST /api/chat` -> LLM proxy (`/v1/chat/completions` first, fallback to `/v1/completions`)
- `POST /api/tts` -> existing TTS proxy

### Run and demo (Windows PowerShell)

```powershell
cd services\pipecat_voice
python server.py
```

Then open:

- `http://localhost:8300/`

Type a message, click send, and the assistant response will be spoken via `<audio>` autoplay.

## 目錄結構

```
services/pipecat_voice/
├── server.py              # FastAPI + SmallWebRTC signaling
├── bot.py                 # Pipecat Pipeline (核心)
├── config.py              # 環境變數設定
├── whisper_http_stt.py    # 遠端 STT HTTP 介面
├── vibevoice_tts.py       # VibeVoice TTS HTTP 介面
├── llm_proxy.py           # LLM 代理（X-Voice-Mode header）
├── Dockerfile
├── requirements.txt
├── .env.sample
└── README.md
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/offer` | WebRTC SDP offer（建立語音連線） |
| `PATCH` | `/api/offer` | WebRTC ICE candidate trickle |
| `POST` | `/api/tts` | TTS 代理（優先走 WebSocket 串流，失敗時回退 HTTP） |
| `GET` | `/health` | 健康檢查 |

## STT 模式

| 模式 | 環境變數 | GPU | 說明 |
|------|---------|-----|------|
| **HTTP**（預設） | `STT_MODE=http` + `STT_BASE_URL` | 不需 | 呼叫遠端 STT 服務 |
| **Local** | `STT_MODE=local` | 需要 | Pipecat 內建 WhisperSTTService |

## Docker 部署

```bash
docker build -t labelnine_site/pipecat_voice .
docker run -d \
  -p 8300:8300 \
  -p 40000-40100:40000-40100/udp \
  -e STT_BASE_URL=http://stt:8100/v1 \
  -e TTS_BASE_URL=http://tts:8200/v1 \
  -e LLM_BASE_URL=http://aiassistant:3082/v1 \
  labelnine_site/pipecat_voice

# 或透過 docker-stack.yml 部署
```

## 環境變數一覽

| 變數 | 預設值 | 說明 |
|------|-------|------|
| `STT_MODE` | `http` | STT 模式（`http` 或 `local`） |
| `STT_BASE_URL` | `http://localhost:8100/v1` | 遠端 STT 服務 URL |
| `TTS_BASE_URL` | `http://localhost:8200/v1` | TTS URL |
| `TTS_STREAM_BASE_URL` | `` | Realtime WebSocket TTS URL（例：`http://localhost:8201/v1`） |
| `TTS_MODEL` | `kokoro` | TTS 模型（`kokoro` / `piper`） |
| `TTS_VOICE` | `zf_001` | TTS 聲線名稱（kokoro） |
| `TTS_SAMPLE_RATE` | `24000` | TTS 輸出取樣率 |
| `LLM_BASE_URL` | `http://localhost:3082/v1` | openai_agent LLM URL |
| `LLM_MODEL` | `default` | LLM 模型名稱 |
| `PIPECAT_HOST` | `0.0.0.0` | 服務監聽位址 |
| `PIPECAT_PORT` | `8300` | 服務監聽埠 |
| `PIPECAT_ICE_SERVERS` | `stun:stun.l.google.com:19302` | WebRTC ICE servers（STUN/TURN，逗號或空白分隔） |
| `VAD_STOP_SECS` | `0.5` | VAD 停頓判定秒數 |

## WebRTC 連線建議

- 若前端一直停在「連線中」，通常是 NAT/ICE 問題，不是 STT/TTS API 壞掉。
- 請至少設定一組 STUN（例如 `stun:stun.l.google.com:19302`）。
- 跨網段或企業網環境建議配置 TURN（可在 `PIPECAT_ICE_SERVERS` 用 JSON 物件帶 `username` / `credential`）。

## Dual-Channel Streaming (Text + Audio)

### Overview

This project supports two independent channels per user request:

```text
User input
  |-- Text Channel (WS): /api/chat/stream?request_id=<id>
  |     -> JSON events: delta/done/error (all carry request_id)
  |
  |-- Audio Channel (WS): /api/voice/stream?request_id=<id>
        -> binary audio chunks + JSON done/error (carry request_id)
```

Text and audio are intentionally separated so text rendering is never blocked by audio playback.

### request_id Rules

- Frontend generates `request_id` for every user message.
- Frontend sends the same `request_id` to both channels.
- `/api/chat/stream` includes `request_id` in every event.
- `/api/voice/stream` accepts `request_id` and returns it in done/error events.
- Frontend ignores events with mismatched request_id to avoid cross-request mixing.

### Local Test

1. Start services

```bash
# STT / TTS should be up first
# Then start pipecat_voice
cd services/pipecat_voice
python server.py
```

2. Open UI

- `http://localhost:8300/`

3. Input this sentence

- `你好，請用三句話介紹你自己`

4. Acceptance checks

- Text starts appending incrementally before final completion.
- Audio starts playing while text is still streaming.
- Both channels finish with the same request_id.

5. Stop / retry

- Click `停止` to close both text and audio streams.
- If one channel fails, the other continues.

6. Simulate audio WS failure (fallback check)

- Stop realtime TTS endpoint (8201) but keep HTTP TTS endpoint (8200).
- Send another prompt from UI.
- Expect text stream to continue.
- Audio channel reports error and UI falls back to `/api/tts` once full text is available.

## Runtime LLM Config + Latency Metric (Demo UI)

- Demo UI now supports per-request LLM configuration:
  - `LLM Base URL` (optional): forwarded to backend as `llm_base_url`
  - `LLM Model`: forwarded as `model`
- If `LLM Base URL` is empty, backend uses `LLM_BASE_URL` from env/config.

### Latency Metric in UI

The metric line now reports:

- `first text`: time when first assistant text delta is rendered
- `first audio`: estimated first audio playback start time
- `text->audio`: `first audio - first text` (requested latency metric)
- `bytes`: total audio bytes received on audio channel

Example:

`Streaming done: first text 520 ms, first audio 1180 ms, text->audio 660 ms, bytes 4130400`
