"""
server.py — Pipecat Voice Service FastAPI server

提供 SmallWebRTC signaling 端點 (POST/PATCH /api/offer)。
每個 WebRTC 連線由 bot.py 的 run_bot() 處理。
"""

import argparse
import asyncio
import contextlib
import io
import json
import logging
from pathlib import Path
import re
import struct
import sys
import time
import wave
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    IceServer,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot
from config import config

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipecat-voice-server")
_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
_LLM_TIMEOUT_SECS = 60.0
_VOICE_FIRST_MIN_CHARS = 16
_VOICE_FIRST_MAX_WAIT_SECS = 0.40
_VOICE_NEXT_MIN_CHARS = 50
_VOICE_NEXT_MAX_WAIT_SECS = 1.20
_VOICE_DEBOUNCE_SECS = 0.40
_VOICE_MIN_PUNCT_SEGMENT_CHARS = 10
_VOICE_HARD_PUNCTUATION = set("。！？!?")
_VOICE_WEAK_PUNCTUATION = set("，、：；,;:\n")

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Pipecat Voice Service",
    description="SmallWebRTC-based voice conversation service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── SmallWebRTC Request Handler ──────────────────────────
def _build_ice_servers(raw_value: str) -> list[IceServer]:
    raw = (raw_value or "").strip()
    if not raw:
        return []

    parsed_items: list[object] = []
    if raw.startswith("["):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                parsed_items = decoded
            else:
                logger.warning("PIPECAT_ICE_SERVERS JSON must be a list, got %s", type(decoded).__name__)
        except Exception as exc:
            logger.warning("PIPECAT_ICE_SERVERS JSON parse failed, fallback to split parsing: %s", exc)

    if not parsed_items:
        parsed_items = [item for item in re.split(r"[,\s]+", raw) if item]

    ice_servers: list[IceServer] = []
    for item in parsed_items:
        if isinstance(item, str):
            ice_servers.append(IceServer(urls=item))
            continue
        if isinstance(item, dict):
            urls = item.get("urls") or item.get("url")
            if not urls:
                logger.warning("Ignore invalid ICE server entry without urls/url: %s", item)
                continue
            ice_servers.append(
                IceServer(
                    urls=urls,
                    username=(item.get("username") or None),
                    credential=(item.get("credential") or None),
                )
            )
            continue
        logger.warning("Ignore unsupported ICE server entry type: %s", type(item).__name__)
    return ice_servers


_ICE_SERVERS = _build_ice_servers(config.PIPECAT_ICE_SERVERS)
request_handler = SmallWebRTCRequestHandler(ice_servers=(_ICE_SERVERS or None))
bot_tasks: set[asyncio.Task] = set()


class RequestTextBus:
    def __init__(self):
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.done = False
        self.created_at = time.time()


_REQUEST_TEXT_BUSES: dict[str, RequestTextBus] = {}


def _get_or_create_request_bus(request_id: str) -> RequestTextBus:
    bus = _REQUEST_TEXT_BUSES.get(request_id)
    if bus is None:
        bus = RequestTextBus()
        _REQUEST_TEXT_BUSES[request_id] = bus
    return bus


async def _request_bus_push_text(request_id: str, text_chunk: str) -> None:
    bus = _get_or_create_request_bus(request_id)
    if bus.done:
        return
    await bus.queue.put(text_chunk)


async def _request_bus_mark_done(request_id: str) -> None:
    bus = _get_or_create_request_bus(request_id)
    if bus.done:
        return
    bus.done = True
    await bus.queue.put(None)


def _request_bus_cleanup(request_id: str) -> None:
    _REQUEST_TEXT_BUSES.pop(request_id, None)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice name override")
    model: Optional[str] = Field(default=None, description="Model name override")
    language: Optional[str] = Field(default="Chinese", description="Language hint")
    speed: Optional[float] = Field(default=None, description="Speech speed override")


class ChatMessage(BaseModel):
    role: str = Field(..., description="chat role: system/user/assistant")
    content: str = Field(..., description="message text")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="chat history")
    model: Optional[str] = Field(default=None, description="model name override")
    llm_base_url: Optional[str] = Field(default=None, description="LLM base URL override")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key override")


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str


class STTTestFileRequest(BaseModel):
    path: str = Field(..., description="Absolute or workspace-relative path to a static audio file")
    model: str = Field(default="whisper-1", description="STT model name")
    language: Optional[str] = Field(default=None, description="Language hint")
    prompt: Optional[str] = Field(default=None, description="Optional initial prompt")
    response_format: str = Field(default="json", description="json/text/verbose_json")
    temperature: float = Field(default=0.0, description="Sampling temperature")


_TTS_MODEL_ALIASES = {
    "kokoro": "kokoro",
    "piper": "piper",
    "cosyvoice": "cosyvoice",
    "cosy": "cosyvoice",
}


def _normalize_tts_model(raw_model: Optional[str]) -> str:
    target = str(raw_model or config.TTS_MODEL or "kokoro").strip().lower()
    if not target:
        target = "kokoro"
    return _TTS_MODEL_ALIASES.get(target, target)


def _sample_rate_for_model(model: str) -> int:
    if model in {"piper", "cosyvoice"}:
        return 22050
    return config.TTS_SAMPLE_RATE


def _infer_stream_base_url() -> str:
    configured = (config.TTS_STREAM_BASE_URL or "").strip()
    if configured:
        return configured.rstrip("/")

    base = config.TTS_BASE_URL.rstrip("/")
    if ":8200" in base:
        return base.replace(":8200", ":8201", 1)
    if ":8202" in base:
        return base.replace(":8202", ":8201", 1)
    return base


def _build_ws_stream_url(
    text: str,
    voice: Optional[str],
    model: Optional[str],
    language: str,
    speed: Optional[float] = None,
    non_streaming_mode: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    stream_base = _infer_stream_base_url()
    stream_http_url = f"{stream_base}/audio/stream"
    if stream_http_url.startswith("https://"):
        stream_ws_url = "wss://" + stream_http_url[len("https://") :]
    elif stream_http_url.startswith("http://"):
        stream_ws_url = "ws://" + stream_http_url[len("http://") :]
    elif stream_http_url.startswith("wss://") or stream_http_url.startswith("ws://"):
        stream_ws_url = stream_http_url
    else:
        stream_ws_url = f"ws://{stream_http_url.lstrip('/')}"

    query_params = {
        "text": text,
        "language": language or "Chinese",
    }
    if model:
        query_params["model"] = model
    if voice:
        query_params["voice"] = voice
    if speed is not None and speed > 0:
        query_params["speed"] = f"{speed:.2f}".rstrip("0").rstrip(".")
    if non_streaming_mode is not None:
        query_params["non_streaming_mode"] = (
            "true" if non_streaming_mode else "false"
        )
    if max_new_tokens is not None and max_new_tokens > 0:
        query_params["max_new_tokens"] = str(max_new_tokens)

    query = urlencode(query_params)
    return f"{stream_ws_url}?{query}"


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()


def _wav_to_pcm16_bytes(wav_data: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise HTTPException(
            status_code=502,
            detail=f"TTS HTTP fallback returned unsupported sample width: {sample_width}",
        )
    if channels <= 1:
        return raw

    total_samples = len(raw) // 2
    samples = struct.unpack(f"<{total_samples}h", raw)
    mono: list[int] = []
    for i in range(0, total_samples, channels):
        frame = samples[i : i + channels]
        mono.append(int(sum(frame) / len(frame)))
    return struct.pack(f"<{len(mono)}h", *mono)


async def _fetch_tts_via_ws(
    text: str,
    voice: Optional[str],
    model: str,
    language: str,
    speed: float,
) -> bytes:
    ws_url = _build_ws_stream_url(
        text=text,
        voice=voice,
        model=model,
        language=language,
        speed=speed,
        non_streaming_mode=False,
        max_new_tokens=512,
    )
    headers = {}
    if config.TTS_API_KEY:
        headers["Authorization"] = f"Bearer {config.TTS_API_KEY}"

    logger.info("TTS websocket stream request -> %s", ws_url.split("?", 1)[0])
    timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT_SECS)
    pcm_data = bytearray()
    text_frames = 0

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=20,
                max_msg_size=0,
            ) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        pcm_data.extend(msg.data)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        text_frames += 1
                        raw = (msg.data or "").strip()
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            payload = {"error": raw}
                        err = str(payload.get("error") or payload.get("detail") or "").strip()
                        if err:
                            raise HTTPException(status_code=502, detail=f"TTS websocket error: {err}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise HTTPException(status_code=502, detail="TTS websocket closed with error")
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"TTS websocket timeout after {config.TTS_TIMEOUT_SECS:.1f}s",
        ) from exc
    except aiohttp.ClientError as exc:
        raise HTTPException(status_code=502, detail=f"TTS websocket unreachable: {exc}") from exc

    if not pcm_data:
        raise HTTPException(status_code=502, detail="TTS websocket returned empty audio stream")

    logger.info(
        "TTS websocket stream received chunks: bytes=%s text_frames=%s",
        len(pcm_data),
        text_frames,
    )
    return _pcm_to_wav(bytes(pcm_data), sample_rate=_sample_rate_for_model(model))


async def _relay_tts_ws_stream(
    client_ws: WebSocket,
    text: str,
    voice: Optional[str],
    model: str,
    language: str,
    speed: Optional[float] = None,
    non_streaming_mode: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
) -> tuple[int, int]:
    ws_url = _build_ws_stream_url(
        text=text,
        voice=voice,
        model=model,
        language=language,
        speed=speed,
        non_streaming_mode=non_streaming_mode,
        max_new_tokens=max_new_tokens,
    )
    headers = {}
    if config.TTS_API_KEY:
        headers["Authorization"] = f"Bearer {config.TTS_API_KEY}"

    timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT_SECS)
    total_bytes = 0
    chunk_count = 0
    start_ts = time.perf_counter()
    first_chunk_secs = -1.0
    logger.info(
        "TTS stream relay start -> %s (text_len=%s)",
        ws_url.split("?", 1)[0],
        len(text),
    )

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=20,
                max_msg_size=0,
            ) as upstream:
                async for msg in upstream:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        payload = bytes(msg.data)
                        if not payload:
                            continue
                        if first_chunk_secs < 0:
                            first_chunk_secs = time.perf_counter() - start_ts
                            logger.info(
                                "TTS stream relay first chunk: %.3fs (%s bytes)",
                                first_chunk_secs,
                                len(payload),
                            )
                        await client_ws.send_bytes(payload)
                        total_bytes += len(payload)
                        chunk_count += 1
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        raw = (msg.data or "").strip()
                        await client_ws.send_text(raw)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise HTTPException(
                            status_code=502, detail="TTS upstream websocket closed with error"
                        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"TTS upstream websocket timeout after {config.TTS_TIMEOUT_SECS:.1f}s",
        ) from exc
    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=502, detail=f"TTS upstream websocket unreachable: {exc}"
        ) from exc

    elapsed = time.perf_counter() - start_ts
    logger.info(
        "TTS stream relay done: chunks=%s bytes=%s elapsed=%.3fs first_chunk=%.3fs",
        chunk_count,
        total_bytes,
        elapsed,
        first_chunk_secs,
    )
    return chunk_count, total_bytes


async def _fetch_tts_via_http(
    text: str,
    voice: Optional[str],
    model: str,
    speed: float,
) -> tuple[bytes, str]:
    endpoint = f"{config.TTS_BASE_URL.rstrip('/')}/audio/speech"
    payload = {
        "model": model,
        "input": text,
        "speed": speed,
        "response_format": "wav",
    }
    if voice:
        payload["voice"] = voice
    headers = {"Content-Type": "application/json"}
    if config.TTS_API_KEY:
        headers["Authorization"] = f"Bearer {config.TTS_API_KEY}"

    timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT_SECS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as upstream:
                body = await upstream.read()
                if upstream.status >= 400:
                    detail = body.decode("utf-8", errors="ignore")[:500]
                    raise HTTPException(
                        status_code=upstream.status,
                        detail=detail or "TTS upstream error",
                    )
                media_type = upstream.headers.get("Content-Type", "audio/wav")
                return body, media_type
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"TTS upstream timeout after {config.TTS_TIMEOUT_SECS:.1f}s",
        ) from exc
    except aiohttp.ClientError as exc:
        raise HTTPException(status_code=502, detail=f"TTS upstream unreachable: {exc}") from exc


def _normalized_llm_base_url(override_base_url: Optional[str] = None) -> str:
    base = str(override_base_url or config.LLM_BASE_URL or "").strip().rstrip("/")
    if not base:
        return base
    if not re.match(r"^https?://", base):
        raise HTTPException(status_code=400, detail="llm_base_url must start with http:// or https://")
    if re.match(r"^https?://[^/]+$", base):
        return f"{base}/v1"
    return base


def _auth_headers(override_api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = (override_api_key if override_api_key is not None else config.LLM_API_KEY or "").strip()
    if token and token.lower() != "not-needed":
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for item in messages:
        role = (item.role or "user").strip().lower()
        content = (item.content or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _coerce_chat_messages(raw_messages: object) -> list[ChatMessage]:
    if not isinstance(raw_messages, list):
        return []
    result: list[ChatMessage] = []
    for item in raw_messages:
        if isinstance(item, ChatMessage):
            if str(item.content or "").strip():
                result.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        result.append(ChatMessage(role=role, content=content))
    return result


async def _llm_chat_completions(
    messages: list[ChatMessage],
    model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    endpoint = f"{_normalized_llm_base_url(llm_base_url)}/chat/completions"
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=_LLM_TIMEOUT_SECS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            endpoint,
            json=payload,
            headers=_auth_headers(llm_api_key),
        ) as resp:
            body = await resp.read()
            if resp.status >= 400:
                detail = body.decode("utf-8", errors="ignore")[:500]
                raise HTTPException(status_code=resp.status, detail=detail or "LLM upstream error")
            try:
                payload = json.loads(body.decode("utf-8", errors="ignore"))
                text = (
                    payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                text = str(text).strip()
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"invalid LLM response: {exc}") from exc
            if not text:
                raise HTTPException(status_code=502, detail="empty assistant reply from LLM")
            return text


async def _llm_completions(
    messages: list[ChatMessage],
    model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    endpoint = f"{_normalized_llm_base_url(llm_base_url)}/completions"
    payload = {
        "model": model,
        "prompt": _messages_to_prompt(messages),
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=_LLM_TIMEOUT_SECS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            endpoint,
            json=payload,
            headers=_auth_headers(llm_api_key),
        ) as resp:
            body = await resp.read()
            if resp.status >= 400:
                detail = body.decode("utf-8", errors="ignore")[:500]
                raise HTTPException(status_code=resp.status, detail=detail or "LLM upstream error")
            try:
                payload = json.loads(body.decode("utf-8", errors="ignore"))
                text = payload.get("choices", [{}])[0].get("text", "")
                text = str(text).strip()
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"invalid LLM response: {exc}") from exc
            if not text:
                raise HTTPException(status_code=502, detail="empty assistant reply from LLM")
            return text


def _extract_stream_delta_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    delta = first.get("delta") or {}
    content = delta.get("content")
    if content is None:
        content = first.get("text")
    if content is None:
        content = (first.get("message") or {}).get("content")
    return str(content or "")


async def _iter_llm_chat_completions_stream(
    messages: list[ChatMessage],
    model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    endpoint = f"{_normalized_llm_base_url(llm_base_url)}/chat/completions"
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "stream": True,
    }
    timeout = aiohttp.ClientTimeout(total=_LLM_TIMEOUT_SECS)
    saw_token = False

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            endpoint,
            json=payload,
            headers=_auth_headers(llm_api_key),
        ) as resp:
            if resp.status >= 400:
                body = await resp.read()
                detail = body.decode("utf-8", errors="ignore")[:500]
                raise HTTPException(status_code=resp.status, detail=detail or "LLM upstream error")

            pending = ""
            async for raw_chunk in resp.content.iter_chunked(2048):
                if not raw_chunk:
                    continue
                pending += raw_chunk.decode("utf-8", errors="ignore")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        return
                    try:
                        frame = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    token = _extract_stream_delta_text(frame)
                    if token:
                        saw_token = True
                        yield token

            tail = pending.strip()
            if tail.startswith("data:"):
                data = tail[5:].strip()
                if data and data != "[DONE]":
                    with contextlib.suppress(Exception):
                        frame = json.loads(data)
                        token = _extract_stream_delta_text(frame)
                        if token:
                            saw_token = True
                            yield token

    if not saw_token:
        raise HTTPException(status_code=502, detail="LLM stream returned no tokens")


def _split_text_chunks(text: str, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        current = "".join(buf)
        if ch in _VOICE_FLUSH_PUNCTUATION or len(current) >= chunk_chars:
            chunk = current.strip()
            if chunk:
                chunks.append(chunk)
            buf = []

    remain = "".join(buf).strip()
    if remain:
        chunks.append(remain)
    return chunks


def _safe_int(raw: Optional[str], default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(str(raw or "").strip())
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _safe_float(raw: Optional[str], default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(str(raw or "").strip())
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


async def _relay_tts_chunk_to_client(
    client_ws: WebSocket,
    text: str,
    voice: Optional[str],
    model: str,
    language: str,
    speed: float,
    session: Optional[aiohttp.ClientSession] = None,
) -> tuple[int, int, float]:
    ws_url = _build_ws_stream_url(
        text=text,
        voice=voice,
        model=model,
        language=language,
        speed=speed,
        non_streaming_mode=False,
        max_new_tokens=512,
    )
    headers = {}
    if config.TTS_API_KEY:
        headers["Authorization"] = f"Bearer {config.TTS_API_KEY}"

    timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT_SECS)
    total_bytes = 0
    chunk_count = 0
    start_ts = time.perf_counter()
    first_chunk_secs = -1.0

    owns_session = session is None
    active_session = session or aiohttp.ClientSession(timeout=timeout)

    try:
        async with active_session.ws_connect(
            ws_url,
            headers=headers,
            heartbeat=20,
            max_msg_size=0,
        ) as upstream:
            async for msg in upstream:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    payload = bytes(msg.data)
                    if not payload:
                        continue
                    if first_chunk_secs < 0:
                        first_chunk_secs = time.perf_counter() - start_ts
                    await client_ws.send_bytes(payload)
                    total_bytes += len(payload)
                    chunk_count += 1
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    raw = (msg.data or "").strip()
                    if raw:
                        await client_ws.send_text(raw)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise HTTPException(status_code=502, detail="TTS upstream websocket closed with error")
    finally:
        if owns_session:
            await active_session.close()

    if chunk_count <= 0:
        raise HTTPException(status_code=502, detail="TTS websocket returned no audio chunks")
    return chunk_count, total_bytes, first_chunk_secs


def segment_text_stream(
    *,
    buffer: str,
    is_first_segment: bool,
    trigger: str,
    now: float,
    last_token_ts: float,
    debounce_until: float,
    has_hard_punctuation: bool,
    has_weak_punctuation: bool,
) -> tuple[bool, str]:
    """
    Two-phase chunking policy:
    - First segment: prioritize fast start (min chars=16, max wait=400ms).
    - Following segments: larger chunks (min chars=50, max wait=1200ms).
    Debounce blocks flushes for 400ms after each send.
    """
    text = (buffer or "").strip()
    if not text:
        return False, "empty"

    if now < debounce_until:
        return False, "debounce"

    min_chars = _VOICE_FIRST_MIN_CHARS if is_first_segment else _VOICE_NEXT_MIN_CHARS
    max_wait = _VOICE_FIRST_MAX_WAIT_SECS if is_first_segment else _VOICE_NEXT_MAX_WAIT_SECS

    if has_hard_punctuation:
        if len(text) < _VOICE_MIN_PUNCT_SEGMENT_CHARS and trigger != "final":
            return False, "short_punctuation_hold"
        return True, "hard_punctuation"

    if len(text) >= min_chars:
        return True, "min_chars"

    waited = max(0.0, now - last_token_ts)
    if trigger in {"timeout", "final"} and waited >= max_wait:
        return True, "max_wait"

    if has_weak_punctuation:
        return False, "weak_punctuation_candidate"

    return False, "hold"


async def _webrtc_connection_callback(webrtc_connection):
    """收到新連線時，非阻塞啟動 bot pipeline。"""
    logger.info("WebRTC connection initialized, scheduling bot pipeline")
    task = asyncio.create_task(run_bot(webrtc_connection))
    bot_tasks.add(task)
    task.add_done_callback(bot_tasks.discard)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_path))


@app.post("/api/offer")
async def offer(request: Request):
    """WebRTC SDP offer — 建立新語音連線"""
    body = await request.json()
    if "sdp" not in body or "type" not in body:
        raise HTTPException(
            status_code=422,
            detail="Invalid offer payload: missing required fields 'sdp' and/or 'type'",
        )
    webrtc_request = SmallWebRTCRequest.from_dict(body)
    sdp_answer = await request_handler.handle_web_request(
        webrtc_request, _webrtc_connection_callback
    )
    return JSONResponse(content=sdp_answer)


@app.patch("/api/offer")
async def ice(request: Request):
    """WebRTC ICE candidate trickle"""
    body = await request.json()
    raw_candidates = body.get("candidates") or []
    candidates = []
    for item in raw_candidates:
        candidates.append(
            IceCandidate(
                candidate=item["candidate"],
                sdp_mid=item.get("sdpMid") or item.get("sdp_mid", ""),
                sdp_mline_index=item.get("sdpMLineIndex")
                if item.get("sdpMLineIndex") is not None
                else item.get("sdp_mline_index", 0),
            )
        )

    patch_request = SmallWebRTCPatchRequest(
        pc_id=body.get("pc_id") or body.get("pcId"),
        candidates=candidates,
    )
    await request_handler.handle_patch_request(patch_request)
    return JSONResponse(content={"status": "ok"})


@app.get("/health")
async def health():
    """健康檢查"""
    return {
        "status": "ok",
        "service": "pipecat-voice",
        "stt_mode": config.STT_MODE,
        "stt_url": config.STT_BASE_URL if config.STT_MODE == "http" else "local",
        "tts_url": config.TTS_BASE_URL,
        "tts_stream_url": _infer_stream_base_url(),
        "tts_model": _normalize_tts_model(config.TTS_MODEL),
        "tts_voice": config.TTS_VOICE,
        "tts_speed": config.TTS_SPEED,
        "llm_url": _normalized_llm_base_url(),
        "ice_servers": [str(server.urls) for server in _ICE_SERVERS],
    }


@app.get("/api/health")
async def api_health():
    """Backward-compatible health route for /api/voice/health through nginx rewrite."""
    return await health()


@app.post("/api/stt/transcriptions")
async def stt_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """
    Proxy browser-uploaded audio to existing STT service.
    Keep STT API key and upstream endpoint on server side.
    """
    base_url = (config.STT_BASE_URL or "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="STT_BASE_URL is not configured")

    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    debug_id = uuid4().hex[:12]
    started_at = time.perf_counter()

    logger.info(
        "STT proxy request[%s]: filename=%s content_type=%s bytes=%s model=%s language=%s response_format=%s endpoint=%s",
        debug_id,
        file.filename or "audio.webm",
        file.content_type or "application/octet-stream",
        len(audio_data),
        model or "whisper-1",
        language or "-",
        response_format or "json",
        f"{base_url}/audio/transcriptions",
    )

    endpoint = f"{base_url}/audio/transcriptions"
    headers: dict[str, str] = {"X-Request-Id": debug_id}
    if config.STT_API_KEY:
        headers["Authorization"] = f"Bearer {config.STT_API_KEY}"

    form = aiohttp.FormData()
    form.add_field(
        "file",
        audio_data,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "application/octet-stream",
    )
    form.add_field("model", model or "whisper-1")
    form.add_field("response_format", response_format or "json")
    form.add_field("temperature", str(temperature))
    if language:
        form.add_field("language", language)
    if prompt:
        form.add_field("prompt", prompt)

    timeout = aiohttp.ClientTimeout(total=45)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, data=form, headers=headers) as upstream:
                body = await upstream.read()
                preview = body.decode("utf-8", errors="ignore")[:200]
                logger.info(
                    "STT proxy response[%s]: status=%s content_type=%s latency_ms=%.1f body_preview=%s",
                    debug_id,
                    upstream.status,
                    upstream.headers.get("Content-Type", "application/octet-stream"),
                    (time.perf_counter() - started_at) * 1000.0,
                    preview,
                )
                if upstream.status >= 400:
                    detail = body.decode("utf-8", errors="ignore")[:800]
                    raise HTTPException(
                        status_code=upstream.status,
                        detail=detail or "STT upstream error",
                    )
                content_type = upstream.headers.get("Content-Type", "application/json")
                return Response(
                    content=body,
                    media_type=content_type,
                    headers={"X-STT-Debug-Id": debug_id},
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="STT upstream timeout")
    except aiohttp.ClientError as exc:
        raise HTTPException(status_code=502, detail=f"STT upstream unreachable: {exc}") from exc


@app.post("/api/stt/test-file")
async def stt_test_file(request: STTTestFileRequest):
    raw_path = str(request.path or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="path must not be empty")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (_BASE_DIR / raw_path).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"audio file not found: {candidate}")

    content_type = "audio/wav"
    suffix = candidate.suffix.lower()
    if suffix == ".webm":
        content_type = "audio/webm"
    elif suffix == ".ogg":
        content_type = "audio/ogg"
    elif suffix == ".mp3":
        content_type = "audio/mpeg"
    elif suffix in {".m4a", ".mp4"}:
        content_type = "audio/mp4"

    logger.info("STT static file test requested: path=%s size=%s", candidate, candidate.stat().st_size)
    with candidate.open("rb") as fh:
        upload = UploadFile(filename=candidate.name, file=fh, headers={"content-type": content_type})
        response = await stt_transcriptions(
            file=upload,
            model=request.model,
            language=request.language,
            prompt=request.prompt,
            response_format=request.response_format,
            temperature=request.temperature,
        )

    body_preview = ""
    if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
        body_preview = bytes(response.body).decode("utf-8", errors="ignore")[:200]
    logger.info("STT static file test finished: path=%s preview=%s", candidate, body_preview)
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [m for m in request.messages if str(m.content or "").strip()]
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    model = (request.model or config.LLM_MODEL or "default").strip()
    llm_base_url = (request.llm_base_url or "").strip() or None
    llm_api_key = request.llm_api_key
    try:
        reply = await _llm_chat_completions(
            messages,
            model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        return ChatResponse(reply=reply, model=model, provider="chat.completions")
    except HTTPException as chat_exc:
        logger.warning("chat/completions failed, fallback to /completions: %s", chat_exc.detail)
        reply = await _llm_completions(
            messages,
            model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        return ChatResponse(reply=reply, model=model, provider="completions")


@app.websocket("/api/chat/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    request_id = (websocket.query_params.get("request_id") or "").strip() or uuid4().hex
    bus_ready = False
    done_marked = False
    try:
        incoming = await websocket.receive_text()
        payload = json.loads(incoming)
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    try:
        request_id = str(payload.get("request_id") or request_id).strip() or request_id
        logger.info(
            "Chat stream init: request_id=%s model=%s message_count=%s llm_base_url=%s",
            request_id,
            payload.get("model") or config.LLM_MODEL or "default",
            len(payload.get("messages") or []),
            payload.get("llm_base_url") or "",
        )
        _get_or_create_request_bus(request_id)
        bus_ready = True
        model = str(payload.get("model") or config.LLM_MODEL or "default").strip()
        llm_base_url = str(payload.get("llm_base_url") or "").strip() or None
        llm_api_key = payload.get("llm_api_key")
        messages = _coerce_chat_messages(payload.get("messages"))
        if not messages:
            await websocket.send_json(
                {"event": "error", "request_id": request_id, "error": "messages must not be empty"}
            )
            await websocket.close(code=1008)
            return

        full_text = ""
        provider = "chat.completions.stream"
        stream_failed_exc: Optional[Exception] = None
        try:
            async for token in _iter_llm_chat_completions_stream(
                messages,
                model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
            ):
                full_text += token
                await _request_bus_push_text(request_id, token)
                logger.info("[SYNC][%s] UI_CHUNK: %s", request_id, token[:80].replace("\n", " "))
                await websocket.send_json(
                    {"event": "delta", "request_id": request_id, "delta": token}
                )
        except Exception as exc:
            stream_failed_exc = exc
            logger.warning("chat stream failed, fallback to non-stream: %s", exc)

        if stream_failed_exc is not None:
            try:
                full_text = await _llm_chat_completions(
                    messages,
                    model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                )
                provider = "chat.completions"
            except HTTPException as chat_exc:
                logger.warning(
                    "chat stream fallback chat/completions failed, fallback /completions: %s",
                    chat_exc.detail,
                )
                full_text = await _llm_completions(
                    messages,
                    model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                )
                provider = "completions"

            # Keep UX streaming-like when LLM stream is unavailable.
            for ch in full_text:
                await _request_bus_push_text(request_id, ch)
                logger.info("[SYNC][%s] UI_CHUNK: %s", request_id, ch[:80].replace("\n", " "))
                await websocket.send_json(
                    {"event": "delta", "request_id": request_id, "delta": ch}
                )

        await websocket.send_json(
            {
                "event": "done",
                "request_id": request_id,
                "text": full_text,
                "model": model,
                "provider": provider,
            }
        )
        await _request_bus_mark_done(request_id)
        done_marked = True
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logger.info("Chat stream client disconnected (request_id=%s)", request_id)
    except HTTPException as exc:
        detail = str(exc.detail or "chat stream failed")
        if bus_ready and not done_marked:
            await _request_bus_mark_done(request_id)
            done_marked = True
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"event": "error", "request_id": request_id, "error": detail}
            )
            await websocket.close(code=1011)
    except Exception as exc:
        logger.error("Chat stream unexpected error: %s", exc, exc_info=True)
        if bus_ready and not done_marked:
            await _request_bus_mark_done(request_id)
            done_marked = True
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"event": "error", "request_id": request_id, "error": str(exc)}
            )
            await websocket.close(code=1011)
    finally:
        if bus_ready and not done_marked:
            with contextlib.suppress(Exception):
                await _request_bus_mark_done(request_id)


@app.post("/api/tts")
async def tts(request: TTSRequest):
    """Proxy TTS request to remote service configured in stack.env."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    model = _normalize_tts_model(request.model)
    explicit_voice = (request.voice or "").strip() or None
    default_voice = (config.TTS_VOICE or "").strip() or None
    # Piper only needs model selection; avoid forcing Kokoro voice into Piper requests.
    voice = explicit_voice if explicit_voice is not None else (
        None if model == "piper" else default_voice
    )
    language = request.language or "Chinese"
    speed = request.speed if request.speed is not None and request.speed > 0 else config.TTS_SPEED
    ws_error = ""

    # WebSocket synthesis path is tuned for Kokoro (24k). Other models use HTTP to keep WAV headers accurate.
    if model == "kokoro":
        try:
            wav_data = await _fetch_tts_via_ws(
                text=text,
                voice=voice,
                model=model,
                language=language,
                speed=speed,
            )
            return Response(
                content=wav_data,
                media_type="audio/wav",
                headers={"X-TTS-Transport": "websocket"},
            )
        except HTTPException as exc:
            ws_error = str(exc.detail or "").strip()
            logger.warning("TTS websocket path failed, fallback to HTTP: %s", ws_error)
    else:
        ws_error = f"skip websocket for model={model}"

    body, media_type = await _fetch_tts_via_http(
        text=text,
        voice=voice,
        model=model,
        speed=speed,
    )
    headers = {"X-TTS-Transport": "http-fallback"}
    if ws_error:
        headers["X-TTS-WS-Error"] = ws_error[:180].replace("\n", " ")
    return Response(content=body, media_type=media_type, headers=headers)


@app.websocket("/api/tts/stream")
async def tts_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        params = websocket.query_params
        text = (params.get("text") or "").strip()
        if not text:
            await websocket.send_json({"error": "text must not be empty"})
            await websocket.close(code=1008)
            return
        voice = (
            (params.get("voice") or params.get("speaker_name") or "").strip() or None
        )
        model = _normalize_tts_model(params.get("model"))
        if voice is None and model != "piper":
            voice = (config.TTS_VOICE or "").strip() or None
        language = (params.get("language") or "Chinese").strip() or "Chinese"
        speed_raw = (params.get("speed") or "").strip()
        speed: float = config.TTS_SPEED
        if speed_raw:
            try:
                parsed_speed = float(speed_raw)
                if parsed_speed > 0:
                    speed = parsed_speed
            except ValueError:
                speed = config.TTS_SPEED
        non_streaming_mode_raw = (params.get("non_streaming_mode") or "").strip().lower()
        non_streaming_mode: Optional[bool] = None
        if non_streaming_mode_raw in {"1", "true", "yes", "on"}:
            non_streaming_mode = True
        elif non_streaming_mode_raw in {"0", "false", "no", "off"}:
            non_streaming_mode = False

        max_new_tokens: Optional[int] = None
        max_new_tokens_raw = (params.get("max_new_tokens") or "").strip()
        if max_new_tokens_raw:
            try:
                parsed_tokens = int(max_new_tokens_raw)
                if parsed_tokens > 0:
                    max_new_tokens = parsed_tokens
            except ValueError:
                max_new_tokens = None

        await _relay_tts_ws_stream(
            client_ws=websocket,
            text=text,
            voice=voice,
            model=model,
            language=language,
            speed=speed,
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
        )
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logger.info("TTS client websocket disconnected")
    except HTTPException as exc:
        detail = str(exc.detail or "TTS stream relay failed")
        logger.warning("TTS stream relay failed: %s", detail)
        with contextlib.suppress(Exception):
            await websocket.send_json({"error": detail})
            await websocket.close(code=1011)
    except Exception as exc:
        logger.error("TTS stream relay unexpected error: %s", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.send_json({"error": f"TTS stream relay error: {exc}"})
            await websocket.close(code=1011)


@app.websocket("/api/voice/stream")
async def voice_stream(websocket: WebSocket):
    tts_session: Optional[aiohttp.ClientSession] = None
    request_id = ""
    try:
        await websocket.accept()
        logger.info("[VOICE] ws accepted")
        req_started = time.perf_counter()
        logger.info("[VOICE] t0=%.3f", req_started)

        async def reject_validation(detail: str, code: int = 1008):
            await websocket.send_json({"error": detail, "request_id": request_id})
            await websocket.close(code=code)

        params = websocket.query_params
        request_id = (params.get("request_id") or "").strip()
        tts_model = _normalize_tts_model(params.get("tts_model") or config.TTS_MODEL)
        language = (params.get("language") or "Chinese").strip() or "Chinese"
        speed = _safe_float(params.get("speed"), config.TTS_SPEED, 0.2, 3.0)

        voice = ((params.get("voice") or params.get("speaker_name") or "").strip() or None)
        if voice is None and tts_model != "piper":
            voice = (config.TTS_VOICE or "").strip() or None

        if not request_id:
            incoming_text = await websocket.receive_text()
            try:
                payload = json.loads(incoming_text)
                if isinstance(payload, dict):
                    tts_model = _normalize_tts_model(str(payload.get("tts_model") or tts_model))
                    request_id = str(payload.get("request_id") or request_id).strip()
                    logger.info(
                        "[VOICE] init payload request_id=%s model=%s tts_model=%s voice=%s",
                        request_id,
                        payload.get("model") or config.LLM_MODEL or "default",
                        tts_model,
                        payload.get("voice") or voice or "",
                    )
            except Exception:
                pass

        if not request_id:
            await reject_validation("request_id must not be empty", code=1008)
            return
        logger.info("[VOICE] request_id=%s", request_id)
        bus = _get_or_create_request_bus(request_id)

        first_flush_latency = -1.0
        tts_first_audio_latency = -1.0
        total_audio_bytes = 0
        total_audio_chunks = 0
        total_tts_calls = 0
        first_tts_sent_ts = -1.0
        first_segment_sent = False
        first_segment_cut_ts = -1.0
        first_audio_sent_ts = -1.0
        debounce_until = 0.0
        last_token_ts = time.perf_counter()
        has_hard_punctuation = False
        has_weak_punctuation = False
        first_text_chunk_latency = -1.0

        tts_timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT_SECS)
        tts_session = aiohttp.ClientSession(timeout=tts_timeout)

        async def flush_text_chunk(chunk: str, reason: str):
            nonlocal first_flush_latency
            nonlocal tts_first_audio_latency
            nonlocal total_audio_bytes
            nonlocal total_audio_chunks
            nonlocal total_tts_calls
            nonlocal first_tts_sent_ts
            nonlocal first_segment_sent
            nonlocal first_segment_cut_ts
            nonlocal first_audio_sent_ts
            nonlocal debounce_until

            cleaned = chunk.strip()
            if not cleaned:
                return
            logger.info("[SYNC][%s] TTS_CHUNK: %s", request_id, cleaned[:80].replace("\n", " "))

            if first_flush_latency < 0:
                first_flush_latency = time.perf_counter() - req_started
                first_segment_cut_ts = time.perf_counter()
                logger.info("voice stream first text flush latency: %.3fs", first_flush_latency)
                logger.info(
                    "[VOICE] t_first_segment=%.3f delta_ms=%.1f",
                    first_segment_cut_ts,
                    (first_segment_cut_ts - req_started) * 1000.0,
                )

            if first_tts_sent_ts < 0:
                first_tts_sent_ts = time.perf_counter()

            total_tts_calls += 1
            logger.info("[VOICE] tts send chars=%s reason=%s", len(cleaned), reason)
            try:
                chunk_count, byte_count, first_chunk_secs = await _relay_tts_chunk_to_client(
                    client_ws=websocket,
                    text=cleaned,
                    voice=voice,
                    model=tts_model,
                    language=language,
                    speed=speed,
                    session=tts_session,
                )
            except HTTPException as ws_exc:
                logger.warning("[VOICE] TTS websocket failed, fallback HTTP: %s", ws_exc.detail)
                wav_body, _ = await _fetch_tts_via_http(
                    text=cleaned,
                    voice=voice,
                    model=tts_model,
                    speed=speed,
                )
                pcm_body = _wav_to_pcm16_bytes(wav_body)
                if not pcm_body:
                    raise HTTPException(status_code=502, detail="TTS HTTP fallback returned empty audio")
                await websocket.send_bytes(pcm_body)
                chunk_count = 1
                byte_count = len(pcm_body)
                first_chunk_secs = time.perf_counter() - first_tts_sent_ts
            total_audio_chunks += chunk_count
            total_audio_bytes += byte_count
            first_segment_sent = True
            debounce_until = time.perf_counter() + _VOICE_DEBOUNCE_SECS

            if tts_first_audio_latency < 0 and first_chunk_secs >= 0 and first_tts_sent_ts >= 0:
                tts_first_audio_latency = first_chunk_secs
                logger.info("voice stream TTFA (first TTS audio chunk): %.3fs", tts_first_audio_latency)
            if first_audio_sent_ts < 0:
                first_audio_sent_ts = first_tts_sent_ts + max(first_chunk_secs, 0.0)
                logger.info(
                    "[VOICE] t_first_audio_sent=%.3f estimated_TTFA_ms=%.1f",
                    first_audio_sent_ts,
                    (first_audio_sent_ts - req_started) * 1000.0,
                )

        buffer = ""
        chat_done = False
        while True:
            chunk = await asyncio.wait_for(bus.queue.get(), timeout=_LLM_TIMEOUT_SECS)
            if chunk is None:
                chat_done = True
                break

            if first_text_chunk_latency < 0:
                first_text_chunk_latency = time.perf_counter() - req_started
                logger.info(
                    "voice stream first text chunk latency (from chat queue): %.3fs",
                    first_text_chunk_latency,
                )

            buffer += chunk
            last_token_ts = time.perf_counter()
            if any(ch in _VOICE_HARD_PUNCTUATION for ch in chunk):
                has_hard_punctuation = True
            if any(ch in _VOICE_WEAK_PUNCTUATION for ch in chunk):
                has_weak_punctuation = True

            should_flush, reason = segment_text_stream(
                buffer=buffer,
                is_first_segment=(not first_segment_sent),
                trigger="token",
                now=last_token_ts,
                last_token_ts=last_token_ts,
                debounce_until=debounce_until,
                has_hard_punctuation=has_hard_punctuation,
                has_weak_punctuation=has_weak_punctuation,
            )
            if should_flush:
                await flush_text_chunk(buffer, reason=reason)
                buffer = ""
                has_hard_punctuation = False
                has_weak_punctuation = False
            elif reason == "debounce":
                logger.info("[VOICE] debounce triggered; keep accumulating")

        if chat_done and buffer.strip():
            now_ts = time.perf_counter()
            should_flush, reason = segment_text_stream(
                buffer=buffer,
                is_first_segment=(not first_segment_sent),
                trigger="final",
                now=now_ts,
                last_token_ts=last_token_ts,
                debounce_until=debounce_until,
                has_hard_punctuation=has_hard_punctuation,
                has_weak_punctuation=has_weak_punctuation,
            )
            if not should_flush and reason == "debounce":
                logger.info("[VOICE] final flush waiting for debounce")
                await asyncio.sleep(max(0.0, debounce_until - time.perf_counter()))
                reason = "final_after_debounce"
            await flush_text_chunk(buffer, reason=reason if should_flush else "final")

        total_secs = time.perf_counter() - req_started
        logger.info(
            "voice stream done: first_queue_text=%.3fs first_flush=%.3fs ttfa=%.3fs total=%.3fs tts_calls=%s audio_chunks=%s audio_bytes=%s",
            first_text_chunk_latency,
            first_flush_latency,
            tts_first_audio_latency,
            total_secs,
            total_tts_calls,
            total_audio_chunks,
            total_audio_bytes,
        )
        await websocket.send_json({"event": "done", "request_id": request_id})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logger.info("Voice stream client disconnected")
    except HTTPException as exc:
        detail = str(exc.detail or "voice stream failed")
        logger.warning("Voice stream failed: %s", detail)
        with contextlib.suppress(Exception):
            await websocket.send_json({"error": detail, "request_id": request_id})
            await websocket.close(code=1008)
    except Exception as exc:
        logger.error("Voice stream unexpected error: %s", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"error": f"voice stream error: {exc}", "request_id": request_id}
            )
            await websocket.close(code=1011)
    finally:
        if tts_session and not tts_session.closed:
            with contextlib.suppress(Exception):
                await tts_session.close()
        if request_id:
            _request_bus_cleanup(request_id)


@app.websocket("/ws-test")
async def ws_test(ws: WebSocket):
    await ws.accept()
    await ws.send_text("ok")
    await ws.close()


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("Pipecat Voice Service starting")
    logger.info(f"  STT mode:  {config.STT_MODE}")
    if config.STT_MODE != "http":
        logger.info(f"  STT model: {config.WHISPER_MODEL} ({config.WHISPER_DEVICE})")
    logger.info(f"  STT BASE URL:       {config.STT_BASE_URL}")
    logger.info(f"  TTS BASE URL:       {config.TTS_BASE_URL}")
    logger.info(f"  TTS STREAM BASE URL:{_infer_stream_base_url()}")
    logger.info(f"  TTS model: {_normalize_tts_model(config.TTS_MODEL)}")
    logger.info(f"  TTS voice: {config.TTS_VOICE}")
    logger.info(f"  LLM BASE URL:       {_normalized_llm_base_url()}")
    if _ICE_SERVERS:
        logger.info("  ICE:       %s", ", ".join(str(server.urls) for server in _ICE_SERVERS))
    else:
        logger.warning("  ICE:       none configured (WebRTC may fail across NAT)")
    logger.info(f"  Listening:  {config.HOST}:{config.PORT}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    await request_handler.close()


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Pipecat Voice Service")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
