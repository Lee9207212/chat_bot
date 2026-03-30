"""Pipecat Voice Service 設定"""

import os


def _rewrite_localhost_for_docker(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return value

    mapping = {
        "http://localhost": "http://host.docker.internal",
        "https://localhost": "https://host.docker.internal",
        "ws://localhost": "ws://host.docker.internal",
        "wss://localhost": "wss://host.docker.internal",
        "http://127.0.0.1": "http://host.docker.internal",
        "https://127.0.0.1": "https://host.docker.internal",
        "ws://127.0.0.1": "ws://host.docker.internal",
        "wss://127.0.0.1": "wss://host.docker.internal",
    }
    for src, dst in mapping.items():
        if value.startswith(src):
            return dst + value[len(src) :]
    return value


class Config:
    """語音服務設定，透過環境變數設定"""

    # ─── STT ──────────────────────────────────────────────
    # "local" = 用 Pipecat 內建 WhisperSTTService（需 GPU）
    # "http"  = 呼叫遠端 STT HTTP API（預設，不需 GPU）
    STT_MODE: str = os.getenv("STT_MODE", "http")
    STT_BASE_URL: str = _rewrite_localhost_for_docker(
        os.getenv("STT_BASE_URL", "http://localhost:8100/v1")
    )
    STT_API_KEY: str = os.getenv("STT_API_KEY", "")
    STT_SAMPLE_RATE: int = int(os.getenv("STT_SAMPLE_RATE", "16000"))
    STT_PROMPT: str = os.getenv("STT_PROMPT", "")
    STT_MIN_AUDIO_SECS: float = float(os.getenv("STT_MIN_AUDIO_SECS", "0.35"))
    STT_MIN_AUDIO_RMS: float = float(os.getenv("STT_MIN_AUDIO_RMS", "160"))

    # local 模式的 whisper 設定
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "zh")

    # ─── TTS ──────────────────────────────────────────────
    TTS_BASE_URL: str = _rewrite_localhost_for_docker(
        os.getenv("TTS_BASE_URL", "http://localhost:8200/v1")
    )
    # Realtime WebSocket TTS endpoint (optional). Example: http://host:8201/v1
    # If empty, server will try to infer from TTS_BASE_URL.
    TTS_STREAM_BASE_URL: str = _rewrite_localhost_for_docker(
        os.getenv("TTS_STREAM_BASE_URL", "")
    )
    TTS_API_KEY: str = os.getenv("TTS_API_KEY", "")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "kokoro")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "zf_001")
    TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    TTS_TIMEOUT_SECS: float = float(os.getenv("TTS_TIMEOUT_SECS", "45"))

    # ─── LLM ─────────────────────────────────────────────
    LLM_BASE_URL: str = _rewrite_localhost_for_docker(
        os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
    )
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "default")

    # ─── Voice Service ────────────────────────────────────
    HOST: str = os.getenv("PIPECAT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PIPECAT_PORT", "8300"))
    # Comma/space/newline separated ICE server URLs.
    # Example: "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
    PIPECAT_ICE_SERVERS: str = os.getenv(
        "PIPECAT_ICE_SERVERS",
        "stun:stun.l.google.com:19302",
    )

    # ─── VAD ──────────────────────────────────────────────
    VAD_STOP_SECS: float = float(os.getenv("VAD_STOP_SECS", "0.5"))
    ALLOW_INTERRUPTIONS: bool = (
        os.getenv("ALLOW_INTERRUPTIONS", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    # ─── System Prompt ────────────────────────────────────
    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一個友善的 AI 語音助手。用繁體中文回答，保持簡潔。"
        "使用工具前，先簡短告知使用者你正在做什麼。",
    )


config = Config()
