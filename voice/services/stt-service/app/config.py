"""STT Service 設定"""

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """STT 服務設定，可透過環境變數覆蓋"""

    # faster-whisper 模型設定
    MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "large-v3-turbo")
    DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "zh")

    # 伺服器設定
    HOST: str = os.getenv("STT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("STT_PORT", "8100"))

    # API Key（可選）
    API_KEY: str = os.getenv("STT_API_KEY", "")

    # VAD 設定
    VAD_FILTER: bool = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"

    # 模型下載目錄（HuggingFace cache）
    HF_HOME: str = os.getenv("HF_HOME", "/app/models")

    # 啟動預熱（降低首請求延遲）
    STARTUP_WARMUP: bool = _env_bool("STT_STARTUP_WARMUP", True)
    WARMUP_SECONDS: float = float(os.getenv("STT_WARMUP_SECONDS", "0.8"))
    WARMUP_LANGUAGE: str = os.getenv("STT_WARMUP_LANGUAGE", "zh")

    # 中文輸出字形轉換（OpenCC）
    OPENCC_CONFIG: str = os.getenv("STT_OPENCC_CONFIG", "s2twp")
    OPENCC_ONLY_ZH: bool = _env_bool("STT_OPENCC_ONLY_ZH", True)


settings = Settings()
