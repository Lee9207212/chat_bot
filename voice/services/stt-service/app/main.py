"""
STT Service — OpenAI Whisper 相容 API

提供 POST /v1/audio/transcriptions 端點，
使用 faster-whisper 進行語音辨識。

用法：
    uvicorn main:app --host 0.0.0.0 --port 8100
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
import traceback

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from config import settings
from stt_engine import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("stt-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入模型"""
    logger.info("=== STT Service starting ===")
    engine.load_model()
    logger.info(f"=== STT Service ready on port {settings.PORT} ===")
    yield
    logger.info("=== STT Service shutting down ===")


app = FastAPI(
    title="STT Service",
    description="OpenAI Whisper 相容 API，基於 faster-whisper",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(authorization: Optional[str] = None):
    """驗證 API Key（可選）"""
    if not settings.API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    if token != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ─── Health ───────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.is_loaded else "loading",
        "model": engine.model_name,
        "model_path": settings.MODEL_PATH,
        "device": settings.DEVICE,
        "compute_type": settings.COMPUTE_TYPE,
        "warmed_up": engine.is_warmed_up,
        "opencc_enabled": engine.opencc_enabled,
        "opencc_config": engine.opencc_config,
    }


# ─── OpenAI-compatible: GET /v1/models ────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name,
                "object": "model",
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


# ─── OpenAI-compatible: POST /v1/audio/transcriptions ─────

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorization: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    """
    OpenAI Whisper 相容端點。

    支援格式：WAV, MP3, FLAC, OGG, M4A, WEBM
    回傳格式：json, text, verbose_json
    """
    _check_api_key(authorization)

    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading")

    # 讀取上傳的音檔
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    logger.info(
        f"Transcribe request[{x_request_id or '-'}]: file={file.filename}, "
        f"content_type={file.content_type}, "
        f"size={len(audio_data)} bytes, "
        f"language={language}, format={response_format}"
    )

    try:
        result = engine.transcribe(
            audio_data=audio_data,
            filename=file.filename,
            content_type=file.content_type,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            request_id=x_request_id,
        )
    except Exception as e:
        logger.error("Transcription traceback:\n%s", traceback.format_exc())
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # 根據 response_format 回傳
    if response_format == "text":
        return PlainTextResponse(result["text"])

    return JSONResponse(result)


# ─── 直接執行 ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )
