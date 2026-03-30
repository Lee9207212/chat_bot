"""
WhisperHTTPSTTService — 透過 HTTP 呼叫遠端 STT 服務

用於無 GPU 環境（如 MacBook），呼叫遠端 services/stt-service 的
POST /v1/audio/transcriptions 端點。

Pipecat STT Service 介面實作。
"""

import io
import logging
import re
import wave
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import AsyncGenerator

import aiohttp
import numpy as np

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService

logger = logging.getLogger("whisper-http-stt")


class WhisperHTTPSTTService(SegmentedSTTService):
    """透過 HTTP API 呼叫遠端 faster-whisper STT 服務"""

    def __init__(
        self,
        base_url: str = "http://localhost:8100/v1",
        api_key: str = "",
        language: str = "zh",
        model: str = "whisper-1",
        sample_rate: int = 16000,
        prompt: str = "",
        min_audio_secs: float = 0.35,
        min_audio_rms: float = 160.0,
        **kwargs,
    ):
        # 關閉 audio passthrough，避免原始音訊 frame 干擾下游 aggregator。
        super().__init__(sample_rate=sample_rate, audio_passthrough=False, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._language = language
        self._model = model
        self._sample_rate = sample_rate
        self._prompt = prompt
        self._prompt_normalized = self._normalize_text(prompt)
        self._min_audio_secs = min_audio_secs
        self._min_audio_rms = min_audio_rms
        self._session: aiohttp.ClientSession | None = None
        self._last_text: str = ""
        self._last_text_normalized: str = ""
        self._last_text_ts: datetime | None = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 保留中英數，移除空白與標點，讓相似比對更穩定。
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text or "").lower().strip()

    def _looks_like_prompt_echo(self, normalized_text: str) -> bool:
        if not normalized_text or not self._prompt_normalized:
            return False
        # 僅在 prompt 足夠長時啟用，避免短詞誤判。
        if len(self._prompt_normalized) < 8:
            return False
        if normalized_text in self._prompt_normalized:
            return True
        prefix_len = min(max(8, len(self._prompt_normalized) // 2), 14)
        if len(normalized_text) >= prefix_len and normalized_text.startswith(
            self._prompt_normalized[:prefix_len]
        ):
            return True
        if self._ngram_overlap(normalized_text, self._prompt_normalized, n=3) >= 0.72:
            return True
        similarity = SequenceMatcher(
            None, normalized_text, self._prompt_normalized
        ).ratio()
        return similarity >= 0.84

    @staticmethod
    def _ngram_overlap(a: str, b: str, n: int = 3) -> float:
        if not a or not b:
            return 0.0
        if len(a) < n or len(b) < n:
            return SequenceMatcher(None, a, b).ratio()
        a_grams = {a[i : i + n] for i in range(len(a) - n + 1)}
        b_grams = {b[i : i + n] for i in range(len(b) - n + 1)}
        if not b_grams:
            return 0.0
        return len(a_grams & b_grams) / len(b_grams)

    def _looks_like_recent_duplicate(self, normalized_text: str, now: datetime) -> bool:
        if (
            not self._last_text_ts
            or not self._last_text_normalized
            or (now - self._last_text_ts).total_seconds() > 2.0
        ):
            return False
        if normalized_text == self._last_text_normalized:
            return True
        similarity = SequenceMatcher(
            None, normalized_text, self._last_text_normalized
        ).ratio()
        return similarity >= 0.9

    def _prepare_wav(self, audio: bytes) -> tuple[io.BytesIO, float, float]:
        # SegmentedSTTService 預設會傳入完整 WAV bytes；保留 raw PCM fallback 以兼容。
        if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            wav_bytes = audio
        else:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self._sample_rate)
                wf.writeframes(audio)
            wav_bytes = wav_buffer.getvalue()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            frame_rate = wf.getframerate() or self._sample_rate
            duration = wf.getnframes() / float(frame_rate)
            if frames:
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(np.square(samples))))
            else:
                rms = 0.0

        return io.BytesIO(wav_bytes), duration, rms

    async def start(self, frame: Frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        frame_sr = getattr(frame, "audio_in_sample_rate", None)
        logger.info(
            "STT service started (frame_sr=%s, stt_sample_rate=%s, min_secs=%.2f, min_rms=%.1f)",
            frame_sr,
            self.sample_rate,
            self._min_audio_secs,
            self._min_audio_rms,
        )

    async def stop(self, frame: EndFrame):
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        接收音訊片段（預設為 WAV bytes），呼叫遠端 STT API，回傳 TranscriptionFrame。
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            wav_buffer, duration, rms = self._prepare_wav(audio)
            if duration < self._min_audio_secs:
                logger.debug(
                    "Short segment skipped: %.3fs < %.3fs",
                    duration,
                    self._min_audio_secs,
                )
                return
            if rms < self._min_audio_rms:
                logger.debug(
                    "Low-energy segment skipped: rms=%.1f < %.1f (duration=%.2fs)",
                    rms,
                    self._min_audio_rms,
                    duration,
                )
                return

            wav_buffer.seek(0)

            # 準備 multipart form data
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer,
                filename="audio.wav",
                content_type="audio/wav",
            )
            form.add_field("model", self._model)
            form.add_field("language", self._language)
            form.add_field("response_format", "json")
            if self._prompt:
                form.add_field("prompt", self._prompt)

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            url = f"{self._base_url}/audio/transcriptions"
            logger.debug(
                "Sending segment to STT: bytes=%d duration=%.2fs rms=%.1f url=%s",
                len(audio),
                duration,
                rms,
                url,
            )

            async with self._session.post(
                url, data=form, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"STT API error {resp.status}: {error_text}")
                    yield ErrorFrame(f"STT API error: {resp.status}")
                    return

                result = await resp.json()
                text = result.get("text", "").strip()

                if text:
                    now = datetime.now(timezone.utc)
                    normalized_text = self._normalize_text(text)
                    if self._looks_like_prompt_echo(normalized_text):
                        logger.debug(
                            "Prompt-like transcription skipped: '%s'",
                            text,
                        )
                        return

                    # 短時間內重複（或高度相似）的句子不再送出，避免 context 污染。
                    if self._looks_like_recent_duplicate(normalized_text, now):
                        logger.debug(f"Duplicate transcription skipped: '{text}'")
                        return

                    logger.info(f"Transcription: '{text}'")
                    yield TranscriptionFrame(
                        text=text,
                        user_id="user",
                        timestamp=now.isoformat(),
                    )
                    self._last_text = text
                    self._last_text_normalized = normalized_text
                    self._last_text_ts = now
                else:
                    logger.debug("Empty transcription, skipping")

        except aiohttp.ClientError as e:
            logger.error(f"STT HTTP error: {e}")
            yield ErrorFrame(f"STT connection error: {e}")
        except Exception as e:
            logger.error(f"STT unexpected error: {e}", exc_info=True)
            yield ErrorFrame(f"STT error: {e}")
