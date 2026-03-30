"""
VibeVoiceHTTPTTSService — 透過 HTTP 呼叫遠端 VibeVoice TTS 服務

呼叫 services/tts-service 的 POST /v1/audio/speech 端點（OpenAI 相容）。
VibeVoice (microsoft/VibeVoice-1.5B) — 繁體中文 TTS，24kHz WAV 輸出。
Pipecat TTS Service 介面實作。
"""

import io
import logging
import struct
import wave
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService

logger = logging.getLogger("vibevoice-http-tts")


class VibeVoiceHTTPTTSService(TTSService):
    """透過 HTTP API 呼叫遠端 VibeVoice TTS 服務"""

    def __init__(
        self,
        base_url: str = "http://localhost:8200/v1",
        api_key: str = "",
        voice: str = "Xinran",
        model: str = "vibevoice",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self._sample_rate = sample_rate
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame: Frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame: EndFrame):
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def run_tts(
        self, text: str, context_id: str = ""
    ) -> AsyncGenerator[Frame, None]:
        """
        接收文字，呼叫遠端 TTS API，回傳 TTSAudioRawFrame。
        """
        if not text.strip():
            return

        if not self._session:
            self._session = aiohttp.ClientSession()

        yield TTSStartedFrame()

        try:
            payload = {
                "model": self._model,
                "input": text,
                "voice": self._voice,
                "response_format": "wav",
            }

            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            url = f"{self._base_url}/audio/speech"
            logger.info(f"TTS request: '{text[:60]}...' → {url}")

            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"TTS API error {resp.status}: {error_text}")
                    yield ErrorFrame(f"TTS API error: {resp.status}")
                    yield TTSStoppedFrame()
                    return

                # 讀取完整 WAV 回應
                wav_data = await resp.read()
                logger.debug(f"Received {len(wav_data)} bytes WAV")

                # 解析 WAV -> raw PCM
                try:
                    wav_io = io.BytesIO(wav_data)
                    with wave.open(wav_io, "rb") as wf:
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        n_channels = wf.getnchannels()
                        pcm_data = wf.readframes(wf.getnframes())

                    # 轉為 mono 16-bit PCM（如果需要）
                    if n_channels > 1:
                        # 簡單取左聲道
                        samples = struct.unpack(
                            f"<{len(pcm_data) // sample_width}h", pcm_data
                        )
                        mono = samples[::n_channels]
                        pcm_data = struct.pack(f"<{len(mono)}h", *mono)

                    # 分 chunk 送出（每 chunk ~20ms）
                    chunk_samples = sample_rate // 50  # 20ms
                    chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

                    for i in range(0, len(pcm_data), chunk_bytes):
                        chunk = pcm_data[i : i + chunk_bytes]
                        if chunk:
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=sample_rate,
                                num_channels=1,
                            )

                except Exception as e:
                    logger.error(f"WAV parsing failed: {e}")
                    yield ErrorFrame(f"TTS WAV parsing error: {e}")

        except aiohttp.ClientError as e:
            logger.error(f"TTS HTTP error: {e}")
            yield ErrorFrame(f"TTS connection error: {e}")
        except Exception as e:
            logger.error(f"TTS unexpected error: {e}", exc_info=True)
            yield ErrorFrame(f"TTS error: {e}")

        yield TTSStoppedFrame()
