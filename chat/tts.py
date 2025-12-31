from __future__ import annotations

import base64
import importlib.util
import io
import math
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_SAMPLE_RATE = 24000


@dataclass
class SynthesizedAudio:
    """Represents synthesized audio ready to send to the frontend."""

    audio_base64: str
    mime_type: str


def _load_custom_tts_module():
    """Load an optional custom TTS module if it exists."""

    spec = importlib.util.find_spec("chat.custom_tts_model")
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CUSTOM_TTS_MODULE = _load_custom_tts_module()


def synthesize_speech(text: str, emotion: Optional[str] = None) -> SynthesizedAudio:
    """Turn text into audio.

    If a `chat/custom_tts_model.py` module is present, it should expose a
    `generate_audio(text: str, emotion: Optional[str]) -> bytes` function
    and an optional `MIME_TYPE` string. That will be used to generate the
    audio bytes, allowing you to load並使用你自訂的 `.pt` 聲音模型。

    When no custom module is provided, this falls back to generating a short
    sine-wave tone so the API仍會提供有效的音訊回傳。
    """

    custom_result = _maybe_generate_with_custom_module(text, emotion)
    if custom_result is not None:
        return custom_result

    waveform = _text_to_waveform(text, emotion=emotion)
    wav_bytes = _waveform_to_wav_bytes(waveform, sample_rate=DEFAULT_SAMPLE_RATE)
    audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
    return SynthesizedAudio(audio_base64=audio_base64, mime_type="audio/wav")


def _maybe_generate_with_custom_module(
    text: str, emotion: Optional[str]
) -> Optional[SynthesizedAudio]:
    if CUSTOM_TTS_MODULE is None or not hasattr(CUSTOM_TTS_MODULE, "generate_audio"):
        return None

    audio_bytes = CUSTOM_TTS_MODULE.generate_audio(text, emotion=emotion)
    if not isinstance(audio_bytes, (bytes, bytearray)):
        return None

    mime_type = getattr(CUSTOM_TTS_MODULE, "MIME_TYPE", "audio/wav")
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    return SynthesizedAudio(audio_base64=audio_base64, mime_type=mime_type)


def _text_to_waveform(text: str, *, emotion: Optional[str]) -> np.ndarray:
    """Generate a simple waveform as a fallback demo."""

    duration_seconds = min(3.0, max(0.6, len(text) * 0.06))
    base_frequency = 220 + (hash(text + (emotion or "")) % 120)
    time_axis = np.linspace(0, duration_seconds, int(DEFAULT_SAMPLE_RATE * duration_seconds), False)

    envelope = np.linspace(0.2, 0.05, time_axis.size)
    waveform = 0.5 * np.sin(2 * math.pi * base_frequency * time_axis) * envelope
    return waveform


def _waveform_to_wav_bytes(waveform: np.ndarray, *, sample_rate: int) -> bytes:
    """Convert a numpy waveform into WAV bytes."""

    clamped = np.clip(waveform, -1.0, 1.0)
    int16_wave = np.int16(clamped * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_wave.tobytes())

    return buffer.getvalue()