"""faster-whisper STT 引擎封裝"""

import io
import logging
import math
import os
from pathlib import Path
import subprocess
import struct
import tempfile
import time
import wave
from typing import Optional

from faster_whisper import WhisperModel
from opencc import OpenCC

from config import settings

logger = logging.getLogger("stt-engine")


class STTEngine:
    """封裝 faster-whisper 模型，提供 transcribe 方法"""

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self._model_info: Optional[str] = None
        self._is_warmed_up: bool = False
        self._opencc: Optional[OpenCC] = None
        self._init_opencc()

    def _init_opencc(self):
        cfg = (settings.OPENCC_CONFIG or "").strip()
        if not cfg:
            logger.info("OpenCC disabled: STT_OPENCC_CONFIG is empty.")
            return
        try:
            self._opencc = OpenCC(cfg)
            logger.info("OpenCC enabled with config: %s", cfg)
        except Exception as exc:
            logger.warning("OpenCC init failed (config=%s): %s", cfg, exc)
            self._opencc = None

    def _convert_text(self, text: str, lang: Optional[str]) -> str:
        if not text or self._opencc is None:
            return text
        if settings.OPENCC_ONLY_ZH and not (lang or "").lower().startswith("zh"):
            return text
        try:
            return self._opencc.convert(text)
        except Exception as exc:
            logger.warning("OpenCC convert failed, return original text: %s", exc)
            return text

    def _build_warmup_wav(self) -> bytes:
        sample_rate = 16000
        duration = max(0.2, float(settings.WARMUP_SECONDS))
        n_samples = int(sample_rate * duration)
        freq = 440.0
        amplitude = 0.18

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                val = int(
                    32767
                    * amplitude
                    * math.sin(2.0 * math.pi * freq * (i / sample_rate))
                )
                frames += struct.pack("<h", val)
            wf.writeframes(bytes(frames))
        return buf.getvalue()

    @staticmethod
    def _guess_suffix(filename: Optional[str], content_type: Optional[str]) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix:
            return suffix
        mime = (content_type or "").lower()
        if "webm" in mime:
            return ".webm"
        if "ogg" in mime or "opus" in mime:
            return ".ogg"
        if "wav" in mime:
            return ".wav"
        if "mpeg" in mime or "mp3" in mime:
            return ".mp3"
        if "mp4" in mime or "m4a" in mime:
            return ".m4a"
        if "flac" in mime:
            return ".flac"
        return ".bin"

    def _decode_to_wav(
        self,
        audio_data: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> bytes:
        suffix = self._guess_suffix(filename, content_type)
        src_path = None
        dst_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
                src.write(audio_data)
                src_path = src.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
                dst_path = dst.name

            logger.info(
                "STT decode input: filename=%s content_type=%s suffix=%s temp_in=%s bytes=%s",
                filename or f"upload{suffix}",
                content_type or "application/octet-stream",
                suffix,
                src_path,
                len(audio_data),
            )

            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    src_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    dst_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                logger.error(
                    "ffmpeg decode failed: rc=%s stderr=%s stdout=%s temp_in=%s",
                    proc.returncode,
                    proc.stderr.strip(),
                    proc.stdout.strip(),
                    src_path,
                )
                raise RuntimeError(
                    f"ffmpeg decode failed (suffix={suffix}, content_type={content_type or '<none>'}): {proc.stderr.strip() or 'unknown error'}"
                )

            with wave.open(dst_path, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                frame_count = wf.getnframes()
            with open(dst_path, "rb") as f:
                wav_data = f.read()

            logger.info(
                "STT decode output: temp_out=%s bytes=%s channels=%s sample_width=%s frame_rate=%s frames=%s",
                dst_path,
                len(wav_data),
                channels,
                sample_width,
                frame_rate,
                frame_count,
            )
            return wav_data
        finally:
            for path in (src_path, dst_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError as exc:
                        logger.warning("Failed to remove temp file %s: %s", path, exc)

    def _warmup(self):
        if self.model is None:
            return
        logger.info(
            "Running STT warmup (seconds=%.2f, language=%s)",
            settings.WARMUP_SECONDS,
            settings.WARMUP_LANGUAGE,
        )
        t0 = time.time()
        audio_file = io.BytesIO(self._build_warmup_wav())
        segments, _ = self.model.transcribe(
            audio_file,
            language=settings.WARMUP_LANGUAGE
            if settings.WARMUP_LANGUAGE != "auto"
            else None,
            temperature=0.0,
            vad_filter=False,
            beam_size=1,
            word_timestamps=False,
        )
        # Force execution of generator to complete warmup path.
        _ = list(segments)
        self._is_warmed_up = True
        logger.info("STT warmup finished in %.2fs", time.time() - t0)

    def load_model(self):
        """載入 faster-whisper 模型"""
        logger.info(
            f"Loading faster-whisper model: {settings.MODEL_PATH} "
            f"(device={settings.DEVICE}, compute_type={settings.COMPUTE_TYPE})"
        )
        t0 = time.time()
        self.model = WhisperModel(
            settings.MODEL_PATH,
            device=settings.DEVICE,
            compute_type=settings.COMPUTE_TYPE,
            download_root=settings.HF_HOME,
        )
        elapsed = time.time() - t0
        self._model_info = settings.MODEL_PATH
        logger.info(f"Model loaded in {elapsed:.1f}s")

        if settings.STARTUP_WARMUP:
            try:
                self._warmup()
            except Exception as exc:
                logger.warning(f"Warmup failed, continuing without blocking startup: {exc}")

    def transcribe(
        self,
        audio_data: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
        request_id: Optional[str] = None,
    ) -> dict:
        """
        轉錄音訊

        Args:
            audio_data: 音檔 bytes（WAV, MP3, FLAC, etc.）
            language: 語言代碼（如 "zh"），None 為自動偵測
            prompt: 初始提示文字
            temperature: 取樣溫度
            response_format: "json" | "text" | "verbose_json"

        Returns:
            dict with "text" key (and optional segments for verbose_json)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        lang = language or settings.LANGUAGE
        t0 = time.time()

        logger.info(
            "STT transcribe start[%s]: filename=%s content_type=%s bytes=%s language=%s response_format=%s",
            request_id or "-",
            filename or "<upload>",
            content_type or "application/octet-stream",
            len(audio_data),
            lang,
            response_format,
        )
        wav_data = self._decode_to_wav(audio_data, filename, content_type)
        audio_file = io.BytesIO(wav_data)
        decode_elapsed = time.time() - t0
        logger.info(
            "STT decode done[%s]: wav_bytes=%s elapsed=%.3fs",
            request_id or "-",
            len(wav_data),
            decode_elapsed,
        )

        segments, info = self.model.transcribe(
            audio_file,
            language=lang if lang != "auto" else None,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=settings.VAD_FILTER,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            beam_size=5,
            word_timestamps=response_format == "verbose_json",
        )

        # 收集所有 segments
        all_segments = []
        full_text_parts = []
        detected_lang = info.language or lang or ""
        for seg in segments:
            seg_text = self._convert_text(seg.text.strip(), detected_lang)
            all_segments.append({
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg_text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            })
            full_text_parts.append(seg_text)

        full_text = " ".join(full_text_parts)
        elapsed = time.time() - t0

        logger.info(
            "STT transcribe done[%s]: audio_secs=%.1f total=%.1fs decode=%.1fs model_lang=%s prob=%.2f text_len=%s",
            request_id or "-",
            info.duration,
            elapsed,
            decode_elapsed,
            info.language,
            info.language_probability,
            len(full_text),
        )

        result = {"text": full_text}

        if response_format == "verbose_json":
            result.update({
                "language": info.language,
                "duration": round(info.duration, 3),
                "segments": all_segments,
            })

        return result

    @property
    def model_name(self) -> str:
        return self._model_info or "not-loaded"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    @property
    def opencc_enabled(self) -> bool:
        return self._opencc is not None

    @property
    def opencc_config(self) -> str:
        return settings.OPENCC_CONFIG


# 單例
engine = STTEngine()
