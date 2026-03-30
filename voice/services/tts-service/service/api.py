import asyncio
import io
import json
import os
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _str_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _safe_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_alias_map(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(","):
        pair = item.strip()
        if not pair or "=" not in pair:
            continue
        src, target = pair.split("=", 1)
        src_key = src.strip().lower()
        target_val = target.strip()
        if src_key and target_val:
            result[src_key] = target_val
    return result


def _parse_csv(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _speech_to_waveform(audio: object) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)

    arr = arr.squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def _waveform_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> tuple[bytes, float]:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)

    pcm16 = (arr * 32767.0).astype(np.int16)
    duration = float(len(pcm16)) / float(sample_rate)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue(), duration


def _waveform_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def _wav_bytes_to_waveform(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    audio = (pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)
    return audio, int(sample_rate)


def _download_file(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.part"
    with urlopen(url, timeout=300) as resp, open(tmp_path, "wb") as f:
        f.write(resp.read())
    os.replace(tmp_path, dst_path)


@dataclass
class Settings:
    enabled_backends_raw: str = os.getenv(
        "TTS_ENABLED_BACKENDS", "kokoro,piper,cosyvoice"
    )
    default_backend: str = os.getenv("TTS_DEFAULT_BACKEND", "kokoro").strip().lower()
    startup_backends_raw: str = os.getenv("TTS_STARTUP_BACKENDS", "kokoro,piper")

    default_speaker: str = os.getenv(
        "TTS_DEFAULT_SPEAKER",
        os.getenv("COSYVOICE_DEFAULT_SPEAKER", "中文女"),
    )
    default_speed: float = float(
        os.getenv("TTS_DEFAULT_SPEED", os.getenv("COSYVOICE_DEFAULT_SPEED", "1.0"))
    )
    text_frontend: bool = _env_bool(
        "TTS_TEXT_FRONTEND", _env_bool("COSYVOICE_TEXT_FRONTEND", True)
    )
    stream_enabled: bool = _env_bool(
        "TTS_STREAM_ENABLED", _env_bool("COSYVOICE_STREAM_ENABLED", True)
    )
    default_stream: bool = _env_bool(
        "TTS_DEFAULT_STREAM", _env_bool("COSYVOICE_DEFAULT_STREAM", False)
    )

    startup_warmup: bool = _env_bool("TTS_STARTUP_WARMUP", True)
    warmup_text: str = os.getenv("TTS_WARMUP_TEXT", "你好，這是 TTS 預熱語句。")

    kokoro_repo_id: str = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M-v1.1-zh")
    kokoro_lang_code: str = os.getenv("KOKORO_LANG_CODE", "z")
    kokoro_default_voice: str = os.getenv("KOKORO_DEFAULT_VOICE", "zf_001")
    kokoro_device: str = os.getenv("KOKORO_DEVICE", "auto")
    kokoro_torch_num_threads: int = int(os.getenv("KOKORO_TORCH_NUM_THREADS", "8"))

    piper_model_path: str = os.getenv(
        "PIPER_MODEL_PATH", "/data/piper_models/zh_CN-huayan-x_low.onnx"
    )
    piper_config_path: str = os.getenv(
        "PIPER_CONFIG_PATH", "/data/piper_models/zh_CN-huayan-x_low.onnx.json"
    )
    piper_model_url: str = os.getenv(
        "PIPER_MODEL_URL",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx",
    )
    piper_config_url: str = os.getenv(
        "PIPER_CONFIG_URL",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json",
    )
    piper_use_cuda: bool = _env_bool("PIPER_USE_CUDA", False)
    piper_omp_num_threads: int = int(os.getenv("PIPER_OMP_NUM_THREADS", "4"))
    piper_default_speaker: int = int(os.getenv("PIPER_DEFAULT_SPEAKER", "0"))

    cosy_model_dir: str = os.getenv(
        "COSYVOICE_MODEL_DIR", "/data/cosyvoice_models/CosyVoice-300M-SFT"
    )
    cosy_model_id: str = os.getenv("COSYVOICE_MODEL_ID", "").strip()
    cosy_mode: str = os.getenv("COSYVOICE_MODE", "auto")
    cosy_default_instruct: str = os.getenv(
        "COSYVOICE_DEFAULT_INSTRUCT",
        "You are a helpful assistant. 请用台湾普通话表达。<|endofprompt|>",
    )
    cosy_prompt_text: str = os.getenv(
        "COSYVOICE_PROMPT_TEXT", "希望你以后能够做的比我还好呦。"
    )
    cosy_prompt_wav: str = os.getenv(
        "COSYVOICE_PROMPT_WAV", "/opt/CosyVoice/asset/zero_shot_prompt.wav"
    )
    cosy_load_jit: bool = _env_bool("COSYVOICE_LOAD_JIT", False)
    cosy_load_trt: bool = _env_bool("COSYVOICE_LOAD_TRT", False)
    cosy_fp16: bool = _env_bool("COSYVOICE_FP16", False)
    cosy_trt_concurrent: int = int(os.getenv("COSYVOICE_TRT_CONCURRENT", "1"))
    cosy_speaker_aliases_raw: str = os.getenv(
        "COSYVOICE_SPEAKER_ALIASES", "vivian=中文女,xinran=中文女"
    )


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    backend: str | None = None
    speaker_name: str | None = None
    voice: str | None = None
    speed: float | None = None
    stream: bool | None = None
    non_streaming_mode: bool | None = None
    text_frontend: bool | None = None

    instruct: str | None = None
    prompt_text: str | None = None
    prompt_wav: str | None = None

    language: str | None = None
    seed: int | None = None
    max_new_tokens: int | None = None
    disable_prefill: bool = False
    cfg_scale: float | None = None


class SpeechCompatRequest(BaseModel):
    input: str = Field(min_length=1)
    model: str | None = None
    backend: str | None = None
    voice: str | None = None
    speaker_name: str | None = None
    speed: float | None = None
    stream: bool | None = None
    non_streaming_mode: bool | None = None
    text_frontend: bool | None = None

    instruct: str | None = None
    instructions: str | None = None
    prompt_text: str | None = None
    prompt_wav: str | None = None

    language: str | None = None
    seed: int | None = None
    max_new_tokens: int | None = None
    disable_prefill: bool = False
    cfg_scale: float | None = None


class BaseTTSBackend:
    name: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.sample_rate = 0
        self._load_lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            self.load()

    def load(self) -> None:
        raise NotImplementedError

    def list_voices(self) -> list[str]:
        return []

    def generate(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> tuple[np.ndarray, int, dict]:
        raise NotImplementedError

    def stream_pcm16(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        stopper = stop_event or threading.Event()
        wav, _, _ = self.generate(
            text=text,
            speaker_name=speaker_name,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
        )
        pcm = _waveform_to_pcm16_bytes(wav)
        chunk_size = 3200
        for i in range(0, len(pcm), chunk_size):
            if stopper.is_set():
                break
            yield pcm[i : i + chunk_size]

    def reset_state(self) -> None:
        return

    def close(self) -> None:
        return


class KokoroBackend(BaseTTSBackend):
    name = "kokoro"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.pipeline = None
        self.device = "cpu"
        self.sample_rate = 24000

    def load(self) -> None:
        from kokoro import KPipeline

        if self.settings.kokoro_torch_num_threads > 0:
            torch.set_num_threads(self.settings.kokoro_torch_num_threads)

        device = self.settings.kokoro_device.strip().lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(
            "Loading Kokoro backend. repo=%s voice=%s device=%s",
            self.settings.kokoro_repo_id,
            self.settings.kokoro_default_voice,
            self.device,
        )
        self.pipeline = KPipeline(
            lang_code=self.settings.kokoro_lang_code,
            repo_id=self.settings.kokoro_repo_id,
            device=self.device,
        )
        self.ready = True

        if self.settings.startup_warmup:
            try:
                self.generate(
                    text=self.settings.warmup_text,
                    speaker_name=self.settings.kokoro_default_voice,
                    speed=self.settings.default_speed,
                    stream=False,
                    text_frontend=False,
                    instruct=None,
                    prompt_text=None,
                    prompt_wav=None,
                )
            except Exception as exc:
                logger.warning("Kokoro warmup failed: %s", exc)

    def list_voices(self) -> list[str]:
        return [self.settings.kokoro_default_voice]

    def generate(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> tuple[np.ndarray, int, dict]:
        self.ensure_loaded()
        voice = (speaker_name or self.settings.kokoro_default_voice).strip()
        if not voice:
            voice = self.settings.kokoro_default_voice
        use_speed = float(speed if speed and speed > 0 else self.settings.default_speed)

        start = time.time()
        segments = list(self.pipeline(text, voice=voice, speed=use_speed))
        audio_parts = [seg.audio.detach().cpu().numpy() for seg in segments if seg.audio is not None]
        if not audio_parts:
            raise RuntimeError("Kokoro returned no audio output.")
        full_wav = np.concatenate(audio_parts, axis=0).astype(np.float32)

        elapsed = time.time() - start
        duration = float(full_wav.shape[0]) / float(self.sample_rate)
        meta = {
            "backend": self.name,
            "mode": "kokoro",
            "speaker_name": voice,
            "generation_time_sec": round(elapsed, 4),
            "audio_duration_sec": round(duration, 4),
            "sample_rate": int(self.sample_rate),
            "model_dir": self.settings.kokoro_repo_id,
            "speed": use_speed,
            "stream": stream,
            "text_frontend": False,
        }
        return full_wav, int(self.sample_rate), meta

    def stream_pcm16(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        self.ensure_loaded()
        stopper = stop_event or threading.Event()
        voice = (speaker_name or self.settings.kokoro_default_voice).strip()
        if not voice:
            voice = self.settings.kokoro_default_voice
        use_speed = float(speed if speed and speed > 0 else self.settings.default_speed)

        if stream:
            for seg in self.pipeline(text, voice=voice, speed=use_speed):
                if stopper.is_set():
                    break
                if seg.audio is None:
                    continue
                yield _waveform_to_pcm16_bytes(_speech_to_waveform(seg.audio))
            return

        wav, _, _ = self.generate(
            text=text,
            speaker_name=voice,
            speed=use_speed,
            stream=False,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
        )
        yield _waveform_to_pcm16_bytes(wav)


class PiperBackend(BaseTTSBackend):
    name = "piper"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.sample_rate = 16000
        self._worker_lock = threading.Lock()
        self._worker_cv = threading.Condition(self._worker_lock)
        self._worker_queue: list[dict[str, Any]] = []
        self._worker_thread: threading.Thread | None = None
        self._worker_shutdown = threading.Event()
        self._worker_ready = threading.Event()
        self._worker_error: Exception | None = None
        self._worker_voice: Any = None
        self._worker_generation = 0
        self._run_lock = threading.Lock()

    def _ensure_model_files(self) -> None:
        if not os.path.exists(self.settings.piper_model_path):
            if not self.settings.piper_model_url:
                raise RuntimeError(
                    f"Piper model not found: {self.settings.piper_model_path}"
                )
            logger.info(
                "Downloading Piper model from %s to %s",
                self.settings.piper_model_url,
                self.settings.piper_model_path,
            )
            _download_file(self.settings.piper_model_url, self.settings.piper_model_path)

        if not os.path.exists(self.settings.piper_config_path):
            if self.settings.piper_config_url:
                logger.info(
                    "Downloading Piper config from %s to %s",
                    self.settings.piper_config_url,
                    self.settings.piper_config_path,
                )
                _download_file(self.settings.piper_config_url, self.settings.piper_config_path)

        if os.path.exists(self.settings.piper_config_path):
            try:
                with open(self.settings.piper_config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                audio_cfg = cfg.get("audio", {})
                maybe_sr = int(audio_cfg.get("sample_rate", self.sample_rate))
                if maybe_sr > 0:
                    self.sample_rate = maybe_sr
            except Exception:
                pass

    def load(self) -> None:
        self._ensure_model_files()
        self._start_worker()

        self.ready = True

        if self.settings.startup_warmup:
            try:
                self.generate(
                    text=self.settings.warmup_text,
                    speaker_name=str(self.settings.piper_default_speaker),
                    speed=self.settings.default_speed,
                    stream=False,
                    text_frontend=False,
                    instruct=None,
                    prompt_text=None,
                    prompt_wav=None,
                )
            except Exception as exc:
                logger.warning("Piper warmup failed: %s", exc)

    def _start_worker(self) -> None:
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._worker_error = None
            self._worker_queue.clear()
            self._worker_shutdown.clear()
            self._worker_ready.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="piper-worker",
                daemon=True,
            )
            self._worker_thread.start()

        if not self._worker_ready.wait(timeout=60):
            raise RuntimeError("Piper worker startup timeout")
        if self._worker_error is not None:
            raise RuntimeError(f"Piper worker startup failed: {self._worker_error}")
        logger.warning("Piper worker initialized (persistent model)")

    def _worker_loop(self) -> None:
        try:
            self._init_worker_voice()
        except Exception as exc:
            with self._worker_lock:
                self._worker_error = exc
                self._worker_ready.set()
                self._worker_cv.notify_all()
            return

        with self._worker_lock:
            self._worker_ready.set()
            self._worker_cv.notify_all()

        while not self._worker_shutdown.is_set():
            with self._worker_cv:
                while not self._worker_queue and not self._worker_shutdown.is_set():
                    self._worker_cv.wait(timeout=0.5)
                if self._worker_shutdown.is_set():
                    break
                job = self._worker_queue.pop(0)
                current_generation = self._worker_generation

            if job["generation"] != current_generation:
                job["error"] = RuntimeError("Piper job cancelled by END")
                job["event"].set()
                continue

            try:
                pcm_bytes = self._synthesize_pcm16(
                    text=job["text"],
                    speaker_id=job["speaker_id"],
                    length_scale=job["length_scale"],
                )
                job["pcm_bytes"] = pcm_bytes
                job["sample_rate"] = int(self.sample_rate)
            except Exception as exc:
                job["error"] = exc
            finally:
                job["event"].set()

    def _init_worker_voice(self) -> None:
        try:
            from piper.voice import PiperVoice
        except Exception as exc:
            raise RuntimeError("Failed to import piper.voice.PiperVoice") from exc

        load_kwargs: dict[str, Any] = {"use_cuda": self.settings.piper_use_cuda}
        if os.path.exists(self.settings.piper_config_path):
            load_kwargs["config_path"] = self.settings.piper_config_path

        self._worker_voice = PiperVoice.load(
            self.settings.piper_model_path,
            **load_kwargs,
        )

        cfg = getattr(self._worker_voice, "config", None)
        cfg_sr = getattr(cfg, "sample_rate", None)
        try:
            parsed_sr = int(cfg_sr)
            if parsed_sr > 0:
                self.sample_rate = parsed_sr
        except Exception:
            pass

    def _synthesize_pcm16(
        self,
        *,
        text: str,
        speaker_id: int,
        length_scale: float,
    ) -> bytes:
        if self._worker_voice is None:
            raise RuntimeError("Piper worker voice not initialized")

        with self._run_lock:
            # Newer Piper Python API: synthesize() yields AudioChunk with PCM bytes.
            if hasattr(self._worker_voice, "synthesize"):
                try:
                    from piper.config import SynthesisConfig

                    syn_cfg = SynthesisConfig(
                        speaker_id=speaker_id,
                        length_scale=length_scale,
                    )
                    out_iter = self._worker_voice.synthesize(text, syn_config=syn_cfg)
                except Exception:
                    out_iter = self._worker_voice.synthesize(text)

                chunk_bytes: list[bytes] = []
                last_sr: int | None = None
                for chunk in out_iter:
                    payload = getattr(chunk, "audio_int16_bytes", b"")
                    if payload:
                        chunk_bytes.append(bytes(payload))
                    try:
                        sr = int(getattr(chunk, "sample_rate", 0))
                        if sr > 0:
                            last_sr = sr
                    except Exception:
                        pass

                pcm_bytes = b"".join(chunk_bytes)
                if pcm_bytes:
                    if last_sr and last_sr > 0:
                        self.sample_rate = last_sr
                    return pcm_bytes

            if hasattr(self._worker_voice, "synthesize_stream_raw"):
                raw_iter = self._worker_voice.synthesize_stream_raw(text)
                chunks: list[bytes] = []
                for chunk in raw_iter:
                    if chunk:
                        chunks.append(bytes(chunk))
                pcm_bytes = b"".join(chunks)
                if pcm_bytes:
                    return pcm_bytes

            raise RuntimeError("Piper voice API did not return any PCM output")

    def _submit_worker_job(
        self,
        *,
        text: str,
        speaker_id: int,
        length_scale: float,
    ) -> tuple[bytes, int]:
        if not self.ready:
            self.ensure_loaded()
        if self._worker_error is not None:
            raise RuntimeError(f"Piper worker is unavailable: {self._worker_error}")

        job: dict[str, Any] = {
            "text": text,
            "speaker_id": speaker_id,
            "length_scale": length_scale,
            "event": threading.Event(),
            "error": None,
            "pcm_bytes": b"",
            "sample_rate": int(self.sample_rate),
            "generation": self._worker_generation,
        }

        with self._worker_cv:
            self._worker_queue.append(job)
            self._worker_cv.notify()

        if not job["event"].wait(timeout=180):
            raise RuntimeError("Piper worker inference timeout")

        if job["error"] is not None:
            raise RuntimeError(f"Piper synthesis failed: {job['error']}") from job["error"]

        pcm_bytes = bytes(job["pcm_bytes"])
        if not pcm_bytes:
            raise RuntimeError("Piper synthesis returned empty audio")

        sample_rate = int(job.get("sample_rate") or self.sample_rate)
        return pcm_bytes, sample_rate

    def list_voices(self) -> list[str]:
        return [f"speaker_{self.settings.piper_default_speaker}"]

    def _resolve_speaker_id(self, speaker_name: str | None) -> int:
        if not speaker_name:
            return self.settings.piper_default_speaker
        cleaned = speaker_name.strip().lower().replace("speaker_", "")
        try:
            return int(cleaned)
        except ValueError:
            return self.settings.piper_default_speaker

    def generate(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> tuple[np.ndarray, int, dict]:
        self.ensure_loaded()
        use_speed = float(speed if speed and speed > 0 else self.settings.default_speed)
        speaker_id = self._resolve_speaker_id(speaker_name)
        length_scale = 1.0 / use_speed if use_speed > 0 else 1.0
        logger.warning(
            "Piper request via worker: mode=generate speaker=%s text_len=%s",
            speaker_id,
            len(text),
        )

        start = time.time()
        pcm_bytes, sr = self._submit_worker_job(
            text=text,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        wav = (pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)

        elapsed = time.time() - start
        duration = float(wav.shape[0]) / float(sr)
        meta = {
            "backend": self.name,
            "mode": "piper",
            "speaker_name": f"speaker_{speaker_id}",
            "generation_time_sec": round(elapsed, 4),
            "audio_duration_sec": round(duration, 4),
            "sample_rate": int(sr),
            "model_dir": self.settings.piper_model_path,
            "speed": use_speed,
            "stream": stream,
            "text_frontend": False,
        }
        return wav, int(sr), meta

    def stream_pcm16(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        self.ensure_loaded()
        stopper = stop_event or threading.Event()
        use_speed = float(speed if speed and speed > 0 else self.settings.default_speed)
        speaker_id = self._resolve_speaker_id(speaker_name)
        length_scale = 1.0 / use_speed if use_speed > 0 else 1.0
        logger.warning(
            "Piper request via worker: mode=stream speaker=%s text_len=%s",
            speaker_id,
            len(text),
        )
        pcm_bytes, _ = self._submit_worker_job(
            text=text,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )
        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
            if stopper.is_set():
                break
            yield pcm_bytes[i : i + chunk_size]

    def reset_state(self) -> None:
        # Piper is stateless between requests; END means drop queued stale jobs.
        with self._worker_cv:
            self._worker_generation += 1
            for pending in self._worker_queue:
                if pending.get("event") and not pending["event"].is_set():
                    pending["error"] = RuntimeError("Piper job cancelled by END")
                    pending["event"].set()
            self._worker_queue.clear()
            self._worker_cv.notify_all()

    def close(self) -> None:
        with self._worker_cv:
            self._worker_shutdown.set()
            self._worker_cv.notify_all()
        worker = self._worker_thread
        if worker and worker.is_alive():
            worker.join(timeout=5)
        self._worker_thread = None
        self._worker_voice = None


class CosyVoiceBackend(BaseTTSBackend):
    name = "cosyvoice"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = None
        self.supported_speakers: list[str] = []
        self.sample_rate = 22050
        self.model_ref = self.settings.cosy_model_dir
        self._lock = threading.Lock()
        self.alias_map = _parse_alias_map(self.settings.cosy_speaker_aliases_raw)

    def _resolve_mode(self) -> str:
        mode = self.settings.cosy_mode.strip().lower()
        if mode in {"sft", "instruct2", "zero_shot"}:
            return mode

        if self.supported_speakers:
            return "sft"
        if hasattr(self.model, "inference_instruct2"):
            return "instruct2"
        return "zero_shot"

    def _resolve_speaker(self, speaker_name: str | None) -> str:
        target = (speaker_name or self.settings.default_speaker).strip()
        if not target:
            target = self.settings.default_speaker

        alias_hit = self.alias_map.get(target.lower())
        if alias_hit:
            target = alias_hit

        if not self.supported_speakers:
            return target

        target_lower = target.lower()
        for spk in self.supported_speakers:
            if spk.lower() == target_lower:
                return spk

        default_lower = self.settings.default_speaker.lower()
        for spk in self.supported_speakers:
            if spk.lower() == default_lower:
                return spk
        return self.supported_speakers[0]

    def _resolve_speed(self, speed: float | None) -> float:
        if speed is None or speed <= 0:
            return self.settings.default_speed
        return float(speed)

    def _resolve_prompt_wav(self, prompt_wav: str | None) -> str:
        target = (prompt_wav or self.settings.cosy_prompt_wav).strip()
        if not target:
            raise RuntimeError("prompt_wav is required for current mode")
        if not os.path.exists(target):
            raise RuntimeError(f"prompt_wav not found: {target}")
        return target

    @staticmethod
    def _looks_like_local_path(model_ref: str) -> bool:
        text = (model_ref or "").strip()
        if not text:
            return False
        if text.startswith(("~", ".", "/")):
            return True
        # Relative local paths (e.g. data/cosyvoice_models/CosyVoice-300M-SFT)
        # can otherwise be mistaken as remote ids.
        if "/" in text and text.count("/") >= 2:
            return True
        if "\\" in text:
            return True
        if ":" in text and "/" not in text:
            return True
        return os.path.isabs(text)

    @staticmethod
    def _looks_like_remote_model_id(model_ref: str) -> bool:
        text = (model_ref or "").strip()
        if not text or text.startswith(("/", ".", "~")) or "\\" in text:
            return False
        parts = [p for p in text.split("/") if p]
        return len(parts) == 2

    def load(self) -> None:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "Failed to import CosyVoice. Ensure CosyVoice source and dependencies are installed."
            ) from exc

        configured_model_dir = (self.settings.cosy_model_dir or "").strip()
        configured_model_id = (self.settings.cosy_model_id or "").strip()

        model_ref = configured_model_dir
        if self._looks_like_local_path(configured_model_dir):
            if os.path.isdir(configured_model_dir):
                logger.info(
                    "Loading CosyVoice backend from local model dir: %s",
                    configured_model_dir,
                )
            elif configured_model_id and self._looks_like_remote_model_id(configured_model_id):
                logger.warning(
                    "CosyVoice local model dir not found (%s), fallback to COSYVOICE_MODEL_ID=%s",
                    configured_model_dir,
                    configured_model_id,
                )
                model_ref = configured_model_id
            else:
                raise RuntimeError(
                    "CosyVoice local model dir does not exist: "
                    f"{configured_model_dir}. "
                    "Provide files under COSYVOICE_MODEL_DIR or set COSYVOICE_MODEL_ID "
                    "to a valid remote model id (org/model)."
                )
        else:
            logger.info("Loading CosyVoice backend from model id: %s", model_ref)

        self.model_ref = model_ref
        model_kwargs: dict[str, object] = {"model_dir": model_ref}
        cosy1_yaml = os.path.join(model_ref, "cosyvoice.yaml")
        cosy2_yaml = os.path.join(model_ref, "cosyvoice2.yaml")
        cosy3_yaml = os.path.join(model_ref, "cosyvoice3.yaml")

        if os.path.exists(cosy1_yaml) or os.path.exists(cosy2_yaml):
            model_kwargs.update(
                {
                    "load_jit": self.settings.cosy_load_jit,
                    "load_trt": self.settings.cosy_load_trt,
                    "fp16": self.settings.cosy_fp16,
                    "trt_concurrent": self.settings.cosy_trt_concurrent,
                }
            )
        elif os.path.exists(cosy3_yaml):
            model_kwargs.update(
                {
                    "load_trt": self.settings.cosy_load_trt,
                    "fp16": self.settings.cosy_fp16,
                    "trt_concurrent": self.settings.cosy_trt_concurrent,
                }
            )

        self.model = AutoModel(**model_kwargs)
        self.supported_speakers = self.model.list_available_spks() or []
        self.sample_rate = int(getattr(self.model, "sample_rate", 22050))
        self.ready = True

        if self.settings.startup_warmup:
            try:
                self.generate(
                    text=self.settings.warmup_text,
                    speaker_name=self.settings.default_speaker,
                    speed=self.settings.default_speed,
                    stream=False,
                    text_frontend=self.settings.text_frontend,
                    instruct=self.settings.cosy_default_instruct,
                    prompt_text=self.settings.cosy_prompt_text,
                    prompt_wav=self.settings.cosy_prompt_wav,
                )
            except Exception as exc:
                logger.warning("CosyVoice warmup failed: %s", exc)

    def list_voices(self) -> list[str]:
        return self.supported_speakers

    def _iter_outputs(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> Iterator[dict]:
        if not self.ready or self.model is None:
            raise RuntimeError("CosyVoice backend is not ready")

        selected_mode = self._resolve_mode()
        selected_speaker = self._resolve_speaker(speaker_name)
        selected_speed = self._resolve_speed(speed)
        use_text_frontend = (
            self.settings.text_frontend if text_frontend is None else text_frontend
        )

        with self._lock:
            if selected_mode == "sft":
                for out in self.model.inference_sft(
                    text,
                    selected_speaker,
                    stream=stream,
                    speed=selected_speed,
                    text_frontend=use_text_frontend,
                ):
                    yield out
                return

            resolved_prompt_wav = self._resolve_prompt_wav(prompt_wav)

            if selected_mode == "instruct2":
                use_instruct = (instruct or self.settings.cosy_default_instruct).strip()
                for out in self.model.inference_instruct2(
                    text,
                    use_instruct,
                    resolved_prompt_wav,
                    stream=stream,
                    speed=selected_speed,
                    text_frontend=use_text_frontend,
                ):
                    yield out
                return

            use_prompt_text = (prompt_text or self.settings.cosy_prompt_text).strip()
            for out in self.model.inference_zero_shot(
                text,
                use_prompt_text,
                resolved_prompt_wav,
                stream=stream,
                speed=selected_speed,
                text_frontend=use_text_frontend,
            ):
                yield out

    def generate(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> tuple[np.ndarray, int, dict]:
        self.ensure_loaded()

        selected_speaker = self._resolve_speaker(speaker_name)
        selected_speed = self._resolve_speed(speed)
        use_text_frontend = (
            self.settings.text_frontend if text_frontend is None else text_frontend
        )

        chunks: list[np.ndarray] = []
        start = time.time()
        for model_output in self._iter_outputs(
            text=text,
            speaker_name=selected_speaker,
            speed=selected_speed,
            stream=stream,
            text_frontend=use_text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
        ):
            wav = _speech_to_waveform(model_output.get("tts_speech"))
            if wav.size > 0:
                chunks.append(wav)

        if not chunks:
            raise RuntimeError("CosyVoice model returned no audio output.")

        full_wav = np.concatenate(chunks, axis=0)
        elapsed = time.time() - start
        duration = float(full_wav.shape[0]) / float(self.sample_rate)

        meta = {
            "backend": self.name,
            "mode": self._resolve_mode(),
            "speaker_name": selected_speaker,
            "generation_time_sec": round(elapsed, 4),
            "audio_duration_sec": round(duration, 4),
            "sample_rate": int(self.sample_rate),
            "model_dir": self.model_ref,
            "speed": selected_speed,
            "stream": stream,
            "text_frontend": use_text_frontend,
        }
        return full_wav, int(self.sample_rate), meta

    def stream_pcm16(
        self,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        self.ensure_loaded()
        stopper = stop_event or threading.Event()
        for model_output in self._iter_outputs(
            text=text,
            speaker_name=speaker_name,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
        ):
            if stopper.is_set():
                break
            wav = _speech_to_waveform(model_output.get("tts_speech"))
            if wav.size == 0:
                continue
            yield _waveform_to_pcm16_bytes(wav)


class MultiBackendTTSService:
    ALIASES = {
        "kokoro": "kokoro",
        "kokoro-82m": "kokoro",
        "hexgrad/kokoro-82m-v1.1-zh": "kokoro",
        "piper": "piper",
        "piper1-gpl": "piper",
        "cosy": "cosyvoice",
        "cosyvoice": "cosyvoice",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backends: dict[str, BaseTTSBackend] = {}
        self.enabled_backends = _parse_csv(self.settings.enabled_backends_raw)
        self.startup_backends = _parse_csv(self.settings.startup_backends_raw)

        if "kokoro" in self.enabled_backends:
            self.backends["kokoro"] = KokoroBackend(settings)
        if "piper" in self.enabled_backends:
            self.backends["piper"] = PiperBackend(settings)
        if "cosyvoice" in self.enabled_backends:
            self.backends["cosyvoice"] = CosyVoiceBackend(settings)

        if not self.backends:
            raise RuntimeError("No TTS backend enabled. Set TTS_ENABLED_BACKENDS.")

        self.default_backend = self._normalize_backend(self.settings.default_backend)
        if self.default_backend not in self.backends:
            self.default_backend = next(iter(self.backends.keys()))

    @property
    def ready(self) -> bool:
        backend = self.backends.get(self.default_backend)
        return bool(backend and backend.ready)

    def _normalize_backend(self, raw: str | None) -> str:
        if not raw:
            return self.default_backend
        key = raw.strip().lower()
        return self.ALIASES.get(key, key)

    def resolve_backend_name(self, backend: str | None, model: str | None = None) -> str:
        candidate = backend or model or self.default_backend
        normalized = self._normalize_backend(candidate)
        if normalized not in self.backends:
            raise RuntimeError(
                f"Unsupported backend/model: {candidate}. enabled={','.join(self.backends.keys())}"
            )
        return normalized

    def ensure_backend(self, backend_name: str) -> BaseTTSBackend:
        backend = self.backends[backend_name]
        backend.ensure_loaded()
        return backend

    def load_startup(self) -> None:
        targets = []
        for name in self.startup_backends:
            norm = self._normalize_backend(name)
            if norm in self.backends and norm not in targets:
                targets.append(norm)

        if self.default_backend not in targets:
            targets.insert(0, self.default_backend)

        for name in targets:
            logger.info("Preloading backend: %s", name)
            self.ensure_backend(name)

    def generate(
        self,
        backend_name: str,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
    ) -> tuple[np.ndarray, int, dict]:
        backend = self.ensure_backend(backend_name)
        return backend.generate(
            text=text,
            speaker_name=speaker_name,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
        )

    def stream_pcm16(
        self,
        backend_name: str,
        text: str,
        speaker_name: str | None,
        speed: float | None,
        stream: bool,
        text_frontend: bool | None,
        instruct: str | None,
        prompt_text: str | None,
        prompt_wav: str | None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[bytes]:
        backend = self.ensure_backend(backend_name)
        return backend.stream_pcm16(
            text=text,
            speaker_name=speaker_name,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stop_event=stop_event,
        )

    def health(self) -> dict:
        backends = {}
        for name, backend in self.backends.items():
            backends[name] = {
                "ready": backend.ready,
                "sample_rate": backend.sample_rate,
                "voices_count": len(backend.list_voices()),
            }
        return {
            "status": "ok" if self.ready else "loading",
            "ready": self.ready,
            "default_backend": self.default_backend,
            "enabled_backends": list(self.backends.keys()),
            "backends": backends,
        }

    def reset_backend_state(self, backend_name: str) -> None:
        backend = self.backends.get(backend_name)
        if backend is None:
            return
        backend.reset_state()

    def shutdown(self) -> None:
        for backend in self.backends.values():
            try:
                backend.close()
            except Exception as exc:
                logger.warning("Backend close failed (%s): %s", backend.name, exc)


settings = Settings()
service = MultiBackendTTSService(settings)


def _resolve_http_stream(stream: bool | None, non_streaming_mode: bool | None) -> bool:
    if not settings.stream_enabled:
        return False
    if stream is not None:
        return stream
    if non_streaming_mode is not None:
        return not non_streaming_mode
    return settings.default_stream


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load_startup()
    try:
        yield
    finally:
        service.shutdown()


app = FastAPI(title="Multi-Backend TTS API", version="4.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return service.health()


@app.get("/v1/voices")
def voices(
    backend: str | None = None,
    all_backends: bool = Query(default=False, alias="all"),
) -> dict:
    if all_backends:
        data: dict[str, list[str]] = {}
        for name, b in service.backends.items():
            data[name] = b.list_voices() if b.ready else []
        return {"voices": data}

    selected = service.resolve_backend_name(backend)
    b = service.ensure_backend(selected)
    return {"backend": selected, "voices": b.list_voices()}


async def _close_ws_safely(websocket: WebSocket, code: int = 1000, reason: str = ""):
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    try:
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        pass


@app.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()

    text = (websocket.query_params.get("text") or "").strip()
    if not text:
        await _close_ws_safely(websocket, code=1008, reason="text is required")
        return

    speaker_name = websocket.query_params.get("speaker_name") or websocket.query_params.get(
        "voice"
    )
    speed = _safe_float(websocket.query_params.get("speed"), settings.default_speed)
    text_frontend = _str_bool(
        websocket.query_params.get("text_frontend"), settings.text_frontend
    )
    non_streaming_mode = _str_bool(websocket.query_params.get("non_streaming_mode"), False)
    stream = settings.stream_enabled and not non_streaming_mode

    backend_raw = websocket.query_params.get("backend")
    model_raw = websocket.query_params.get("model")
    try:
        backend_name = service.resolve_backend_name(backend_raw, model_raw)
    except Exception as exc:
        await _close_ws_safely(websocket, code=1008, reason=str(exc))
        return

    instruct = websocket.query_params.get("instruct")
    prompt_text = websocket.query_params.get("prompt_text")
    prompt_wav = websocket.query_params.get("prompt_wav")

    stop_signal = threading.Event()
    try:
        iterator = service.stream_pcm16(
            backend_name=backend_name,
            text=text,
            speaker_name=speaker_name,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stop_event=stop_signal,
        )
    except Exception as exc:
        await _close_ws_safely(websocket, code=1011, reason=str(exc))
        return

    sentinel = object()
    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            await websocket.send_bytes(chunk)
    except WebSocketDisconnect:
        stop_signal.set()
    except Exception as exc:
        stop_signal.set()
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": str(exc)})
    finally:
        stop_signal.set()
        iterator_close = getattr(iterator, "close", None)
        if callable(iterator_close):
            await asyncio.to_thread(iterator_close)
        # END semantics: clear backend request state but keep worker alive.
        try:
            await run_in_threadpool(service.reset_backend_state, backend_name)
        except Exception:
            pass
        await _close_ws_safely(websocket)


@app.post("/v1/tts")
async def tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    speaker = request.speaker_name or request.voice or settings.default_speaker
    stream = _resolve_http_stream(request.stream, request.non_streaming_mode)

    try:
        backend_name = service.resolve_backend_name(request.backend)
        wav, sample_rate, meta = await run_in_threadpool(
            service.generate,
            backend_name,
            text,
            speaker,
            request.speed,
            stream,
            request.text_frontend,
            request.instruct,
            request.prompt_text,
            request.prompt_wav,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    wav_bytes, _ = _waveform_to_wav_bytes(wav, sample_rate=sample_rate)
    headers = {
        "X-Backend": str(meta.get("backend", backend_name)),
        "X-Generation-Time-Sec": str(meta.get("generation_time_sec", "")),
        "X-Audio-Duration-Sec": str(meta.get("audio_duration_sec", "")),
        "X-Mode": str(meta.get("mode", "")),
        "X-Speaker-Name": quote(str(meta.get("speaker_name", "")), safe=""),
        "X-Sample-Rate": str(meta.get("sample_rate", sample_rate)),
        "X-Model-Dir": str(meta.get("model_dir", "")),
        "X-Speed": str(meta.get("speed", "")),
        "X-Stream": str(meta.get("stream", stream)).lower(),
        "X-Text-Frontend": str(meta.get("text_frontend", False)).lower(),
    }
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers=headers)


@app.post("/v1/audio/speech")
async def audio_speech(request: SpeechCompatRequest):
    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty")

    requested_model = (request.model or "").strip() or settings.default_backend
    speaker = request.speaker_name or request.voice or settings.default_speaker
    stream = _resolve_http_stream(request.stream, request.non_streaming_mode)
    instruct = request.instructions or request.instruct

    try:
        backend_name = service.resolve_backend_name(request.backend, requested_model)
        wav, sample_rate, meta = await run_in_threadpool(
            service.generate,
            backend_name,
            text,
            speaker,
            request.speed,
            stream,
            request.text_frontend,
            instruct,
            request.prompt_text,
            request.prompt_wav,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    wav_bytes, _ = _waveform_to_wav_bytes(wav, sample_rate=sample_rate)
    headers = {
        "X-Backend": str(meta.get("backend", backend_name)),
        "X-Generation-Time-Sec": str(meta.get("generation_time_sec", "")),
        "X-Audio-Duration-Sec": str(meta.get("audio_duration_sec", "")),
        "X-Mode": str(meta.get("mode", "")),
        "X-Speaker-Name": quote(str(meta.get("speaker_name", "")), safe=""),
        "X-Sample-Rate": str(meta.get("sample_rate", sample_rate)),
        "X-Model-Dir": str(meta.get("model_dir", "")),
        "X-Speed": str(meta.get("speed", "")),
        "X-Stream": str(meta.get("stream", stream)).lower(),
        "X-Text-Frontend": str(meta.get("text_frontend", False)).lower(),
        "X-Requested-Model": requested_model,
    }
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers=headers)
