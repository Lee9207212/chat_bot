from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf

from audio_separation.ffmpeg_utils import ensure_ffmpeg_available, run_ffmpeg_to_wav
from audio_separation.logging_utils import log

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def validate_input_path(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported input format. Supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_input_wav(
    input_path: Path,
    temp_dir: Path,
    sample_rate: int,
    channels: int,
) -> Path:
    ensure_ffmpeg_available()
    prepared_wav_path = temp_dir / "prepared_input.wav"

    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        log("Input detected as video. Extracting audio track with ffmpeg.")
    else:
        log("Input detected as audio. Converting to normalized WAV with ffmpeg.")

    run_ffmpeg_to_wav(
        input_path=input_path,
        output_wav_path=prepared_wav_path,
        sample_rate=sample_rate,
        channels=channels,
    )
    return prepared_wav_path


def get_audio_info(audio_path: Path) -> dict[str, Any]:
    info = sf.info(str(audio_path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
        "frames": int(info.frames),
        "format": info.format,
        "subtype": info.subtype,
    }


def write_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
