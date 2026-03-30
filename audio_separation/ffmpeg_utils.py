from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is required but not installed."""


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg was not found in PATH. Please install ffmpeg and ensure "
            "'ffmpeg -version' works in your terminal before running this tool."
        )


def run_ffmpeg_to_wav(
    input_path: Path,
    output_wav_path: Path,
    sample_rate: int,
    channels: int,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_wav_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown ffmpeg error."
        raise RuntimeError(f"ffmpeg failed while converting input to WAV: {stderr}")
