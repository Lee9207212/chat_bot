'''這個程式的功能是從影片或音訊檔案中分離出專注於對話的聲樂部分。
它支援多種輸入格式（如 .mp4、.mkv、.wav、.mp3），並使用 Demucs 分離器來處理聲音分離。程式會將分離出的聲樂和伴奏分別儲存為 vocals.wav 和 accompaniment.wav，並在指定的輸出目錄中生成一個 metadata.json 文件，記錄處理過程中的相關資訊和結果。
使用者可以透過命令列參數來指定輸入檔案、輸出目錄、分離器模型、執行裝置、取樣率和聲道數等選項。'''

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from audio_separation.io_utils import (
    ensure_output_dir,
    get_audio_info,
    prepare_input_wav,
    validate_input_path,
    write_metadata,
)
from audio_separation.logging_utils import log
from audio_separation.separators import DemucsSeparator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate dialogue-focused vocals from video or audio input."
    )
    parser.add_argument("--input", required=True, help="Path to input .mp4/.mkv/.wav/.mp3 file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write vocals.wav, accompaniment.wav, and metadata.json.",
    )
    parser.add_argument(
        "--separator",
        default="demucs",
        choices=["demucs"],
        help="Separator backend. Default: demucs",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Model name for the selected separator. Default: htdemucs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device for Demucs CLI. Default: cpu",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Target sample rate for the prepared WAV input. Default: 44100",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Target number of channels for the prepared WAV input. Default: 2",
    )
    return parser


def create_separator(name: str, model_name: str, device: str):
    if name == "demucs":
        return DemucsSeparator(model_name=model_name, device=device)
    raise ValueError(f"Unsupported separator backend: {name}")


def build_initial_metadata(args: argparse.Namespace, input_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "extracted_wav_path": None,
        "output_vocals_path": str(output_dir / "vocals.wav"),
        "output_accompaniment_path": str(output_dir / "accompaniment.wav"),
        "model_name": args.model,
        "sample_rate": None,
        "duration_seconds": None,
        "processing_time_seconds": None,
        "success": False,
        "error_message": None,
        "separator": args.separator,
        "command_args": {
            "input": args.input,
            "output_dir": args.output_dir,
            "separator": args.separator,
            "model": args.model,
            "device": args.device,
            "sample_rate": args.sample_rate,
            "channels": args.channels,
        },
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_path = output_dir / "metadata.json"
    ensure_output_dir(output_dir)
    metadata = build_initial_metadata(args, input_path, output_dir)

    start_time = time.perf_counter()
    temp_dir_path: Path | None = None

    try:
        log(f"Validating input: {input_path}")
        validate_input_path(input_path)

        separator = create_separator(args.separator, args.model, args.device)

        with tempfile.TemporaryDirectory(prefix="audio_separation_") as temp_dir_str:
            temp_dir_path = Path(temp_dir_str)
            prepared_wav_path = prepare_input_wav(
                input_path=input_path,
                temp_dir=temp_dir_path,
                sample_rate=args.sample_rate,
                channels=args.channels,
            )
            metadata["extracted_wav_path"] = str(prepared_wav_path)

            audio_info = get_audio_info(prepared_wav_path)
            metadata["sample_rate"] = audio_info["sample_rate"]
            metadata["duration_seconds"] = audio_info["duration_seconds"]

            log("Starting source separation.")
            separation_result = separator.separate(
                input_wav=str(prepared_wav_path),
                output_dir=str(output_dir),
            )

            metadata["model_name"] = separation_result["model_name"]
            metadata["output_vocals_path"] = separation_result["output_vocals_path"]
            metadata["output_accompaniment_path"] = separation_result["output_accompaniment_path"]
            metadata["success"] = True
            metadata["temporary_files_cleaned"] = True

        elapsed = time.perf_counter() - start_time
        metadata["processing_time_seconds"] = round(elapsed, 3)
        write_metadata(metadata, metadata_path)
        log(f"Finished successfully. Outputs written to: {output_dir}")
        return 0

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        metadata["processing_time_seconds"] = round(elapsed, 3)
        metadata["error_message"] = str(exc)
        if temp_dir_path is not None:
            metadata["temporary_files_cleaned"] = not temp_dir_path.exists()
        else:
            metadata["temporary_files_cleaned"] = True
        write_metadata(metadata, metadata_path)
        log(f"Failed: {exc}")
        return 1
