"""這個程式的功能是將一個專注於對話的聲樂音訊檔案（如 vocals.wav）分割成多個較短的訓練片段，方便後續的審核和使用。
程式會根據指定的參數來分析音訊，檢測出聲樂片段，並將它們儲存到指定的輸出目錄中，同時生成 segments.csv 和 segments.json 以記錄每個片段的相關資訊。"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_separation.io_utils import get_audio_info
from audio_separation.logging_utils import log
from audio_separation.segmenter import split_audio_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a vocals-first audio file into reviewable training segments."
    )
    parser.add_argument("--input", required=True, help="Path to vocals.wav or another audio file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where segments/, segments.csv, and segments.json will be written.",
    )
    parser.add_argument(
        "--min_duration_ms",
        type=int,
        default=800,
        help="Minimum kept segment length in milliseconds. Default: 800",
    )
    parser.add_argument(
        "--max_duration_ms",
        type=int,
        default=8000,
        help="Maximum target segment length in milliseconds. Default: 8000",
    )
    parser.add_argument(
        "--pad_ms",
        type=int,
        default=120,
        help="Padding added before and after each detected segment. Default: 120",
    )
    parser.add_argument(
        "--frame_ms",
        type=int,
        default=30,
        help="Analysis frame size in milliseconds. Default: 30",
    )
    parser.add_argument(
        "--hop_ms",
        type=int,
        default=10,
        help="Analysis hop size in milliseconds. Default: 10",
    )
    parser.add_argument(
        "--max_silence_ms",
        type=int,
        default=250,
        help="Bridge silent gaps shorter than this value. Default: 250",
    )
    parser.add_argument(
        "--threshold_db",
        type=float,
        default=None,
        help="Optional manual RMS threshold in dBFS. Default: auto",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    audio_info = get_audio_info(input_path)
    log(
        "Splitting audio. "
        f"duration={audio_info['duration_seconds']:.2f}s "
        f"sample_rate={audio_info['sample_rate']} "
        f"channels={audio_info['channels']}"
    )

    metadata = split_audio_file(
        input_path=input_path,
        output_dir=output_dir,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        pad_ms=args.pad_ms,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        max_silence_ms=args.max_silence_ms,
        threshold_db=args.threshold_db,
    )
    log(
        f"Finished splitting. segment_count={metadata['segment_count']} "
        f"threshold_db={metadata['detection']['threshold_db']}"
    )
    return 0
