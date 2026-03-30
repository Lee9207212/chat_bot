"""這個程式的功能是讓使用者逐一審核分割後的 WAV 檔案，並將審核結果（如 yes、no、maybe）寫入 segments.csv 中對應的行。
使用者可以透過按鍵來標記每個片段的狀態，並且可以選擇從特定的檔案開始審核，或是從未標記過的第一個檔案開始。
程式會在審核過程中即時更新 CSV 文件，以確保資料的完整性和持久性。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import winsound
from pathlib import Path

import msvcrt
import soundfile as sf

from audio_separation.logging_utils import log

LABEL_MAP = {
    "y": "yes",
    "n": "no",
    "m": "maybe",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review segmented WAV files one by one and write labels into segments.csv."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Segment review folder containing segments/ and segments.csv.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional starting WAV file name, for example 0042.wav.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path = output_dir / "segments.csv"
    segments_dir = output_dir / "segments"

    if not csv_path.exists():
        raise FileNotFoundError(f"segments.csv not found: {csv_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"segments directory not found: {segments_dir}")

    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No segment rows found in: {csv_path}")

    start_index = _resolve_start_index(rows, args.start)
    reviewed_count = 0

    print("Controls: y=yes n=no m=maybe r=replay q=quit")
    print()

    for index in range(start_index, len(rows)):
        row = rows[index]
        file_name = row["file_name"]
        wav_path = segments_dir / file_name
        if not wav_path.exists():
            raise FileNotFoundError(f"Segment file not found: {wav_path}")

        duration_seconds = _resolve_duration_seconds(row, wav_path)
        label = _review_one(
            wav_path=wav_path,
            file_name=file_name,
            index=index + 1,
            total=len(rows),
            duration_seconds=duration_seconds,
        )
        if label is None:
            _stop_audio()
            _write_rows(csv_path, rows)
            log(f"Stopped review. reviewed_in_this_run={reviewed_count}")
            return 0

        row["keep"] = label
        reviewed_count += 1
        _write_rows(csv_path, rows)

    _stop_audio()
    log(f"Finished review. reviewed_in_this_run={reviewed_count}")
    return 0


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_start_index(rows: list[dict[str, str]], start_file_name: str | None) -> int:
    if start_file_name is None:
        for index, row in enumerate(rows):
            if not row.get("keep", "").strip():
                return index
        return 0

    target = start_file_name.strip().lower()
    for index, row in enumerate(rows):
        if row["file_name"].strip().lower() == target:
            return index
    raise ValueError(f"Start file not found in segments.csv: {start_file_name}")


def _review_one(
    *,
    wav_path: Path,
    file_name: str,
    index: int,
    total: int,
    duration_seconds: float,
) -> str | None:
    while True:
        _drain_key_buffer()
        print(f"[{index}/{total}] Playing {file_name} ({duration_seconds:.2f}s)")
        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key in LABEL_MAP:
                    _stop_audio()
                    print(f"Marked {file_name} -> {LABEL_MAP[key]}")
                    print()
                    return LABEL_MAP[key]
                if key == "r":
                    _stop_audio()
                    print(f"Replay {file_name}")
                    print()
                    break
                if key == "q":
                    print()
                    return None
            time.sleep(0.03)


def _drain_key_buffer() -> None:
    while msvcrt.kbhit():
        msvcrt.getwch()


def _resolve_duration_seconds(row: dict[str, str], wav_path: Path) -> float:
    raw_value = row.get("duration_seconds", "").strip()
    if raw_value:
        try:
            return float(raw_value)
        except ValueError:
            pass

    info = sf.info(str(wav_path))
    return float(info.duration)


def _stop_audio() -> None:
    winsound.PlaySound(None, winsound.SND_PURGE)


if __name__ == "__main__":
    raise SystemExit(main())
