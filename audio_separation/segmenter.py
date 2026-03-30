from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_separation.io_utils import ensure_output_dir
from audio_separation.logging_utils import log


@dataclass
class Segment:
    segment_id: int
    file_name: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    keep: str = ""
    note: str = ""


def split_audio_file(
    input_path: Path,
    output_dir: Path,
    *,
    min_duration_ms: int = 800,
    max_duration_ms: int = 8000,
    pad_ms: int = 120,
    frame_ms: int = 30,
    hop_ms: int = 10,
    max_silence_ms: int = 250,
    threshold_db: float | None = None,
) -> dict[str, Any]:
    ensure_output_dir(output_dir)
    segments_dir = output_dir / "segments"
    ensure_output_dir(segments_dir)

    audio, sample_rate = sf.read(str(input_path), always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Input audio is empty: {input_path}")

    mono = audio.mean(axis=1, dtype=np.float32)
    frame_length = _ms_to_samples(frame_ms, sample_rate)
    hop_length = _ms_to_samples(hop_ms, sample_rate)
    min_duration_samples = _ms_to_samples(min_duration_ms, sample_rate)
    max_duration_samples = _ms_to_samples(max_duration_ms, sample_rate)
    pad_samples = _ms_to_samples(pad_ms, sample_rate)
    max_silence_frames = max(1, round(max_silence_ms / hop_ms))

    frame_db = _frame_rms_db(mono, frame_length, hop_length)
    actual_threshold_db = _resolve_threshold_db(frame_db, threshold_db)
    speech_mask = frame_db >= actual_threshold_db
    speech_mask = _bridge_short_silences(speech_mask, max_silence_frames)

    raw_segments = _mask_to_segments(
        speech_mask=speech_mask,
        hop_length=hop_length,
        frame_length=frame_length,
        total_samples=len(mono),
    )
    padded_segments = _apply_padding(raw_segments, pad_samples, len(mono))
    merged_segments = _merge_overlapping_segments(padded_segments)
    bounded_segments = _split_long_segments(
        mono=mono,
        segments=merged_segments,
        max_duration_samples=max_duration_samples,
        min_duration_samples=min_duration_samples,
        hop_length=hop_length,
    )
    final_segments = [
        (start, end)
        for start, end in bounded_segments
        if end - start >= min_duration_samples
    ]

    metadata_segments: list[Segment] = []
    for index, (start, end) in enumerate(final_segments, start=1):
        segment_audio = audio[start:end]
        file_name = f"{index:04d}.wav"
        output_path = segments_dir / file_name
        sf.write(str(output_path), segment_audio, sample_rate, subtype="PCM_16")

        peak_dbfs, rms_dbfs = _segment_levels(segment_audio)
        metadata_segments.append(
            Segment(
                segment_id=index,
                file_name=file_name,
                start_seconds=round(start / sample_rate, 3),
                end_seconds=round(end / sample_rate, 3),
                duration_seconds=round((end - start) / sample_rate, 3),
                peak_dbfs=round(peak_dbfs, 2),
                rms_dbfs=round(rms_dbfs, 2),
            )
        )

    metadata = {
        "input_path": str(input_path.resolve()),
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "segment_count": len(metadata_segments),
        "detection": {
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "threshold_db": round(actual_threshold_db, 2),
            "max_silence_ms": max_silence_ms,
            "min_duration_ms": min_duration_ms,
            "max_duration_ms": max_duration_ms,
            "pad_ms": pad_ms,
        },
        "segments": [asdict(segment) for segment in metadata_segments],
    }

    _write_segments_json(output_dir / "segments.json", metadata)
    _write_segments_csv(output_dir / "segments.csv", metadata_segments)
    log(f"Wrote {len(metadata_segments)} segments to: {segments_dir}")
    return metadata


def _ms_to_samples(milliseconds: int, sample_rate: int) -> int:
    return max(1, round(sample_rate * milliseconds / 1000))


def _frame_rms_db(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(audio) <= frame_length:
        rms = np.sqrt(np.mean(np.square(audio), dtype=np.float64))
        return np.array([_safe_db(rms)], dtype=np.float32)

    values: list[float] = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start : start + frame_length]
        rms = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
        values.append(_safe_db(rms))
    return np.array(values, dtype=np.float32)


def _resolve_threshold_db(frame_db: np.ndarray, threshold_db: float | None) -> float:
    if threshold_db is not None:
        return float(threshold_db)

    speech_reference_db = float(np.percentile(frame_db, 95))
    noise_reference_db = float(np.percentile(frame_db, 20))
    adaptive_threshold = speech_reference_db - 24.0
    return max(noise_reference_db + 6.0, adaptive_threshold, -55.0)


def _bridge_short_silences(mask: np.ndarray, max_silence_frames: int) -> np.ndarray:
    if len(mask) == 0:
        return mask

    bridged = mask.copy()
    start = 0
    while start < len(mask):
        if bridged[start]:
            start += 1
            continue

        end = start
        while end < len(mask) and not bridged[end]:
            end += 1

        if start > 0 and end < len(mask) and (end - start) <= max_silence_frames:
            bridged[start:end] = True
        start = end

    return bridged


def _mask_to_segments(
    *,
    speech_mask: np.ndarray,
    hop_length: int,
    frame_length: int,
    total_samples: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_index: int | None = None

    for frame_index, is_speech in enumerate(speech_mask):
        if is_speech and start_index is None:
            start_index = frame_index
        elif not is_speech and start_index is not None:
            start_sample = start_index * hop_length
            end_sample = min(total_samples, frame_index * hop_length + frame_length)
            segments.append((start_sample, end_sample))
            start_index = None

    if start_index is not None:
        start_sample = start_index * hop_length
        segments.append((start_sample, total_samples))

    return segments


def _apply_padding(
    segments: list[tuple[int, int]],
    pad_samples: int,
    total_samples: int,
) -> list[tuple[int, int]]:
    return [
        (max(0, start - pad_samples), min(total_samples, end + pad_samples))
        for start, end in segments
    ]


def _merge_overlapping_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_long_segments(
    *,
    mono: np.ndarray,
    segments: list[tuple[int, int]],
    max_duration_samples: int,
    min_duration_samples: int,
    hop_length: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in segments:
        result.extend(
            _split_single_segment(
                mono=mono,
                start=start,
                end=end,
                max_duration_samples=max_duration_samples,
                min_duration_samples=min_duration_samples,
                hop_length=hop_length,
            )
        )
    return result


def _split_single_segment(
    *,
    mono: np.ndarray,
    start: int,
    end: int,
    max_duration_samples: int,
    min_duration_samples: int,
    hop_length: int,
) -> list[tuple[int, int]]:
    duration = end - start
    if duration <= max_duration_samples:
        return [(start, end)]

    midpoint = start + max_duration_samples
    search_radius = max(hop_length, max_duration_samples // 5)
    search_start = max(start + min_duration_samples, midpoint - search_radius)
    search_end = min(end - min_duration_samples, midpoint + search_radius)
    if search_end <= search_start:
        return _hard_split_segment(start, end, max_duration_samples, min_duration_samples)

    split_at = _find_low_energy_split(mono, search_start, search_end, hop_length)
    if split_at - start < min_duration_samples or end - split_at < min_duration_samples:
        return _hard_split_segment(start, end, max_duration_samples, min_duration_samples)

    return _split_single_segment(
        mono=mono,
        start=start,
        end=split_at,
        max_duration_samples=max_duration_samples,
        min_duration_samples=min_duration_samples,
        hop_length=hop_length,
    ) + _split_single_segment(
        mono=mono,
        start=split_at,
        end=end,
        max_duration_samples=max_duration_samples,
        min_duration_samples=min_duration_samples,
        hop_length=hop_length,
    )


def _find_low_energy_split(
    mono: np.ndarray,
    search_start: int,
    search_end: int,
    hop_length: int,
) -> int:
    best_index = search_start
    best_energy = float("inf")

    for index in range(search_start, search_end, hop_length):
        left = max(0, index - hop_length)
        right = min(len(mono), index + hop_length)
        window = mono[left:right]
        if len(window) == 0:
            continue
        energy = float(np.mean(np.abs(window), dtype=np.float64))
        if energy < best_energy:
            best_energy = energy
            best_index = index

    return best_index


def _hard_split_segment(
    start: int,
    end: int,
    max_duration_samples: int,
    min_duration_samples: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_duration_samples:
        next_cursor = cursor + max_duration_samples
        if end - next_cursor < min_duration_samples:
            break
        result.append((cursor, next_cursor))
        cursor = next_cursor
    result.append((cursor, end))
    return result


def _segment_levels(segment_audio: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(segment_audio)))
    rms = float(np.sqrt(np.mean(np.square(segment_audio), dtype=np.float64)))
    return _safe_db(peak), _safe_db(rms)


def _safe_db(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-8))


def _write_segments_json(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_segments_csv(path: Path, segments: list[Segment]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_id",
                "file_name",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "peak_dbfs",
                "rms_dbfs",
                "keep",
                "note",
            ],
        )
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))
