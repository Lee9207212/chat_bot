from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_separation.logging_utils import log
from audio_separation.separators.base import BaseSeparator


class DemucsSeparator(BaseSeparator):
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
        two_stems: str = "vocals",
    ) -> None:
        self._model_name = model_name
        self.device = device
        self.two_stems = two_stems

    @property
    def model_name(self) -> str:
        return self._model_name

    def separate(self, input_wav: str, output_dir: str) -> dict:
        if importlib.util.find_spec("demucs") is None:
            raise RuntimeError(
                "Demucs is not installed. Install dependencies with "
                "'pip install -r requirements.txt' first."
            )

        input_wav_path = Path(input_wav).resolve()
        final_output_dir = Path(output_dir).resolve()
        final_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="demucs_run_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            demucs_root = temp_dir / "demucs_output"
            command = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                self.model_name,
                "-d",
                self.device,
                "--two-stems",
                self.two_stems,
                "-o",
                str(demucs_root),
                str(input_wav_path),
            ]

            log(
                f"Running Demucs CLI with model='{self.model_name}' on device='{self.device}'."
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or result.stdout.strip() or "Unknown Demucs error."
                raise RuntimeError(f"Demucs separation failed: {stderr}")

            stem_root = self._find_track_output_dir(
                demucs_root=demucs_root,
                input_wav_path=input_wav_path,
            )
            vocals_source = stem_root / "vocals.wav"
            accompaniment_source = self._resolve_accompaniment_path(stem_root)

            if not vocals_source.exists():
                raise RuntimeError(f"Demucs did not produce vocals.wav in: {stem_root}")
            if not accompaniment_source.exists():
                raise RuntimeError(
                    f"Demucs did not produce accompaniment output in: {stem_root}"
                )

            vocals_output = final_output_dir / "vocals.wav"
            accompaniment_output = final_output_dir / "accompaniment.wav"
            shutil.copy2(vocals_source, vocals_output)
            shutil.copy2(accompaniment_source, accompaniment_output)

        return {
            "model_name": self.model_name,
            "separator_name": "demucs",
            "output_vocals_path": str(vocals_output),
            "output_accompaniment_path": str(accompaniment_output),
        }

    def _find_track_output_dir(self, demucs_root: Path, input_wav_path: Path) -> Path:
        expected = demucs_root / self.model_name / input_wav_path.stem
        if expected.exists():
            return expected

        candidates = [path for path in (demucs_root / self.model_name).iterdir() if path.is_dir()]
        if len(candidates) == 1:
            return candidates[0]

        raise RuntimeError(
            "Unable to locate Demucs output directory. "
            f"Expected something like: {expected}"
        )

    def _resolve_accompaniment_path(self, stem_root: Path) -> Path:
        no_vocals = stem_root / "no_vocals.wav"
        if no_vocals.exists():
            return no_vocals

        other_stems = [
            path
            for path in stem_root.glob("*.wav")
            if path.name.lower() != "vocals.wav"
        ]
        if not other_stems:
            raise RuntimeError("No accompaniment stems were found in Demucs output.")

        mixed_output = stem_root / "accompaniment.wav"
        self._mix_stems(other_stems, mixed_output)
        return mixed_output

    def _mix_stems(self, stem_paths: list[Path], output_path: Path) -> None:
        log("Demucs returned multi-stem output. Mixing non-vocal stems into accompaniment.wav.")

        mixed_audio = None
        sample_rate = None

        for stem_path in stem_paths:
            audio, current_sample_rate = sf.read(str(stem_path), always_2d=True)
            if sample_rate is None:
                sample_rate = current_sample_rate
                mixed_audio = np.zeros_like(audio, dtype=np.float32)
            elif current_sample_rate != sample_rate:
                raise RuntimeError(
                    f"Sample rate mismatch while mixing stems: {stem_path} "
                    f"uses {current_sample_rate}, expected {sample_rate}."
                )

            assert mixed_audio is not None
            if audio.shape != mixed_audio.shape:
                raise RuntimeError(
                    "Stem shape mismatch while mixing accompaniment. "
                    f"Expected {mixed_audio.shape}, got {audio.shape} from {stem_path}."
                )

            mixed_audio += audio.astype(np.float32)

        assert mixed_audio is not None
        peak = float(np.max(np.abs(mixed_audio)))
        if peak > 1.0:
            mixed_audio /= peak

        sf.write(str(output_path), mixed_audio, sample_rate)
