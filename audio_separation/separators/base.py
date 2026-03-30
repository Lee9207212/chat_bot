from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseSeparator(ABC):
    """Base interface for pluggable source separation backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the backend model identifier."""

    @abstractmethod
    def separate(self, input_wav: str, output_dir: str) -> dict:
        """
        Separate the input WAV into dialogue-focused vocals and accompaniment.

        Returns a dict containing at least output file paths and model metadata.
        """
