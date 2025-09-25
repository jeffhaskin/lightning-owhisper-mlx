"""Configuration helpers for the transcription server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import ClassVar, Tuple



@dataclass(slots=True)
class ServerConfig:
    """Runtime configuration loaded from environment variables."""


    SUPPORTED_MODELS: ClassVar[Tuple[str, ...]] = (
        "tiny",
        "small",
        "distil-small.en",
        "base",
        "medium",
        "distil-medium.en",
        "large",
        "large-v2",
        "distil-large-v2",
        "large-v3",
        "distil-large-v3",
    )


    model: str = os.getenv("LWM_MODEL", "small")
    quantization: str | None = (
        os.getenv("LWM_QUANTIZATION") or os.getenv("LWM_QUANT") or None
    )
    batch_size: int = int(os.getenv("LWM_BATCH_SIZE", "6"))
    sample_rate: int = int(os.getenv("LWM_SAMPLE_RATE", "16000"))
    stream_window_seconds: float = float(
        os.getenv("LWM_STREAM_WINDOW_SECONDS", os.getenv("LWM_STREAM_WINDOW", "2.5"))
    )
    max_workers: int = int(os.getenv("LWM_MAX_WORKERS", "1"))
    default_language: str | None = os.getenv("LWM_DEFAULT_LANGUAGE")
    def __post_init__(self) -> None:
        if self.quantization == "":
            self.quantization = None
        if self.quantization is not None and self.quantization not in {"4bit", "8bit"}:
            raise ValueError(
                "LWM_QUANTIZATION must be empty, '4bit', or '8bit' if provided."
            )

        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(
                "LWM_MODEL must be one of: " + ", ".join(self.SUPPORTED_MODELS)
            )

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")
        if self.stream_window_seconds <= 0:
            raise ValueError("Stream window must be positive")


DEFAULT_CONFIG = ServerConfig()
