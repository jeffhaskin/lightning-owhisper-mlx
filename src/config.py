"""Configuration utilities for the Lightning Whisper Deepgram-compatible server."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field, validator


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    host: str = Field("0.0.0.0", description="Host interface for the server")
    port: int = Field(9000, description="Port for the server to bind to")
    model: str = Field("small", description="Name of the lightning-whisper-mlx model to load")
    batch_size: int = Field(12, description="Batch size used by the transcription engine")
    quant: Optional[str] = Field(
        None,
        description="Optional quantization level to request from the whisper model (4bit or 8bit)",
    )
    chunk_size: float = Field(
        5.0,
        description="Duration, in seconds, of audio buffered before triggering a transcription run.",
    )
    sample_rate: int = Field(
        16000,
        description="Expected input sample rate of the audio stream. Incoming audio will be resampled if necessary.",
    )
    language: Optional[str] = Field(
        None,
        description="Optional language hint forwarded to the transcription engine.",
    )
    enable_word_timestamps: bool = Field(
        False,
        description="When true, attempt to approximate word-level timestamps even when the model does not expose them.",
    )

    class Config:
        env_prefix = "LIGHTNING_WHISPER_"
        env_file = ".env"

    @validator("quant")
    def _validate_quant(cls, value: Optional[str]) -> Optional[str]:  # noqa: D401 - short docstring
        """Ensure the quantization value is valid."""

        if value is None:
            return value
        normalized = value.lower()
        if normalized not in {"4bit", "8bit"}:
            raise ValueError("quant must be either '4bit' or '8bit'")
        return normalized


@lru_cache()
def get_config() -> Config:
    """Return a cached configuration instance."""

    return Config()


__all__ = ["Config", "get_config"]
