"""Configuration management for the Lightning OWhisper MLX server."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for a single speech-to-text model."""

    id: str = Field(..., description="External identifier exposed in the API")
    model: str = Field(..., description="Name passed to LightningWhisperMLX")
    quantization: Optional[str] = Field(
        default=None,
        description="Optional quantization level (None, '4bit', '8bit')",
    )
    batch_size: int = Field(
        default=12, description="Batch size forwarded to LightningWhisperMLX"
    )

    @field_validator("quantization")
    @classmethod
    def _validate_quant(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in {"4bit", "8bit"}:
            raise ValueError("quantization must be either '4bit' or '8bit'")
        return value


class GeneralConfig(BaseModel):
    """General server settings."""

    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key required for incoming requests.",
    )
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=52693, description="Server port")
    sample_rate: int = Field(default=16000, description="Expected input sample rate")


class AppConfig(BaseModel):
    """Top-level configuration model."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    models: List[ModelConfig] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "AppConfig":
        """Load configuration from a YAML file."""

        data = yaml.safe_load(path.read_text())
        return cls(**data)

    def require_model(self, model_id: str) -> ModelConfig:
        """Return the model configuration matching *model_id*."""

        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(f"Model '{model_id}' is not configured")


DEFAULT_CONFIG = AppConfig(
    models=[
        ModelConfig(id="distil-medium-en", model="distil-medium.en"),
        ModelConfig(id="distil-small-en", model="distil-small.en"),
    ]
)


def load_config(path: Optional[Path]) -> AppConfig:
    """Load configuration from *path* or return :data:`DEFAULT_CONFIG`."""

    if path is None:
        return DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return AppConfig.from_file(path)
