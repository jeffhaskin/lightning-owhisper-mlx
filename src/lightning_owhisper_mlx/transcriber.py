"""Utilities for executing lightning-whisper-mlx transcriptions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import anyio
import numpy as np

from .config import ModelConfig

LOGGER = logging.getLogger(__name__)

try:
    from lightning_whisper_mlx import LightningWhisperMLX  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    LightningWhisperMLX = None  # type: ignore


@dataclass
class TranscriptionResult:
    """Container for transcription output."""

    text: str
    words: list[dict]
    language: Optional[str]


class LightningModelWrapper:
    """Lazy wrapper around :class:`LightningWhisperMLX`."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._model: Optional[LightningWhisperMLX] = None

    def _ensure_model(self) -> LightningWhisperMLX:
        if LightningWhisperMLX is None:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "lightning-whisper-mlx is not installed or not available on this platform"
            )
        if self._model is None:
            LOGGER.info(
                "loading lightning-whisper-mlx model %s (quant=%s, batch=%s)",
                self._config.model,
                self._config.quantization,
                self._config.batch_size,
            )
            self._model = LightningWhisperMLX(
                model=self._config.model,
                batch_size=self._config.batch_size,
                quant=self._config.quantization,
            )
        return self._model

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> TranscriptionResult:
        model = self._ensure_model()
        result = model.transcribe(audio_path=audio, language=language)
        text = result.get("text", "")
        segments = result.get("segments", []) or []

        words: list[dict] = []
        for segment in segments:
            if "words" in segment and segment["words"]:
                words.extend(segment["words"])
            else:
                words.append(
                    {
                        "word": segment.get("text", "").strip(),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", segment.get("start", 0.0)),
                        "confidence": segment.get("avg_logprob", 0.0),
                    }
                )

        language = result.get("language")
        return TranscriptionResult(text=text, words=words, language=language)


class TranscriberCache:
    """Cache of instantiated :class:`LightningWhisperMLX` models."""

    def __init__(self):
        self._cache: Dict[str, LightningModelWrapper] = {}

    def get(self, config: ModelConfig) -> LightningModelWrapper:
        if config.id not in self._cache:
            self._cache[config.id] = LightningModelWrapper(config)
        return self._cache[config.id]


class TranscriberService:
    """Service responsible for executing transcriptions asynchronously."""

    def __init__(self, cache: Optional[TranscriberCache] = None):
        self._cache = cache or TranscriberCache()

    async def transcribe(
        self, config: ModelConfig, audio: np.ndarray, language: Optional[str]
    ) -> TranscriptionResult:
        wrapper = self._cache.get(config)
        return await anyio.to_thread.run_sync(wrapper.transcribe, audio, language)
