"""Async wrapper around lightning-whisper-mlx."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .config import ServerConfig

try:  # pragma: no cover - optional heavy dependency is imported lazily at runtime
    from lightning_whisper_mlx import LightningWhisperMLX, transcribe_audio
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "lightning-whisper-mlx must be installed to run the transcription server"
    ) from exc


class LightningMLXTranscriber:
    """Threaded helper that keeps a Lightning Whisper MLX model warm."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._model: LightningWhisperMLX | None = None
        self._model_ready = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._model_dir: Path | None = None

    @property
    def model_name(self) -> str:
        return self._model.name if self._model is not None else self._config.model

    @property
    def quantization(self) -> str | None:
        if self._model is not None and "-" in self._model.name:
            if self._model.name.endswith("-4-bit"):
                return "4bit"
            if self._model.name.endswith("-8-bit"):
                return "8bit"
        return self._config.quantization

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._model_ready:
            if self._model is None:
                model = LightningWhisperMLX(
                    self._config.model,
                    batch_size=self._config.batch_size,
                    quant=self._config.quantization,
                )
                self._model = model
                self._model_dir = Path("./mlx_models") / model.name

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
        initial_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run transcription on ``audio`` asynchronously."""

        if audio.ndim != 1:
            raise ValueError("Audio must be a 1-D numpy array of PCM samples")
        await self._ensure_model()
        assert self._model_dir is not None
        params: dict[str, Any] = {
            "path_or_hf_repo": str(self._model_dir),
            "language": language or self._config.default_language,
            "word_timestamps": word_timestamps,
            "initial_prompt": initial_prompt,
            "batch_size": self._config.batch_size,
        }
        params.update(kwargs)

        loop = asyncio.get_running_loop()
        func = partial(transcribe_audio, audio, **params)
        return await loop.run_in_executor(self._executor, func)

    async def aclose(self) -> None:
        """Release the underlying thread pool."""
        self._executor.shutdown(wait=False, cancel_futures=True)
