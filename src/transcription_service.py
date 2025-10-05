"""Integration with lightning-whisper-mlx for audio transcription."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import threading

import numpy as np

try:  # pragma: no cover - optional dependency runtime import
    from importlib.metadata import PackageNotFoundError, version
except ModuleNotFoundError:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency runtime import
    from lightning_whisper_mlx import LightningWhisperMLX
    from lightning_whisper_mlx.audio import HOP_LENGTH, SAMPLE_RATE as MODEL_SAMPLE_RATE
    from lightning_whisper_mlx.transcribe import transcribe_audio
except Exception:  # pragma: no cover - handled gracefully
    LightningWhisperMLX = None  # type: ignore[assignment]
    transcribe_audio = None  # type: ignore[assignment]
    HOP_LENGTH = 160
    MODEL_SAMPLE_RATE = 16000

try:  # pragma: no cover - optional dependency
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover
    resample_poly = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass
class Word:
    """Internal representation of recognised words."""

    word: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionChunkResult:
    """Structured output from the transcription service."""

    text: str
    start: float
    end: float
    words: List[Word]
    language: Optional[str]
    is_speech: bool


class TranscriptionService:
    """Service that wraps lightning-whisper-mlx for audio transcription."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        quant: Optional[str],
        language: Optional[str],
        enable_word_timestamps: bool,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.quant = quant
        self.language = language
        self.enable_word_timestamps = enable_word_timestamps
        self._model_wrapper: Optional[LightningWhisperMLX] = None
        self._model_path: Optional[Path] = None
        self._load_lock = threading.Lock()

    @property
    def model_version(self) -> str:
        """Return the installed lightning-whisper-mlx version if available."""

        try:
            return version("lightning-whisper-mlx")
        except PackageNotFoundError:  # pragma: no cover - optional dependency
            return "unknown"

    async def transcribe_chunk(
        self,
        chunk: np.ndarray,
        *,
        start: float,
        end: float,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionChunkResult:
        """Transcribe an audio chunk asynchronously."""

        return await asyncio.to_thread(
            self._transcribe_chunk_sync,
            chunk,
            start,
            end,
            sample_rate,
            language,
        )

    def _transcribe_chunk_sync(
        self,
        samples: np.ndarray,
        start: float,
        end: float,
        sample_rate: int,
        language: Optional[str],
    ) -> TranscriptionChunkResult:
        if samples.size == 0:
            return TranscriptionChunkResult(
                text="",
                start=start,
                end=end,
                words=[],
                language=self.language,
                is_speech=False,
            )

        waveform = self._resample(samples, source_rate=sample_rate)
        energy = float(np.sqrt(np.mean(np.square(waveform))))
        speech_detected = bool(waveform.size and energy > 0.0005)

        model_result = self._run_inference(waveform, language_override=language)

        text = model_result.get("text", "").strip() if isinstance(model_result, dict) else ""
        detected_language = model_result.get("language") if isinstance(model_result, dict) else None
        segments = model_result.get("segments", []) if isinstance(model_result, dict) else []

        words = self._segments_to_words(segments, start_offset=start)

        if self.enable_word_timestamps and not words and text:
            words = self._approximate_words(text, start, end)

        if not text and words:
            text = " ".join(word.word for word in words)

        return TranscriptionChunkResult(
            text=text,
            start=start,
            end=end,
            words=words,
            language=detected_language or language or self.language,
            is_speech=speech_detected or bool(text),
        )

    def _resample(self, samples: np.ndarray, *, source_rate: int) -> np.ndarray:
        """Resample incoming audio to the model's expected sample rate."""

        if source_rate == MODEL_SAMPLE_RATE:
            return samples.astype(np.float32)

        if resample_poly is None:
            LOGGER.warning(
                "scipy is not available; falling back to naive interpolation for resampling from %s to %s",
                source_rate,
                MODEL_SAMPLE_RATE,
            )
            duration = samples.shape[0] / float(source_rate)
            target_length = int(duration * MODEL_SAMPLE_RATE)
            if target_length <= 0:
                return samples.astype(np.float32)
            x_old = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
            x_new = np.linspace(0.0, duration, num=target_length, endpoint=False)
            return np.interp(x_new, x_old, samples).astype(np.float32)

        gcd = math.gcd(source_rate, MODEL_SAMPLE_RATE)
        up = MODEL_SAMPLE_RATE // gcd
        down = source_rate // gcd
        resampled = resample_poly(samples, up, down)
        return resampled.astype(np.float32)

    def _run_inference(self, waveform: np.ndarray, *, language_override: Optional[str]) -> dict:
        if LightningWhisperMLX is None or transcribe_audio is None:
            raise RuntimeError(
                "lightning-whisper-mlx is not installed. Please install the package to enable transcription."
            )

        if self._model_wrapper is None:
            with self._load_lock:
                if self._model_wrapper is None:
                    self._model_wrapper = LightningWhisperMLX(self.model_name, batch_size=self.batch_size, quant=self.quant)
                    model_dir = Path("./mlx_models") / getattr(self._model_wrapper, "name", self.model_name)
                    self._model_path = model_dir

        assert self._model_path is not None  # for type checkers

        kwargs = {
            "audio": waveform,
            "path_or_hf_repo": str(self._model_path),
            "batch_size": self.batch_size,
        }
        chosen_language = language_override or self.language
        if chosen_language:
            kwargs["language"] = chosen_language

        return transcribe_audio(**kwargs)

    def _segments_to_words(self, segments: List[list], *, start_offset: float) -> List[Word]:
        words: List[Word] = []
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) < 3:
                continue
            start_seek, end_seek, text = segment[0], segment[1], segment[2]
            if not text:
                continue
            start = start_offset + float(start_seek * HOP_LENGTH / MODEL_SAMPLE_RATE)
            end = start_offset + float(end_seek * HOP_LENGTH / MODEL_SAMPLE_RATE)
            words.extend(self._approximate_words(str(text), start, end))
        return words

    def _approximate_words(self, text: str, start: float, end: float) -> List[Word]:
        tokens = [token for token in text.split() if token]
        if not tokens:
            return []
        duration = max(end - start, 0.0)
        interval = duration / max(len(tokens), 1)
        words: List[Word] = []
        for index, token in enumerate(tokens):
            word_start = start + index * interval
            word_end = word_start + interval
            words.append(Word(word=token, start=word_start, end=word_end if interval else word_start))
        return words


__all__ = ["TranscriptionService", "TranscriptionChunkResult", "Word"]
