"""Audio buffering and preprocessing utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AudioChunk:
    """Represents a chunk of audio ready for transcription."""

    samples: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int

    @property
    def duration(self) -> float:
        """Return the duration of the chunk in seconds."""

        return float(len(self.samples) / self.sample_rate)


class AudioProcessor:
    """Incrementally consumes PCM16 audio bytes and yields uniform chunks."""

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_size: float,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self._buffer = bytearray()
        self._processed_samples = 0
        self._bytes_per_sample = 2 * channels  # pcm16
        self._chunk_bytes = int(math.ceil(self.chunk_size * self.sample_rate * self._bytes_per_sample))

    def append(self, data: bytes) -> List[AudioChunk]:
        """Append PCM16 bytes to the processor and return ready chunks."""

        if not data:
            return []

        self._buffer.extend(data)
        return self._consume_ready_chunks()

    def flush(self) -> Optional[AudioChunk]:
        """Flush any remaining samples into a final chunk."""

        if not self._buffer:
            return None

        chunk = self._buffer[:]
        self._buffer.clear()
        return self._as_chunk(chunk)

    def _consume_ready_chunks(self) -> List[AudioChunk]:
        chunks: List[AudioChunk] = []
        while len(self._buffer) >= self._chunk_bytes:
            raw = self._buffer[: self._chunk_bytes]
            del self._buffer[: self._chunk_bytes]
            chunks.append(self._as_chunk(raw))
        return chunks

    def _as_chunk(self, raw: bytes) -> AudioChunk:
        """Convert raw PCM16 bytes into an :class:`AudioChunk`."""

        if self.channels == 1:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        else:
            multi = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            samples = multi.reshape(-1, self.channels).mean(axis=1)

        samples /= np.float32(32768.0)
        sample_count = len(samples)
        start_time = self._processed_samples / self.sample_rate
        end_time = (self._processed_samples + sample_count) / self.sample_rate
        self._processed_samples += sample_count
        return AudioChunk(samples=samples, start_time=start_time, end_time=end_time, sample_rate=self.sample_rate)


__all__ = ["AudioProcessor", "AudioChunk"]
