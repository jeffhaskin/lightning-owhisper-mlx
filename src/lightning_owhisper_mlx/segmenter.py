"""Audio segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AudioSegment:
    """A contiguous block of audio samples."""

    samples: np.ndarray
    start_time: float
    end_time: float
    channel_index: int


@dataclass
class Segmenter:
    """Simple RMS-based speech segmenter."""

    sample_rate: int
    redemption_time: float
    channel_index: int
    energy_threshold: float = 0.01
    max_buffer_duration: float = 30.0
    _buffer: List[np.ndarray] = field(default_factory=list, init=False)
    _segment_active: bool = field(default=False, init=False)
    _segment_start: float = field(default=0.0, init=False)
    _silence_duration: float = field(default=0.0, init=False)
    _stream_time: float = field(default=0.0, init=False)

    def submit(self, chunk: np.ndarray) -> List[AudioSegment]:
        """Process a chunk and return completed segments if any."""

        if chunk.ndim != 1:
            raise ValueError("audio chunk must be one-dimensional")

        duration = len(chunk) / float(self.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        segments: List[AudioSegment] = []

        if rms >= self.energy_threshold:
            if not self._segment_active:
                self._segment_active = True
                self._segment_start = self._stream_time
                self._buffer.clear()
            self._buffer.append(chunk)
            self._silence_duration = 0.0
        elif self._segment_active:
            self._buffer.append(chunk)
            self._silence_duration += duration
            if self._silence_duration >= self.redemption_time:
                segments.append(self._finalize_segment())

        if self._segment_active and self._segment_duration() > self.max_buffer_duration:
            segments.append(self._finalize_segment())

        self._stream_time += duration
        return [seg for seg in segments if seg.samples.size > 0]

    def flush(self) -> List[AudioSegment]:
        """Return the final segment if audio is still buffered."""

        if self._segment_active and self._buffer:
            return [self._finalize_segment()]
        return []

    def _segment_duration(self) -> float:
        total_samples = sum(len(chunk) for chunk in self._buffer)
        return total_samples / float(self.sample_rate)

    def _finalize_segment(self) -> AudioSegment:
        if not self._buffer:
            self._reset()
            return AudioSegment(np.array([], dtype=np.float32), self._segment_start, self._stream_time, self.channel_index)

        audio = np.concatenate(self._buffer)

        if self._silence_duration > 0:
            trim_samples = int(self._silence_duration * self.sample_rate)
            if 0 < trim_samples < audio.size:
                audio = audio[:-trim_samples]

        end_time = self._segment_start + (audio.size / float(self.sample_rate))

        segment = AudioSegment(audio, self._segment_start, end_time, self.channel_index)
        self._reset()
        return segment

    def _reset(self) -> None:
        self._buffer.clear()
        self._segment_active = False
        self._segment_start = self._stream_time
        self._silence_duration = 0.0
