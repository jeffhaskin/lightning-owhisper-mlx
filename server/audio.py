"""Audio conversion helpers."""

from __future__ import annotations

import io
import wave
from typing import Iterable

import numpy as np


def pcm16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to a mono float32 numpy array."""
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    return (samples.astype(np.float32) / 32768.0).copy()


def downmix_to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """Down-mix multi-channel audio into mono."""
    if audio.ndim == 1 or channels == 1:
        return audio.astype(np.float32, copy=False)
    reshaped = audio.reshape(-1, channels)
    return reshaped.mean(axis=1, dtype=np.float32)


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Linearly resample ``audio`` from ``original_rate`` to ``target_rate``."""
    if original_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(original_rate)
    target_length = max(int(round(duration * target_rate)), 1)
    source_indices = np.arange(audio.shape[0], dtype=np.float64)
    target_indices = np.linspace(0, audio.shape[0] - 1, target_length, dtype=np.float64)
    resampled = np.interp(target_indices, source_indices, audio.astype(np.float64))
    return resampled.astype(np.float32)


def decode_audio_bytes(audio_bytes: bytes, expected_sample_rate: int) -> np.ndarray:
    """Decode WAV or raw PCM bytes into a float32 numpy array."""
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        with wave.open(io.BytesIO(audio_bytes)) as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                raise ValueError("Only 16-bit WAV files are supported")
            frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        audio = downmix_to_mono(audio, channels)
        return resample_audio(audio, sample_rate, expected_sample_rate)
    return pcm16le_to_float32(audio_bytes)


def chunk_bytes(chunks: Iterable[bytes]) -> bytes:
    """Concatenate an iterable of byte chunks."""
    return b"".join(chunks)
