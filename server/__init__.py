"""Deepgram-compatible transcription server using lightning-whisper-mlx."""

from .app import create_app

__all__ = ["create_app"]
