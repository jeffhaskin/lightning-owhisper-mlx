"""Conversion helpers that transform internal results into Deepgram-compatible payloads."""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List

from .models import (
    Alternative,
    ChannelResult,
    ListenV1Metadata,
    ListenV1Results,
    ListenV1SpeechStarted,
    ListenV1UtteranceEnd,
    Metadata,
    ModelInfo,
    WordTiming,
)
from .transcription_service import TranscriptionChunkResult, Word


@dataclass
class StreamState:
    """Accumulates the transcript and word history for a connection."""

    transcript: str = ""
    words: List[Word] = field(default_factory=list)

    def append(self, result: TranscriptionChunkResult) -> None:
        if result.text:
            if self.transcript:
                self.transcript = f"{self.transcript} {result.text}".strip()
            else:
                self.transcript = result.text
        if result.words:
            self.words.extend(result.words)

    def reset(self) -> None:
        self.transcript = ""
        self.words.clear()


class DeepgramAdapter:
    """Creates Deepgram-compatible payloads from transcription results."""

    def __init__(self, *, request_id: str, model_name: str, model_version: str) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.model_version = model_version
        self.metadata = Metadata(
            request_id=request_id,
            model_info=ModelInfo(name=model_name, version=model_version, arch=platform.machine()),
        )

    def build_metadata_message(self) -> ListenV1Metadata:
        return ListenV1Metadata(metadata=self.metadata)

    def build_results_message(
        self,
        *,
        chunk_result: TranscriptionChunkResult,
        state: StreamState,
        is_final: bool,
    ) -> ListenV1Results:
        transcript_text = state.transcript or chunk_result.text
        word_timings = [self._to_word_timing(word) for word in state.words] if state.words else [
            self._to_word_timing(word) for word in chunk_result.words
        ]

        alternative = Alternative(transcript=transcript_text, confidence=None, words=word_timings)
        channel = ChannelResult(alternatives=[alternative])
        return ListenV1Results(
            duration=chunk_result.end - chunk_result.start,
            start=chunk_result.start,
            is_final=is_final,
            speech_final=is_final,
            channel=channel,
            metadata=self.metadata,
        )

    def build_speech_started(self) -> ListenV1SpeechStarted:
        return ListenV1SpeechStarted(metadata=self.metadata)

    def build_utterance_end(self) -> ListenV1UtteranceEnd:
        return ListenV1UtteranceEnd(metadata=self.metadata)

    def _to_word_timing(self, word: Word) -> WordTiming:
        return WordTiming(word=word.word, start=word.start, end=word.end, confidence=word.confidence)


__all__ = ["DeepgramAdapter", "StreamState"]
