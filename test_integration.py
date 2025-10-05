"""Lightweight integration checks for core components."""
from __future__ import annotations

import numpy as np

from src.audio_processor import AudioProcessor
from src.deepgram_adapter import DeepgramAdapter, StreamState
from src.transcription_service import TranscriptionChunkResult, Word


def run_basic_checks() -> None:
    processor = AudioProcessor(sample_rate=16000, chunk_size=0.5)
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, endpoint=False)) * 32767).astype(np.int16)
    payload = samples.tobytes()
    chunks = processor.append(payload)
    assert chunks, "Expected at least one chunk from the audio processor"

    result = TranscriptionChunkResult(
        text="hello world",
        start=0.0,
        end=0.5,
        words=[Word(word="hello", start=0.0, end=0.25), Word(word="world", start=0.25, end=0.5)],
        language="en",
        is_speech=True,
    )
    state = StreamState()
    state.append(result)
    adapter = DeepgramAdapter(request_id="test", model_name="tiny", model_version="0.0.0")
    payload = adapter.build_results_message(chunk_result=result, state=state, is_final=True)
    assert payload.type == "results"
    assert payload.channel.alternatives[0].transcript == "hello world"


if __name__ == "__main__":
    run_basic_checks()
    print("Integration checks passed.")
