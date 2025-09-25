"""WebSocket streaming session management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
import uuid

import numpy as np

from .audio import pcm16le_to_float32, resample_audio
from .config import ServerConfig
from .responses import build_streaming_result
from .transcriber import LightningMLXTranscriber

StreamCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class StreamingState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    utterance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sample_rate: int = 16000
    language: str | None = None
    new_samples_threshold: int = 16000
    accumulated_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    new_samples: int = 0
    segments_sent: int = 0


class StreamingSession:
    """Handle chunked streaming transcription using a worker task."""

    def __init__(
        self,
        *,
        transcriber: LightningMLXTranscriber,
        config: ServerConfig,
        send: StreamCallback,
    ) -> None:
        self._transcriber = transcriber
        self._config = config
        self._send = send
        self._state = StreamingState(
            sample_rate=config.sample_rate,
            new_samples_threshold=int(config.sample_rate * config.stream_window_seconds),
        )
        self._queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._closed = False

    @property
    def state(self) -> StreamingState:
        return self._state

    @property
    def closed(self) -> bool:
        return self._closed

    async def configure(self, *, sample_rate: int | None = None, language: str | None = None) -> None:
        if sample_rate:
            self._state.sample_rate = sample_rate
        self._state.new_samples_threshold = int(
            self._config.sample_rate * self._config.stream_window_seconds
        )
        if language:
            self._state.language = language

    async def enqueue_audio(self, chunk: bytes) -> None:
        if self._closed:
            return
        await self._queue.put(("audio", chunk))

    async def finalize(self) -> None:
        if self._closed:
            return
        await self._queue.put(("finalize", None))
        await self._queue.join()
        await self._worker_task
        self._closed = True

    async def abort(self) -> None:
        if self._closed:
            return
        await self._queue.put(("abort", None))
        await self._worker_task
        self._closed = True

    async def _worker_loop(self) -> None:
        while True:
            msg_type, payload = await self._queue.get()
            try:
                if msg_type == "audio" and payload is not None:
                    await self._handle_audio_chunk(payload)
                    if (
                        self._state.new_samples >= self._state.new_samples_threshold
                        and self._state.new_samples > 0
                    ):
                        await self._transcribe(is_final=False)
                elif msg_type == "finalize":
                    await self._transcribe(is_final=True)
                    break
                elif msg_type == "abort":
                    break
            finally:
                self._queue.task_done()
        self._closed = True

    async def _handle_audio_chunk(self, chunk: bytes) -> None:
        samples = pcm16le_to_float32(chunk)
        if samples.size == 0:
            return
        state = self._state
        if state.sample_rate != self._config.sample_rate:
            samples = resample_audio(samples, state.sample_rate, self._config.sample_rate)
        state.accumulated_audio = (
            samples
            if state.accumulated_audio.size == 0
            else np.concatenate([state.accumulated_audio, samples])
        )
        state.new_samples += samples.size

    async def _transcribe(self, *, is_final: bool) -> None:
        state = self._state
        if state.accumulated_audio.size == 0:
            if is_final:
                await self._send(
                    build_streaming_result(
                        [],
                        config=self._config,
                        request_id=state.session_id,
                        utterance_id=state.utterance_id,
                        is_final=True,
                        start_offset=0.0,
                    )
                )
            state.new_samples = 0
            return

        audio = state.accumulated_audio.copy()
        state.new_samples = 0
        result = await self._transcriber.transcribe(
            audio,
            language=state.language,
            word_timestamps=True,
        )
        segments = result.get("segments") or []
        new_segments = segments[state.segments_sent :]
        if not new_segments and is_final and segments:
            # Deepgram clients expect the final transcript to include the
            # complete utterance. If we have already emitted all of the
            # segments as interim results, resend the full list so the
            # closing frame carries the transcript instead of an empty
            # payload.
            new_segments = segments
        state.segments_sent = len(segments)
        if not new_segments and not is_final:
            return
        start_offset = (
            new_segments[0]["start"]
            if new_segments
            else (segments[-1]["end"] if segments else 0.0)
        )
        payload = build_streaming_result(
            new_segments,
            config=self._config,
            request_id=state.session_id,
            utterance_id=state.utterance_id,
            is_final=is_final,
            start_offset=start_offset,
        )
        await self._send(payload)
