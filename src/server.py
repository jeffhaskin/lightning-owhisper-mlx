"""FastAPI based WebSocket server that mimics the Deepgram streaming API."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from .audio_processor import AudioProcessor
from .config import get_config
from .deepgram_adapter import DeepgramAdapter, StreamState
from .transcription_service import TranscriptionChunkResult, TranscriptionService

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Lightning Whisper Deepgram Server")

_config = get_config()
_service = TranscriptionService(
    model_name=_config.model,
    batch_size=_config.batch_size,
    quant=_config.quant,
    language=_config.language,
    enable_word_timestamps=_config.enable_word_timestamps,
)


@app.get("/health", response_class=JSONResponse)
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""

    return {"status": "ok", "model": _config.model}


@app.websocket("/v1/listen")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    request_id = str(uuid.uuid4())
    adapter = DeepgramAdapter(
        request_id=request_id,
        model_name=_config.model,
        model_version=_service.model_version,
    )
    state = StreamState()
    channels = 1
    sample_rate = _config.sample_rate
    language_override: Optional[str] = _config.language
    processor = AudioProcessor(sample_rate=sample_rate, chunk_size=_config.chunk_size, channels=channels)
    speech_started = False
    last_result: Optional[TranscriptionChunkResult] = None

    await websocket.send_json(adapter.build_metadata_message().model_dump(by_alias=True))

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                for chunk in processor.append(message["bytes"]):
                    try:
                        result = await _service.transcribe_chunk(
                            chunk.samples,
                            start=chunk.start_time,
                            end=chunk.end_time,
                            sample_rate=chunk.sample_rate,
                            language=language_override,
                        )
                    except RuntimeError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        LOGGER.error("Transcription error: %s", exc)
                        return
                    if not result.is_speech and not result.text:
                        continue
                    if result.is_speech and not speech_started:
                        await websocket.send_json(adapter.build_speech_started().model_dump(by_alias=True))
                        speech_started = True
                    state.append(result)
                    last_result = result
                    payload = adapter.build_results_message(
                        chunk_result=result,
                        state=state,
                        is_final=False,
                    )
                    await websocket.send_json(payload.model_dump(by_alias=True))
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    LOGGER.warning("Received invalid JSON payload: %s", message["text"])
                    continue

                message_type = _classify_message_type(payload)
                if message_type == "configure":
                    sample_rate, channels, language_override, processor = _apply_configuration(
                        payload,
                        sample_rate,
                        channels,
                        language_override,
                    )
                    state.reset()
                    speech_started = False
                    last_result = None
                    await websocket.send_json({"type": "configured", "request_id": request_id})
                elif message_type == "finalize":
                    final_chunk = processor.flush()
                    if final_chunk is not None:
                        try:
                            result = await _service.transcribe_chunk(
                                final_chunk.samples,
                                start=final_chunk.start_time,
                                end=final_chunk.end_time,
                                sample_rate=final_chunk.sample_rate,
                                language=language_override,
                            )
                        except RuntimeError as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                            LOGGER.error("Transcription error: %s", exc)
                            break
                        if result.is_speech and not speech_started:
                            await websocket.send_json(adapter.build_speech_started().model_dump(by_alias=True))
                            speech_started = True
                        if result.text or result.words:
                            state.append(result)
                            last_result = result
                            payload = adapter.build_results_message(
                                chunk_result=result,
                                state=state,
                                is_final=True,
                            )
                            await websocket.send_json(payload.model_dump(by_alias=True))
                    elif last_result is not None:
                        payload = adapter.build_results_message(
                            chunk_result=last_result,
                            state=state,
                            is_final=True,
                        )
                        await websocket.send_json(payload.model_dump(by_alias=True))
                    await websocket.send_json(adapter.build_utterance_end().model_dump(by_alias=True))
                    state.reset()
                    speech_started = False
                    last_result = None
                    processor = AudioProcessor(sample_rate=sample_rate, chunk_size=_config.chunk_size, channels=channels)
                elif message_type == "close":
                    await websocket.close()
                    break
                elif message_type == "keepalive":
                    await websocket.send_json({"type": "keepalive"})
                else:
                    LOGGER.debug("Ignoring message: %s", payload)
            else:
                LOGGER.debug("Unhandled message type: %s", message)
    except WebSocketDisconnect:
        LOGGER.info("Client disconnected: %s", request_id)
    except Exception as exc:  # pragma: no cover - best effort error handling
        LOGGER.exception("WebSocket error: %s", exc)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason=str(exc))
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()


def _classify_message_type(payload: Dict[str, Any]) -> str:
    message_type = str(payload.get("type", "")).lower()
    if "final" in message_type:
        return "finalize"
    if "close" in message_type:
        return "close"
    if "keep" in message_type:
        return "keepalive"
    if "config" in message_type or "start" in message_type:
        return "configure"
    return "unknown"


def _apply_configuration(
    payload: Dict[str, Any],
    current_sample_rate: int,
    current_channels: int,
    current_language: Optional[str],
) -> tuple[int, int, Optional[str], AudioProcessor]:
    config = payload.get("config", {})
    sample_rate = payload.get("sample_rate") or config.get("sample_rate")
    channels = payload.get("channels") or config.get("channels")
    language = payload.get("language") or config.get("language")

    new_sample_rate = current_sample_rate
    new_channels = current_channels
    new_language = current_language

    if sample_rate:
        new_sample_rate = int(sample_rate)
    if channels:
        new_channels = int(channels)
    if language:
        new_language = str(language)

    processor = AudioProcessor(
        sample_rate=new_sample_rate,
        chunk_size=_config.chunk_size,
        channels=new_channels,
    )

    return new_sample_rate, new_channels, new_language, processor


__all__ = ["app"]
