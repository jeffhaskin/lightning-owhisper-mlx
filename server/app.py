"""FastAPI application exposing Deepgram-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .audio import decode_audio_bytes
from .config import DEFAULT_CONFIG, ServerConfig
from .responses import build_listen_response
from .streaming import StreamingSession
from .transcriber import LightningMLXTranscriber
def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or DEFAULT_CONFIG
    app = FastAPI(
        title="Lightning Whisper MLX Deepgram-compatible Server",
        version="0.1.0",
    )
    transcriber = LightningMLXTranscriber(cfg)
    app.state.config = cfg
    app.state.transcriber = transcriber

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await transcriber.aclose()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": cfg.model,
            "quantization": cfg.quantization,
            "sample_rate": cfg.sample_rate,
            "version": app.version,
        }

    @app.post("/v1/listen")
    async def listen(
        file: UploadFile = File(...),
        language: str | None = None,
    ) -> JSONResponse:
        audio_bytes = await file.read()
        try:
            audio = decode_audio_bytes(audio_bytes, cfg.sample_rate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = await transcriber.transcribe(audio, language=language, word_timestamps=True)
        payload = build_listen_response(result, cfg)
        return JSONResponse(payload)

    @app.websocket("/v1/listen")
    async def websocket_listen(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send_message(message: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(message)

        session = StreamingSession(
            transcriber=transcriber,
            config=cfg,
            send=send_message,
        )

        await send_message(
            {
                "type": "metadata",
                "request_id": session.state.session_id,
                "model": cfg.model,
                "quantization": cfg.quantization,
                "sample_rate": cfg.sample_rate,
            }
        )

        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"]:
                    try:
                        should_close = await _handle_control_message(session, message["text"])
                    except HTTPException as exc:
                        await send_message({"type": "error", "error": exc.detail, "code": exc.status_code})
                        await session.abort()
                        await websocket.close(code=4400, reason=str(exc.detail))
                        return
                    if should_close:
                        break
                elif "bytes" in message and message["bytes"]:
                    await session.enqueue_audio(message["bytes"])
                else:
                    # Keep-alive ping/pong
                    continue
        except WebSocketDisconnect:
            await session.abort()
        except Exception:
            await session.abort()
            raise
        finally:
            if not session.closed:
                await session.finalize()

    return app


async def _handle_control_message(session: StreamingSession, message_text: str) -> bool:
    try:
        payload = json.loads(message_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON control message") from exc

    message_type = (payload.get("type") or "").lower()
    if message_type == "start":
        config = payload.get("config", {})
        await session.configure(
            sample_rate=config.get("sample_rate"),
            language=config.get("language"),
        )
        return False
    if message_type in {"stop", "end", "finalize"}:
        await session.finalize()
        return True
    if message_type in {"ping", "keepalive"}:
        return False
    raise HTTPException(status_code=400, detail=f"Unsupported control message type: {message_type}")
