"""FastAPI application exposing a Deepgram compatible interface."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.websockets import WebSocketState

from .config import AppConfig, ModelConfig
from .responses import build_transcript_response
from .segmenter import Segmenter
from .transcriber import TranscriberService

LOGGER = logging.getLogger(__name__)


class Authenticator:
    """Utility for validating API keys."""

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def verify_headers(self, headers: Dict[str, str], *, allow_missing: bool = False) -> None:
        if self.api_key is None:
            return

        auth = headers.get("authorization")
        if not auth:
            if allow_missing:
                return
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

        token = None
        if auth.lower().startswith("token "):
            token = auth[6:].strip()
        elif auth.lower().startswith("bearer "):
            token = auth[7:].strip()

        if token != self.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    def verify_websocket(self, websocket: WebSocket) -> None:
        if self.api_key is None:
            return
        header = websocket.headers.get("authorization")
        if header is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        token = None
        header_lower = header.lower()
        if header_lower.startswith("token "):
            token = header[6:].strip()
        elif header_lower.startswith("bearer "):
            token = header[7:].strip()
        if token != self.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def create_app(config: AppConfig) -> FastAPI:
    """Create a configured FastAPI application."""

    authenticator = Authenticator(config.general.api_key)
    transcriber = TranscriberService()

    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        LOGGER.addHandler(handler)
    LOGGER.propagate = False

    app = FastAPI(title="Lightning OWhisper MLX", version="0.1.0")

    def _redact_headers(headers: Mapping[str, str]) -> Dict[str, str]:
        redacted = {}
        for key, value in headers.items():
            if key.lower() == "authorization":
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted

    def _truncate(text: str, limit: int = 200) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\u2026"

    async def _auth_dependency(request: Request) -> None:
        authenticator.verify_headers(request.headers)

    async def _optional_auth_dependency(request: Request) -> None:
        authenticator.verify_headers(request.headers, allow_missing=True)

    @app.middleware("http")
    async def _log_http_requests(request: Request, call_next):
        LOGGER.info(
            "HTTP request method=%s path=%s query=%s headers=%s",
            request.method,
            request.url.path,
            request.url.query,
            _redact_headers(dict(request.headers)),
        )
        response = await call_next(request)
        LOGGER.info(
            "HTTP response status=%s method=%s path=%s",
            response.status_code,
            request.method,
            request.url.path,
        )
        return response

    @app.get("/health")
    async def health(_: None = Depends(_optional_auth_dependency)) -> PlainTextResponse:
        return PlainTextResponse("OK")

    @app.get("/v1/status")
    async def status_endpoint(_: None = Depends(_optional_auth_dependency)) -> PlainTextResponse:
        return PlainTextResponse("", status_code=status.HTTP_204_NO_CONTENT)

    def _serialize_models(models: Iterable[ModelConfig]) -> Dict[str, List[Dict[str, str]]]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model.id,
                    "object": "model",
                }
                for model in models
            ],
        }

    @app.get("/models")
    async def list_models(_: None = Depends(_optional_auth_dependency)) -> JSONResponse:
        return JSONResponse(_serialize_models(config.models))

    @app.get("/v1/models")
    async def list_models_v1(_: None = Depends(_optional_auth_dependency)) -> JSONResponse:
        return JSONResponse(_serialize_models(config.models))

    async def handle_websocket(websocket: WebSocket) -> None:
        LOGGER.info(
            "WebSocket connection attempt path=%s client=%s query=%s headers=%s",
            websocket.url.path,
            websocket.client,
            list(websocket.query_params.multi_items()),
            _redact_headers(dict(websocket.headers)),
        )

        try:
            authenticator.verify_websocket(websocket)
        except HTTPException as exc:
            LOGGER.warning(
                "WebSocket authentication failed path=%s client=%s status=%s",
                websocket.url.path,
                websocket.client,
                exc.status_code,
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        query = websocket.query_params
        model_id = query.get("model") or (config.models[0].id if config.models else None)
        if model_id is None:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        try:
            model_config = config.require_model(model_id)
        except KeyError:
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        try:
            channels = int(query.get("channels", "1"))
        except ValueError:
            channels = 1
        channels = max(1, min(channels, 2))

        redemption_ms = query.get("redemption_time_ms")
        try:
            redemption_time = float(int(redemption_ms) / 1000.0) if redemption_ms else 0.4
        except ValueError:
            redemption_time = 0.4

        languages = query.getlist("languages") or []
        language = languages[0] if languages else query.get("language")

        request_id = uuid.uuid4().hex
        await websocket.accept(headers=[("dg-request-id", request_id)])
        LOGGER.info(
            "WebSocket accepted request_id=%s model=%s channels=%d language=%s redemption_time=%.3f",
            request_id,
            model_config.id,
            channels,
            language,
            redemption_time,
        )

        sample_rate = config.general.sample_rate
        segmenters = [
            Segmenter(
                sample_rate=sample_rate,
                redemption_time=redemption_time,
                channel_index=idx,
                energy_threshold=0.01,
            )
            for idx in range(channels)
        ]

        async def _transcribe_and_send(segment, idx):
            if segment.samples.size == 0:
                return
            result = await transcriber.transcribe(
                model_config,
                segment.samples.astype(np.float32),
                language,
            )
            payload = build_transcript_response(
                result=result,
                segment=segment,
                model_name=model_config.model,
                total_channels=channels,
                request_id=request_id,
            )
            alt = payload.get("channel", {}).get("alternatives", [{}])
            transcript = alt[0].get("transcript", "") if alt else ""
            LOGGER.info(
                "Transcribed segment request_id=%s channel=%s start=%.3f duration=%.3f final=%s text=%r",
                request_id,
                payload.get("channel_index"),
                segment.start_time,
                max(segment.end_time - segment.start_time, 0.0),
                payload.get("is_final"),
                _truncate(transcript),
            )
            await websocket.send_text(json.dumps(payload))

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    LOGGER.info(
                        "WebSocket disconnect received path=%s client=%s code=%s",
                        websocket.url.path,
                        websocket.client,
                        message.get("code"),
                    )
                    break

                if message.get("bytes"):
                    chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    LOGGER.info(
                        "WebSocket audio chunk received request_id=%s bytes=%d channels=%d",
                        request_id,
                        len(message["bytes"]),
                        channels,
                    )
                    if channels == 1:
                        float_chunk = chunk.astype(np.float32) / 32768.0
                        for segment in segmenters[0].submit(float_chunk):
                            await _transcribe_and_send(segment, 0)
                    else:
                        if chunk.size % 2 != 0:
                            chunk = chunk[:-1]
                        samples = chunk.reshape(-1, 2)
                        for idx in range(2):
                            float_chunk = samples[:, idx].astype(np.float32) / 32768.0
                            for segment in segmenters[idx].submit(float_chunk):
                                await _transcribe_and_send(segment, idx)
                elif message.get("text"):
                    try:
                        control = json.loads(message["text"])
                    except json.JSONDecodeError:
                        LOGGER.warning(
                            "Failed to parse control message request_id=%s payload=%r",
                            request_id,
                            _truncate(message["text"]),
                        )
                        continue
                    LOGGER.info(
                        "WebSocket control message request_id=%s payload=%s",
                        request_id,
                        control,
                    )
                    control_type = control.get("type")
                    if control_type in {"Finalize", "CloseStream"}:
                        for idx, segmenter in enumerate(segmenters):
                            for segment in segmenter.flush():
                                await _transcribe_and_send(segment, idx)
                        if control_type == "CloseStream":
                            break
                if websocket.application_state == WebSocketState.DISCONNECTED:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            for idx, segmenter in enumerate(segmenters):
                for segment in segmenter.flush():
                    await _transcribe_and_send(segment, idx)

            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()

    app.websocket("/listen")(handle_websocket)
    app.websocket("/v1/listen")(handle_websocket)

    return app
