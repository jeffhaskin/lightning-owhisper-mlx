import base64
import json
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

from lightning_owhisper_mlx.config import AppConfig, GeneralConfig, ModelConfig
from lightning_owhisper_mlx.server import create_app
from lightning_owhisper_mlx.transcriber import TranscriptionResult


def _make_app():
    config = AppConfig(
        general=GeneralConfig(api_key=None, host="127.0.0.1", port=0, sample_rate=16000),
        models=[ModelConfig(id="demo", model="distil-small.en")],
    )

    return create_app(config)


def test_hyprnote_style_audio_flow():
    app = _make_app()
    audio = (np.sin(np.linspace(0, np.pi, 1600)) * 12000).astype("<i2")
    payload = {
        "type": "audio",
        "value": {"data": base64.b64encode(audio.tobytes()).decode("ascii")},
    }

    transcript = TranscriptionResult(
        text="hello world",
        words=[{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9}],
        language="en",
    )

    async_mock = AsyncMock(return_value=transcript)

    with patch(
        "lightning_owhisper_mlx.server.TranscriberService.transcribe",
        new=async_mock,
    ) as mocked_transcribe:
        client = TestClient(app)

        with client.websocket_connect(
            "/v1/listen?model=demo&channels=1&sample_rate=16000&"
            "encoding=linear16&multichannel=true&interim_results=true&redemption_time_ms=200"
        ) as websocket:
            websocket.send_text(json.dumps(payload))
            websocket.send_text(json.dumps({"type": "Finalize"}))
            message = websocket.receive_text()
            data = json.loads(message)

            assert data["channel"]["alternatives"][0]["transcript"] == "hello world"
            websocket.send_text(json.dumps({"type": "CloseStream"}))

        assert mocked_transcribe.await_count == 1
        call = mocked_transcribe.await_args_list[0]
        args = call.args
        # args: (model_config, audio_array, language)
        np.testing.assert_allclose(args[1][:5], (audio.astype(np.float32) / 32768.0)[:5])
