# Lightning Whisper Deepgram Server

This repository provides a drop-in replacement for Deepgram's streaming WebSocket API powered by [lightning-whisper-mlx](https://github.com/Lightning-AI/lightning-whisper-mlx). The server accepts Deepgram-compatible WebSocket messages, streams PCM16 audio, and responds with metadata and transcription results that follow Deepgram's schema.

## Features

- ⚡️ **Lightning Whisper backend** – Uses the Apple MLX-optimised Whisper models for rapid transcription.
- 🔁 **Real-time streaming** – Handles WebSocket media frames, incremental transcription, and utterance finalisation.
- 🔄 **Deepgram compatibility** – Supports `ListenV1Media`, `ListenV1Finalize`, keep-alives, metadata, and result payloads matching Deepgram's schema.
- 🔊 **Audio preprocessing** – Incremental PCM16 chunking, optional resampling, and rudimentary speech detection.
- 🧪 **Test utilities** – Scripts for collecting sample audio, simulating live streaming, and performing smoke tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `lightning-whisper-mlx` package will download model weights on first use. For Apple Silicon the default MLX backend is used automatically.

## Running the server

```bash
uvicorn src.server:app --host 0.0.0.0 --port 9000
```

### Environment variables

Configuration options can be overridden via environment variables (prefixed with `LIGHTNING_WHISPER_`):

| Variable | Description | Default |
| --- | --- | --- |
| `LIGHTNING_WHISPER_MODEL` | Whisper model name (e.g. `small`, `medium`, `distil-small.en`) | `small` |
| `LIGHTNING_WHISPER_BATCH_SIZE` | Batch size for inference | `12` |
| `LIGHTNING_WHISPER_QUANT` | Optional quantisation (`4bit`/`8bit`) | unset |
| `LIGHTNING_WHISPER_CHUNK_SIZE` | Length of buffered audio (seconds) before transcription | `5.0` |
| `LIGHTNING_WHISPER_SAMPLE_RATE` | Expected input sample rate | `16000` |
| `LIGHTNING_WHISPER_LANGUAGE` | Default language hint | unset |
| `LIGHTNING_WHISPER_ENABLE_WORD_TIMESTAMPS` | Approximate word timing output (`true`/`false`) | `false` |

## API Usage

1. **Connect** to `ws://<host>:<port>/v1/listen`.
2. **Send a configure message** (optional) matching Deepgram's schema, e.g.:
   ```json
   {"type": "ListenV1Configure", "sample_rate": 16000, "channels": 1}
   ```
3. **Stream PCM16 audio** as binary WebSocket frames (`ListenV1Media`).
4. **Finalize** the stream by sending `{"type": "ListenV1Finalize"}`.
5. Optionally send keep-alives `{"type": "ListenV1KeepAlive"}`.
6. Close the connection with `{"type": "ListenV1CloseStream"}` or by closing the socket.

Responses include metadata, speech-start notifications, incremental results, and a final `utterance_end` message mirroring Deepgram's API.

## Testing utilities

- `test_audio_collector.py` – Download YouTube audio and split into chunks.
- `test_realtime_simulation.py` – Simulate streaming a WAV file to the server.
- `test_websocket_client.py` – Generate a synthetic sine tone and observe the server's responses.
- `test_integration.py` – Basic smoke tests for the core modules.

Run the scripted checks with:

```bash
python test_integration.py
```

or simulate a real stream:

```bash
python test_realtime_simulation.py sample.wav --chunk-duration 1.5
```

## Project structure

```
src/
  audio_processor.py       # Incremental PCM16 chunking
  config.py                # Environment-driven configuration
  deepgram_adapter.py      # Deepgram schema conversion helpers
  models.py                # Pydantic response models
  transcription_service.py # lightning-whisper-mlx integration
  server.py                # FastAPI WebSocket server
```

## Notes

- Word timings are approximated from segment timings when the model does not provide token-level timestamps.
- The first transcription request downloads the model weights to `./mlx_models/<model-name>`.
- For production deployments consider running behind a process manager (e.g. `systemd`, `supervisord`) and enabling TLS termination.

## License

This repository aggregates open-source components; consult upstream projects for their respective licenses.
