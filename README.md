# Lightning Whisper MLX Deepgram-Compatible Server

This repository provides a self-hosted speech-to-text service that mimics the
[Deepgram API](https://developers.deepgram.com/reference/) while delegating all
transcription work to [lightning-whisper-mlx](https://github.com/lightning-ai/lightning-whisper-mlx).
The server is implemented in Python with [FastAPI](https://fastapi.tiangolo.com/)
and exposes both REST and WebSocket streaming endpoints.

> ⚠️ **Platform requirements:** lightning-whisper-mlx relies on Apple's
> [MLX](https://ml-explore.github.io/mlx/build/html/index.html) framework and is
> primarily intended for Apple Silicon devices. Running the server on other
> platforms may require alternative backends or a CPU-only configuration.

## Features

- `POST /v1/listen`: Deepgram-style single-shot transcription of audio files.
- `WS /v1/listen`: Real-time or near-real-time chunked streaming transcription with
  Deepgram-compatible control messages (`start`, `stop`, `ping`, etc.).
- Configurable model, quantization level, stream window size, and language
  defaults through environment variables.
- Simple CLI entrypoint: `python -m server --host 0.0.0.0 --port 8080`.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The `lightning-whisper-mlx` package will download and cache the selected
   Whisper checkpoints the first time the server processes audio. By default
   models are stored in `./mlx_models/`.

2. **Configure the server (optional)**

   | Variable | Description | Default |
   | --- | --- | --- |
   | `LWM_MODEL` | Whisper model alias (`tiny`, `small`, `distil-small.en`, `base`, `medium`, `distil-medium.en`, `large`, `large-v2`, `distil-large-v2`, `large-v3`, `distil-large-v3`) | `small` |
   | `LWM_QUANTIZATION` | Quantization level (`4bit` or `8bit`) | _none_ |
   | `LWM_BATCH_SIZE` | Batch size passed to lightning-whisper-mlx | `6` |
   | `LWM_SAMPLE_RATE` | Target sample rate (Whisper expects 16000) | `16000` |
   | `LWM_STREAM_WINDOW_SECONDS` | Minimum buffered audio before sending interim transcripts | `2.5` |
   | `LWM_DEFAULT_LANGUAGE` | Force a specific language code | auto-detect |

3. **Run the service**

   ```bash
   python -m server --host 0.0.0.0 --port 8080
   ```

   You can also mount the app with any ASGI server that supports lifespan events
   (e.g. `uvicorn server.app:create_app`).

## REST usage example

```bash
curl -X POST      -H "Content-Type: audio/wav"      --data-binary @sample.wav \
    http://localhost:8080/v1/listen
```

The response payload mirrors Deepgram's structure, including channel and
alternatives data:

```json
{
  "request_id": "…",
  "model": "small",
  "metadata": {
    "model": "small",
    "quantization": null,
    "sample_rate": 16000,
    "language": "en"
  },
  "results": {
    "channels": [
      {
        "channel_index": 0,
        "alternatives": [
          {
            "transcript": "hello world",
            "confidence": null,
            "words": [
              {"word": "hello", "start": 0.0, "end": 0.4, "confidence": 0.92},
              {"word": "world", "start": 0.4, "end": 0.8, "confidence": 0.90}
            ]
          }
        ],
        "duration": 0.8
      }
    ]
  }
}
```

## Streaming usage example

1. **Open a WebSocket connection** to `ws://localhost:8080/v1/listen`. Clients
   may include a Deepgram-style `Authorization` header for compatibility, but
   the server accepts any value.
2. **Send a JSON `start` control message**:

   ```json
   {"type": "start", "config": {"sample_rate": 16000, "language": "en"}}
   ```

3. **Stream binary PCM audio frames** (16-bit little-endian). The server buffers
   roughly `LWM_STREAM_WINDOW_SECONDS` seconds before emitting interim
   transcripts.
4. **Send a JSON `stop` message** to flush the final transcript.

Streaming responses are JSON frames shaped like Deepgram's `Results` messages:

```json
{
  "type": "Results",
  "request_id": "…",
  "channel_index": 0,
  "utterance_id": "…",
  "start": 1.2,
  "duration": 3.4,
  "is_final": false,
  "speech_final": false,
  "channel": {
    "alternatives": [
      {"transcript": "partial text", "confidence": null, "words": [...]}
    ]
  },
  "timestamp": "2024-01-01T12:34:56.789012+00:00"
}
```

When `stop` is received, the server issues one last message with `is_final` set
to `true` to indicate transcription has completed.

## Development notes

- The server intentionally mirrors Deepgram's envelope but does not yet support
  every feature (e.g. diarization, multi-channel audio, alternative encodings).
  Contributions are welcome!
- Streaming currently keeps the full audio history in memory to preserve
  context. Long-running sessions may require additional buffering logic or
  chunk-wise decoding for better scalability.
- Unit tests are not provided; please validate changes locally with representative
  audio samples before deploying to production.
