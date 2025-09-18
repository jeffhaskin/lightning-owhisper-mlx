# lightning-owhisper-mlx

Lightning-OWhisper-MLX is a Deepgram-compatible speech-to-text server tailored
for [Hyprnote](https://github.com/fastrepl/hyprnote). It mirrors the public
surface of `owhisper` while replacing the inference backend with
[`lightning-whisper-mlx`](https://github.com/mustafaaljadery/lightning-whisper-mlx)
so you can run fast Whisper models on Apple Silicon.

## Features

- **Deepgram compatible API** – implements the `/listen`, `/v1/listen`,
  `/models`, `/v1/models`, `/health`, and `/v1/status` endpoints used by
  Hyprnote.
- **WebSocket streaming** – accepts 16 kHz PCM audio and emits Deepgram style
  transcript messages with word-level metadata.
- **Multiple model support** – load any lightning-whisper-mlx model, including
  quantized variants.
- **API key optional** – require `Token` or `Bearer` authorization headers with
  a single config value.

## Installation

```bash
pip install lightning-owhisper-mlx
# Install the MLX backend (optional on non-Apple platforms)
pip install "lightning-owhisper-mlx[mlx]"
```

## Configuration

The server reads a simple YAML file describing available models. By default it
exposes two distilled English models, but you can provide your own configuration
via the `--config` flag.

```yaml
# config.yaml
general:
  api_key: super-secret-token
  host: 0.0.0.0
  port: 52693
  sample_rate: 16000
models:
  - id: distil-medium-en
    model: distil-medium.en
    batch_size: 12
  - id: turbo-en
    model: large-v3
    quantization: 8bit
    batch_size: 4
```

## Running the server

```bash
lightning-owhisper-mlx --config config.yaml
```

The CLI accepts `--host` and `--port` overrides for quick experiments. The
server exposes the following Deepgram-compatible endpoints:

| Method | Path          | Description              |
| ------ | ------------- | ------------------------ |
| GET    | `/health`     | Liveness probe           |
| GET    | `/v1/status`  | Deepgram status endpoint |
| GET    | `/models`     | List configured models   |
| GET    | `/v1/models`  | Deepgram model list      |
| WS     | `/listen`     | Streaming transcription  |
| WS     | `/v1/listen`  | Deepgram streaming API   |

Clients can connect using the existing Hyprnote owhisper client or any Deepgram
compatible client. Audio should be 16 kHz little-endian PCM with one or two
channels.

## Development

Install development dependencies and run the unit test suite:

```bash
pip install -e .
pytest
```

The tests cover the audio segmentation logic and Deepgram response formatting so
they run without requiring the MLX runtime.
