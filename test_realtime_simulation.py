"""Simulate a client streaming audio to the local Deepgram-compatible server."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import wave

import websockets


async def simulate_stream(
    audio_path: Path,
    *,
    uri: str,
    chunk_duration: float,
    language: str | None = None,
) -> None:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames_per_chunk = int(sample_rate * chunk_duration)

        async with websockets.connect(uri, ping_interval=None) as websocket:
            config_payload = {
                "type": "ListenV1Configure",
                "sample_rate": sample_rate,
                "channels": channels,
            }
            if language:
                config_payload["language"] = language
            await websocket.send(json.dumps(config_payload))

            while True:
                frames = wav_file.readframes(frames_per_chunk)
                if not frames:
                    break
                await websocket.send(frames)
                await asyncio.sleep(chunk_duration)

            await websocket.send(json.dumps({"type": "ListenV1Finalize"}))

            async for message in websocket:
                if isinstance(message, bytes):
                    continue
                payload = json.loads(message)
                print(json.dumps(payload, indent=2))
                if payload.get("type") == "utterance_end":
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream audio to the local server.")
    parser.add_argument("audio", type=Path, help="Path to a WAV file to stream")
    parser.add_argument(
        "--uri",
        default="ws://127.0.0.1:9000/v1/listen",
        help="Server WebSocket URI",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.0,
        help="Duration of each chunk in seconds",
    )
    parser.add_argument("--language", default=None, help="Optional language hint")
    args = parser.parse_args()

    asyncio.run(
        simulate_stream(
            args.audio,
            uri=args.uri,
            chunk_duration=args.chunk_duration,
            language=args.language,
        )
    )


if __name__ == "__main__":
    main()
