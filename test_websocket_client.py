"""Minimal WebSocket client for manual smoke testing."""
from __future__ import annotations

import argparse
import asyncio
import json
import math

import numpy as np
import websockets


def generate_sine_wave(duration: float, frequency: float, sample_rate: int = 16000) -> bytes:
    """Generate a PCM16 sine wave for testing."""

    frame_count = int(duration * sample_rate)
    times = np.linspace(0.0, duration, num=frame_count, endpoint=False)
    waveform = 0.2 * np.sin(2 * math.pi * frequency * times)
    pcm = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


async def run_client(uri: str, *, duration: float, frequency: float, language: str | None) -> None:
    payload = generate_sine_wave(duration, frequency)
    async with websockets.connect(uri, ping_interval=None) as websocket:
        config = {"type": "ListenV1Configure", "sample_rate": 16000, "channels": 1}
        if language:
            config["language"] = language
        await websocket.send(json.dumps(config))
        await websocket.send(payload)
        await websocket.send(json.dumps({"type": "ListenV1Finalize"}))

        async for message in websocket:
            if isinstance(message, bytes):
                continue
            data = json.loads(message)
            print(json.dumps(data, indent=2))
            if data.get("type") == "utterance_end":
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple WebSocket client test")
    parser.add_argument(
        "--uri",
        default="ws://127.0.0.1:9000/v1/listen",
        help="Server WebSocket URI",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Tone duration in seconds")
    parser.add_argument("--frequency", type=float, default=440.0, help="Tone frequency in Hz")
    parser.add_argument("--language", default=None, help="Optional language hint")
    args = parser.parse_args()

    asyncio.run(run_client(args.uri, duration=args.duration, frequency=args.frequency, language=args.language))


if __name__ == "__main__":
    main()
