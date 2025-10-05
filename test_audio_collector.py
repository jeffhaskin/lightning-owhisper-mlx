"""Utility script to download audio samples for integration testing."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import wave

from yt_dlp import YoutubeDL


def download_youtube_audio(url: str, destination: Path) -> Path:
    """Download a video's audio track and convert it to WAV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(destination.with_suffix("")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    output = Path(filename).with_suffix(".wav")
    if output != destination:
        output.rename(destination)
        output = destination
    return output


def chunk_audio_file(source: Path, chunk_duration: float, target_dir: Path) -> None:
    """Split a WAV file into equally sized chunks."""

    target_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(source), "rb") as wav_in:
        frame_rate = wav_in.getframerate()
        channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        frames_per_chunk = int(frame_rate * chunk_duration)
        total_frames = wav_in.getnframes()
        chunk_count = math.ceil(total_frames / frames_per_chunk)

        for index in range(chunk_count):
            frames = wav_in.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = target_dir / f"chunk_{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as wav_out:
                wav_out.setnchannels(channels)
                wav_out.setsampwidth(sampwidth)
                wav_out.setframerate(frame_rate)
                wav_out.writeframes(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and chunk audio for testing.")
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument("destination", type=Path, help="Destination WAV file path")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=15.0,
        help="Chunk duration in seconds for optional splitting",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help="Optional directory to store chunked audio files",
    )
    args = parser.parse_args()

    wav_path = download_youtube_audio(args.url, args.destination)
    print(f"Downloaded audio to {wav_path}")

    if args.chunks_dir is not None:
        chunk_audio_file(wav_path, args.chunk_duration, args.chunks_dir)
        print(f"Chunks written to {args.chunks_dir}")


if __name__ == "__main__":
    main()
