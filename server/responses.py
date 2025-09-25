"""Response formatting utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
import uuid

from .config import ServerConfig


def _clean_word(text: str) -> str:
    return text.replace("\n", " ").strip()


def segments_to_words(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            cleaned = _clean_word(word.get("word", ""))
            if not cleaned:
                continue
            words.append(
                {
                    "word": cleaned,
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "confidence": word.get("probability"),
                }
            )
    return words


def segments_to_transcript(segments: Iterable[dict[str, Any]]) -> str:
    return "".join(segment.get("text", "") for segment in segments).strip()


def build_alternative(segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transcript": segments_to_transcript(segments),
        "confidence": None,
        "words": segments_to_words(segments),
    }


def build_listen_response(
    result: dict[str, Any],
    config: ServerConfig,
    request_id: str | None = None,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = result.get("segments") or []
    alternative = build_alternative(segments) if segments else {"transcript": result.get("text", ""), "confidence": None, "words": []}
    duration = segments[-1]["end"] if segments else None
    language = result.get("language")

    return {
        "request_id": request_id or str(uuid.uuid4()),
        "model": config.model,
        "metadata": {
            "model": config.model,
            "quantization": config.quantization,
            "sample_rate": config.sample_rate,
            "language": language,
        },
        "results": {
            "channels": [
                {
                    "channel_index": 0,
                    "alternatives": [alternative],
                    "duration": duration,
                }
            ]
        },
    }


def build_streaming_result(
    segments: list[dict[str, Any]],
    *,
    config: ServerConfig,
    request_id: str,
    utterance_id: str,
    is_final: bool,
    start_offset: float,
) -> dict[str, Any]:
    alternative = build_alternative(segments)
    duration = segments[-1]["end"] if segments else start_offset
    start_time = segments[0]["start"] if segments else start_offset
    return {
        "type": "Results",
        "request_id": request_id,
        "channel_index": 0,
        "utterance_id": utterance_id,
        "start": start_time,
        "duration": duration,
        "is_final": is_final,
        "speech_final": is_final,
        "channel": {"alternatives": [alternative]},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
