"""Helpers for constructing Deepgram compatible responses."""

from __future__ import annotations

import uuid
from typing import Dict, Optional

import numpy as np

from .segmenter import AudioSegment
from .transcriber import TranscriptionResult


def _word_confidence(word: Dict) -> float:
    for key in ("confidence", "probability", "score", "avg_logprob"):
        value = word.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def build_transcript_response(
    *,
    result: TranscriptionResult,
    segment: AudioSegment,
    model_name: str,
    total_channels: int,
    request_id: str,
    model_uuid: Optional[str] = None,
) -> Dict:
    """Return a Deepgram compatible transcript payload."""

    model_uuid = model_uuid or uuid.uuid4().hex
    words = []
    confidences = []

    for word in result.words:
        start = segment.start_time + float(word.get("start", 0.0))
        end = segment.start_time + float(word.get("end", start))
        confidence = _word_confidence(word)
        confidences.append(confidence)
        words.append(
            {
                "word": word.get("word", ""),
                "start": start,
                "end": end,
                "confidence": confidence,
                "speaker": segment.channel_index if total_channels > 1 else None,
                "punctuated_word": word.get("punctuated_word"),
                "language": word.get("language"),
            }
        )

    transcript_text = result.text.strip()
    confidence = float(np.mean(confidences)) if confidences else 0.0
    languages = []
    if result.language:
        languages = [result.language]

    return {
        "type": "Results",
        "start": segment.start_time,
        "duration": max(segment.end_time - segment.start_time, 0.0),
        "is_final": True,
        "speech_final": True,
        "from_finalize": False,
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript_text,
                    "words": words,
                    "confidence": confidence,
                    "languages": languages,
                }
            ]
        },
        "metadata": {
            "request_id": request_id,
            "model_info": {
                "name": model_name,
                "version": "1.0",
                "arch": "mlx",
            },
            "model_uuid": model_uuid,
        },
        "channel_index": [segment.channel_index, total_channels],
    }
