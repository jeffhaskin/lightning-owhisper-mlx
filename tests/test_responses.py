from lightning_owhisper_mlx.responses import build_transcript_response
from lightning_owhisper_mlx.segmenter import AudioSegment
from lightning_owhisper_mlx.transcriber import TranscriptionResult

import numpy as np


def test_build_transcript_response_structure():
    result = TranscriptionResult(
        text="Hello world",
        words=[{"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9}],
        language="en",
    )
    segment = AudioSegment(np.ones(4, dtype=np.float32), start_time=1.0, end_time=1.5, channel_index=0)

    payload = build_transcript_response(
        result=result,
        segment=segment,
        model_name="distil-small.en",
        total_channels=1,
        request_id="abc123",
        model_uuid="uuid",
    )

    assert payload["type"] == "Results"
    assert payload["metadata"]["request_id"] == "abc123"
    alt = payload["channel"]["alternatives"][0]
    assert alt["transcript"] == "Hello world"
    assert alt["words"][0]["start"] == 1.0
    assert alt["words"][0]["confidence"] == 0.9
