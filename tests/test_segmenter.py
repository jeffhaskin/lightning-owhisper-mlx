import numpy as np

from lightning_owhisper_mlx.segmenter import Segmenter


def test_segmenter_detects_segment():
    segmenter = Segmenter(sample_rate=4, redemption_time=0.5, channel_index=0, energy_threshold=0.05)

    # Initial silence
    assert segmenter.submit(np.zeros(4, dtype=np.float32)) == []

    # Speech chunk
    speech = np.ones(4, dtype=np.float32) * 0.2
    segments = segmenter.submit(speech)
    assert segments == []

    # Silence to trigger redemption
    silence = np.zeros(4, dtype=np.float32)
    segments = segmenter.submit(silence)
    assert len(segments) == 1
    segment = segments[0]
    assert np.allclose(segment.samples[:4], speech)
    assert segment.start_time == 1.0  # after initial silence chunk
    assert segment.end_time > segment.start_time

    # Flush should not return additional segments after reset
    assert segmenter.flush() == []
