"""Tests for nemoscribe.streaming"""

from itertools import pairwise
from pathlib import Path

import pytest

from nemoscribe.audio import SAMPLE_RATE, Chunk, load
from nemoscribe.streaming import StreamingSession


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/hello.wav").exists(), reason="dev-machine fixture"
)
def test_streaming_resets_yield_one_event_per_utterance(transcriber):
    audio = load("scratch/hello.wav")
    boundaries = []
    session = StreamingSession(
        transcriber, language="en-US", reset_silence_s=0.3, on_event=boundaries.append
    )
    for i in range(0, len(audio), 1600):
        session.feed(Chunk(samples=audio[i : i + 1600], start=i / SAMPLE_RATE))
    events = session.close()

    assert boundaries == events
    assert len(events) == 3
    for e in events:
        assert "hello" in e.text.lower()
    assert all(a.end <= b.start for a, b in pairwise(events))


@pytest.mark.integration
def test_generation_finished_before_speech_yields_no_event(transcriber):
    from nemoscribe.streaming import _Generation

    gen = _Generation(transcriber, "en-US", lambda _: None)
    assert gen.finish() is None


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/hello.wav").exists(), reason="dev-machine fixture"
)
def test_stream_ending_mid_speech_flushes_final_event(transcriber):
    audio = load("scratch/hello.wav")[: int(1.3 * SAMPLE_RATE)]  # cut inside hello #1
    session = StreamingSession(transcriber, language="en-US", reset_silence_s=0.3)
    for i in range(0, len(audio), 1600):
        session.feed(Chunk(samples=audio[i : i + 1600], start=i / SAMPLE_RATE))
    events = session.close()

    assert len(events) == 1
    assert "hel" in events[0].text.lower()
    assert events[0].end == pytest.approx(1.3, abs=0.11)


def test_session_rejects_unsupported_lookahead():
    with pytest.raises(ValueError) as exc_info:
        StreamingSession(object(), lookahead=0)  # 0 is not supported

    assert "lookahead" in str(exc_info.value)
