"""Tests for nemoscribe.vad.probs_to_segments — pure logic, handcrafted inputs.

Legible parameters throughout: min_silence_ms=96 → 3 frames,
min_speech_ms=64 → 2 frames; one frame = 32 ms = 512 samples.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import nemoscribe.vad as vad
from nemoscribe.audio import SAMPLE_RATE, load
from nemoscribe.vad import FRAME, VadError, probs_to_segments

EASY = dict(min_silence_ms=96, min_speech_ms=64, max_speech_s=1000)


def test_silence_only():
    probs = np.zeros(30, dtype=np.float32)

    assert probs_to_segments(probs, **EASY) == []


def test_single_burst_with_trailing_silence():
    probs = np.zeros(30, dtype=np.float32)
    probs[5:12] = 0.9

    segs = probs_to_segments(probs, **EASY)

    assert segs == [(5 * FRAME, 12 * FRAME)]


def test_hysteresis_dip_does_not_split():
    probs = np.zeros(30, dtype=np.float32)
    probs[5:12] = 0.9
    probs[8] = 0.4

    segs = probs_to_segments(probs, **EASY)

    assert segs == [(5 * FRAME, 12 * FRAME)]


def test_speech_running_to_end_of_array_is_flushed():
    probs = np.zeros(20, dtype=np.float32)
    probs[15:] = 0.9

    segs = probs_to_segments(probs, **EASY)

    assert segs == [(15 * FRAME, 20 * FRAME)]


def test_single_frame_blip_is_dropped():
    probs = np.zeros(30, dtype=np.float32)
    probs[10] = 0.9

    segs = probs_to_segments(probs, **EASY)

    assert segs == []


def test_long_segment_splits_at_the_quietest_frame():
    probs = np.zeros(15, dtype=np.float32)
    probs[0:8] = 0.9
    probs[4] = 0.36

    segs = probs_to_segments(
        probs,
        min_silence_ms=96,
        min_speech_ms=64,
        max_speech_s=0.16,
    )

    assert segs == [(0 * FRAME, 4 * FRAME), (4 * FRAME, 8 * FRAME)]


def test_invalid_params_rejected_at_the_door():
    probs = np.ones(50, dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        probs_to_segments(probs, min_speech_ms=1000, max_speech_s=0.1)

    assert "max_speech_s" in str(exc_info)


def test_cache_dir_respects_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert vad._cache_dir() == tmp_path / "nemoscribe"


def test_ensure_model_rejects_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vad, "_download", lambda url, dest: dest.write_bytes(b"junk"))

    with pytest.raises(VadError) as exc_info:
        vad.ensure_model()

    assert "checksum" in str(exc_info.value)
    assert not (tmp_path / "nemoscribe" / "silero_vad.onnx").exists()


def test_ensure_model_uses_cache_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cached = tmp_path / "nemoscribe" / "silero_vad.onnx"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"pretend model")

    def booby_trap(url, dest):
        raise AssertionError("network touched despite cache hit!")

    monkeypatch.setattr(vad, "_download", booby_trap)

    assert vad.ensure_model() == cached


def test_ensure_model_wraps_network_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def dead_network(url, dest):
        raise OSError("connection refused")

    monkeypatch.setattr(vad, "_download", dead_network)

    with pytest.raises(VadError) as exc_info:
        vad.ensure_model()

    assert "download" in str(exc_info.value)


def test_ensure_model_installs_verified_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    payload = b"fake model bytes"
    monkeypatch.setattr(vad, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(vad, "_download", lambda url, dest: dest.write_bytes(payload))

    path = vad.ensure_model()

    assert path == tmp_path / "nemoscribe" / "silero_vad.onnx"
    assert path.read_bytes() == payload
    assert not path.with_suffix(".part").exists()


@pytest.mark.integration
def test_silence_yields_low_probabilities():
    probs = vad.speech_probs(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert len(probs) == SAMPLE_RATE // FRAME + 1
    assert probs.max() < 0.5


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/hello.wav").exists(), reason="dev-machine fixture"
)
def test_hello_wav_yields_three_speech_segments():
    segs = vad.segments(load("scratch/hello.wav"))

    assert len(segs) == 3
