"""Tests for nemoscribe.audio — fixtures are synthesized by ffmpeg, nothing committed."""

import subprocess
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import nemoscribe
from nemoscribe.audio import SAMPLE_RATE, AudioDecodeError, Chunk, load


def test_package_imports():
    assert nemoscribe.__doc__ is not None


def test_load_resamples_to_project_rate(tmp_path, make_tone_wav):
    wav = tmp_path / "tone.wav"
    make_tone_wav(wav, seconds=1.0, rate=44100)

    a = load(wav)

    assert a.dtype == np.float32
    assert abs(len(a) - SAMPLE_RATE) < SAMPLE_RATE * 0.01
    peak = np.abs(a).max()
    assert peak <= 1.0
    assert peak > 0.1


def test_load_extracts_audio_from_video(tmp_path, make_video_with_audio):
    mp4 = tmp_path / "clip.mp4"
    make_video_with_audio(mp4, seconds=1.0)

    a = load(mp4)

    assert abs(len(a) - SAMPLE_RATE) < SAMPLE_RATE * 0.05
    assert np.abs(a).max() > 0.1


def test_missing_file_raises():
    with pytest.raises(AudioDecodeError):
        load("no/such/file.wav")


def test_garbage_file_raises_with_helpful_message(tmp_path):
    bad = tmp_path / "garbage.bin"
    bad.write_bytes(b"not audio")

    with pytest.raises(AudioDecodeError) as exc_info:
        load(bad)

    assert "garbage.bin" in str(exc_info.value)


def test_chunk_time_arithmetic():
    c = Chunk(samples=np.zeros(8000, dtype=np.float32), start=2.0)
    assert c.duration == 0.5
    assert c.end == 2.5


def test_chunk_is_immutable():
    c = Chunk(samples=np.zeros(160, dtype=np.float32), start=0.0)
    with pytest.raises(FrozenInstanceError):
        c.start = 1.0


def test_missing_ffprobe_raises(monkeypatch):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(AudioDecodeError) as exc_info:
        load("anything.wav")

    assert "ffprobe" in str(exc_info.value)


def test_missing_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr("nemoscribe.audio._channel_count", lambda _: 1)  # probe passes
    monkeypatch.setenv("PATH", "")  # ffmpeg is absent

    with pytest.raises(AudioDecodeError) as exc_info:
        load("anything.wav")

    assert "ffmpeg" in str(exc_info)


def test_load_downmixes_stereo_by_channel_mean(tmp_path, make_tone_wav):
    left, right = tmp_path / "l.wav", tmp_path / "r.wav"
    stereo = tmp_path / "s.wav"
    make_tone_wav(left, rate=16000, frequency=440)
    make_tone_wav(right, rate=16000, frequency=880)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1",
            "-filter_complex",
            "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map",
            "[a]",
            "-ar",
            "16000",
            str(stereo),
        ],
        check=True,
        capture_output=True,
    )

    l, r, s = load(left), load(right), load(stereo)
    n = min(len(l), len(r), len(s))

    assert np.abs(s[:n] - (l[:n] + r[:n]) / 2).max() < 1e-3


def test_decode_failure_after_successful_probe(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.audio._channel_count", lambda _: 1)
    bad = tmp_path / "liar.wav"
    bad.write_bytes(b"not audio")

    with pytest.raises(AudioDecodeError) as exc_info:
        load(bad)

    assert "could not decode" in str(exc_info.value)
