"""Tests for nemoscribe.audio — fixtures are synthesized by ffmpeg, nothing committed."""

import subprocess
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import nemoscribe
from nemoscribe.audio import SAMPLE_RATE, AudioDecodeError, Chunk, load


def make_tone_wav(path, seconds=1.0, rate=44100):
    """Synthesize a 440 Hz test tone; raises immediately if fixture creation fails."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            str(rate),
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def make_video_with_audio(path, seconds=1.0):
    """Synthesize a tiny mp4: black frames + the same tone."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=black:s=64x64:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "mpeg4",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_package_imports():
    assert nemoscribe.__doc__ is not None


def test_load_resamples_to_project_rate(tmp_path):
    wav = tmp_path / "tone.wav"
    make_tone_wav(wav, seconds=1.0, rate=44100)

    a = load(wav)

    assert a.dtype == np.float32
    assert abs(len(a) - SAMPLE_RATE) < SAMPLE_RATE * 0.01
    peak = np.abs(a).max()
    assert peak <= 1.0
    assert peak > 0.1


def test_load_extracts_audio_from_video(tmp_path):
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


def test_missing_ffmpeg_raises(monkeypatch):
    monkeypatch.setenv("PATH", "")  # this tests process finds no binaries

    with pytest.raises(AudioDecodeError) as exc_info:
        load("anything.wav")

    assert "ffmpeg" in str(exc_info)
