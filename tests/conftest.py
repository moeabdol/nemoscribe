"""Shared pytest fixtures."""

import subprocess

import pytest


@pytest.fixture(scope="session")
def transcriber():
    """One model load shared by every integration test in the run."""
    from nemoscribe.engine import Transcriber

    return Transcriber()


@pytest.fixture
def make_tone_wav():
    def _make(path, seconds=1.0, rate=44100, frequency=440):
        """Synthesize a 440 Hz test tone; raises immediately if fixture creation fails."""
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:duration={seconds}",
                "-ar",
                str(rate),
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    return _make


@pytest.fixture
def make_video_with_audio():
    def _make(path, seconds=1.0):
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

    return _make
