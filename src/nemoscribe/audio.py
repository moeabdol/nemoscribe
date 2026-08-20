"""Audio decoding: any format → mono float32 at 16 kHz."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioDecodeError(Exception):
    """Raised when a file cannot be decoded to audio."""


def load(path: str | Path) -> np.ndarray:
    """Decode an audio/video file to mono float32 samples at SAMPLE_RATE."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise AudioDecodeError("ffmpeg not found on PATH - install it") from e
    if p.returncode != 0:
        raise AudioDecodeError(
            f"ffmpeg could not decode {path}: {p.stderr.decode().strip()}"
        )
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


@dataclass(frozen=True, eq=False)
class Chunk:
    """A timestamped span of audio on the session clock."""

    samples: np.ndarray  # mono float32 at SAMPLE_RATE
    start: float  # seconds since session t=0

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.start + self.duration
