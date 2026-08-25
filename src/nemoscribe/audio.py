"""Audio decoding: any format → mono float32 at 16 kHz."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioDecodeError(Exception):
    """Raised when a file cannot be decoded to audio."""


def _channel_count(path: str | Path) -> int:
    """Ask ffprobe how many channels the file's first audio stream has."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise AudioDecodeError(
            "ffprobe not found in PATH — install ffmpeg (ffprobe ships with it)"
        ) from e
    out = p.stdout.decode().strip()
    if p.returncode != 0 or not out:
        raise AudioDecodeError(
            f"no decodable audio stream in {path}: {p.stderr.decode().strip()}"
        )
    return int(out.splitlines()[0])


def load(path: str | Path) -> np.ndarray:
    """Decode an audio/video file to mono float32 samples at SAMPLE_RATE.

    Multi-channel audio is downmixed by exact channel mean. ffmpeg's own downmix
    runs ~3 dB hoter, and streaming inference is gain-sensitive.
    """
    channels = _channel_count(path)
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
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise AudioDecodeError("ffmpeg not found on PATH - install ffmpeg") from e
    if p.returncode != 0:
        raise AudioDecodeError(
            f"ffmpeg could not decode {path}: {p.stderr.decode().strip()}"
        )

    raw = np.frombuffer(p.stdout, dtype=np.float32)
    if channels == 1:
        return raw.copy()
    frames = len(raw) // channels
    return (
        raw[: frames * channels]
        .reshape(frames, channels)
        .mean(axis=1)
        .astype(np.float32)
    )


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


def save_wav(path: str | Path, samples: np.ndarray) -> None:
    """Write mono float32 samples as 16-bit PCM at SAMPLE_RATE."""
    import wave

    data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())
