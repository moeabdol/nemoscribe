"""Audio sources: generators of timestamped Chunks."""

import queue
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, Chunk, load


class SourceError(Exception):
    """Raised when an audio source cannot be opened."""


def _default_input_device() -> str | None:
    """Prefer the PipeWire/Pulse bridge devices — PortAudio's raw default can be
    a silent dead-end on modern Linux"""
    import sounddevice as sd

    names = {d["name"] for d in sd.query_devices()}
    for preferred in ("pipewire", "pulse"):
        if preferred in names:
            return preferred
    return None


def _default_monitor() -> str:
    """Name of the default output sink's monitor source (PipeWire/Pulse)."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-sink"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise SourceError(
            "pactl not found — system capture needs pipewire-pulse"
        ) from e
    if out.returncode != 0 or not out.stdout.strip():
        raise SourceError(
            "could not find the default audio sink (is pactl/pipewire-pulse available?)"
        )
    return out.stdout.strip() + ".monitor"


def file_chunks(
    path: str | Path, *, chunk_s: float = 0.1, realtime: bool = False
) -> Generator[Chunk, None, None]:
    """Replay a file as a stream of Chunks; realtime paces it to the wall clock."""
    audio = load(path)
    step = int(chunk_s * SAMPLE_RATE)
    for i in range(0, len(audio), step):
        chunk = Chunk(samples=audio[i : i + step], start=i / SAMPLE_RATE)
        if realtime:
            time.sleep(chunk.duration)
        yield chunk


def mic_chunks(
    *, chunk_s: float = 0.1, device: str | int | None = None
) -> Generator[Chunk, None, None]:
    """Capture the default (or named) microphone as timestamped Chunks."""
    import sys

    import sounddevice as sd  # deferred: loads PortAudio

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)  # overruns reported, never fatal
        q.put(indata[:, 0].copy())  # mono column; COPY — buffer is reused

    samples_seen = 0
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(chunk_s * SAMPLE_RATE),
        device=device if device is not None else _default_input_device(),
        callback=callback,
    ):
        while True:
            samples = q.get()
            yield Chunk(samples=samples, start=samples_seen / SAMPLE_RATE)
            samples_seen += len(samples)


def system_chunks(
    *, chunk_s: float = 0.1, source_name: str | None = None
) -> Generator[Chunk, None, None]:
    """Capture what the system is playing (the default sink's monitor)."""
    name = source_name or _default_monitor()
    block_bytes = int(chunk_s * SAMPLE_RATE) * 4  # f32le: 4 bytes/sample
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "pulse",
        "-i",
        name,
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None  # guaranteed by stdout=PIPE above
    samples_seen = 0
    try:
        while True:
            data = proc.stdout.read(block_bytes)
            if not data:  # EOF: ffmpeg existed
                break
            samples = np.frombuffer(data, dtype=np.float32).copy()
            yield Chunk(samples=samples, start=samples_seen / SAMPLE_RATE)
            samples_seen += len(samples)
    finally:
        proc.terminate()
        proc.wait()
