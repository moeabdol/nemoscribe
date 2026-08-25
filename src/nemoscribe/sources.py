"""Audio sources: generators of timestamped Chunks."""

import queue
import time
from collections.abc import Generator
from pathlib import Path

from .audio import SAMPLE_RATE, Chunk, load


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


def _default_input_device() -> str | None:
    """Prefer the PipeWire/Pulse bridge devices — PortAudio's raw default can be
    a silent dead-end on modern Linux"""
    import sounddevice as sd

    names = {d["name"] for d in sd.query_devices()}
    for preferred in ("pipewire", "pulse"):
        if preferred in names:
            return preferred
    return None


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
