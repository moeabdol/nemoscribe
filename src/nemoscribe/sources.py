"""Audio sources: generators of timestamped Chunks."""

import time
from collections.abc import Iterator
from pathlib import Path

from .audio import SAMPLE_RATE, Chunk, load


def file_chunks(
    path: str | Path, *, chunk_s: float = 0.1, realtime: bool = False
) -> Iterator[Chunk]:
    """Replay a file as a stream of Chunks; realtime paces it to the wall clock."""
    audio = load(path)
    step = int(chunk_s * SAMPLE_RATE)
    for i in range(0, len(audio), step):
        chunk = Chunk(samples=audio[i : i + step], start=i / SAMPLE_RATE)
        if realtime:
            time.sleep(chunk.duration)
        yield chunk
