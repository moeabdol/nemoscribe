"""Plain-text writer: one event per line."""

from collections.abc import Sequence

from ..events import TranscriptEvent


def write_txt(events: Sequence[TranscriptEvent]) -> str:
    ordered = sorted(events, key=lambda e: e.start)
    return "".join(e.text + "\n" for e in ordered)
