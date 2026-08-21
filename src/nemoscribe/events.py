"""Transcript events — the common currency between engines and writers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """A single recognized word and when it was spoken."""

    text: str  # punctuated & cased as the model emitted it — never normalized
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptEvent:
    """One transcribed segment — the unit every writer consumes."""

    text: str
    start: float
    end: float
    language: str
    source: str
    words: tuple[Word, ...] = ()
