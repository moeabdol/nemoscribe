"""SRT writer: timestamped subtitle cues."""

from collections.abc import Sequence

from ..events import TranscriptEvent, Word


def _timestamp(seconds: float) -> str:
    """Format seconds as SRT's HH:MM:SS,mmm (comma — the format demands it)."""
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _event_cues(
    event: TranscriptEvent, max_chars: int
) -> list[tuple[float, float, str]]:
    """Split one event into (start, end, text) cues at word boundaries.

    With word timing, cues pack greedily up to max_chars and take their times
    from their first/last word. Without timings, the whole event becomes one cue
    — the graceful-degradation contract of events.words.
    """
    if not event.words:
        return [(event.start, event.end, event.text)]

    cues = []
    current: list[Word] = []
    length = 0
    for word in event.words:
        added = len(word.text) + (1 if current else 0)
        if current and length + added > max_chars:
            cues.append(_flush(current))
            current, length = [], 0
            added = len(word.text)
        current.append(word)
        length += added
    if current:
        cues.append(_flush(current))
    return cues


def _flush(words: list[Word]) -> tuple[float, float, str]:
    return words[0].start, words[-1].end, " ".join(w.text for w in words)


def write_srt(
    events: Sequence[TranscriptEvent],
    *,
    max_cue_chars: int = 84,
    lead_in_s: float = 0.3,
) -> str:
    ordered = sorted(events, key=lambda e: e.start)
    cues = [cue for e in ordered for cue in _event_cues(e, max_cue_chars)]

    # presentation only: show each cue slightly before its speech so readers get
    # a head start; also corrects RNNT emission lag (word timestamps run 0.3-1 s
    # behind the acoustic). Never crosses the previous cue's end.
    adjusted, prev_end = [], 0.0
    for start, end, text in cues:
        start = max(start - lead_in_s, prev_end, 0.0)
        adjusted.append((start, end, text))
        prev_end = end

    blocks = [
        f"{i}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(adjusted, 1)
    ]
    return "\n".join(blocks)
