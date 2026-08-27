"""JSONL writer: NeMo-manifest-compatible event log."""

import json
from collections.abc import Mapping, Sequence

from ..events import TranscriptEvent


def write_jsonl(
    events: Sequence[TranscriptEvent], *, audio_filepath: str | Mapping[str, str]
) -> str:
    ordered = sorted(events, key=lambda e: e.start)
    lines = []
    for e in ordered:
        resolved = (
            audio_filepath
            if isinstance(audio_filepath, str)
            else audio_filepath[e.source]  # missing label = loud KeyError, on purpose
        )
        record = {
            "audio_filepath": resolved,
            "offset": round(e.start, 3),
            "duration": round(e.end - e.start, 3),
            "text": e.text,
        }
        if e.language:
            record["target_lang"] = e.language
        if e.source:
            record["source"] = e.source
        lines.append(json.dumps(record, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)
