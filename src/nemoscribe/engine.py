"""Batch transcription engine: audio → VAD segments → model → events."""

import re
import threading

import numpy as np

from . import vad
from .audio import SAMPLE_RATE
from .events import TranscriptEvent, Word

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

RUNWAY_S = 1.0  # trailing silence appended to each segment's DECODE input only

LANG_TAG = re.compile(r"<([a-z]{2}-[A-Z]{2})>")


class EngineError(Exception):
    """Raised when the transcription engine cannot run as requested."""


def _pad_and_clamp(
    segments: list[tuple[int, int]],
    total_samples: int,
    *,
    pad_ms: int = 200,
) -> list[tuple[int, int]]:
    """Widen VAD segments for onset recovery, without invading neighbors.

    Padding recovers the ~300 ms of speech onset silero opens late on; the
    midpoint rule keeps padded neighbors from overlapping: each may grow at most
    to the middle of the gap between them.
    """
    pad = int(pad_ms / 1000 * SAMPLE_RATE)
    out = []
    for i, (s, e) in enumerate(segments):
        s_p = max(0, s - pad)
        e_p = min(total_samples, e + pad)
        if i > 0:
            s_p = max(s_p, (segments[i - 1][1] + s) // 2)
        if i < len(segments) - 1:
            e_p = min(e_p, (e + segments[i + 1][0]) // 2)
        out.append((s_p, e_p))
    return out


def _words_from_pieces(pieces: list[dict], offset: float) -> tuple[Word, ...]:
    """Group token pieces into whitespace-delimited words on the file clock.

    A piece with leading whitespace starts a new word; whitespace-only pieces
    act as boundaries and contribute no text.
    """
    words: list[Word] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        text = "".join(p["token"] for p in group).strip()
        if text:
            words.append(
                Word(
                    text=text,
                    start=group[0]["start"] + offset,
                    end=group[-1]["end"] + offset,
                )
            )
        group.clear()

    for p in pieces:
        if p["token"].isspace():
            flush()
            continue
        if group and p["token"].startswith(" "):
            flush()
        group.append(p)
    flush()
    return tuple(words)


def _clamp_words(words: tuple[Word, ...], end: float) -> tuple[Word, ...]:
    """Clip word times to the segment's real end — runway tokens carry
    timestamps inside the appended silence, which is not part of the file."""
    return tuple(
        Word(text=w.text, start=min(w.start, end), end=min(w.end, end)) for w in words
    )


def _detect_language(raw: str) -> str:
    """Pull the model's emitted locale tag (e.g. <ar-AR>) from a raw decode."""
    m = LANG_TAG.search(raw)
    return m.group(1) if m else ""


class Transcriber:
    """Loads Nemotron once; turns audio arrays into transcript events."""

    def __init__(self, device: str | None = None):
        import torch  # deferred: 2.5 s import must not tax CLI startup
        from transformers import AutoModelForRNNT, AutoProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise EngineError(
                "cuda requested but no CUDA device is available — "
                "use --device cpu or omit --device to auto-detect"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForRNNT.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        # transformers' generate stores per-call state ON the model — never reentrant
        self.decode_lock = threading.Lock()

    def transcribe(
        self, audio: np.ndarray, *, language: str = "en-US"
    ) -> list[TranscriptEvent]:
        """Transcribe a full recording: VAD → padded segments → one event each."""
        speech = _pad_and_clamp(vad.segments(audio), len(audio))
        events = [
            self._transcribe_segment(audio[s:e], s / SAMPLE_RATE, language)
            for s, e in speech
        ]
        return [e for e in events if e is not None]

    def _transcribe_segment(
        self, clip: np.ndarray, offset: float, language: str
    ) -> TranscriptEvent | None:
        runway = np.zeros(int(RUNWAY_S * SAMPLE_RATE), dtype=np.float32)
        inputs = self.processor(
            np.concatenate([clip, runway]),
            sampling_rate=SAMPLE_RATE,
            language=language,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with self.decode_lock:
            out = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=int((len(clip) / SAMPLE_RATE + RUNWAY_S) * 125) + 16,
            )

        texts, durations = self.processor.decode(
            out.sequences, durations=out.durations, skip_special_tokens=True
        )
        text = " ".join(texts[0].split())
        if not text:
            return None

        if language == "auto":
            raw = self.processor.decode(out.sequences, skip_special_tokens=False)
            detected = _detect_language(raw[0])
        else:
            detected = language

        seg_end = offset + len(clip) / SAMPLE_RATE
        return TranscriptEvent(
            text=text,
            start=offset,  # segment bounds, not word
            end=seg_end,
            language=detected,
            source="",
            words=_clamp_words(_words_from_pieces(durations[0], offset), seg_end),
        )
