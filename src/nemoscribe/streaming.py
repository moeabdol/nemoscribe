"""Streaming transcription: a stateful session over the cache-aware decode."""

import queue
from collections import deque
from threading import Thread

import numpy as np

from .audio import SAMPLE_RATE, Chunk
from .engine import RUNWAY_S
from .events import TranscriptEvent
from .vad import StreamVad


class StreamingSession:
    """One live decode: feed Chunks in, get partial text out, finals on close.

    A StremVad watches the fed audio: speech starts a decode generation, primed
    with ~preroll_s of held-back audio (onset recovery); sustained silence
    (reset_silence_s) finalizes it into one event and returns to idle — every
    speech onset meets a fresh decoder.
    """

    def __init__(
        self,
        transcriber,
        *,
        language="en-US",
        lookahead=6,
        on_partial=None,
        on_event=None,
        reset_silence_s=1.0,
        preroll_s=0.3,
        source="",
        max_utterance_s=20.0,
    ):
        if lookahead not in (3, 6, 13):
            raise ValueError(
                f"lookahead {lookahead} unsupported for streaming: 1 is not a model "
                "tier, and 0 breaks the streaming window geometry (first_mel • hop < "
                "n_fft/2 → negative window start); use 3, 6, or 13"
            )

        transcriber.processor.set_num_lookahead_tokens(lookahead)  # shared once
        self._t = transcriber
        self._language = language
        self._on_partial = on_partial or (lambda t: None)
        self._on_event = on_event or (lambda e: None)
        self._reset_silence_s = reset_silence_s
        self._preroll_s = preroll_s
        self._source = source
        self._max_utterance_s = max_utterance_s
        self._vad = StreamVad()
        self._raw: queue.Queue = queue.Queue()
        self._events: list[TranscriptEvent] = []
        self._orchestrator = Thread(target=self._run, daemon=True)
        self._orchestrator.start()

    def feed(self, chunk: Chunk) -> None:
        self._raw.put(chunk)

    def close(self) -> list[TranscriptEvent]:
        self._raw.put(None)
        self._orchestrator.join()
        return self._events

    def _run(self) -> None:
        gen = None
        gen_started = 0.0
        silence_run = 0.0
        preroll: deque = deque()
        last_chunk = None
        while (chunk := self._raw.get()) is not None:
            last_chunk = chunk
            probs = self._vad.feed(chunk.samples)
            if len(probs):
                if (probs < 0.35).all():
                    silence_run += chunk.duration
                else:
                    silence_run = 0.0
            has_speech = len(probs) > 0 and bool((probs >= 0.5).any())

            if gen is None:  # IDLE: hold audio, watch for speech
                preroll.append(chunk)
                while (
                    sum(c.duration for c in preroll) > self._preroll_s + chunk.duration
                ):
                    preroll.popleft()
                if has_speech:
                    gen = _Generation(
                        self._t, self._language, self._on_partial, self._source
                    )
                    gen_started = chunk.start
                    for held in preroll:
                        gen.feed(held)
                    preroll.clear()
                    silence_run = 0.0
            else:  # ACTIVE: feed, watch for silence
                gen.feed(chunk)
                if silence_run >= self._reset_silence_s:
                    if event := gen.finish(end_hint=chunk.end - silence_run):
                        self._events.append(event)
                        self._on_event(event)
                    gen = None
                    silence_run = 0.0
                elif chunk.end - gen_started >= self._max_utterance_s:
                    # rotation: continous speech never pauses, so force-finalize and
                    # continue mid-speech (seam may clip one word — the price of the
                    # bounded cues and bounded lock holds)
                    if event := gen.finish(end_hint=chunk.end):
                        self._events.append(event)
                        self._on_event(event)
                    gen = _Generation(
                        self._t, self._language, self._on_partial, self._source
                    )
                    gen_started = chunk.end

        if gen is not None:  # stream ended mid-speech: flush
            hint = last_chunk.end if last_chunk else None
            if event := gen.finish(end_hint=hint):
                self._events.append(event)
                self._on_event(event)


class _Generation:
    """One decode lifetime: prime → stream → finalize into a single event."""

    def __init__(self, transcriber, language, on_partial, source=""):
        from transformers import TextIteratorStreamer  # deferred

        self._t = transcriber
        self._language = language
        self._on_partial = on_partial
        self._source = source

        p = self._t.processor
        self._first_n = p.num_samples_first_audio_chunk
        self._chunk_n = p.num_samples_per_audio_chunk
        self._hop = p.feature_extractor.hop_length
        self._n_fft = p.feature_extractor.n_fft
        self._first_mel = p.num_mel_frames_first_audio_chunk
        self._mel_per_chunk = p.num_mel_frames_per_audio_chunk

        self._queue: queue.Queue = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0
        self._pieces: list[str] = []
        self._start_time: float | None = None
        self._end_time = 0.0

        self._streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True)
        self._decode_thread = Thread(target=self._run_decode, daemon=True)
        self._drain_thread = Thread(target=self._run_drain, daemon=True)
        self._decode_thread.start()
        self._drain_thread.start()

    def feed(self, chunk: Chunk) -> None:
        if self._start_time is None:
            self._start_time = chunk.start
        self._end_time = chunk.end
        self._queue.put(chunk.samples)

    def finish(self, end_hint: float | None = None) -> TranscriptEvent | None:
        # decode runway: tail frames need right-context before their tokens flush
        self._queue.put(np.zeros(int(RUNWAY_S * SAMPLE_RATE), dtype=np.float32))
        self._queue.put(None)  # end-of-stream sentinel
        self._decode_thread.join()
        self._drain_thread.join()
        text = " ".join("".join(self._pieces).split())
        if not text:
            return None
        return TranscriptEvent(
            text=text,
            start=self._start_time or 0.0,
            end=end_hint if end_hint is not None else self._end_time,
            language=self._language if self._language != "auto" else "",
            source=self._source,
        )

    def _ensure(self, abs_end: int) -> bool:
        """Grow the buffer to cover absolute sample abs_end; False on EOF."""
        while self._base + len(self._buf) < abs_end:
            item = self._queue.get()
            if item is None:
                return False
            self._buf = np.concatenate([self._buf, item])
        return True

    def _window(self, abs_start: int, abs_end: int) -> np.ndarray:
        keep_from = max(0, abs_start - self._n_fft)
        if keep_from > self._base:
            self._buf = self._buf[keep_from - self._base :]
            self._base = keep_from
        return self._buf[abs_start - self._base : abs_end - self._base]

    def _run_decode(self) -> None:
        try:
            if not self._ensure(self._first_n):
                return  # unreachable while RUNWAY_S • SAMPLE_RATE ≥ first_n; kept as guard
            p = self._t.processor
            first = p(
                self._window(0, self._first_n),
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=True,
                language=self._language,
                return_tensors="pt",
            )
            first = first.to(self._t.model.device, dtype=self._t.model.dtype)

            def features():
                yield first.input_features[:, : self._first_mel, :]
                mel_idx = self._first_mel
                while True:
                    start = mel_idx * self._hop - self._n_fft // 2
                    end = start + self._chunk_n
                    if not self._ensure(end):
                        return
                    inputs = p(
                        self._window(start, end),
                        sampling_rate=SAMPLE_RATE,
                        is_streaming=True,
                        is_first_audio_chunk=False,
                        language=self._language,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self._t.model.device, dtype=self._t.model.dtype)
                    yield inputs.input_features
                    mel_idx += self._mel_per_chunk

            kwargs = {
                **first,
                "input_features": features(),
                "streamer": self._streamer,
                "max_new_tokens": 1_000_000_000,  # unbound by design: the feature stream ending stops the decode
            }
            with self._t.decode_lock:
                self._t.model.generate(**kwargs)
        finally:
            self._streamer.end()

    def _run_drain(self) -> None:
        for piece in self._streamer:
            self._pieces.append(piece)
            self._on_partial(piece)
