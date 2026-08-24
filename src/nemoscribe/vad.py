"""Speech detection: silero VAD over 16 kHz mono audio."""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

FRAME = 512  # silero's analysis frame: 512 samples = 32 ms at 16 kHz
CONTEXT = 64

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class VadError(Exception):
    """Raised when the VAD model cannot be obtained or run."""


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "nemoscribe"


def _download(url: str, dest: Path) -> None:  # pragma: no cover
    urllib.request.urlretrieve(url, dest)


def ensure_model() -> Path:
    """Return the local silero model path, downloading and verifying on first use."""
    dest = _cache_dir() / "silero_vad.onnx"
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    try:
        _download(MODEL_URL, part)
    except OSError as e:
        raise VadError(
            f"could not download the silero VAD model from {MODEL_URL} — check "
            "your network connection"
        ) from e

    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        part.unlink()
        raise VadError(
            f"silero VAD model failed checksum verification (expected "
            f"{MODEL_SHA256[:12]}..., got {digest[:12]}...) — partial download "
            "or upstream change; delete nothing and just retry"
        )

    os.replace(part, dest)
    return dest


def probs_to_segments(
    probs,
    *,
    threshold=0.5,
    neg_threshold=0.35,
    min_silence_ms=500,
    min_speech_ms=300,
    max_speech_s=12.0,
) -> list[tuple[int, int]]:
    """Turn per-frame speech probabilities into (start, end) sample ranges.

    Hysteresis: a segment opens when prob >= threshold and closes only after
    min_silence_ms worth of frames below neg_threshold; frames between the two
    thresholds neither open nor close anything.
    """
    ms_per_frame = FRAME / SAMPLE_RATE * 1000.0
    min_sil = max(1, int(min_silence_ms / ms_per_frame))
    min_sp = max(1, int(min_speech_ms / ms_per_frame))
    max_sp = max(2, int(max_speech_s * 1000 / ms_per_frame))

    if max_sp < 2 * min_sp:
        raise ValueError(
            f"max_speech_s ({max_speech_s}) too small: a split segment must fit "
            f"two pieces of min_speech_ms ({min_speech_ms}) each"
        )

    segs, in_speech, start, silence_run = [], False, 0, 0
    for i, p in enumerate(probs):
        if not in_speech:  # in SILENCE
            if p >= threshold:  # strong evidence to OPEN
                in_speech, start, silence_run = True, i, 0
        else:  # in SPEECH
            if p < neg_threshold:  # strong evidence toward CLOSE
                silence_run += 1
                if silence_run >= min_sil:
                    segs.append([start, i - silence_run + 1])
                    in_speech = False
            else:
                silence_run = 0  # anything >= neg_threshold resets silence_run streak

    if in_speech:
        segs.append([start, len(probs)])

    kept = [s for s in segs if s[1] - s[0] >= min_sp]

    final, stack = [], list(reversed(kept))
    while stack:
        s = stack.pop()
        if s[1] - s[0] <= max_sp:
            final.append(s)
            continue
        lo, hi = s[0] + min_sp, min(s[0] + max_sp, s[1] - min_sp)
        cut = lo + int(np.argmin(probs[lo:hi]))
        stack.append([cut, s[1]])
        final.append([s[0], cut])

    return [(s[0] * FRAME, s[1] * FRAME) for s in final]


def speech_probs(audio: np.ndarray, model_path: str | Path | None = None) -> np.ndarray:
    """Per-frame speech probabilities: one float per FRAME samples of audio."""
    n = int(np.ceil(len(audio) / FRAME))
    padded = np.pad(audio, (0, n * FRAME - len(audio)))
    return StreamVad(model_path).feed(padded)


def segments(audio: np.ndarray, **params) -> list[tuple[int, int]]:
    """Speech segments of `audio` as (start, end) sample ranges."""
    return probs_to_segments(speech_probs(audio), **params)


class StreamVad:
    """Incremental silero VAD: feed arbitrary-sized chunks, get per-frame probs."""

    def __init__(self, model_path: str | Path | None = None):
        import onnxruntime as ort  # deferred: keeps CLI startup fast

        self._sess = ort.InferenceSession(
            str(model_path or ensure_model()),
            providers=["CPUExecutionProvider"],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> np.ndarray:
        """Consume new samples: return probs for each COMPLETE frame formed."""
        data = np.concatenate([self._pending, samples])
        n = len(data) // FRAME
        self._pending = data[n * FRAME :]

        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            frame = data[i * FRAME : (i + 1) * FRAME]
            x = np.concatenate([self._context, frame])[None, :]
            out, self._state = self._sess.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            probs[i] = np.asarray(out)[0, 0]
            self._context = frame[-CONTEXT:]
        return probs
