"""Tests for nemoscribe.cli"""

import json

import numpy as np
import pytest

from nemoscribe.audio import AudioDecodeError, Chunk
from nemoscribe.cli import _parse_source, main
from nemoscribe.engine import EngineError
from nemoscribe.events import TranscriptEvent
from nemoscribe.sources import SourceError


class FakeTranscriber:
    def __init__(self, device=None):
        pass

    def transcribe(self, audio, *, language="en-US"):
        return [
            TranscriptEvent(
                text="Hello there.", start=0.5, end=1.5, language=language, source=""
            )
        ]


class ExplodingTranscriber:
    def __init__(self, device=None):
        raise EngineError("no CUDA device available")


class PolyglotFakeTranscriber:
    def __init__(self, device=None):
        pass

    def transcribe(self, _, *, language="en-US"):
        mk = lambda text, lang: TranscriptEvent(
            text=text, start=0.0, end=1.0, language=lang, source=""
        )
        return [
            mk("مرحبا", "ar-AR"),
            mk("سلام", "ar-AR"),
            mk("Hello", "en-US"),
            mk("??", ""),
        ]


def test_transcribe_command_writes_three_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr(
        "nemoscribe.audio.load", lambda _: np.zeros(16_000, dtype=np.float32)
    )

    wav = tmp_path / "talk.wav"
    wav.touch()

    assert main(["transcribe", str(wav)]) == 0
    assert (tmp_path / "talk.srt").exists()
    assert (tmp_path / "talk.jsonl").exists()
    assert (tmp_path / "talk.txt").exists()
    assert "Hello there." in (tmp_path / "talk.txt").read_text()


def test_transcribe_reports_bad_file_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)

    def broken_load(_):
        raise AudioDecodeError("boom")

    monkeypatch.setattr("nemoscribe.audio.load", broken_load)
    wav = tmp_path / "bad.wav"
    wav.touch()

    assert main(["transcribe", str(wav)]) == 1
    assert not (tmp_path / "bad.srt").exists()


def test_transcribe_fails_cleanly_when_engine_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", ExplodingTranscriber)
    wav = tmp_path / "x.wav"
    wav.touch()

    assert main(["transcribe", str(wav)]) == 2
    assert not (tmp_path / "x.srt").exists()


def test_auto_language_prints_scorecard(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", PolyglotFakeTranscriber)
    monkeypatch.setattr(
        "nemoscribe.audio.load", lambda _: np.zeros(16_000, dtype=np.float32)
    )
    wav = tmp_path / "mix.wav"
    wav.touch()

    assert main(["transcribe", "--language", "auto", str(wav)]) == 0

    err = capsys.readouterr().err
    assert "ar-AR 2" in err
    assert "en-US 1" in err
    assert "unknown 1" in err


def test_stream_rejects_unsupported_source(capsys):
    assert main(["stream", "webcam"]) == 2
    err = capsys.readouterr().err
    assert "unsupported source" in err
    assert "loading model" not in err


class FakeStreamingSession:
    fed = 0

    def __init__(
        self, transcriber, *, on_partial=None, on_event=None, source="", **kwargs
    ):
        self._on_partial = on_partial or (lambda _: None)
        self._source = source
        self._on_event = on_event or (lambda _: None)
        FakeStreamingSession.fed = 0

    def feed(self, chunk):
        FakeStreamingSession.fed += 1

    def close(self):
        event = TranscriptEvent(
            text="Hello there.",
            start=0.0,
            end=0.25,
            language="en-US",
            source=self._source,
        )
        self._on_partial("Hello there. ")
        self._on_event(event)
        return [event]


def test_stream_command_feeds_file_and_writes_output(
    tmp_path, monkeypatch, capsys, make_tone_wav
):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    wav = tmp_path / "talk.wav"
    make_tone_wav(wav, seconds=0.25, rate=16_000)

    assert main(["stream", f"file={wav}"]) == 0

    assert FakeStreamingSession.fed == 3
    assert "Hello there." in capsys.readouterr().out
    assert (tmp_path / "talk.srt").exists()


def test_stream_rejects_duplicate_labels(capsys):
    assert main(["stream", "mic", "mic"]) == 2
    err = capsys.readouterr().err
    assert "duplicate" in err
    assert "loading model" not in err


def test_stream_fails_cleanly_when_engine_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", ExplodingTranscriber)
    assert main(["stream", "file=whatever.wav"]) == 2


class SilentStreamingSession(FakeStreamingSession):
    def close(self):
        return []


def test_stream_warns_when_no_speech(tmp_path, monkeypatch, capsys, make_tone_wav):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", SilentStreamingSession)
    wav = tmp_path / "quiet.wav"
    make_tone_wav(wav, seconds=0.25, rate=16_000)

    assert main(["stream", f"file={wav}"]) == 0
    assert "no speech" in capsys.readouterr().err


def fake_mic_chunks(**kwargs):
    yield Chunk(samples=np.zeros(1600, dtype=np.float32), start=0.0)
    yield Chunk(samples=np.zeros(1600, dtype=np.float32), start=0.1)
    # generator ends: simulates the source closing; real Ctrl-C is main-thread
    # only and is verified in live acceptance, not simulatable from here


def test_stream_mic_writes_timestamped_outputs(tmp_path, monkeypatch, make_tone_wav):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    monkeypatch.setattr("nemoscribe.sources.mic_chunks", fake_mic_chunks)
    monkeypatch.chdir(tmp_path)  # mic outputs land in CWD — sandbox it

    assert main(["stream", "mic"]) == 0
    assert list(tmp_path.glob("mic-*.srt"))  # timestamp stem exists
    assert FakeStreamingSession.fed == 2


def test_stream_mic_save_audio_writes_wav_and_manifest_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    monkeypatch.setattr("nemoscribe.sources.mic_chunks", fake_mic_chunks)
    monkeypatch.chdir(tmp_path)

    assert main(["stream", "mic", "--save-audio"]) == 0

    wavs = list(tmp_path.glob("mic-*.wav"))
    assert len(wavs) == 1
    record = json.loads(next(tmp_path.glob("mic-*.jsonl")).read_text().splitlines()[0])
    assert record["audio_filepath"].endswith(".wav")
    assert record["audio_filepath"] == wavs[0].name


def test_parse_source_grammar():
    assert _parse_source("mic") == ("mic", "", "mic")
    assert _parse_source("mic:me") == ("mic", "", "me")
    assert _parse_source("mic:") == ("mic", "", "mic")
    assert _parse_source("system:them") == ("system", "", "them")
    assert _parse_source("file=talk.wav") == ("file", "talk.wav", "talk")
    assert _parse_source("file=/a/b/clip.mp4") == ("file", "/a/b/clip.mp4", "clip")


def test_parse_source_rejects_unknown_kinds():
    with pytest.raises(ValueError) as exc_info:
        _parse_source("webcam")

    assert "unsupported source" in str(exc_info.value)


def test_stream_two_file_sources_merge_and_split(tmp_path, monkeypatch, make_tone_wav):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    monkeypatch.chdir(tmp_path)
    make_tone_wav(tmp_path / "alpha.wav", seconds=0.25, rate=16_000)
    make_tone_wav(tmp_path / "beta.wav", seconds=0.25, rate=16_000)

    assert (
        main(["stream", "file=alpha.wav", "file=beta.wav", "--split", "--save-audio"])
        == 0
    )

    stem = min(tmp_path.glob("stream-*.jsonl"), key=lambda p: len(p.name))
    sources = {json.loads(line)["source"] for line in stem.read_text().splitlines()}
    assert sources == {"alpha", "beta"}
    assert next(tmp_path.glob("stream-*-alpha.wav"), None) is not None
    assert next(tmp_path.glob("stream-*-beta.wav"), None) is not None
    assert stem.with_suffix(".wav").exists()  # the mix — same basename, mpv auto-pairs

    for label in ("alpha", "beta"):
        for ext in (".srt", ".jsonl", ".txt"):
            assert stem.with_name(f"{stem.stem}-{label}{ext}").exists()

    # per-source manifest points at its own stem wav (parse, don't just glob)
    line = json.loads(
        stem.with_name(f"{stem.stem}-alpha.jsonl").read_text().splitlines()[0]
    )
    assert line["audio_filepath"] == f"{stem.stem}-alpha.wav"
    assert line["source"] == "alpha"


def fake_system_chunks(**kwargs):
    yield Chunk(samples=np.zeros(1600, dtype=np.float32), start=0.0)


def test_stream_system_and_file_sources_merge(tmp_path, monkeypatch, make_tone_wav):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    monkeypatch.setattr("nemoscribe.sources.system_chunks", fake_system_chunks)
    monkeypatch.chdir(tmp_path)
    make_tone_wav(tmp_path / "clip.wav", seconds=0.25, rate=16_000)

    assert main(["stream", "system:them", "file=clip.wav"]) == 0

    stem = next(tmp_path.glob("stream-*.jsonl"))
    sources = {json.loads(l)["source"] for l in stem.read_text().splitlines()}
    assert sources == {"them", "clip"}


def test_stream_dead_source_fails_fast_with_status_1(
    tmp_path, monkeypatch, make_tone_wav
):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)
    monkeypatch.chdir(tmp_path)
    make_tone_wav(tmp_path / "good.wav", seconds=0.25, rate=16_000)

    assert main(["stream", "file=good.wav", "file=missing.wav"]) == 1


def test_stream_system_source_setup_failure_exits_2(monkeypatch, capsys):
    monkeypatch.setattr("nemoscribe.engine.Transcriber", FakeTranscriber)
    monkeypatch.setattr("nemoscribe.streaming.StreamingSession", FakeStreamingSession)

    def no_sink(**kwargs):
        raise SourceError("no default sink")

    monkeypatch.setattr("nemoscribe.sources.system_chunks", no_sink)

    assert main(["stream", "system"]) == 2
    assert "no default sink" in capsys.readouterr().err
