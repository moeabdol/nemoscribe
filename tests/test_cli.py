"""Tests for nemoscribe.cli"""

import json

import numpy as np

from nemoscribe.audio import AudioDecodeError, Chunk
from nemoscribe.cli import main
from nemoscribe.engine import EngineError
from nemoscribe.events import TranscriptEvent


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
    assert main(["stream", "mic:me"]) == 2

    err = capsys.readouterr().err
    assert "unsupported source" in err
    assert "loading model" not in err


class FakeStreamingSession:
    fed = 0

    def __init__(self, transcriber, *, on_partial=None, **kwargs):
        self._on_partial = on_partial or (lambda _: None)
        FakeStreamingSession.fed = 0

    def feed(self, chunk):
        FakeStreamingSession.fed += 1

    def close(self):
        self._on_partial("Hello there. ")
        return [
            TranscriptEvent(
                text="Hello there.", start=0.0, end=0.25, language="en-US", source=""
            )
        ]


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


def test_stream_rejects_multiple_sources(capsys):
    assert main(["stream", "file=a.wav", "file=b.wav"]) == 2
    assert "one source at a time" in capsys.readouterr().err


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
    raise KeyboardInterrupt  # simulate Ctrl-C mid-capture


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
