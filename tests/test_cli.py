"""Tests for nemoscribe.cli"""

import numpy as np

from nemoscribe.audio import AudioDecodeError
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

    def transcribe(self, audio, *, language="en-US"):
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
        "nemoscribe.audio.load", lambda path: np.zeros(16_000, dtype=np.float32)
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

    def broken_load(path):
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
        "nemoscribe.audio.load", lambda p: np.zeros(16_000, dtype=np.float32)
    )
    wav = tmp_path / "mix.wav"
    wav.touch()

    assert main(["transcribe", "--language", "auto", str(wav)]) == 0

    err = capsys.readouterr().err
    assert "ar-AR 2" in err
    assert "en-US 1" in err
    assert "unknown 1" in err
