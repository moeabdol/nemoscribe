"""Tests for nemoscribe.engine"""

from itertools import pairwise
from pathlib import Path

import pytest

from nemoscribe.audio import load
from nemoscribe.engine import (
    _clamp_words,
    _detect_language,
    _pad_and_clamp,
    _words_from_pieces,
)
from nemoscribe.events import Word


def test_pad_empty_list():
    assert _pad_and_clamp([], 100_000) == []


def test_pad_extends_both_sides():
    assert _pad_and_clamp([(16_000, 32_000)], 100_000) == [(12_800, 35_200)]


def test_pad_clamps_at_file_start():
    assert _pad_and_clamp([(1_600, 16_000)], 100_000) == [(0, 19_200)]


def test_pad_clamps_at_file_end():
    assert _pad_and_clamp([(1_600, 16_000)], 17_000) == [(0, 17_000)]


def test_neighbors_split_a_narrow_gap_at_the_midpoint():
    segs = [(0, 16_000), (17_000, 30_000)]  # 1,000-sample gap, mid = 16,500

    padded = _pad_and_clamp(segs, 40_000)

    assert padded == [(0, 16_500), (16_500, 33_200)]
    assert padded[0][1] <= padded[1][0]


def test_wide_gap_gets_full_padding_no_clamping():
    segs = [(10_000, 20_000), (40_000, 50_000)]  # 20,000-sample gap » 2 pad

    assert _pad_and_clamp(segs, 60_000) == [(6_800, 23_200), (36_800, 53_200)]


def piece(token, start, end):
    return {"token": token, "start": start, "end": end}


def test_words_from_hello_there_pieces():
    pieces = [
        piece("H", 1.36, 1.44),
        piece("el", 1.36, 1.44),
        piece("lo", 1.52, 1.60),
        piece(" there", 1.84, 1.92),
        piece(".", 2.00, 2.08),
        piece(" ", 2.00, 2.08),
    ]

    assert _words_from_pieces(pieces, 0.0) == (
        Word(text="Hello", start=1.36, end=1.60),
        Word(text="there.", start=1.84, end=2.08),
    )


def test_words_offset_shifts_to_file_clock():
    words = _words_from_pieces([piece("Hi", 0.5, 0.7)], 10.0)

    assert words == ((Word(text="Hi", start=10.5, end=10.7)),)


def test_whitespace_piece_is_a_boundary():
    pieces = [piece("foo", 0.0, 0.1), piece(" ", 0.1, 0.2), piece("bar", 0.2, 0.3)]

    assert [w.text for w in _words_from_pieces(pieces, 0.0)] == ["foo", "bar"]


def test_no_pieces_no_words():
    assert _words_from_pieces([], 0.0) == ()


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/hello.wav").exists(), reason="dev-machine fixture"
)
def test_transcriber_hello_wav_end_to_end(transcriber):
    events = transcriber.transcribe(load("scratch/hello.wav"), language="en-US")

    assert len(events) == 3
    for e in events:
        assert "hello" in e.text.lower()
        assert "there" in e.text.lower()
        assert e.words
        assert e.language == "en-US"
    assert all(a.end <= b.start for a, b in pairwise(events))


def test_detect_language_finds_tag():
    assert _detect_language("مرحبا. <ar-AR>") == "ar-AR"


def test_detect_language_empty_when_absent():
    assert _detect_language("Hello there. ") == ""


def test_detect_language_ignores_malformed_tags():
    assert _detect_language("Hello <En-us> <en> there") == ""


def test_detect_language_requires_complete_tag():
    assert _detect_language("maybe <en-USA> tag") == ""


def test_clamp_words_clip_runway_timestamps():
    words = (
        Word(text="Hello", start=1.0, end=1.4),
        Word(text="there.", start=1.6, end=2.7),
    )

    assert _clamp_words(words, 2.0) == (
        Word(text="Hello", start=1.0, end=1.4),
        Word(text="there.", start=1.6, end=2.0),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/hello.wav").exists(), reason="dev-machine fixture"
)
def test_auto_detects_english_hellos(transcriber):
    events = transcriber.transcribe(load("scratch/hello.wav"), language="auto")

    assert [e.language for e in events] == ["en-US"] * 3


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("scratch/arabic.wav").exists(), reason="dev-machine fixture"
)
def test_auto_on_arabic_gives_text_without_tags(transcriber):
    # Documents a measured model limitation (2026-08-23): short Arabic utterances
    # get correct text but no punctuation/tag ritual, even with decode runway.
    # If a future model version starts tagging, this test SHOULD fail — good
    # news arriving as a red test.
    events = transcriber.transcribe(load("scratch/arabic.wav"), language="auto")

    assert events
    assert all(e.language == "" for e in events)
    assert any("\u0600" <= ch <= "\u06ff" for ch in events[0].text)
