"""Tests for nemoscribe.events"""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from nemoscribe.events import TranscriptEvent, Word


def test_construction_and_access():
    words = (
        Word(text="Hello", start=1.36, end=1.60),
        Word(text="there.", start=1.84, end=1.92),
    )
    e = TranscriptEvent(
        text="Hello there.",
        start=1.36,
        end=1.92,
        language="en-US",
        source="",
        words=words,
    )

    assert e.text == "Hello there."
    assert e.words[1].text == "there."
    assert e.end - e.start == pytest.approx(0.56)


def test_equality_is_by_value():
    a = TranscriptEvent(text="hi", start=0.0, end=1.0, language="en-US", source="")
    b = TranscriptEvent(text="hi", start=0.0, end=1.0, language="en-US", source="")
    c = TranscriptEvent(text="hi", start=0.0, end=1.5, language="en-US", source="")

    assert a == b
    assert a != c


def test_immutable_and_words_default():
    e = TranscriptEvent(text="hi", start=0.0, end=1.0, language="", source="")

    assert e.words == ()
    with pytest.raises(FrozenInstanceError):
        e.text = "edited"
