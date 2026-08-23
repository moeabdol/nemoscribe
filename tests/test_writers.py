"""Tests for nemoscribe.writers"""

import json

from nemoscribe.events import TranscriptEvent, Word
from nemoscribe.writers import write_txt
from nemoscribe.writers.jsonl import write_jsonl
from nemoscribe.writers.srt import _event_cues, _timestamp, write_srt


def make_event(**overrides):
    defaults = {"text": "hi", "start": 0.0, "end": 1.0, "language": "", "source": ""}
    return TranscriptEvent(**{**defaults, **overrides})


def test_txt_of_no_events_is_empty_file():
    assert write_txt([]) == ""


def test_txt_one_line_per_event_sorted_by_time():
    events = [
        make_event(text="Second one.", start=2.0, end=3.0),
        make_event(text="First one.", start=0.5),
    ]

    assert write_txt(events) == "First one.\nSecond one.\n"


def test_timestamp_zero():
    assert _timestamp(0.0) == "00:00:00,000"


def test_timestamp_non_zero():
    assert _timestamp(1.36) == "00:00:01,360"


def test_timestamp_rolls_over_cleanly():
    assert _timestamp(59.9995) == "00:01:00,000"


def test_timestamp_hours_minutes_seconds_ms():
    assert _timestamp(3661.5) == "01:01:01,500"


def test_srt_event_without_words_is_one_cue():
    e = make_event(text="Hello, there.", start=0.86, end=1.6)

    assert _event_cues(e, 84) == [(0.86, 1.6, "Hello, there.")]


def test_srt_packs_words_greedily_within_budget():
    words = (
        Word(text="alpha", start=0.0, end=0.4),
        Word(text="beta", start=0.5, end=0.9),
        Word(text="gamma", start=1.0, end=1.4),
        Word(text="delta", start=1.5, end=1.9),
    )
    e = make_event(text="alpha beta gamma delta", start=0.0, end=1.9, words=words)

    assert _event_cues(e, 11) == [(0.0, 0.9, "alpha beta"), (1.0, 1.9, "gamma delta")]


def test_srt_oversized_word_gets_its_own_cue():
    words = (
        Word(text="w" * 90, start=0.0, end=1.0),  # longer than any budget
        Word(text="ok", start=1.1, end=1.3),
    )
    e = make_event(text="irrelevant", start=0.0, end=1.3, words=words)

    cues = _event_cues(e, 84)

    assert len(cues) == 2
    assert cues[0] == (0.0, 1.0, "w" * 90)


def test_srt_cue_times_come_from_words_not_event_bounds():
    # event bounds include VAD padding; the word knows when speech happened
    words = (Word(text="Hi.", start=1.0, end=1.2),)
    e = make_event(text="Hi.", start=0.8, end=1.6, words=words)

    assert _event_cues(e, 84) == [(1.0, 1.2, "Hi.")]


def test_srt_golden_output():
    events = [
        make_event(text="Hello, there.", start=0.86, end=1.6),
        make_event(text="Hello, again.", start=2.34, end=3.23),
    ]

    expected = (
        "1\n"
        "00:00:00,560 --> 00:00:01,600\n"
        "Hello, there.\n"
        "\n"
        "2\n"
        "00:00:02,040 --> 00:00:03,230\n"
        "Hello, again.\n"
    )
    assert write_srt(events) == expected


def test_jsonl_records_have_manifest_keys():
    events = [
        make_event(text="Hello.", start=0.86, end=1.6, source="me", language="en-US")
    ]

    lines = write_jsonl(events, audio_filepath="hello.wav").splitlines()
    record = json.loads(lines[0])

    assert record == {
        "audio_filepath": "hello.wav",
        "offset": 0.86,
        "duration": 0.74,
        "text": "Hello.",
        "source": "me",
        "target_lang": "en-US",
    }


def test_jsonl_omits_empty_optionals():
    record = json.loads(write_jsonl([make_event()], audio_filepath="a.wav"))

    assert "target_lang" not in record
    assert "source" not in record


def test_jsonl_keeps_arabic_readable():
    out = write_jsonl([make_event(text="مرحبا بالعالم")], audio_filepath="ar.wav")

    assert "مرحبا بالعالم" in out
    assert "\\u" not in out


def test_srt_lead_in_shifts_cue_start_early():
    events = [make_event(text="Hi.", start=1.0, end=2.0)]

    assert "00:00:00,700 --> 00:00:02,000" in write_srt(events)


def test_srt_lead_in_clamps_at_zero():
    events = [make_event(text="Hi.", start=0.1, end=1.0)]

    assert "00:00:00,000 --> 00:00:01,000" in write_srt(events)


def test_srt_lead_in_never_overlaps_previous_cue():
    events = [
        make_event(text="One.", start=0.5, end=2.0),
        make_event(text="Two.", start=2.1, end=3.0),
    ]

    assert "00:00:02,000 --> 00:00:03,000" in write_srt(events)
