"""Tests for nemoscribe.sources"""

import io
import sys
import types

import numpy as np
import pytest

from nemoscribe.audio import load
from nemoscribe.sources import (
    SourceError,
    _default_input_device,
    _default_monitor,
    file_chunks,
    mic_chunks,
    system_chunks,
)


def test_file_chunks_tile_the_file_with_timestamps(tmp_path, make_tone_wav):
    wav = tmp_path / "tone.wav"

    # 4000 samples → 2 full + 1 ragged chunk
    make_tone_wav(wav, seconds=0.25, rate=16_000)

    chunks = list(file_chunks(wav, chunk_s=0.1))

    assert [c.start for c in chunks] == [0.0, 0.1, 0.2]
    assert [len(c.samples) for c in chunks][:2] == [1600, 1600]
    assert np.array_equal(np.concatenate([c.samples for c in chunks]), load(wav))


def _stub_sd(devices=(), stream_cls=None):
    return types.SimpleNamespace(
        query_devices=lambda: list(devices),
        InputStream=stream_cls,
    )


def test_default_input_prefers_pipewire_bridge(monkeypatch):
    stub = _stub_sd(
        devices=[{"name": "sof-hda"}, {"name": "pipewire"}, {"name": "pulse"}]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", stub)

    assert _default_input_device() == "pipewire"


def test_default_input_falls_back_to_portaudio_default(monkeypatch):
    stub = _stub_sd(devices=[{"name": "Microphone (USB Audio)"}])  # a Windows-ish world
    monkeypatch.setitem(sys.modules, "sounddevice", stub)

    assert _default_input_device() is None


class _StubStream:
    """Delivers three known blocks through the callback, then idles."""

    def __init__(self, callback, **kwargs):
        self._callback = callback

    def __enter__(self):
        for i in range(3):
            block = np.full((1600, 1), 0.1 * (i + 1), dtype=np.float32)
            self._callback(block, 1600, None, None)
        return self

    def __exit__(self, *exc):
        return False


def test_mic_chunks_sample_clock_and_copy(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _stub_sd(stream_cls=_StubStream))

    gen = mic_chunks()
    chunks = [next(gen) for _ in range(3)]
    gen.close()

    assert [c.start for c in chunks] == [0.0, 0.1, 0.2]
    assert all(len(c.samples) == 1600 for c in chunks)
    assert [float(c.samples[0]) for c in chunks] == pytest.approx([0.1, 0.2, 0.3])


def test_default_monitor_appends_suffix(monkeypatch):
    stub = lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="alsa_output.usb\n"
    )
    monkeypatch.setattr("nemoscribe.sources.subprocess.run", stub)

    assert _default_monitor() == "alsa_output.usb.monitor"


def test_default_monitor_failure_raises(monkeypatch):
    stub = lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr("nemoscribe.sources.subprocess.run", stub)

    with pytest.raises(SourceError):
        _default_monitor()


class _FakePopen:
    last: "_FakePopen | None" = None

    def __init__(self, cmd, stdout=None):
        self.stdout = io.BytesIO(np.arange(4000, dtype=np.float32).tobytes())
        self.terminated = False
        self.waited = False
        _FakePopen.last = self

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.waited = True


def test_system_chunks_tile_and_teardown(monkeypatch):
    monkeypatch.setattr("nemoscribe.sources.subprocess.Popen", _FakePopen)

    # explicit name: pactl never runs
    chunks = list(system_chunks(source_name="fake.monitor"))

    joined = np.concatenate([c.samples for c in chunks])
    assert np.array_equal(joined, np.arange(4000, dtype=np.float32))

    # 2 full + 1 ragged
    assert [c.start for c in chunks] == pytest.approx([0.0, 0.1, 0.2])

    proc = _FakePopen.last
    assert proc is not None
    assert proc.terminated and proc.waited


def test_default_monitor_missing_pactl_raises(monkeypatch):
    def no_pactl(*a, **k):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr("nemoscribe.sources.subprocess.run", no_pactl)

    with pytest.raises(SourceError):
        _default_monitor()
