"""Tests for nemoscribe.sources"""

import numpy as np

from nemoscribe.audio import load
from nemoscribe.sources import file_chunks


def test_file_chunks_tile_the_file_with_timestamps(tmp_path, make_tone_wav):
    wav = tmp_path / "tone.wav"

    # 4000 samples → 2 full + 1 ragged chunk
    make_tone_wav(wav, seconds=0.25, rate=16_000)

    chunks = list(file_chunks(wav, chunk_s=0.1))

    assert [c.start for c in chunks] == [0.0, 0.1, 0.2]
    assert [len(c.samples) for c in chunks][:2] == [1600, 1600]
    assert np.array_equal(np.concatenate([c.samples for c in chunks]), load(wav))
