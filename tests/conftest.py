import functools

import numpy as np
import pytest

from clackless import synth


@functools.cache
def _tts(sr: int):
    return synth.tts_speech(sr)


@pytest.fixture
def tts_speech():
    """Real synthesized speech from the OS voice, or skip if the OS has none."""

    def get(sr: int) -> np.ndarray:
        x = _tts(sr)
        if x is None:
            pytest.skip("no text-to-speech engine on this machine")
        return x

    return get


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
