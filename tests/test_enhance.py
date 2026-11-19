"""Neural voice cleanup (DeepFilterNet3): typing *under* a voice.

The enhancer is not a per-bin gain, so it is measured directly: the voice
part of the output is its projection onto the clean voice, the rest counts as
left-over typing (which also includes any voice distortion: conservative).
"""

from pathlib import Path

import numpy as np
import pytest

from clackless import metrics, synth
from clackless.dsp import Settings, Suppressor, process_array
from clackless.enhance import SpeechEnhancer
from clackless.evaluate import load_mono, split_voice

DATA = Path(__file__).parent / "data"
REAL = ["laptop", "computer", "keychron"]
DEFAULT = Settings(mute_between_speech=True, enhance_speech=True)  # what `listen`/`mic` use


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_enhancer_delay_is_exact_and_block_size_independent(sr, tts_speech):
    voice = tts_speech(sr)[: 4 * sr]
    e = SpeechEnhancer(sr)
    lat = e.latency_samples
    x = np.concatenate([voice, np.zeros(lat + 100)])
    y = e.process(x)
    lags = range(lat - 50, lat + 51)
    best = max(lags, key=lambda d: float(np.dot(y[d : d + voice.size], voice)))
    assert best == lat
    assert metrics.si_sdr(y[lat : lat + voice.size], voice) >= 15.0

    e2, rng, parts, i = SpeechEnhancer(sr), np.random.default_rng(0), [], 0
    while i < x.size:
        n = int(rng.integers(1, 900))
        parts.append(e2.process(x[i : i + n]))
        i += n
    assert np.allclose(np.concatenate(parts), y, atol=1e-6)


def test_enhancer_bypass_is_a_pure_delay():
    sr = 48000
    x = np.random.default_rng(1).standard_normal(sr) * 0.1
    e = SpeechEnhancer(sr)
    y = e.process(x, bypass=True)
    assert np.allclose(y[e.latency_samples :], x[: x.size - e.latency_samples])


def test_typing_under_a_voice_is_reduced(tts_speech):
    """Real typing as loud as the voice, while the voice talks: the case the
    signal processing alone cannot fix. Judged over the three keyboards (one
    clip alone is noisy: how much typing falls inside words varies)."""
    sr = 48000
    voice = np.concatenate([np.zeros(sr // 2), tts_speech(sr)[: 6 * sr]])
    gains, removed = [], []
    for name in REAL:
        keys = load_mono(str(DATA / f"keyboard_{name}.flac"), sr)[: voice.size]
        keys = keys / np.max(np.abs(keys)) * np.max(np.abs(voice))
        out = process_array(voice + keys, sr, DEFAULT)
        plain = process_array(voice + keys, sr, Settings(mute_between_speech=True))
        sdr, sdr_plain = metrics.si_sdr(out, voice), metrics.si_sdr(plain, voice)
        assert sdr >= 12.0 and sdr >= sdr_plain - 0.5, name
        gains.append(sdr - sdr_plain)
        removed.append(
            metrics.reduction_dba(keys, split_voice(out, voice)[1], sr)
            - metrics.reduction_dba(keys, split_voice(plain, voice)[1], sr)
        )
    assert np.mean(gains) >= 4.0  # measured 5.6 (macOS voice), 8 on natural voices
    assert np.mean(removed) >= 3.0  # measured 5.6


@pytest.mark.parametrize("name", REAL)
def test_default_silences_real_typing(name):
    sr = 48000
    x = load_mono(str(DATA / f"keyboard_{name}.flac"), sr)
    assert metrics.reduction_dba(x, process_array(x, sr, DEFAULT), sr) >= 40.0


def test_default_keeps_a_clean_voice(tts_speech):
    sr = 48000
    voice = np.concatenate([np.zeros(sr), tts_speech(sr)])
    room = synth.stationary_noise(sr, voice.size / sr + 0.1, seed=4)[: voice.size] * 1e-3
    out = process_array(voice + room, sr, DEFAULT)
    assert metrics.si_sdr(out, voice) >= 18.0
    onset = slice(sr, sr + sr // 5)
    assert metrics.reduction_db(voice[onset], out[onset]) <= 1.5


def test_default_latency_unchanged_by_enhancer():
    """The voice detector hears the raw input, so the enhancer's 40 ms comes out
    of the mute look-ahead: the total stays within the mute-mode bound."""
    for sr in (16000, 44100, 48000):
        with_enh = Suppressor(sr, DEFAULT)
        assert with_enh.enhancing
        assert with_enh.latency_ms <= 120.0
        assert abs(with_enh.latency_ms - Suppressor(sr, Settings(mute_between_speech=True)).latency_ms) <= 3.0
