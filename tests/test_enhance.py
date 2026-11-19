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
from clackless.evaluate import load_mono, split_voice, typing_mask

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
    gains, removed, report = [], [], []
    for name in REAL:
        keys = load_mono(str(DATA / f"keyboard_{name}.flac"), sr)[: voice.size]
        keys = keys / np.max(np.abs(keys)) * np.max(np.abs(voice))
        out = process_array(voice + keys, sr, DEFAULT)
        plain = process_array(voice + keys, sr, Settings(mute_between_speech=True))
        sdr, sdr_plain = metrics.si_sdr(out, voice), metrics.si_sdr(plain, voice)
        report.append(f"{name}: SI-SDR {sdr_plain:.1f} -> {sdr:.1f}")
        assert sdr >= 7.0 and sdr >= sdr_plain - 0.5, report  # robotic CI voices (espeak, SAPI): 8.5-10
        gains.append(sdr - sdr_plain)
        removed.append(
            metrics.reduction_dba(keys, split_voice(out, voice)[1], sr)
            - metrics.reduction_dba(keys, split_voice(plain, voice)[1], sr)
        )
    assert np.mean(gains) >= 4.0, report  # measured 5.6 (macOS voice), 8 on natural voices
    assert np.mean(removed) >= 3.0, (removed, report)  # measured 5.6


@pytest.mark.parametrize("name", REAL)
def test_default_silences_real_typing(name):
    """Every keystroke; the room tone between bursts of typing is (rightly) kept."""
    sr = 48000
    x = load_mono(str(DATA / f"keyboard_{name}.flac"), sr)
    keys = typing_mask(x, sr)
    y = process_array(x, sr, DEFAULT)
    assert metrics.reduction_dba(x[keys], y[keys], sr) >= 40.0  # measured 89-100


def _untouched(y, x):
    """Share of the samples that come out exactly as they went in."""
    return float(np.mean(np.abs(y - x) < 1e-5))


def test_a_voice_nobody_types_over_is_not_processed(tts_speech):
    """No typing, no cleanup: every sound of the voice passes as it came in
    (after the first second, which is cleaned in case a stream starts mid-typing)."""
    sr = 48000
    x = np.concatenate([np.zeros(2 * sr), tts_speech(sr)])
    x = x + synth.stationary_noise(sr, x.size / sr + 0.1, seed=4)[: x.size] * 1e-3
    y = process_array(x, sr, DEFAULT)
    assert _untouched(y[2 * sr :], x[2 * sr :]) >= 0.8  # measured 1.0; a false click would cost ~1 s


def test_typing_mid_sentence_is_cleaned_from_its_first_keystroke():
    """She talks, then starts typing mid-sentence: the voice before it is
    untouched, and the cleanup is on before the first click is heard."""
    sr = 48000
    voice = synth.speech(sr, 8.0, seed=3)
    typing, onsets = synth.keyboard(sr, 3.0, seed=5)
    start = 4 * sr - onsets[0]  # first keystroke at 4 s
    kb = np.zeros(voice.size)
    kb[start : start + typing.size] = typing
    x = voice + kb
    gated = process_array(x, sr, DEFAULT)
    always = process_array(
        x, sr, Settings(mute_between_speech=True, enhance_speech=True, only_while_typing=False)
    )

    before = slice(int(1.5 * sr), 4 * sr - sr // 20)
    assert _untouched(gated[before], x[before]) == 1.0

    first = slice(4 * sr - sr // 200, 4 * sr + sr // 20)  # the first keystroke
    left = metrics.reduction_dba(kb[first], split_voice(gated[first], voice[first])[1], sr)
    best = metrics.reduction_dba(kb[first], split_voice(always[first], voice[first])[1], sr)
    assert left >= best - 1.0, (left, best)


def test_gated_output_is_block_size_independent(tts_speech):
    sr = 16000
    voice = tts_speech(sr)[: 3 * sr]
    typing, _ = synth.keyboard(sr, 2.0, seed=9)
    x = np.concatenate([voice, typing, voice])
    whole = Suppressor(sr, DEFAULT).process(x)
    sup, rng, parts, i = Suppressor(sr, DEFAULT), np.random.default_rng(3), [], 0
    while i < x.size:
        n = int(rng.integers(1, 700))
        parts.append(sup.process(x[i : i + n]))
        i += n
    assert np.allclose(np.concatenate(parts), whole, atol=1e-6)


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
