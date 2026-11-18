"""Signal-processing invariants: reconstruction, latency, streaming, robustness."""

import numpy as np
import pytest

from clackless import synth
from clackless.dsp import Settings, Suppressor, process_array

RATES = [8000, 16000, 22050, 44100, 48000, 96000]


def _run_stream(sup: Suppressor, x: np.ndarray, sizes) -> np.ndarray:
    out, pos, i = [], 0, 0
    while pos < x.size:
        n = sizes[i % len(sizes)]
        out.append(sup.process(x[pos : pos + n]))
        pos += n
        i += 1
    return np.concatenate(out)


@pytest.mark.parametrize("sr", RATES)
def test_bypass_is_exact_delay(sr, rng):
    x = rng.standard_normal(sr) * 0.3
    sup = Suppressor(sr, Settings(bypass=True))
    y = sup.process(x)
    lat = sup.latency_samples
    np.testing.assert_allclose(y[lat:], x[: x.size - lat], atol=1e-5)
    assert np.max(np.abs(y[:lat])) < 1e-6


@pytest.mark.parametrize("sr", [16000, 48000])
def test_disabled_processing_reconstructs_input(sr, rng):
    x = rng.standard_normal(sr) * 0.3
    y = process_array(x, sr, Settings(keystroke_strength=0.0, noise_reduction=0.0))
    np.testing.assert_allclose(y, x, atol=1e-5)


@pytest.mark.parametrize("sr", RATES)
def test_latency_bound(sr):
    """NFR-2: added algorithmic latency <= 40 ms."""
    assert Suppressor(sr).latency_ms <= 40.0


def test_block_size_independence(rng):
    """NFR-6: output must not depend on the OS buffer size."""
    sr = 48000
    speech = synth.speech(sr, 2.0, seed=3)
    typing, _ = synth.keyboard(sr, 2.0, seed=3)
    x = speech + typing
    ref = Suppressor(sr, Settings(noise_reduction=0.5)).process(x)
    for sizes in ([1, 7, 64], [480], [4096, 3], list(rng.integers(1, 2000, 50))):
        y = _run_stream(Suppressor(sr, Settings(noise_reduction=0.5)), x, sizes)
        np.testing.assert_allclose(y, ref, atol=1e-6)


def test_output_length_always_matches_input(rng):
    sup = Suppressor(48000)
    for n in [0, 1, 5, 127, 128, 129, 1000, 48000]:
        assert sup.process(rng.standard_normal(n)).shape == (n,)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(48000),
        np.full(48000, np.nan),
        np.full(48000, np.inf),
        np.concatenate([np.ones(24000) * 50, -np.ones(24000) * 50]),
        np.sign(np.sin(np.arange(48000) / 5.0)),  # full-scale square wave, clipped
        np.random.default_rng(0).standard_normal(48000) * 1e-9,  # near-silence
    ],
    ids=["silence", "nan", "inf", "huge", "clipped", "tiny"],
)
def test_pathological_input_never_crashes(x):
    """NFR-7: robust against bad input."""
    y = process_array(x, 48000, Settings(noise_reduction=1.0))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 8.0


def test_silence_stays_silent():
    y = process_array(np.zeros(48000), 48000, Settings(noise_reduction=1.0))
    assert np.max(np.abs(y)) == 0.0


def test_reset_restores_initial_state(rng):
    sr = 16000
    x = rng.standard_normal(sr) * 0.1
    sup = Suppressor(sr)
    a = sup.process(x)
    sup.process(rng.standard_normal(sr))
    sup.reset()
    np.testing.assert_array_equal(sup.process(x), a)
    assert sup.stats.frames > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(keystroke_strength=1.5),
        dict(keystroke_strength=-0.1),
        dict(noise_reduction=2.0),
        dict(detect_low_db=9, detect_high_db=4),
        dict(lookahead_ms=1),
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Suppressor(48000, Settings(**kwargs))


def test_too_low_sample_rate_rejected():
    with pytest.raises(ValueError):
        Suppressor(4000)


def test_gains_never_amplify(rng):
    """Suppression only ever removes energy, so it cannot cause clipping."""
    sr = 48000
    typing, _ = synth.keyboard(sr, 3.0, seed=5)
    x = synth.speech(sr, 3.0, seed=5) + typing

    class Spy(Suppressor):
        max_gain = 0.0

        def _gain(self, mag, smag):
            g = super()._gain(mag, smag)
            Spy.max_gain = max(Spy.max_gain, float(np.max(g)))
            return g

    Spy(sr, Settings(noise_reduction=1.0)).process(x)
    assert Spy.max_gain <= 1.0 + 1e-12


def test_keystroke_counter_is_plausible():
    sr = 48000
    typing, onsets = synth.keyboard(sr, 8.0, seed=2)
    room = synth.stationary_noise(sr, 8.0, seed=2) * 10 ** (-60 / 20)
    sup = Suppressor(sr)
    sup.process(typing + room)
    # Each keystroke has a press and a release; fast rollover may merge some events.
    assert len(onsets) <= sup.stats.keystrokes <= 2 * len(onsets) + 2


def test_strength_scales_suppression():
    from clackless import metrics

    sr = 48000
    typing, _ = synth.keyboard(sr, 4.0, seed=4)
    room = synth.stationary_noise(sr, 4.0, seed=4) * 10 ** (-55 / 20)
    reductions = [
        metrics.reduction_dba(
            typing + room, process_array(typing + room, sr, Settings(keystroke_strength=s)), sr
        )
        for s in (0.0, 0.5, 1.0)
    ]
    assert reductions[0] == pytest.approx(0.0, abs=0.01)
    assert 0.0 < reductions[1] < reductions[2]
