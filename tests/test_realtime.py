"""Live-pipeline logic without real audio hardware."""

import numpy as np
import pytest

from clackless import synth
from clackless.devices import Device
from clackless.dsp import Settings
from clackless.realtime import Pipeline, RingBuffer


def test_ring_waits_for_target_then_plays_in_order():
    rb = RingBuffer(target=100, high_water=300)
    rb.write(np.arange(50, dtype=np.float32))
    assert np.all(rb.read(10) == 0)  # not primed yet
    rb.write(np.arange(50, 150, dtype=np.float32))
    np.testing.assert_array_equal(rb.read(150), np.arange(150, dtype=np.float32))
    assert rb.underruns == 0


def test_ring_underrun_pads_with_silence_and_counts():
    rb = RingBuffer(target=10, high_water=40)
    rb.write(np.ones(10, dtype=np.float32))
    out = rb.read(15)
    np.testing.assert_array_equal(out, np.r_[np.ones(10), np.zeros(5)].astype(np.float32))
    assert rb.underruns == 1


def test_ring_overrun_drops_oldest_back_to_target():
    rb = RingBuffer(target=100, high_water=200)
    rb.write(np.arange(180, dtype=np.float32))
    rb.write(np.arange(180, 230, dtype=np.float32))  # exceeds high water
    assert rb.overruns == 1
    assert rb.fill == 100
    np.testing.assert_array_equal(rb.read(100), np.arange(130, 230, dtype=np.float32))


def test_ring_wraparound_preserves_data(rng):
    rb = RingBuffer(target=64, high_water=200)
    expected, got = [], []
    counter = 0
    for _ in range(500):
        n = int(rng.integers(1, 60))
        chunk = np.arange(counter, counter + n, dtype=np.float32)
        counter += n
        rb.write(chunk)
        expected.append(chunk)
        got.append(rb.read(n))
    data = np.concatenate(got)
    data = data[np.argmax(data > 0) - 1 :] if data.any() else data
    # Whatever was played is a contiguous, in-order slice of what was written.
    assert np.all(np.diff(data[data > 0]) == 1)
    assert rb.overruns == 0


def test_ring_rejects_bad_config():
    with pytest.raises(ValueError):
        RingBuffer(target=0)
    with pytest.raises(ValueError):
        RingBuffer(target=100, high_water=50)


def _dev(index, name, ins, outs):
    return Device(index, name, "Core Audio", ins, outs, 48000.0)


def test_pipeline_refuses_feedback_loop():
    d = _dev(1, "BlackHole 2ch", 2, 2)
    with pytest.raises(ValueError):
        Pipeline(d, d, 48000, Settings())
    Pipeline(d, d, 48000, Settings(), allow_same_device=True)  # Linux pulse routing


def test_pipeline_callbacks_clean_audio_end_to_end():
    """Drive the PortAudio callbacks by hand, as the audio threads would."""
    sr = 48000
    inp = _dev(1, "BlackHole 2ch", 2, 0)
    out = _dev(2, "Speakers", 0, 2)
    pipe = Pipeline(inp, out, sr, Settings(), blocksize_ms=10, buffer_ms=30)
    typing, _ = synth.keyboard(sr, 4.0, seed=8)
    stereo_in = np.stack([typing, typing], axis=1).astype(np.float32)

    played = []
    block = pipe.blocksize
    for i in range(0, stereo_in.shape[0] - block, block):
        pipe._input_cb(stereo_in[i : i + block], block, None, None)
        outdata = np.zeros((block, 2), dtype=np.float32)
        pipe._output_cb(outdata, block, None, None)
        played.append(outdata.copy())
    played = np.concatenate(played)

    assert np.array_equal(played[:, 0], played[:, 1])  # mono duplicated to both channels
    from clackless import metrics

    assert metrics.reduction_dba(typing[: played.shape[0]], played[:, 0], sr) > 15
    assert pipe.ring.underruns == 0 and pipe.ring.overruns == 0
    st = pipe.status()
    assert st.keystrokes > 10 and not st.bypass


def test_pipeline_runtime_controls():
    pipe = Pipeline(_dev(1, "a", 1, 0), _dev(2, "b", 0, 1), 16000, Settings())
    pipe.set_bypass(True)
    assert pipe.settings.bypass
    pipe.adjust(keystroke_strength=5, noise_reduction=-1)
    assert pipe.settings.keystroke_strength == 1.0
    assert pipe.settings.noise_reduction == 0.0
