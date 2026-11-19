"""Neural speech enhancement: DeepFilterNet3, streamed frame by frame.

DeepFilterNet3 (Schröter et al., MIT / Apache-2.0) estimates, every 10 ms,
which parts of the spectrum are voice and which are anything else, and rebuilds
the voice with a short complex filter. It removes typing *under* a voice, the
case the signal-processing suppressor cannot reach without damaging the voice.

The model runs at 48 kHz. Other rates are resampled in and out with a
stateful polyphase filter, so any block size gives identical output.
``scripts/export_deepfilternet.py`` produces the bundled ONNX file.
"""

from __future__ import annotations

from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy import signal

__all__ = ["SpeechEnhancer", "make_enhancer"]

_RATE = 48000
_HOP = 480  # 10 ms
_MODEL_DELAY = 1440  # samples at 48 kHz: STFT window + 2 frames of look-ahead
_STATES = [
    ("erb_norm", lambda: np.linspace(-60.0, -90.0, 32, dtype=np.float32)),
    ("unit_norm", lambda: np.linspace(0.001, 0.0001, 96, dtype=np.float32).reshape(1, 96, 1)),
    ("analysis_mem", lambda: np.zeros(480, np.float32)),
    ("synthesis_mem", lambda: np.zeros(480, np.float32)),
    ("erb_buf", lambda: np.zeros((1, 1, 3, 32), np.float32)),
    ("spec_feat_buf", lambda: np.zeros((1, 2, 3, 96), np.float32)),
    ("c0_buf", lambda: np.zeros((1, 64, 5, 96), np.float32)),
    ("spec_buf_x", lambda: np.zeros((5, 481, 2), np.float32)),
    ("spec_buf_y", lambda: np.zeros((7, 481, 2), np.float32)),
    ("enc_hidden", lambda: np.zeros((1, 1, 256), np.float32)),
    ("erb_dec_hidden", lambda: np.zeros((2, 1, 256), np.float32)),
    ("df_dec_hidden", lambda: np.zeros((2, 1, 256), np.float32)),
]


class _Resampler:
    """Streaming rational resampler (up by `up`, low-pass, down by `down`).

    Output sample m sits at input time m * down / up minus a fixed group delay
    of `half` / up input samples, independent of how the input is chunked."""

    def __init__(self, up: int, down: int, h: np.ndarray):
        self.up, self.down = up, down
        taps = -(-h.size // up)
        poly = np.zeros(taps * up)
        poly[: h.size] = h * up
        self._poly = poly.reshape(taps, up).T[:, ::-1].copy()  # [phase, tap], oldest sample first
        self._taps = taps
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(self._taps - 1)  # samples before the stream started are silence
        self._n_in = 0
        self._m = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._hist, x])
        start = self._n_in - (self._taps - 1)  # absolute index of buf[0]
        self._n_in += x.size
        # Outputs whose newest input sample is now available.
        m_end = (self._n_in * self.up + self.down - 1) // self.down
        m = np.arange(self._m, m_end)
        self._m = m_end
        if m.size:
            t = m * self.down
            j = t // self.up  # newest input sample
            phase = t - j * self.up
            idx = (j - start - (self._taps - 1))[:, None] + np.arange(self._taps)[None, :]
            y = np.einsum("ij,ij->i", buf[idx], self._poly[phase])
        else:
            y = np.zeros(0)
        self._hist = buf[buf.size - (self._taps - 1) :] if self._taps > 1 else np.zeros(0)
        return y


def _lowpass(up: int, rate: int, half_taps: int) -> np.ndarray:
    """Kaiser-windowed sinc at the intermediate rate (rate * up), with a group
    delay of exactly `half_taps` samples at `rate`. Passes 0.9 x the lower Nyquist."""
    n = 2 * half_taps * up + 1
    cutoff = 0.9 * min(rate, _RATE) / (rate * up)
    return signal.firwin(n, cutoff, window=("kaiser", 8.0))


class SpeechEnhancer:
    """Streams DeepFilterNet3. ``process`` returns as many samples as it is given,
    delayed by exactly ``latency_samples`` (at rates where 30 ms is a whole
    number of samples, i.e. every common rate)."""

    def __init__(self, sample_rate: int):
        from importlib.resources import files

        import onnxruntime as ort

        model = files("clackless").joinpath("models/deepfilternet3_streaming.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sr = int(sample_rate)
        g = gcd(self.sr, _RATE)
        self._up, self._down = _RATE // g, self.sr // g  # input -> 48 kHz
        half = 16  # taps per side, in input samples
        self._resample = self.sr != _RATE
        if self._resample:
            h = _lowpass(self._up, self.sr, half)
            self._in_rs = _Resampler(self._up, self._down, h)
            self._out_rs = _Resampler(self._down, self._up, h)
        # Output prefill: covers the 10 ms framing (+1 sample for resampler rounding).
        self._prefill = -(-_HOP * self.sr // _RATE) + (1 if self._resample else 0)
        model_delay = _MODEL_DELAY * self.sr / _RATE
        self.latency_samples = int(round(self._prefill + model_delay + (2 * half if self._resample else 0)))
        self.reset()

    def reset(self) -> None:
        self._state = {name: make() for name, make in _STATES}
        self._fifo = np.zeros(0, np.float32)
        self._out = np.zeros(self._prefill)
        self._raw_delay = np.zeros(self.latency_samples)
        if self._resample:
            self._in_rs.reset()
            self._out_rs.reset()

    def process(self, x: np.ndarray, bypass: bool = False) -> np.ndarray:
        """Enhance a block; with ``bypass`` return the input delayed by the same amount."""
        enhanced, delayed = self.process_pair(x)
        return delayed if bypass else enhanced

    def process_pair(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(enhanced block, input block delayed by the same amount)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n = x.size
        y = self._in_rs.push(x) if self._resample else x
        fifo = np.concatenate([self._fifo, y.astype(np.float32)])
        k = fifo.size // _HOP
        frames = []
        for i in range(k):
            res = self._sess.run(None, {"frame": fifo[i * _HOP : (i + 1) * _HOP], **self._state})
            frames.append(res[0])
            self._state = {name: v for (name, _), v in zip(_STATES, res[1:], strict=True)}
        self._fifo = fifo[k * _HOP :]
        if frames:
            z = np.concatenate(frames).astype(np.float64)
            z = self._out_rs.push(z) if self._resample else z
            self._out = np.concatenate([self._out, z])
        out, self._out = self._out[:n], self._out[n:]
        if out.size < n:  # cannot happen with the prefill, but never return short
            out = np.concatenate([out, np.zeros(n - out.size)])
        raw = np.concatenate([self._raw_delay, x])
        delayed, self._raw_delay = raw[:n], raw[n:]
        return out, delayed


def make_enhancer(sample_rate: int) -> Optional[SpeechEnhancer]:
    """The enhancer, or None when onnxruntime or the model is unavailable."""
    try:
        return SpeechEnhancer(sample_rate)
    except Exception:
        return None
