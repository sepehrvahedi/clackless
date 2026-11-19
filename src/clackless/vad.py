"""Streaming voice-activity detector tuned to tell voices from typing.

Voiced speech is periodic (a glottal pulse every 2.5-20 ms) with a pitch that
stays stable for tens of milliseconds. A keystroke can ring periodically too
(key caps, the case, the desk), but only for a few milliseconds. So "voice"
means: in the 150-1000 Hz band (the harmonics of any human pitch, above
desk and chassis booms), the normalised autocorrelation peaks at >= 0.7 for a
pitch-stable run of >= 80 ms, at least 9 dB above the 2-second background
floor (a steady hum is the floor itself). Measured on real recordings, typing
triggers this on < 1 % of frames, voices on 30-60 %.

A confirmed voice keeps the detector active for 2 s (consonants, pauses, and
moments where loud typing masks the pitch). 30 ms of very clean periodicity (>= 0.8) already
activates it for 150 ms, so the start of a word is protected while the 80 ms
run is still being confirmed; keyboard ringing practically never gets there.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

__all__ = ["VoiceDetector"]

_BAND_HZ = (150.0, 1000.0)
_PITCH_HZ = (50.0, 400.0)  # down to vocal fry
_WINDOW_S = 0.040
_BLOCK_S = 0.010
_VOICING = 0.7
_ABOVE_FLOOR_DB = 9.0
_MIN_LEVEL_DBFS = -70.0  # quieter than this is never a voice (e.g. filter ringing in digital silence)
_CONFIRM_BLOCKS = 8  # 80 ms of stable pitch = a voice
_EARLY_BLOCKS = 3  # 30 ms of very clean periodicity: a voice is starting
_EARLY_VOICING = 0.8
_HANG_CONFIRMED_S = 2.0  # loud typing can mask a voice's pitch mid-sentence
_HANG_EARLY_S = 0.15
_FLOOR_S = 2.0


class VoiceDetector:
    def __init__(
        self, sample_rate: int, hold_s: float = _HANG_CONFIRMED_S, early_hold_s: float = _HANG_EARLY_S
    ):
        self.sr = int(sample_rate)
        self.dec = max(1, self.sr // 8000)
        self.fs = self.sr / self.dec
        hi = min(_BAND_HZ[1], 0.45 * self.fs)
        self._sos = signal.butter(4, [_BAND_HZ[0], hi], "band", fs=self.sr, output="sos")
        self.win = int(_WINDOW_S * self.fs)
        self.min_lag = int(self.fs / _PITCH_HZ[1])
        self.max_lag = int(self.fs / _PITCH_HZ[0])
        self.block = int(_BLOCK_S * self.fs)
        self._min_energy = self.win * 10 ** (_MIN_LEVEL_DBFS / 10)
        self._hang_confirmed = int(hold_s / _BLOCK_S)
        self._hang_early = int(early_hold_s / _BLOCK_S)
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros((self._sos.shape[0], 2))
        self._phase = 0
        self._buf = np.zeros(self.win + self.max_lag)
        self._pending = 0
        self._energies: deque = deque(maxlen=int(_FLOOR_S / _BLOCK_S))
        self._run = 0
        self._strong_run = 0
        self._prev_lag = 0
        self._hang = 0

    @property
    def active(self) -> bool:
        """True while a voice is present (or was, within the hang-over time)."""
        return self._hang > 0

    def push(self, chunk: np.ndarray) -> None:
        y, self._zi = signal.sosfilt(self._sos, chunk, zi=self._zi)
        d = y[self._phase :: self.dec]
        self._phase = (self._phase - y.size) % self.dec
        if d.size == 0:
            return
        self._buf = np.concatenate([self._buf, d])[-(self.win + self.max_lag) :]
        self._pending += d.size
        while self._pending >= self.block:
            self._pending -= self.block
            self._decide()

    def _decide(self) -> None:
        a = self._buf[-self.win :]
        e0 = float(np.dot(a, a)) + 1e-18
        self._energies.append(e0)
        # Earlier windows, lag = min_lag .. max_lag samples back.
        views = sliding_window_view(self._buf, self.win)  # row i starts at sample i
        earlier = views[self.max_lag - np.arange(self.min_lag, self.max_lag + 1)]
        num = earlier @ a
        den = np.sqrt(e0 * (np.einsum("ij,ij->i", earlier, earlier) + 1e-18))
        r = num / den
        best = int(np.argmax(r))
        lag = self.min_lag + best
        floor = max(min(self._energies), self._min_energy)
        voiced = r[best] >= _VOICING and e0 >= floor * 10 ** (_ABOVE_FLOOR_DB / 10)
        # Same pitch as the previous block, or an octave jump (the autocorrelation
        # may lock onto double or half the period).
        ratio = lag / self._prev_lag if self._prev_lag else 0.0
        continuous = any(abs(ratio - m) <= 0.15 * m for m in (0.5, 1.0, 2.0))
        if voiced and self._run > 0 and continuous:
            self._run += 1
        else:
            self._run = 1 if voiced else 0
        stable = voiced and self._run > 1
        # A word starting is steady or rising; a resonance (a key's thump) is dying away.
        steady = len(self._energies) < 3 or e0 >= 0.5 * self._energies[-3]
        strong = r[best] >= _EARLY_VOICING and voiced and steady
        self._strong_run = self._strong_run + 1 if strong and (stable or self._strong_run == 0) else 0
        self._prev_lag = lag if voiced else 0
        if self._run >= _CONFIRM_BLOCKS:
            self._hang = self._hang_confirmed
        elif self._strong_run >= _EARLY_BLOCKS:
            self._hang = max(self._hang, self._hang_early)
        elif self._hang > 0:
            self._hang -= 1


_SILERO_RATE = 16000
_SILERO_CHUNK = 512  # 32 ms
_SILERO_CONTEXT = 64
_SPEECH_ON = 0.6  # speech starts at >= 0.6 for two chunks in a row ...
_SPEECH_ON_SURE = 0.8  # ... or >= 0.8 in a single chunk (one-chunk blips at 0.5-0.8 are clicks/pings)
_SPEECH_OFF = 0.35


class SileroVoiceDetector:
    """Neural voice-activity detector (Silero VAD v5, MIT) via onnxruntime.

    Far more robust than pitch tracking: on real typing recordings it reports
    speech < 0.5 % of the time, on voices (including under loud typing) 60-85 %.
    Same interface as :class:`VoiceDetector`.
    """

    def __init__(
        self, sample_rate: int, hold_s: float = _HANG_CONFIRMED_S, early_hold_s: float = _HANG_EARLY_S
    ):
        from importlib.resources import files

        import onnxruntime as ort  # optional dependency; callers fall back to VoiceDetector

        model = files("clackless").joinpath("models/silero_vad.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sr = int(sample_rate)
        self._hold_chunks = max(1, int(round(hold_s * _SILERO_RATE / _SILERO_CHUNK)))
        # Anti-alias low-pass before resampling to 16 kHz.
        if self.sr > _SILERO_RATE:
            self._sos = signal.butter(6, 7000, "low", fs=self.sr, output="sos")
        else:
            self._sos = None
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(_SILERO_CONTEXT, np.float32)
        self._pos = 0.0  # fractional read position (input samples) of the next 16 kHz sample
        self._tail = np.zeros(1)  # last input sample, for interpolation across blocks
        self._buf16 = np.zeros(0, np.float32)
        self._zi = None if self._sos is None else np.zeros((self._sos.shape[0], 2))
        self._speaking = False
        self._hang = 0
        self.prob = 0.0

    @property
    def active(self) -> bool:
        return self._speaking or self._hang > 0

    def push(self, chunk: np.ndarray) -> None:
        x = np.asarray(chunk, dtype=np.float64)
        if self._sos is not None:
            x, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        # Streaming linear-interpolation resampler (plenty for voice detection).
        xs = np.concatenate([self._tail, x])
        step = self.sr / _SILERO_RATE
        n_out = int(np.floor((xs.size - 1 - self._pos) / step)) + 1 if xs.size - 1 >= self._pos else 0
        if n_out > 0:
            t = self._pos + step * np.arange(n_out)
            i = np.floor(t).astype(int)
            frac = t - i
            y = xs[i] * (1 - frac) + xs[np.minimum(i + 1, xs.size - 1)] * frac
            self._pos = t[-1] + step - (xs.size - 1)
            self._buf16 = np.concatenate([self._buf16, y.astype(np.float32)])
        else:
            self._pos -= xs.size - 1
        self._tail = xs[-1:]
        while self._buf16.size >= _SILERO_CHUNK:
            self._run(self._buf16[:_SILERO_CHUNK])
            self._buf16 = self._buf16[_SILERO_CHUNK:]

    def _run(self, chunk: np.ndarray) -> None:
        inp = np.concatenate([self._context, chunk])[None, :]
        out, self._state = self._sess.run(
            None, {"input": inp, "state": self._state, "sr": np.array(_SILERO_RATE, np.int64)}
        )
        self._context = inp[0, -_SILERO_CONTEXT:]
        prev, self.prob = self.prob, float(out[0, 0])
        if self.prob >= _SPEECH_ON_SURE or (
            self.prob >= _SPEECH_ON and (prev >= _SPEECH_ON or self._speaking)
        ):
            self._speaking = True
        elif self.prob < _SPEECH_OFF:
            if self._speaking:
                self._hang = self._hold_chunks
            self._speaking = False
        if not self._speaking and self._hang > 0:
            self._hang -= 1


def make_voice_detector(sample_rate: int, hold_s: float, early_hold_s: float, prefer_neural: bool = True):
    """Silero (neural) when onnxruntime is available, else the built-in detector."""
    if prefer_neural:
        try:
            return SileroVoiceDetector(sample_rate, hold_s=hold_s, early_hold_s=early_hold_s)
        except Exception:  # onnxruntime missing or model unreadable: degrade gracefully
            pass
    return VoiceDetector(sample_rate, hold_s=hold_s, early_hold_s=early_hold_s)
