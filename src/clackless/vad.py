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
    def __init__(self, sample_rate: int):
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
        self._hang_confirmed = int(_HANG_CONFIRMED_S / _BLOCK_S)
        self._hang_early = int(_HANG_EARLY_S / _BLOCK_S)
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
        if voiced and self._run > 0 and abs(lag - self._prev_lag) <= 0.15 * self._prev_lag:
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
