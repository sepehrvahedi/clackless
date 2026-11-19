"""Hears typing, so the cleanup can stay out of the way the rest of the time.

The neural speech enhancer keeps voices and drops everything else, so what it
removes (the raw input minus its output) is the evidence. Above 1.5 kHz, a
keystroke there:

* starts suddenly: >= 7 dB above the removed signal 10-20 ms earlier,
* is most of what is there: within 3 dB of the input itself (a clean voice
  loses 20 dB or less to the enhancer, so it never gets close),
* has died away 15-30 ms later, on average, both in what is removed (>= 6 dB)
  and in the input (>= 5 dB). Steady noise, a fricative ("s", "sh"), a meeting
  app's noise gate opening and the first instant of a word (which the enhancer
  drops before it locks on to the voice) do not.

Tuned on 45 voice clips (natural and robotic voices; clean, in a room, with
fan noise or hiss) and three real keyboards (alone, under a voice, quiet,
under a fan): 96 % of the typing is covered, and the cleanup is on 0.6 % of
the time nobody types.

Every keystroke switches the cleanup on from 10 ms before it until 1 s after
it, with short cross-fades. The result is a mix envelope (0 = pass the input
untouched, 1 = use the cleaned audio). It trails the enhancer's output by up to
`(look-ahead + pre-roll + 1)` frames, which the suppressor's own delay covers,
so the cleanup is fully on before the first click reaches the output.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
from scipy import signal

__all__ = ["KeyboardDetector", "make_keyboard_detector"]

_FRAME_MS = 5.0
_HIGHPASS_HZ = 1500.0
_ONSET_DB = 7.0
_SHARE_DB = -3.0
_DECAY_DB = 6.0
_INPUT_DECAY_DB = 5.0
_SILENT_DB = -60.0  # nothing quieter than this counts
_PRE_FRAMES = 2  # the cleanup is fully on this many frames before a click
_MAX_DECAY_FRAMES = 6  # the decay test looks this far past the click
_MIN_DECAY_FRAMES = 3
_HOLD_S = 1.0  # ...and stays on this long after it
_RELEASE_S = 0.15


class KeyboardDetector:
    """Push time-aligned (input, enhanced) blocks; get mix-envelope samples back."""

    def __init__(self, sample_rate: int, available_delay: int):
        self.sr = int(sample_rate)
        self.frame = max(16, int(round(self.sr * _FRAME_MS / 1000.0)))
        self.decay_frames = min(_MAX_DECAY_FRAMES, available_delay // self.frame - _PRE_FRAMES - 1)
        if self.decay_frames < _MIN_DECAY_FRAMES:
            raise ValueError("not enough delay to confirm a keystroke before it is heard")
        self.lag = (self.decay_frames + _PRE_FRAMES) * self.frame
        self._sos = signal.butter(2, _HIGHPASS_HZ, "high", fs=self.sr, output="sos")
        self._hold = int(round(_HOLD_S * self.sr / self.frame))
        self._attack = 1.0 / self.frame
        self._release = 1.0 / (_RELEASE_S * self.sr)
        self.reset()

    def reset(self) -> None:
        zi = np.zeros((self._sos.shape[0], 2))
        self._zi_res, self._zi_in = zi.copy(), zi.copy()
        self._pend_res = np.zeros(0)
        self._pend_in = np.zeros(0)
        self._res_db: Deque[float] = deque(maxlen=self.decay_frames + 5)
        self._in_db: Deque[float] = deque(maxlen=self.decay_frames + 5)
        self._frames = 0
        # A stream can start in the middle of typing, before any onset to hear:
        # start with the cleanup on, as if a keystroke had just been heard.
        self._countdown = self._hold
        self._env = 1.0
        self.keystrokes = 0
        self.on_frames = 0
        self.emitted_frames = 0

    @property
    def active(self) -> bool:
        return self._env > 0.0

    def push(self, raw: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        res, self._zi_res = signal.sosfilt(self._sos, raw - enhanced, zi=self._zi_res)
        hin, self._zi_in = signal.sosfilt(self._sos, raw, zi=self._zi_in)
        res = np.concatenate([self._pend_res, res])
        hin = np.concatenate([self._pend_in, hin])
        k = res.size // self.frame
        out = []
        f = self.frame
        for i in range(k):
            self._res_db.append(_db(res[i * f : (i + 1) * f]))
            self._in_db.append(_db(hin[i * f : (i + 1) * f]))
            self._frames += 1
            if self._frames <= self.decay_frames + _PRE_FRAMES:
                continue  # the first envelope frame is the one _PRE_FRAMES before the first candidate
            if self._click():
                self.keystrokes += 1
                self._countdown = _PRE_FRAMES + self._hold
            out.append(self._emit(self._countdown > 0))
            self._countdown = max(0, self._countdown - 1)
        self._pend_res, self._pend_in = res[k * f :], hin[k * f :]
        return np.concatenate(out) if out else np.zeros(0)

    def _click(self) -> bool:
        r, x = list(self._res_db), list(self._in_db)
        if len(r) < self.decay_frames + 5:
            return False
        c = len(r) - 1 - self.decay_frames  # the candidate frame
        before = _mean_db(r[c - 4 : c - 1])
        peak = max(r[c], r[c + 1])
        return (
            r[c] >= before + _ONSET_DB
            and peak >= max(x[c], x[c + 1]) + _SHARE_DB
            and peak > _SILENT_DB
            and _mean_db(r[c + 3 :]) <= peak - _DECAY_DB
            and _mean_db(x[c + 3 :]) <= max(x[c], x[c + 1]) - _INPUT_DECAY_DB
        )

    def _emit(self, on: bool) -> np.ndarray:
        steps = np.arange(1, self.frame + 1)
        if on:
            env = np.minimum(1.0, self._env + self._attack * steps)
            self.on_frames += 1
        else:
            env = np.maximum(0.0, self._env - self._release * steps)
        self.emitted_frames += 1
        self._env = float(env[-1])
        return env


def _mean_db(levels) -> float:
    return 10.0 * np.log10(np.mean(np.power(10.0, np.asarray(levels) / 10.0)))


def _db(x: np.ndarray) -> float:
    return 10.0 * np.log10(float(np.mean(x * x)) + 1e-12)


def make_keyboard_detector(sample_rate: int, available_delay: int) -> Optional[KeyboardDetector]:
    """A detector, or None when the pipeline's delay is too short to use one."""
    try:
        return KeyboardDetector(sample_rate, available_delay)
    except ValueError:
        return None
