"""Streaming keystroke suppressor and stationary noise reducer.

How it works (docs/HOW_IT_WORKS.md has the long version with the reasoning)
---------------------------------------------------------------------------
Audio is analysed with a short-time Fourier transform (16 ms frames, 75 %
overlap) and a ~20 ms look-ahead. For every frame we estimate what the signal
would look like *without* a keystroke, from:

* the median of the last ~120 ms of already-cleaned frames (background),
* the cleaned frames just before the current analysis window,
* the frames 10-20 ms in the future.

A keystroke is louder than the past *and* has died away again within the
look-ahead; speech is sustained (a word ending is not louder than 10 ms
earlier, a word onset is still there 20 ms later). Per frequency bin we test
whether it falls > 8 dB below its look-ahead peak, on frequency-smoothed
spectra so random noise fluctuation does not look like decay. Decaying bins
are compared to the past only; sustained bins may also use the future level.

An event starts when the frame sticks out above that reference by >= 4 dB in
at least two of the 1-2.5 / 2.5-5 / 5-8 / 8-12 kHz bands (clicks are broadband,
consonants usually are not). Two guards protect voices: the raw input one
pitch period earlier masks the current frame (a deep voice is a train of
click-like glottal pulses, keystrokes are >= 50 ms apart), and anything that
sticks out continuously for > 50 ms is never treated as a keystroke.

During an event every bin is clamped down to the spectrum captured just
before it started (+3 dB tolerance), so speech harmonics that were already
present survive and only the click energy above them is removed. The event is
held for 25 ms to catch the tail; after the click's own first 20 ms, only
bins that stay faded are clamped, which protects a voice that starts right
after a keystroke.

An optional decision-directed Wiener filter against a minimum-statistics noise
floor then removes stationary noise (fans, hum, hiss).

The module contains no audio-device code; :class:`Suppressor` accepts and
returns NumPy arrays of any length, and its output does not depend on how the
input is split into blocks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from .vad import make_voice_detector

__all__ = ["Settings", "Suppressor", "Stats", "process_array"]


@dataclass
class Settings:
    """User-facing parameters. Defaults are tuned for 16-48 kHz speech."""

    keystroke_strength: float = 1.0
    """0 disables keystroke suppression, 1 removes as much as possible."""

    noise_reduction: float = 0.0
    """0 disables stationary noise reduction, 1 is the maximum (gain floor -18 dB)."""

    bypass: bool = False
    """Pass audio through untouched (still delayed by the same latency)."""

    frame_ms: float = 16.0
    """STFT frame length. 16 ms averages over >= 1 glottal pulse of a deep voice."""

    lookahead_ms: float = 20.0
    """How far ahead the detector looks. Adds directly to latency."""

    history_ms: float = 120.0
    """Length of the past window used for the background reference."""

    detect_low_db: float = 4.0
    """Excess over the reference where suppression starts to fade in."""

    detect_high_db: float = 9.0
    """Excess over the reference where suppression is fully applied."""

    detect_bands_hz: tuple = ((1000.0, 2500.0), (2500.0, 5000.0), (5000.0, 8000.0), (8000.0, 12000.0))
    """Detector sub-bands (those above the Nyquist limit are dropped). A keystroke
    must stick out in at least two of them. The 8-12 kHz band matters while the
    other person talks: their voice fills 1-5 kHz, a click still sticks out up top."""

    max_attenuation_db: float = 40.0
    """Deepest attenuation applied to any bin of a keystroke frame."""

    hold_ms: float = 25.0
    """Keep suppressing this long after the detector stops firing (click tails)."""

    mute_between_speech: bool = False
    """Silence the stream (typing, room tone, everything) whenever nobody is
    speaking, and pass speech through untouched. Strongest removal of typing
    while the other person is silent (real keyboards: 50+ dB). Adds 80 ms of
    delay so the decision can look ahead and keep the very start of each word.
    Typing *during* speech is left as is."""

    mute_depth_db: float = 40.0
    """How far below the room level the stream goes while muted."""

    voice_hold_ms: Optional[float] = None
    """How long after the last detected voice the stream counts as "speaking".
    Default: 300 ms with muting (short, so typing right after a sentence is
    silenced), 600 ms otherwise."""

    neural_vad: bool = True
    """Use the bundled Silero neural voice detector (needs onnxruntime); falls back
    to the built-in pitch-based detector when it is unavailable."""

    typing_gate: bool = True
    """While nobody is speaking, clamp everything that decays (keystrokes and their
    room echo) to the room-noise floor. Sustained sounds - the start of a word -
    are always kept."""

    def validate(self) -> "Settings":
        if not 0.0 <= self.keystroke_strength <= 1.0:
            raise ValueError("keystroke_strength must be within 0..1")
        if not 0.0 <= self.noise_reduction <= 1.0:
            raise ValueError("noise_reduction must be within 0..1")
        if self.detect_high_db <= self.detect_low_db:
            raise ValueError("detect_high_db must be greater than detect_low_db")
        if not 0.0 <= self.mute_depth_db <= 80.0:
            raise ValueError("mute_depth_db must be within 0..80")
        if self.voice_hold_ms is not None and self.voice_hold_ms < 100:
            raise ValueError("voice_hold_ms must be >= 100")
        if self.lookahead_ms < 5 or self.frame_ms < 4:
            raise ValueError("lookahead_ms must be >= 5 and frame_ms >= 4")
        return self


@dataclass
class Stats:
    frames: int = 0
    transient_frames: int = 0
    keystrokes: int = 0
    muted_frames: int = 0

    @property
    def muted_fraction(self) -> float:
        return self.muted_frames / self.frames if self.frames else 0.0


# Internal constants, tuned with scripts/evaluate.py (see docs/HOW_IT_WORKS.md).
_DECAY_RATIO = 0.4  # bin counts as "decaying" if it falls below -8 dB of its look-ahead peak
_DECAY_SPAN_MS = 16.0  # ...within the last 16 ms of look-ahead
_FUTURE_SPAN_MS = 8.0  # level used as the "sustained" reference for non-decaying bins
_PITCH_PERIOD_MS = 14.0  # longest glottal period we protect (deep voices / vocal fry ~ 70 Hz)
_CLICK_TAIL_MS = 20.0  # a click's own tail: clamped hard; later in an event, sustained sound is protected
_MAX_BURST_MS = 50.0  # continuous "excess" longer than this is speech/noise, not a keystroke
_BG_QUANTILE = 0.5  # background = this quantile of the cleaned history
_MUTE_PREROLL_MS = 80.0  # mute: extra delay so a word is unmuted from its very start
_GATE_TOLERANCE = 1.6  # typing gate: keep the room tone up to +4 dB above its floor
_GATE_TRIGGER_DB = 3.0  # typing gate acts only when something sticks out this far above the room
_GATE_HOLD_MS = 60.0  # ...and for this long afterwards (the click's tail and room echo)
_SUSTAIN_RATIO = 0.6  # typing gate: a bin is "sustained" if it stays >= -4.4 dB over the look-ahead
_SMOOTH_HZ = 400.0  # frequency smoothing for the decay test
_TOLERANCE = 1.2  # +1.6 dB headroom above the reference so natural fluctuation is kept
_NOISE_BIAS = 2.5  # minimum-statistics bias compensation
_NOISE_FLOOR_DB = -18.0  # Wiener gain floor at noise_reduction == 1
_SILENCE_FLOOR = 1e-4  # ~ -80 dBFS: below this nothing counts as "louder"


class Suppressor:
    """Block-size independent streaming processor for mono float audio."""

    def __init__(self, sample_rate: int, settings: Optional[Settings] = None):
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz")
        self.sample_rate = int(sample_rate)
        self.settings = (settings or Settings()).validate()
        s = self.settings

        # Exact frame length (a multiple of 4 for the 75 % overlap), not a power
        # of two: the time resolution must be the same at every sample rate.
        self.n_fft = max(16, 4 * int(round(sample_rate * s.frame_ms / 1000.0 / 4)))
        self.hop = self.n_fft // 4
        self.n_bins = self.n_fft // 2 + 1
        frame_rate = sample_rate / self.hop

        def frames(ms: float, minimum: int = 1) -> int:
            return max(minimum, int(round(ms / 1000.0 * frame_rate)))

        self.lookahead = frames(s.lookahead_ms, 3)
        self.history = frames(s.history_ms, 5)
        self.hold_frames = frames(s.hold_ms, 0)
        self._decay_k = min(self.lookahead, frames(_DECAY_SPAN_MS, 2))
        self._fut_k = min(self.lookahead, frames(_FUTURE_SPAN_MS, 2))
        # Frames whose analysis window ends where the current one begins.
        self._recent = (4, 6)
        # One maximal pitch period (80 Hz voice) of history before that window.
        self._period_k = frames(_PITCH_PERIOD_MS, 2)
        self._max_burst = frames(_MAX_BURST_MS, 4)
        self._tail_k = frames(_CLICK_TAIL_MS, 2)
        self._smooth_bins = max(3, int(round(_SMOOTH_HZ * self.n_fft / sample_rate)) | 1)

        # sqrt-Hann analysis/synthesis pair: perfect reconstruction at 75 % overlap.
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)
        self._win = np.sqrt(hann)
        self._ola_gain = self.hop / np.sum(hann)

        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        nyq = 0.95 * sample_rate / 2
        self._bands = [
            sel
            for lo, hi in s.detect_bands_hz
            for sel in [(freqs >= lo) & (freqs < min(hi, nyq))]
            if sel.sum() >= 2
        ] or [freqs >= nyq / 4]
        # Use the second-loudest band (i.e. require two bands) when possible.
        self._rank = min(1, len(self._bands) - 1)
        self._floor_pow = _SILENCE_FLOOR**2 * self.n_fft / 2

        self._subwin = frames(300.0, 4)  # noise tracker sub-window (~0.3 s, 5 of them)
        # When muting, an early (not yet confirmed) voice must hold the stream open
        # longer: a wrong mute inside a sentence is worse than a short leak.
        early_hold = 0.5 if s.mute_between_speech else 0.15
        hold_ms = (
            s.voice_hold_ms if s.voice_hold_ms is not None else (300.0 if s.mute_between_speech else 600.0)
        )
        self._vad = make_voice_detector(
            sample_rate, hold_s=hold_ms / 1000.0, early_hold_s=early_hold, prefer_neural=s.neural_vad
        )
        self._gate_hold_frames = frames(_GATE_HOLD_MS, 1)
        self._min_gain = 10 ** (-s.max_attenuation_db / 20.0)
        self._mute_delay = frames(_MUTE_PREROLL_MS, 1) if s.mute_between_speech else 0
        self.latency_samples = self.n_fft + (self.lookahead + self._mute_delay) * self.hop
        self.reset()

    # ------------------------------------------------------------------ API
    @property
    def speaking(self) -> bool:
        """Whether the voice detector currently hears speech (for status displays)."""
        return bool(self._vad.active)

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sample_rate

    def reset(self) -> None:
        """Forget all history (e.g. after a device change)."""
        self._in_buf = np.zeros(self.n_fft)
        self._out_acc = np.zeros(self.n_fft)
        self._in_fifo = np.zeros(0)
        self._out_fifo = np.zeros(self.hop)
        self._specs: Deque[np.ndarray] = deque()
        self._mags: Deque[np.ndarray] = deque()
        self._smags: Deque[np.ndarray] = deque()
        # Cleaned magnitudes of past frames: circular, bins x frames (newest at _hist_pos - 1).
        self._hist = np.zeros((self.n_bins, self.history))
        self._hist_len = 0
        self._hist_pos = 0
        self._bg = np.zeros(self.n_bins)
        self._raw = np.zeros((self.n_bins, self._recent[0] + self._period_k + 1))
        self._raw_pos = 0
        self._hold = 0
        self._burst = 0
        self._event_age = 0
        self._disarmed = False
        self._pre_event = np.zeros(self.n_bins)
        self._psd_smooth = np.zeros(self.n_bins)
        self._prev_clean_pow = np.zeros(self.n_bins)
        self._sub_min = np.full(self.n_bins, np.inf)
        self._sub_count = 0
        self._sub_mins: Deque[np.ndarray] = deque(maxlen=5)
        self._room = _FloorTracker(self.n_bins, self._subwin)
        self._vad.reset()
        self._gate_hold = 0
        self._gate_trig = False
        # Mute stage: frames wait here so the decision can use the voice detector's
        # verdict from up to `_mute_delay` frames *later* (the start of a word).
        self._mq: Deque = deque()
        for _ in range(self._mute_delay):
            self._mq.append(
                (
                    np.zeros(self.n_bins, dtype=complex),
                    np.ones(self.n_bins),
                    np.zeros(self.n_bins),
                    np.full(self.n_bins, np.inf),
                    False,
                )
            )
        self._mute_mix = 1.0
        self.stats = Stats()
        # Pre-roll so the very first frames have a full look-ahead window.
        for _ in range(self.lookahead):
            self._specs.append(np.zeros(self.n_bins, dtype=complex))
            self._mags.append(np.zeros(self.n_bins))
            self._smags.append(np.zeros(self.n_bins))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Process a mono block of any length; returns a float32 block of equal length."""
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        n = x.size
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), -4.0, 4.0)
        fifo = np.concatenate([self._in_fifo, x])
        hop = self.hop
        steps = fifo.size // hop
        produced = [self._step(fifo[i * hop : (i + 1) * hop]) for i in range(steps)]
        self._in_fifo = fifo[steps * hop :].copy()
        if produced:
            self._out_fifo = np.concatenate([self._out_fifo, *produced])
        out = self._out_fifo[:n]
        self._out_fifo = self._out_fifo[n:]
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """Push enough silence through to drain the internal delay line."""
        return self.process(np.zeros(self.latency_samples))

    # ------------------------------------------------------------- internals
    def _step(self, chunk: np.ndarray) -> np.ndarray:
        hop = self.hop
        self._in_buf[:-hop] = self._in_buf[hop:]
        self._in_buf[-hop:] = chunk
        self._vad.push(chunk)  # sees the newest audio: centred on the frame being output
        spec = np.fft.rfft(self._in_buf * self._win)
        mag = np.abs(spec)
        self._specs.append(spec)
        self._mags.append(mag)
        self._smags.append(uniform_filter1d(mag, self._smooth_bins, mode="nearest"))

        # The frame being output is `lookahead` frames old; the rest is look-ahead.
        center_spec = self._specs.popleft()
        center_mag = self._mags.popleft()
        center_smag = self._smags.popleft()

        if self.settings.bypass:
            out_spec = center_spec
        else:
            gain = self._gain(center_mag, center_smag)
            if self._mute_delay:
                center_spec, gain = self._mute_stage(center_spec, gain, center_mag)
            out_spec = self._apply(center_spec, gain)

        self._out_acc += np.fft.irfft(out_spec, self.n_fft) * self._win
        out = self._out_acc[:hop] * self._ola_gain
        self._out_acc[:-hop] = self._out_acc[hop:]
        self._out_acc[-hop:] = 0.0
        return out

    def _gain(self, mag: np.ndarray, smag: np.ndarray) -> np.ndarray:
        s = self.settings
        self.stats.frames += 1
        gain = np.ones(self.n_bins)
        room = self._room.update(mag)

        gate_ready = np.isfinite(room[0])  # until the room is known, use the per-keystroke logic
        # With muting on, the (delayed, look-ahead) mute stage handles non-speech, so
        # the gate is skipped: it would clip the start of a word before the voice
        # detector has confirmed it.
        use_gate = s.typing_gate and not s.mute_between_speech
        if s.mute_between_speech:
            # Mute mode: non-speech is silenced by the (look-ahead) mute stage and
            # speech passes untouched. Per-keystroke clamping during speech costs
            # natural voices noticeable fidelity for ~2 dB on real typing, so it
            # is not used here.
            pass
        elif s.keystroke_strength > 0 and use_gate and gate_ready and not self._vad.active:
            gain = self._typing_gate(mag, smag, room)
        elif s.keystroke_strength > 0 and self._hist_len >= 3:
            self._gate_hold = 0
            ahead = np.asarray(self._mags)
            sahead = np.asarray(self._smags)
            fut = np.mean(ahead[-self._fut_k :], axis=0)

            # Click-like bins fall well below their look-ahead peak. "dips": at
            # some point in the next ~16 ms (catches clicks in fast typing);
            # "fades": on average (a deep voice dips between glottal pulses but
            # does not fade).
            speak = np.maximum(smag, np.max(sahead, axis=0))
            span = sahead[-self._decay_k :]
            dips = np.min(span, axis=0) < _DECAY_RATIO * speak
            fades = np.mean(span, axis=0) < _DECAY_RATIO * speak

            # Live reference: long-term background, and the moment just before
            # the current window (a word ending is not louder than ~10 ms earlier).
            pre = detect_pre = self._bg
            a, b = self._recent
            if self._hist_len >= b:
                cols = [(self._hist_pos - i) % self.history for i in range(a, b + 1)]
                pre = np.maximum(pre, self._hist[:, cols].mean(axis=1))
                # Voiced speech is a train of glottal pulses (every 2.5-14 ms);
                # with a deep voice each pulse looks like a tiny click. The
                # previous pulse, one pitch period earlier, masks the current
                # one. Keystrokes are >= 50 ms apart, so they are not masked.
                # This uses the *raw* input: if it used the cleaned history, one
                # wrongly suppressed pulse would unmask the next one.
                n = min(self._period_k, self._hist_len - a + 1)
                cols = [(self._raw_pos - i) % self._raw.shape[1] for i in range(a, a + n)]
                detect_pre = np.maximum(pre, self._raw[:, cols].max(axis=1))
            else:
                detect_pre = pre
            live_ref = np.where(dips, pre, np.maximum(pre, fut))
            if self._hold > 0 and self._event_age <= self._tail_k:
                # Within one click's tail, its own pulse must not mask the tail.
                # Later, continuing needs a new onset that passes the pitch mask.
                excess_db = self._band_excess_db(mag, live_ref)
            else:
                detect_ref = np.where(dips, detect_pre, np.maximum(detect_pre, fut))
                excess_db = self._band_excess_db(mag, detect_ref)
            triggered = excess_db >= s.detect_low_db and not self._disarmed

            if self._hold > 0:
                # Inside an event the recent past is the click itself: clamp to the
                # spectrum captured just before the event started ...
                decayed = fades if self._event_age > self._tail_k else dips
                ref = np.where(decayed, self._pre_event, np.maximum(self._pre_event, fut))
                tail_db = self._band_excess_db(mag, ref)
                p = float(np.clip(max(tail_db, excess_db), 0.0, 1.0))
            else:
                ref = live_ref
                p = float(np.clip((excess_db - s.detect_low_db) / (s.detect_high_db - s.detect_low_db), 0, 1))
                p = p if triggered else 0.0
            p *= s.keystroke_strength

            # ... but (re)triggering is always judged against the live reference,
            # so an event cannot chain itself through a word that follows it.
            if triggered:
                if self._hold == 0:
                    self.stats.keystrokes += 1
                    self._pre_event = pre
                    self._event_age = 0
                self._hold = self.hold_frames
                self._burst += 1
                if self._burst > self._max_burst:
                    # No keystroke sticks out continuously for this long: it is
                    # speech or noise. Stand down until the signal settles.
                    self._hold, self._disarmed, p = 0, True, 0.0
            else:
                self._burst = 0
                self._hold = max(0, self._hold - 1)
            self._event_age += 1
            if self._disarmed and excess_db < s.detect_low_db:
                self._disarmed = False

            if p > 0:
                self.stats.transient_frames += 1
                clamp = np.clip(_TOLERANCE * ref / (mag + 1e-12), self._min_gain, 1.0)
                gain = 1.0 - p * (1.0 - clamp)

        clean_mag = mag * gain
        self._push_history(clean_mag, mag)

        if s.noise_reduction > 0:
            gain = gain * self._noise_gain(clean_mag)
        return gain

    def _apply(self, spec: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Final gain application (a hook so evaluation can record/replay gains)."""
        return spec * gain

    def _mute_stage(self, spec: np.ndarray, gain: np.ndarray, mag: np.ndarray):
        """Delay frames by `_mute_delay`, then silence them unless a voice is
        present within that window (so the start of every word is kept).
        Returns the delayed spectrum and its final gain."""
        self._mq.append((spec, gain, mag, self._room._floor.copy(), self._vad.active))
        out_spec, out_gain, out_mag, out_room, _ = self._mq.popleft()
        speaking = any(entry[4] for entry in self._mq) or not np.isfinite(out_room[0])
        target = 1.0 if speaking else 0.0
        # Fade in fast (8 ms), out a little slower (24 ms): no clicks of our own.
        step = 0.5 if target > self._mute_mix else 1.0 / 6
        self._mute_mix += float(np.clip(target - self._mute_mix, -step, step))
        if self._mute_mix >= 1.0:
            return out_spec, out_gain
        if self._mute_mix <= 0.0:
            self.stats.muted_frames += 1
        quiet = np.clip(out_room * 10 ** (-self.settings.mute_depth_db / 20) / (out_mag + 1e-12), 0.0, 1.0)
        return out_spec, out_gain * (self._mute_mix + (1.0 - self._mute_mix) * quiet)

    def _typing_gate(self, mag: np.ndarray, smag: np.ndarray, room: np.ndarray) -> np.ndarray:
        """Nobody is speaking: clamp what decays to the room floor around each event."""
        # Leave the per-keystroke event logic in a clean state for when a voice returns.
        self._hold = self._burst = 0
        self._disarmed = False
        floor = _GATE_TOLERANCE * room
        # Only act while something sticks out above the room (and during its
        # tail), so the room tone between keystrokes is left untouched.
        triggered = self._band_excess_db(mag, floor) >= _GATE_TRIGGER_DB
        if triggered:
            if not self._gate_trig:
                self.stats.keystrokes += 1  # a new onset (fast keys can share one held event)
            self._gate_hold = self._gate_hold_frames
        elif self._gate_hold > 0:
            self._gate_hold -= 1
        self._gate_trig = triggered
        if self._gate_hold == 0:
            return np.ones(self.n_bins)
        self.stats.transient_frames += 1
        sahead = np.asarray(self._smags)
        # Sustained bins (the start of a word, a hiss like "s") are kept.
        sustained = np.min(sahead, axis=0) >= _SUSTAIN_RATIO * smag
        fut = np.mean(np.asarray(self._mags)[-self._fut_k :], axis=0)
        ref = np.where(sustained, np.maximum(floor, fut), floor)
        clamp = np.clip(ref / (mag + 1e-12), self._min_gain, 1.0)
        return 1.0 - self.settings.keystroke_strength * (1.0 - clamp)

    def _band_excess_db(self, mag: np.ndarray, ref: np.ndarray) -> float:
        """How far `mag` sticks out above `ref`, in the 2nd-loudest detector band."""
        cur_pow = mag**2 + self._floor_pow
        ref_pow = ref**2 + self._floor_pow
        band_db = sorted(
            (10.0 * np.log10(np.sum(cur_pow[sel]) / np.sum(ref_pow[sel])) for sel in self._bands),
            reverse=True,
        )
        return band_db[self._rank]

    def _push_history(self, clean_mag: np.ndarray, raw_mag: np.ndarray) -> None:
        self._raw[:, self._raw_pos] = raw_mag
        self._raw_pos = (self._raw_pos + 1) % self._raw.shape[1]
        self._hist[:, self._hist_pos] = clean_mag
        self._hist_pos = (self._hist_pos + 1) % self.history
        self._hist_len = min(self._hist_len + 1, self.history)
        # The background median moves slowly; refreshing it every other frame
        # (~5 ms) halves the dominant cost with no measurable quality change.
        if self.stats.frames % 2 == 1 or self._hist_len < self.history:
            k = int(self._hist_len * _BG_QUANTILE)
            self._bg = np.partition(self._hist[:, : self._hist_len], k, axis=1)[:, k]

    def _noise_gain(self, mag: np.ndarray) -> np.ndarray:
        """Decision-directed Wiener gain against a minimum-statistics noise floor."""
        p = mag**2
        if not np.any(p):  # digital silence: nothing to learn or remove
            return np.ones(self.n_bins)
        self._psd_smooth = p if not np.any(self._psd_smooth) else 0.85 * self._psd_smooth + 0.15 * p
        self._sub_min = np.minimum(self._sub_min, self._psd_smooth)
        self._sub_count += 1
        if self._sub_count >= self._subwin:
            self._sub_mins.append(self._sub_min)
            self._sub_min = np.full(self.n_bins, np.inf)
            self._sub_count = 0
        floor = self._sub_min
        if self._sub_mins:
            floor = np.minimum(floor, np.min(np.asarray(self._sub_mins), axis=0))
        noise = np.maximum(floor * _NOISE_BIAS, 1e-20)
        snr_post = p / noise
        snr_prio = 0.98 * self._prev_clean_pow / noise + 0.02 * np.maximum(snr_post - 1.0, 0.0)
        g = snr_prio / (1.0 + snr_prio)
        g = np.maximum(g, 10 ** (_NOISE_FLOOR_DB * self.settings.noise_reduction / 20.0))
        self._prev_clean_pow = (g**2) * p
        return g


class _FloorTracker:
    """Minimum-statistics estimate of the stationary background magnitude per bin."""

    def __init__(self, n_bins: int, subwin: int):
        self._smooth = np.zeros(n_bins)
        self._sub_min = np.full(n_bins, np.inf)
        self._mins: Deque[np.ndarray] = deque(maxlen=5)
        self._count = 0
        self._subwin = subwin
        self._seen = 0
        self._floor = np.full(n_bins, np.inf)

    def update(self, mag: np.ndarray) -> np.ndarray:
        p = mag**2
        # The zero pre-roll and the first partially-filled analysis windows would
        # fake a near-zero floor for seconds: skip them (no floor = gate stays off).
        # Later digital silence is real (e.g. a meeting app's own noise gate).
        if self._seen == 0 and not np.any(p):
            return self._floor
        self._seen += 1
        if self._seen <= 4:
            return self._floor
        self._smooth = p if self._seen == 5 else 0.85 * self._smooth + 0.15 * p
        self._sub_min = np.minimum(self._sub_min, self._smooth)
        self._count += 1
        if self._count >= self._subwin:
            self._mins.append(self._sub_min)
            self._sub_min = np.full(self._smooth.size, np.inf)
            self._count = 0
        floor = (
            self._sub_min
            if not self._mins
            else np.minimum(self._sub_min, np.min(np.asarray(self._mins), axis=0))
        )
        self._floor = np.sqrt(np.maximum(floor * _NOISE_BIAS, 1e-20))
        return self._floor


def process_array(audio: np.ndarray, sample_rate: int, settings: Optional[Settings] = None) -> np.ndarray:
    """Offline helper: clean a whole mono array; output is aligned with the input."""
    sup = Suppressor(sample_rate, settings)
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    y = np.concatenate([sup.process(x), sup.flush()])
    return y[sup.latency_samples : sup.latency_samples + x.size]
