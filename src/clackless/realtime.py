"""Live audio routing: input device -> Suppressor -> output device.

Input and output are opened as two independent streams joined by a small
jitter buffer. A single duplex stream would be simpler, but many host APIs
(WASAPI, ALSA) cannot open one across two different devices, and two devices
never share a clock exactly. The buffer absorbs scheduling jitter and the
slow clock drift by padding (underrun) or dropping (overrun) samples, which
is counted and shown to the user instead of crashing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .devices import Device
from .dsp import Settings, Suppressor

__all__ = ["RingBuffer", "Pipeline", "LiveStatus"]


class RingBuffer:
    """Thread-safe single-producer/single-consumer float32 FIFO with drift control.

    * Playback starts once `target` samples are buffered (jitter cushion).
    * If the reader runs dry, it outputs silence and re-buffers (underrun).
    * If the fill exceeds `high_water` (input clock faster than output clock),
      the oldest audio is dropped back down to `target` (overrun), so latency
      can never creep up by more than `high_water - target`.
    """

    def __init__(self, target: int, high_water: Optional[int] = None):
        if target <= 0:
            raise ValueError("target must be positive")
        self._target = target
        self._high = high_water if high_water is not None else 3 * target
        if self._high <= target:
            raise ValueError("high_water must be above target")
        self._cap = self._high * 2
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._read = 0
        self._fill = 0
        self._lock = threading.Lock()
        self._primed = False
        self.underruns = 0
        self.overruns = 0

    @property
    def fill(self) -> int:
        return self._fill

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size > self._high:
            x = x[-self._high :]
        n = x.size
        with self._lock:
            if self._fill + n > self._high:
                drop = min(self._fill, self._fill + n - self._target)
                self._read = (self._read + drop) % self._cap
                self._fill -= drop
                self.overruns += 1
            start = (self._read + self._fill) % self._cap
            first = min(n, self._cap - start)
            self._buf[start : start + first] = x[:first]
            self._buf[: n - first] = x[first:]
            self._fill += n

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            if not self._primed:
                if self._fill < self._target:
                    return out
                self._primed = True
            take = min(n, self._fill)
            first = min(take, self._cap - self._read)
            out[:first] = self._buf[self._read : self._read + first]
            out[first:take] = self._buf[: take - first]
            self._read = (self._read + take) % self._cap
            self._fill -= take
            if take < n:
                self.underruns += 1
                self._primed = False
        return out


@dataclass
class LiveStatus:
    running: bool
    bypass: bool
    input_dbfs: float
    keystrokes: int
    underruns: int
    overruns: int
    latency_ms: float
    """Latency Clackless adds: DSP look-ahead + jitter buffer + one audio block.
    (The devices' own latency is not included - the meeting app would incur it anyway.)"""
    keystroke_strength: float
    noise_reduction: float
    speaking: bool = False
    muted: bool = False
    muted_fraction: float = 0.0
    """The voice detector hears speech (with muting on, everything else is silenced)."""


class Pipeline:
    """Owns the two PortAudio streams and the suppressor."""

    def __init__(
        self,
        input_device: Device,
        output_device: Device,
        samplerate: int,
        settings: Settings,
        blocksize_ms: float = 10.0,
        buffer_ms: float = 30.0,
        allow_same_device: bool = False,
    ):
        # Linux routes through the single ALSA "pulse" device in both directions.
        if input_device.index == output_device.index and not allow_same_device:
            raise ValueError(
                "Input and output must be different devices (this would create a feedback loop)."
            )
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = int(samplerate)
        self.settings = settings
        self.suppressor = Suppressor(self.samplerate, settings)
        self.blocksize = max(64, int(self.samplerate * blocksize_ms / 1000))
        target = int(self.samplerate * buffer_ms / 1000)
        self.ring = RingBuffer(target=target, high_water=target + 3 * self.blocksize)
        self.in_channels = max(1, min(2, input_device.max_input_channels))
        self.out_channels = max(1, min(2, output_device.max_output_channels))
        self._level = 1e-10
        self._streams = []
        self._lock = threading.Lock()
        self.buffer_ms = buffer_ms

    # -------------------------------------------------------------- callbacks
    def _input_cb(self, indata, frames, time_info, status):  # PortAudio thread
        if status and status.input_overflow:
            self.ring.overruns += 1
        mono = indata.mean(axis=1) if indata.ndim == 2 else indata
        self._level = 0.9 * self._level + 0.1 * float(np.mean(mono.astype(np.float64) ** 2) + 1e-12)
        with self._lock:
            out = self.suppressor.process(mono)
        self.ring.write(out)

    def _output_cb(self, outdata, frames, time_info, status):  # PortAudio thread
        block = self.ring.read(frames)
        outdata[:] = np.repeat(block[:, None], outdata.shape[1], axis=1)

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        import sounddevice as sd

        common = dict(samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", latency="low")
        self._streams = [
            sd.OutputStream(
                device=self.output_device.index,
                channels=self.out_channels,
                callback=self._output_cb,
                **common,
            ),
            sd.InputStream(
                device=self.input_device.index, channels=self.in_channels, callback=self._input_cb, **common
            ),
        ]
        for s in self._streams:
            s.start()

    def stop(self) -> None:
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:  # pragma: no cover - teardown best effort
                pass
        self._streams = []

    @property
    def running(self) -> bool:
        return bool(self._streams) and all(s.active for s in self._streams)

    def set_bypass(self, on: bool) -> None:
        with self._lock:
            self.settings.bypass = on

    def adjust(
        self, keystroke_strength: Optional[float] = None, noise_reduction: Optional[float] = None
    ) -> None:
        with self._lock:
            if keystroke_strength is not None:
                self.settings.keystroke_strength = float(np.clip(keystroke_strength, 0, 1))
            if noise_reduction is not None:
                self.settings.noise_reduction = float(np.clip(noise_reduction, 0, 1))

    def status(self) -> LiveStatus:
        added_ms = self.suppressor.latency_ms + self.buffer_ms + 1000.0 * self.blocksize / self.samplerate
        return LiveStatus(
            running=self.running,
            bypass=self.settings.bypass,
            input_dbfs=10 * np.log10(self._level),
            keystrokes=self.suppressor.stats.keystrokes,
            underruns=self.ring.underruns,
            overruns=self.ring.overruns,
            latency_ms=added_ms,
            keystroke_strength=self.settings.keystroke_strength,
            noise_reduction=self.settings.noise_reduction,
            speaking=self.suppressor.speaking,
            muted=self.settings.mute_between_speech,
            muted_fraction=self.suppressor.stats.muted_fraction,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def wait_forever(pipeline: Pipeline, render: Callable[[LiveStatus], None], interval: float = 0.25) -> None:
    while pipeline.running:
        render(pipeline.status())
        time.sleep(interval)
