"""Objective evaluation of the suppressor on generated test material.

The suppressor is non-linear, so to measure "how much keyboard was removed"
and "how much speech was damaged" separately we use *shadow filtering*: the
gains are computed on the mixture, then exactly the same gains are applied to
each clean component on its own.

The neural speech enhancer is not a per-bin gain, so shadow filtering cannot
split its output. It is measured directly instead (section 5 of the benchmark):
the voice part of the output is its projection onto the clean voice, and the
remainder counts as left-over typing (conservative: it includes any voice
distortion too).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics, synth
from .dsp import Settings, Suppressor

__all__ = ["shadow_process", "Check", "run_benchmark", "format_report", "load_mono"]


class _Recorder(Suppressor):
    """Records the final per-frame gain (after every stage, including muting)."""

    def __init__(self, *a, **k):
        self.recorded: List[np.ndarray] = []
        super().__init__(*a, **k)

    def _apply(self, spec, gain):
        self.recorded.append(gain)
        return spec * gain


class _Replayer(Suppressor):
    """Applies recorded gains to another signal, frame by frame (same delays)."""

    def __init__(self, gains: Sequence[np.ndarray], *a, **k):
        self._it = iter(gains)
        super().__init__(*a, **k)

    def _gain(self, mag, smag):
        return np.ones(self.n_bins)

    def _mute_stage(self, spec, gain, mag):
        self._mq.append((spec, gain, mag, None, False))
        out = self._mq.popleft()
        return out[0], out[1]

    def _apply(self, spec, gain):
        return spec * next(self._it)


def _run(sup: Suppressor, x: np.ndarray) -> np.ndarray:
    lat = sup.latency_samples
    y = sup.process(np.concatenate([x, np.zeros(lat)]))
    return y[lat : lat + x.size]


def shadow_process(
    mixture: np.ndarray,
    components: Sequence[np.ndarray],
    sample_rate: int,
    settings: Optional[Settings] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Process `mixture`, then replay the identical gains on each component.
    (Without the neural enhancer, which cannot be split this way.)"""
    settings = replace(settings or Settings(), enhance_speech=False)
    rec = _Recorder(sample_rate, settings)
    out = _run(rec, np.asarray(mixture, dtype=np.float64))
    outs = [
        _run(_Replayer(rec.recorded, sample_rate, settings), np.asarray(c, dtype=np.float64))
        for c in components
    ]
    return out, outs


@dataclass
class Check:
    name: str
    value: float
    unit: str
    threshold: float
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold


def run_benchmark(
    sample_rate: int = 48000,
    duration: float = 10.0,
    seed: int = 1,
    use_tts: bool = True,
    settings: Optional[Settings] = None,
    progress: Optional[Callable[[str], None]] = None,
    save: Optional[Dict[str, np.ndarray]] = None,
    typing_recording: Optional[np.ndarray] = None,
    speech_recording: Optional[np.ndarray] = None,
) -> List[Check]:
    """Run the standard scenarios and return the checks with their thresholds.

    Thresholds are the acceptance criteria from docs/REQUIREMENTS.md (NFR-4/5).
    If `save` is a dict, example before/after signals are stored into it.
    `typing_recording` / `speech_recording` (mono, at `sample_rate`) replace the
    generated keyboard / voice with real recordings, e.g. from `clackless record`.
    """
    import time

    sr = sample_rate
    say = progress or (lambda _m: None)
    checks: List[Check] = []
    full = settings or Settings(enhance_speech=True)
    settings = replace(full, enhance_speech=False)  # sections 1-4: the signal processing alone

    real_typing = typing_recording is not None
    if real_typing:
        duration = typing_recording.size / sr
    say("preparing speech and typing")
    speech = synth.speech(sr, duration, seed=seed)
    speech_label = "synthetic"
    if speech_recording is not None:
        speech, speech_label = _fit(speech_recording, int(duration * sr)), "your voice"
    if real_typing:
        # A real recording already contains its own room tone.
        typing = np.asarray(typing_recording, dtype=np.float64)
        room = np.zeros_like(typing)
    else:
        typing, _ = synth.keyboard(sr, duration, seed=seed)
        room = synth.stationary_noise(sr, duration, seed=seed + 2) * 10 ** (-55 / 20)
    tts = synth.tts_speech(sr) if use_tts else None
    if tts is not None:
        tts = tts[: speech.size]

    # 1. Primary scenario: the other person types while silent (room tone only).
    say("scenario: typing over room tone")
    mix = typing + room
    t0 = time.perf_counter()
    out, (typing_out, room_out) = shadow_process(mix, [typing, room], sr, settings)
    rtf = (time.perf_counter() - t0) / duration / 3  # three passes in shadow_process
    if real_typing:
        # The recording's own room tone is (correctly) kept, which caps the number.
        removed = metrics.reduction_dba(typing, typing_out, sr)
        checks.append(Check("Typing removed (your recording)", removed, "dB(A)", 15.0))
    else:
        removed = metrics.reduction_dba(typing, typing_out, sr)
        checks.append(Check("Typing removed (typist silent)", removed, "dB(A)", 18.0))
        checks.append(Check("Room tone change", metrics.reduction_db(room, room_out), "dB", 2.0, False))
    if save is not None:
        save["typing_before"], save["typing_after"] = mix, out

    # 2. Transparency: speech without typing must pass unchanged.
    for label, sig in ((speech_label, speech), ("TTS", tts)):
        if sig is None:
            continue
        say(f"scenario: {label} speech only")
        _, (sig_out,) = shadow_process(sig, [sig], sr, settings)
        checks.append(
            Check(f"Speech preserved, no typing ({label})", metrics.si_sdr(sig_out, sig), "dB SI-SDR", 20.0)
        )

    # 3. Typing while talking (keyboard peaks as loud as the voice peaks).
    for label, sig in ((speech_label, speech), ("TTS", tts)):
        if sig is None:
            continue
        say(f"scenario: typing while talking ({label})")
        mix, kb = synth.mix(sig, typing, 0.0)
        ref = sig[: mix.size]
        out, (sp_out, kb_out) = shadow_process(mix, [ref, kb], sr, settings)
        checks.append(
            Check(
                f"Typing removed while talking ({label})", metrics.reduction_dba(kb, kb_out, sr), "dB(A)", 5.0
            )
        )
        checks.append(
            Check(f"Speech preserved while typing ({label})", metrics.si_sdr(sp_out, ref), "dB SI-SDR", 15.0)
        )
        if save is not None and label == speech_label:
            save["talking_before"], save["talking_after"] = mix, out

    # 4. Mute mode: silence everything between speech (the `listen`/`mic` default).
    say("scenario: mute between speech")
    mute = Settings(**{**settings.__dict__, "mute_between_speech": True})
    mix = typing + room
    _, (typing_out,) = shadow_process(mix, [typing], sr, mute)
    checks.append(
        Check(
            "Typing removed, mute mode (typist silent)",
            metrics.reduction_dba(typing, typing_out, sr),
            "dB(A)",
            30.0,
        )
    )
    for label, sig in ((speech_label, speech), ("TTS", tts)):
        if sig is None:
            continue
        lead = np.concatenate([np.zeros(sr), sig])  # speech starting after silence
        _, (out,) = shadow_process(lead, [lead], sr, mute)
        checks.append(
            Check(f"Speech preserved, mute mode ({label})", metrics.si_sdr(out, lead), "dB SI-SDR", 18.0)
        )

    checks.append(Check("Real-time factor", rtf, "x", 0.25, False))
    checks.append(Check("Added latency", Suppressor(sr, settings).latency_ms, "ms", 40.0, False))
    checks.append(Check("Added latency, mute mode", Suppressor(sr, mute).latency_ms, "ms", 120.0, False))

    # 5. The `listen`/`mic` default: mute mode with the neural voice cleanup.
    best = replace(full, mute_between_speech=True, enhance_speech=True)
    if full.enhance_speech and Suppressor(sr, best).enhancing:
        say("scenario: neural voice cleanup (default)")
        mix = typing + room
        t0 = time.perf_counter()
        out = _run(Suppressor(sr, best), mix)
        rtf = (time.perf_counter() - t0) / duration
        checks.append(
            Check(
                "Typing removed, default (typist silent)",
                metrics.reduction_dba(typing, out, sr),
                "dB(A)",
                40.0,
            )
        )
        for label, sig in ((speech_label, speech), ("TTS", tts)):
            if sig is None:
                continue
            lead = np.concatenate([np.zeros(sr), sig])
            out = _run(Suppressor(sr, best), lead)
            checks.append(
                Check(f"Speech preserved, default ({label})", metrics.si_sdr(out, lead), "dB SI-SDR", 15.0)
            )
            say(f"scenario: typing while talking, default ({label})")
            mix, kb = synth.mix(sig, typing, 0.0)
            ref = sig[: mix.size]
            out = _run(Suppressor(sr, best), mix)
            voice, rest = split_voice(out, ref)
            checks.append(
                Check(
                    f"Typing removed while talking, default ({label})",
                    metrics.reduction_dba(kb, rest, sr),
                    "dB(A)",
                    5.0,
                )
            )
            checks.append(
                Check(
                    f"Speech preserved while typing, default ({label})",
                    metrics.si_sdr(out, ref),
                    "dB SI-SDR",
                    12.0,
                )
            )
            if save is not None and label == speech_label:
                save["talking_after_default"] = out
        checks.append(Check("Real-time factor, default", rtf, "x", 0.7, False))
        checks.append(Check("Added latency, default", Suppressor(sr, best).latency_ms, "ms", 120.0, False))
    return checks


def split_voice(out: np.ndarray, voice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a processed mixture into (the part explained by the clean voice, the rest)."""
    a = float(np.dot(out, voice) / (np.dot(voice, voice) + 1e-12))
    return a * voice, out - a * voice


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Loop or trim `x` to exactly n samples."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("empty recording")
    reps = int(np.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def load_mono(path: str, sample_rate: int) -> np.ndarray:
    """Read any soundfile-supported file as mono float at `sample_rate`."""
    from math import gcd

    import soundfile as sf
    from scipy import signal

    data, sr = sf.read(str(path), always_2d=True, dtype="float64")
    x = data.mean(axis=1)
    if sr != sample_rate:
        g = gcd(int(sr), int(sample_rate))
        x = signal.resample_poly(x, sample_rate // g, int(sr) // g)
    return x


def format_report(checks: Sequence[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        op = ">=" if c.higher_is_better else "<="
        mark = "PASS" if c.passed else "FAIL"
        lines.append(
            f"  [{mark}] {c.name:<{width}}  {c.value:7.2f} {c.unit:<10} (target {op} {c.threshold:g})"
        )
    return "\n".join(lines)
