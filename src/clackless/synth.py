"""Deterministic test-signal generators: speech-like audio and keyboard typing.

These exist so the suppressor can be *measured* (in CI, with no audio files
or network access) and so users can run ``clackless selftest`` on their own
machine. The speech is a source-filter (formant) synthesizer with voiced
vowels, fricatives, plosives and pauses; the keyboard model produces press
and release clicks built from a sharp broadband impulse, several damped
mechanical resonances and a low "bottom-out" thump, typed in word-like
bursts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal

__all__ = ["speech", "keyboard", "stationary_noise", "mix", "tts_speech"]

# (F1, F2, F3) in Hz for a handful of vowels (adult average).
_VOWELS = [
    (730, 1090, 2440),  # a
    (530, 1840, 2480),  # e
    (270, 2290, 3010),  # i
    (570, 840, 2410),  # o
    (300, 870, 2240),  # u
    (660, 1720, 2410),  # ae
    (520, 1190, 2390),  # uh
]


def _resonator(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return signal.lfilter(b, a, x)


def _bandnoise(rng, n, lo, hi, sr):
    hi = min(hi, 0.45 * sr)
    lo = min(lo, 0.5 * hi)  # keep the band valid at low sample rates
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, rng.standard_normal(n))


def speech(sr: int, duration: float, seed: int = 0, f0: Optional[float] = None) -> np.ndarray:
    """Return speech-like audio normalised to a peak of 0.5.

    Source-filter model: a sawtooth glottal source (-6 dB/octave, i.e. glottal
    flow -12 dB/oct plus lip radiation +6 dB/oct, as in real speech) through
    three formant resonators. `f0` fixes the mean pitch (default: random
    95-210 Hz); use ~80 Hz to stress-test deep voices.
    """
    rng = np.random.default_rng(seed)
    n_total = int(sr * duration)
    out = np.zeros(n_total)
    t = int(0.15 * sr)
    base_f0 = rng.uniform(95, 210)
    if f0 is not None:
        base_f0 = float(f0)
    words_left = rng.integers(3, 8)
    while t < n_total:
        # --- optional consonant onset
        kind = rng.choice(["none", "fric", "plos"], p=[0.35, 0.35, 0.30])
        if kind == "fric":
            n = int(rng.uniform(0.05, 0.12) * sr)
            lo = rng.uniform(2500, 4500)
            seg = _bandnoise(rng, n, lo, lo * 2.2, sr)
            env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
            seg *= env * rng.uniform(0.04, 0.09)
            end = min(n_total, t + n)
            out[t:end] += seg[: end - t]
            t = end
        elif kind == "plos":
            n = int(0.012 * sr)
            seg = _bandnoise(rng, n, 500, 6000, sr) * np.exp(-np.arange(n) / (0.003 * sr))
            seg *= rng.uniform(0.05, 0.12)
            end = min(n_total, t + n)
            out[t:end] += seg[: end - t]
            t = end + int(0.01 * sr)
        if t >= n_total:
            break
        # --- voiced vowel
        n = int(rng.uniform(0.09, 0.26) * sr)
        f0_start = base_f0 * rng.uniform(0.85, 1.2)
        f0_end = f0_start * rng.uniform(0.8, 1.15)
        f0 = np.linspace(f0_start, f0_end, n) * (1 + 0.01 * rng.standard_normal(n).cumsum() / np.sqrt(n))
        phase = np.cumsum(f0 / sr)
        src = 2.0 * (phase % 1.0) - 1.0
        src += 0.02 * rng.standard_normal(n)  # breathiness
        f1, f2, f3 = _VOWELS[rng.integers(len(_VOWELS))]
        jitter = rng.uniform(0.92, 1.08)
        v = (
            _resonator(src, f1 * jitter, 80, sr)
            + 0.6 * _resonator(src, f2 * jitter, 110, sr)
            + 0.3 * _resonator(src, f3 * jitter, 160, sr)
        )
        attack = int(0.02 * sr)
        release = int(0.04 * sr)
        env = np.ones(n)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        v *= env * rng.uniform(0.6, 1.0)
        end = min(n_total, t + n)
        out[t:end] += v[: end - t] / (np.max(np.abs(v)) + 1e-9) * 0.5
        t = end
        # --- gaps
        words_left -= 1
        if words_left <= 0:
            t += int(rng.uniform(0.3, 0.7) * sr)  # phrase pause
            words_left = rng.integers(3, 8)
        elif rng.random() < 0.5:
            t += int(rng.uniform(0.04, 0.12) * sr)  # word gap
    peak = np.max(np.abs(out)) + 1e-12
    return out / peak * 0.5


def _click(rng, sr: int, loud: float, space: bool) -> np.ndarray:
    n = int((0.06 if space else 0.04) * sr)
    tt = np.arange(n) / sr
    x = np.zeros(n)
    # Sharp contact impulse.
    x += rng.standard_normal(n) * np.exp(-tt / rng.uniform(0.0002, 0.0007))
    # Mechanical resonances (key cap, switch, case).
    for _ in range(rng.integers(3, 7)):
        f = np.exp(rng.uniform(np.log(900 if space else 1500), np.log(min(10000, 0.45 * sr))))
        tau = rng.uniform(0.0015, 0.010)
        x += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.3)) * np.exp(-tt / tau)
    # Bottom-out thump.
    f = rng.uniform(80, 180) if space else rng.uniform(150, 350)
    x += (0.8 if space else 0.4) * np.sin(2 * np.pi * f * tt) * np.exp(-tt / rng.uniform(0.006, 0.015))
    if space:  # stabiliser rattle
        x += 0.3 * rng.standard_normal(n) * np.exp(-tt / 0.012) * (tt > 0.005)
    return x / (np.max(np.abs(x)) + 1e-12) * loud


def keyboard(
    sr: int, duration: float, seed: int = 0, keys_per_second: float = 6.0
) -> Tuple[np.ndarray, List[int]]:
    """Return (typing audio with peak 0.5, list of keystroke onset samples)."""
    rng = np.random.default_rng(seed + 1000)
    n_total = int(sr * duration)
    out = np.zeros(n_total)
    onsets: List[int] = []
    mean_ioi = 1.0 / keys_per_second
    t = rng.uniform(0.05, 0.4)
    while t < duration - 0.1:
        word_len = rng.integers(2, 9)
        for _ in range(word_len):
            space = rng.random() < 0.12
            loud = rng.uniform(0.6, 1.0)
            onset = int(t * sr)
            press = _click(rng, sr, loud, space)
            release = _click(rng, sr, loud * rng.uniform(0.25, 0.6), space)
            r_onset = onset + int(rng.uniform(0.05, 0.13) * sr)
            for start, wave in ((onset, press), (r_onset, release)):
                end = min(n_total, start + wave.size)
                if start < n_total:
                    out[start:end] += wave[: end - start]
            onsets.append(onset)
            t += max(0.06, rng.normal(mean_ioi * 0.75, mean_ioi * 0.25))
            if t >= duration - 0.1:
                break
        t += rng.uniform(0.15, 0.6)  # pause between words
        if rng.random() < 0.15:
            t += rng.uniform(0.5, 1.5)  # thinking pause
    peak = np.max(np.abs(out)) + 1e-12
    return out / peak * 0.5, onsets


def stationary_noise(sr: int, duration: float, seed: int = 0, color: str = "pink") -> np.ndarray:
    """Fan/hiss-like noise with RMS 1."""
    rng = np.random.default_rng(seed + 2000)
    n = int(sr * duration)
    x = rng.standard_normal(n)
    if color == "pink":
        # Paul Kellet's economy pink filter.
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        x = signal.lfilter(b, a, x)
    elif color == "hum":
        tt = np.arange(n) / sr
        x = 0.3 * x + sum(np.sin(2 * np.pi * 50 * k * tt) / k for k in range(1, 6))
    return x / (np.sqrt(np.mean(x**2)) + 1e-12)


def mix(speech_sig: np.ndarray, noise_sig: np.ndarray, peak_ratio_db: float) -> np.ndarray:
    """Add noise so its peak is `peak_ratio_db` relative to the speech peak."""
    n = min(speech_sig.size, noise_sig.size)
    s = speech_sig[:n]
    k = noise_sig[:n]
    scale = np.max(np.abs(s)) / (np.max(np.abs(k)) + 1e-12) * 10 ** (peak_ratio_db / 20)
    return s + scale * k, scale * k


def _tts_to_wav(text: str, wav: Path) -> bool:
    """Render `text` to `wav` with whatever TTS engine this OS has."""
    tmp = wav.parent
    if sys.platform == "darwin" and shutil.which("say"):
        aiff = tmp / "tts.aiff"
        if subprocess.run(["say", "-o", str(aiff), text], capture_output=True).returncode != 0:
            return False
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(wav)]
        return subprocess.run(cmd, capture_output=True).returncode == 0
    if sys.platform.startswith("win"):
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            return False
        txt = tmp / "tts.txt"
        txt.write_text(text, encoding="utf-8")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{wav}'); "
            f"$s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()"
        )
        res = subprocess.run([ps, "-NoProfile", "-Command", script], capture_output=True)
        return res.returncode == 0 and wav.exists()
    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        return False
    return subprocess.run([engine, "-w", str(wav), text], capture_output=True).returncode == 0


def tts_speech(sr: int, text: Optional[str] = None) -> Optional[np.ndarray]:
    """Real synthesized speech from the OS voice (macOS `say`, Windows SAPI, espeak-ng).

    Returns None when no engine is available.
    """
    import soundfile as sf

    text = text or (
        "Thanks for joining today. Could you walk me through a project you are proud of, "
        "and tell me what you would do differently if you started it again? "
        "Take your time, there is no rush, I am just writing a few notes while you talk."
    )
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "tts.wav"
        try:
            if not _tts_to_wav(text, wav):
                return None
            data, file_sr = sf.read(str(wav), always_2d=True)
        except (OSError, RuntimeError):
            return None
    x = data.mean(axis=1)
    if x.size == 0 or not np.any(x):
        return None
    if file_sr != sr:
        from math import gcd

        g = gcd(int(file_sr), int(sr))
        x = signal.resample_poly(x, sr // g, int(file_sr) // g)
    return x / (np.max(np.abs(x)) + 1e-12) * 0.5
