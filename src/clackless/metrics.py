"""Objective quality measures used by the tests and `clackless selftest`."""

from __future__ import annotations

import numpy as np

__all__ = ["si_sdr", "energy_db", "reduction_db", "a_weight", "reduction_dba"]


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (higher is better)."""
    n = min(estimate.size, reference.size)
    est = np.asarray(estimate[:n], dtype=np.float64)
    ref = np.asarray(reference[:n], dtype=np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-20) / (np.dot(noise, noise) + 1e-20)))


def energy_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(np.sum(x * x) + 1e-20))


def reduction_db(before: np.ndarray, after: np.ndarray) -> float:
    """How much quieter `after` is than `before`, in dB."""
    return energy_db(before) - energy_db(after)


def a_weight(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the IEC 61672 A-weighting curve (models perceived loudness)."""
    from scipy import signal

    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997
    num = [(2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20), 0, 0, 0, 0]
    den = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    b, a = signal.bilinear(num, den, sample_rate)
    return signal.lfilter(b, a, np.asarray(x, dtype=np.float64))


def reduction_dba(before: np.ndarray, after: np.ndarray, sample_rate: int) -> float:
    """Loudness reduction in dB(A): how much quieter `after` sounds."""
    return reduction_db(a_weight(before, sample_rate), a_weight(after, sample_rate))
