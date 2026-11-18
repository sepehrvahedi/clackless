"""Regenerate docs/spectrogram.png (before/after spectrograms for the README).

pip install matplotlib && python scripts/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import signal  # noqa: E402

from clackless import synth  # noqa: E402
from clackless.dsp import process_array  # noqa: E402

SR, DUR = 48000, 4.0
# Single-hue sequential ramp (light = quiet, dark = loud).
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
INK, MUTED = "#1f1f1d", "#6b6b66"


def spec_db(x):
    f, t, z = signal.stft(x, SR, nperseg=1024, noverlap=768)
    return f / 1000, t, 20 * np.log10(np.abs(z) + 1e-9)


def main():
    typing, _ = synth.keyboard(SR, DUR, seed=3, keys_per_second=7)
    room = synth.stationary_noise(SR, DUR, seed=3) * 10 ** (-55 / 20)
    speech = synth.tts_speech(SR)
    speech = speech[: int(DUR * SR)] if speech is not None else synth.speech(SR, DUR, seed=3)
    scenarios = [
        ("Other person typing, silent", typing + room),
        ("Other person typing while talking", synth.mix(speech, typing, 0.0)[0]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True, sharey=True, constrained_layout=True)
    vmax = None
    for col, (title, x) in enumerate(scenarios):
        for row, (label, sig) in enumerate((("Before", x), ("After Clackless", process_array(x, SR)))):
            f, t, z = spec_db(sig)
            vmax = vmax if vmax is not None else z.max()
            ax = axes[row, col]
            im = ax.pcolormesh(t, f, z, cmap=CMAP, vmin=vmax - 70, vmax=vmax, shading="auto", rasterized=True)
            ax.set_ylim(0, 12)
            ax.set_title(f"{label} - {title.lower()}", loc="left", fontsize=10, color=INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (kHz)", fontsize=9, color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=9, color=MUTED)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Level (dB, relative)", fontsize=9, color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    out = Path(__file__).resolve().parent.parent / "docs" / "spectrogram.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
