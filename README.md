# Clackless

**Remove the other person's keyboard clatter from your calls — and your own from theirs.**

Clackless sits between your meeting app (Teams, Zoom, Google Meet, Slack, Discord, …) and your
speakers or microphone, and removes keystroke sounds in real time. It runs locally on
Windows, macOS and Linux, adds about 36 ms of delay, uses about 2 % of one CPU core, and needs no GPU,
no account and no internet connection.

![Spectrograms before and after Clackless](docs/spectrogram.png)

*Top: someone typing during a call. Bottom: after Clackless. Left: the typist is silent (e.g. taking
notes while you talk): the clicks are gone and the room tone is untouched. Right: typing while
talking: most clicks are removed, while some under loud speech remain (see [Limitations](#limitations)).*

## Why

Interviews, stand-ups, 1:1s: the other side is taking notes and every keystroke lands in your
ears. Meeting apps' noise suppression targets steady noise (fans, hum) and lets many keystrokes
through. Clackless targets the keystrokes specifically.

| Mode | For | Path |
|---|---|---|
| `clackless listen` | **hearing** a call without the other person's typing | meeting app → virtual cable → Clackless → your headphones |
| `clackless mic` | **sending** your voice without your typing (+ background-noise reduction) | your mic → Clackless → virtual cable → meeting app |
| `clackless file` | cleaning a recording | `in.wav` → `out.wav` |

## Quick start

**1. Install** (Python 3.10+):

```bash
pipx install git+https://github.com/sepehrvahedi/clackless
```

or download a standalone binary from the [Releases](https://github.com/sepehrvahedi/clackless/releases)
page (no Python needed). On Debian/Ubuntu also run `sudo apt install libportaudio2`.

**2. Check that it works on your machine** (no call needed):

```bash
clackless selftest --save demo
```

It generates speech and typing, cleans them, prints measured results, and writes
`demo/typing_before.wav` and `demo/typing_after.wav` so you can listen for yourself.

**3. Install a free virtual audio cable** (one time). `clackless setup` prints the exact steps for your OS:

| OS | Cable | Notes |
|---|---|---|
| macOS | [BlackHole](https://existential.audio/blackhole/) — `brew install blackhole-2ch` | use `blackhole-16ch` as a second cable for mic mode |
| Windows | [VB-CABLE](https://vb-audio.com/Cable/) | reboot after installing |
| Linux | none needed — `clackless setup --apply` | creates PipeWire/PulseAudio virtual devices with `pactl` |

**4. Run it during a call:**

```bash
clackless listen
```

and set your meeting app's **speaker** to the cable (e.g. "BlackHole 2ch" / "CABLE Input"). Type
`b` + Enter to toggle bypass and hear the difference; `+`/`-` change the strength; `q` quits.

To stop sending **your own** typing (e.g. you are the interviewer taking notes):

```bash
clackless mic --output "BlackHole 16ch"
```

and set the meeting app's **microphone** to that cable. Mic mode also turns on gentle
background-noise reduction (`--noise 0.5`; use `--noise 0` to disable).

## How well it works

Every number below is reproduced by the test-suite (which CI runs on Windows, macOS and Linux), using
*shadow filtering*: removal and speech damage are measured separately, on the same processing.
The ranges cover 3 sample rates (16, 44.1, 48 kHz) × 3 different typists/voices.

| Scenario | Result | Acceptance threshold |
|---|---|---|
| Typing while the typist is silent (main case) | **18.5–22.6 dB(A) quieter** (mean 20.9) | ≥ 18 per case |
| Room tone while typing | changed by ≈ 1 dB or less | ≤ 2 dB |
| Speech with no typing (synthetic voice, macOS and espeak-ng TTS voices) | 21–40 dB SI-SDR (essentially untouched) | ≥ 20 |
| Deep voice (80 Hz) / vocal fry (65 Hz), no typing | 24–38 / 18.6–23.5 dB SI-SDR | ≥ 20 / ≥ 15 |
| Typing while talking | 7–13 dB(A) quieter | ≥ 5 |
| Speech while typing | 24–36 dB SI-SDR | ≥ 18 |
| Added latency | 36 ms | ≤ 40 |
| CPU | 1–2 % of real time | ≤ 25 % |

About 20 dB(A) means the clicks sound roughly a quarter as loud (perceived loudness halves every
~10 dB) and carry 1/100 of the energy. Details of the method and evaluation:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Product requirements:
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

### Measure it with *your* keyboard

The keyboard in the tests is a physical model. To check against your real keyboard:

```bash
clackless record my_typing.wav --seconds 15     # type normally, don't talk
clackless selftest --typing my_typing.wav --save demo
```

If the result looks off for your keyboard, please open an issue with the numbers (and the file
if you are happy to share it). That is the fastest way to make Clackless better.

## Commands

| Command | What it does |
|---|---|
| `clackless listen [-i IN] [-o OUT]` | clean incoming call audio (auto-detects the cable and your speakers) |
| `clackless mic [-i IN] [-o OUT]` | clean your microphone into the cable |
| `clackless file IN OUT` | clean an audio file (WAV/FLAC/OGG, any channels) |
| `clackless selftest [--save DIR] [--typing F] [--speech F]` | measure performance on this machine |
| `clackless record OUT [--seconds N]` | record your mic (to test with your own keyboard) |
| `clackless devices` | list audio devices (virtual cables are marked) |
| `clackless setup [--apply\|--remove]` | per-OS setup guide; on Linux creates/removes virtual devices |

Common options: `--strength 0..1` (keystroke suppression, default 1), `--noise 0..1`
(background-noise reduction), `--samplerate`, `--buffer-ms`. Devices can be given by index or by
part of their name.

## Privacy

Everything runs locally. Clackless makes no network connections, has no telemetry, and in live
modes writes nothing to disk.

## Limitations

* **Real keyboards vary.** The evaluation uses a physical keyboard model, a speech synthesizer and
  real OS text-to-speech voices — not yet a large set of real recordings. Use
  `clackless record` / `selftest --typing` and share results.
* **Typing while talking is reduced, not removed.** Where the voice is louder than a click,
  the click is left alone on purpose, so the voice is never damaged.
* **Vocal fry** (very low, creaky voice, ~65 Hz) is the hardest case for false detections. It
  is still protected (≥ 15 dB SI-SDR), with a smaller margin than typical voices.
* **You need a virtual audio cable** on macOS and Windows (free, one-time install). Use
  separate cables for `listen` and `mic` if you run both.
* **Linux live routing** relies on PortAudio's ALSA `pulse` device (pipewire-alsa or the
  PulseAudio ALSA plugin). It is covered by unit tests, but has had less real-world use than
  macOS and Windows so far.
* Bluetooth headsets add their own latency; Clackless's 36 ms comes on top of that.

## Development

```bash
git clone https://github.com/sepehrvahedi/clackless && cd clackless
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                       # ~100 tests: DSP invariants, effectiveness, real-time, CLI
python scripts/evaluate.py   # the full evaluation grid; run before/after any DSP change
```

The algorithm lives in `src/clackless/dsp.py` and has no audio-device code; `realtime.py` does
the device I/O. Pull requests are welcome, especially real keyboard recordings and evaluation
results, a small GUI, and testing on more Linux setups.

Releases: push a tag `vX.Y.Z` and GitHub Actions builds and tests standalone binaries for
Windows, macOS and Linux and attaches them to the release (optional PyPI publishing is described in
`.github/workflows/release.yml`).

## License

MIT. Free to use, modify and share.
