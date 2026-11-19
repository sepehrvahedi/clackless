# Clackless

**Silence the other person's keyboard on your calls — and your own on theirs.**

Clackless sits between your meeting app (Teams, Zoom, Google Meet, Slack, Discord, …) and your
speakers or microphone. While the other person is quiet, a neural voice detector keeps the line
silent, so their typing simply isn't heard. When they speak, a neural speech enhancer
(DeepFilterNet3) strips the typing from under their voice. It runs locally on Windows, macOS and
Linux, needs no GPU, account or internet connection, and adds about 0.1 s of delay.

![Spectrograms before and after Clackless](docs/spectrogram.png)

*Keystroke suppression in `--no-mute` mode, on the keyboard model (top: before, bottom: after). In
the default mode the left half (typist silent) is simply silent; see
[How well it works](#how-well-it-works) for measurements on real keyboards.*

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

Every number below comes from **real keyboard recordings** (a laptop, a desktop keyboard and a
mechanical Keychron, see [tests/data](tests/data/README.md)) and natural voices, and the test-suite
(which CI runs on Windows, macOS and Linux) guards each one. For the signal processing, removal and
voice damage are measured separately on the same processing ("shadow filtering"); the neural
enhancer is measured directly (the voice part of the output is its projection onto the clean voice,
everything else counts as left-over typing, so the numbers are conservative).

### Default: mute between speech (`clackless listen`, `clackless mic`)

| Scenario | Result |
|---|---|
| They type while you talk (they're silent) | **50+ dB(A) quieter: silent** |
| They type *while* talking (typing as loud as their voice) | **~7.5 dB(A) quieter**, and their voice much clearer: 8 → 17 dB SI-SDR |
| They type more quietly while talking (half as loud) | ~4.6 dB(A) quieter; voice 14 → 20.5 dB SI-SDR |
| Their voice with no typing | kept: 24–35 dB SI-SDR on natural voices, the same as without the cleanup |
| The start of their words | kept (≤ 1.5 dB lost in the first 200 ms) |
| Right after they stop talking | 40–52 dB(A) quieter (before: 3–26, while the line was still open) |
| Their room tone and background noise | removed |
| Your own mic (`mic`): your typing + a fan, while you talk | your voice 2 → 13 dB SI-SDR |
| Added delay | ~116 ms |
| CPU | about a quarter of one core on a MacBook Air |

How it works, in two stages:

1. **Between speech, silence.** A small neural voice-activity detector
   ([Silero VAD](https://github.com/snakers4/silero-vad), bundled, MIT) decides whether someone is
   speaking. On real typing it fires < 0.5 % of the time; on voices, even under loud typing,
   60–85 %. The decision looks 80 ms ahead, so the line opens *before* a word starts.
2. **During speech, clean the voice.** [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet)
   (bundled, MIT/Apache-2.0), streamed in 10 ms steps, rebuilds the voice and drops what isn't voice:
   typing, hum, fans. It needs 40 ms, which comes out of the 80 ms look-ahead, so the total delay
   is the same as without it. `--no-enhance` turns it off.

### `--no-mute --no-enhance`: lowest delay (36 ms), keystroke suppression only

`--no-mute` alone keeps the neural cleanup (76 ms): real typing while the typist is silent then
drops 48–52 dB(A).

| Scenario | Real keyboards | Keyboard model (evaluation grid) |
|---|---|---|
| Typing while the typist is silent | 8–11 dB(A) quieter (about half as loud) | 23.6–26.6 dB(A) |
| Typing while the typist talks | 2–5 dB(A) | 7–13 dB(A) |
| Room tone | changed by ≤ 1.3 dB | |
| Speech with no typing | 15–40 dB SI-SDR (natural voices lowest) | ≥ 20 on the grid voices |

10 dB(A) quieter sounds roughly half as loud. Details of the method and evaluation:
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

Common options: `--no-mute` (keep the line open between speech; 76 ms delay instead of ~116 ms),
`--no-enhance` (turn off the neural voice cleanup: less CPU, and 36 ms with `--no-mute`),
`--mute-depth DB` (how far to turn it down, default 40), `--strength 0..1` (keystroke suppression),
`--noise 0..1` (extra background-noise reduction; default 0, or 0.5 for `mic --no-enhance`),
`--samplerate`, `--buffer-ms`. `clackless file` takes `--mute` and `--no-enhance` too. Devices can be given by index or by
part of their name.

## Privacy

Everything runs locally. Clackless makes no network connections, has no telemetry, and in live
modes writes nothing to disk.

## Limitations

* **Typing *while* they talk is reduced, not removed**: ~7.5 dB(A) for typing as loud as their
  voice (it sounds about half as loud). Keystrokes that land exactly on a vowel are the hardest.
* **Delay**: the default mode adds ~116 ms (the look-ahead that keeps word starts). In a lively
  conversation you can trade that for `--no-mute` (76 ms) or `--no-mute --no-enhance` (36 ms,
  typing only partly removed).
* **CPU**: the neural cleanup needs about an eighth of one core (MacBook Air); on old machines use
  `--no-enhance` if you hear dropouts.
* **Silence between sentences**: with the default mode their room tone disappears between
  sentences, like many meeting apps' own noise suppression does.
* **You need a virtual audio cable** on macOS and Windows (free, one-time install). Use
  separate cables for `listen` and `mic` if you run both.
* **Linux live routing** relies on PortAudio's ALSA `pulse` device (pipewire-alsa or the
  PulseAudio ALSA plugin). It is covered by unit tests, but has had less real-world use than
  macOS and Windows so far.
* Bluetooth headsets add their own latency on top of Clackless's.

## Roadmap

* **More removal under speech**: a model trained specifically on keyboard noise (DeepFilterNet3
  is a general speech enhancer).
* A small GUI with a bypass button and a level meter.

## Development

```bash
git clone https://github.com/sepehrvahedi/clackless && cd clackless
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                       # ~150 tests: DSP invariants, effectiveness, neural cleanup, real-time, CLI
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
