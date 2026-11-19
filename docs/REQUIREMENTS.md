# Clackless — Requirements

## Problem

In video calls (Teams, Zoom, Meet, Slack huddles, …) the sound of the other
person typing is transmitted to you. Keyboard clicks are short, loud,
broadband transients; the noise suppression in most meeting apps targets
*stationary* noise (fans, hum) and lets many keystrokes through. They break
concentration, especially in interviews where one side is taking notes.

The same problem exists in the other direction: when *you* type (e.g. an
interviewer writing feedback) your microphone sends the clicks to everyone else.

## Users and modes

| Mode | Who | Signal path |
|------|-----|-------------|
| **Incoming** (`listen`) | Anyone listening to a call (e.g. a candidate) | meeting app → virtual cable → **Clackless** → your speakers/headphones |
| **Outgoing** (`mic`) | Anyone speaking on a call (e.g. an interviewer typing notes) | your microphone → **Clackless** → virtual cable → meeting app |
| **File** (`file`) | Anyone with a recording | `in.wav` → **Clackless** → `out.wav` |

## Functional requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Suppress keyboard keystroke sounds (press and release clicks, spacebar thumps) in a live audio stream while keeping speech intelligible and natural. |
| FR-2 | Optionally reduce stationary background noise (fan, hum, hiss, AC) with an adjustable strength; off by default for incoming audio, on by default for the microphone. |
| FR-3 | **Incoming mode**: read meeting audio from a virtual audio device and play the cleaned audio to a chosen output device. |
| FR-4 | **Outgoing mode**: read a physical microphone and write the cleaned audio to a virtual audio device the meeting app uses as its microphone. |
| FR-5 | **File mode**: clean WAV/FLAC/OGG files offline, producing output identical to the live pipeline (same algorithm, same code path). |
| FR-6 | List audio devices and auto-detect common virtual cables (BlackHole, VB-Cable, Loopback, PulseAudio/PipeWire null sinks). |
| FR-7 | Print step-by-step virtual-cable setup instructions for the current OS (`clackless setup`). |
| FR-8 | Adjustable keystroke suppression strength (0–1) and noise-reduction strength (0–1); a bypass switch for A/B comparison. |
| FR-9 | Show live status: number of keystrokes suppressed, input level, dropped buffers. |
| FR-10 | A built-in self-test (`clackless selftest`) that generates speech + typing, cleans it and reports the measured improvement, so users can verify it works on their machine without a call. |
| FR-11 | Handle any device sample rate (16 kHz–96 kHz) and mono/stereo devices. |
| FR-12 | Default mode: silence the stream whenever the other person is not speaking and pass their speech through, keeping the start of every word; `--no-mute` for the low-delay keystroke-suppression mode. |
| FR-13 | Remove typing (and other noise) *under* speech with a neural speech enhancer, on by default, without adding delay to the default mode; `--no-enhance` to turn it off; fall back gracefully without onnxruntime. |

## Non-functional requirements

| ID | Category | Requirement | How it is verified |
|----|----------|-------------|--------------------|
| NFR-1 | Portability | Runs on Windows, macOS and Linux (x86-64 and ARM64) with Python ≥ 3.10; pure Python + NumPy/SciPy + onnxruntime + PortAudio (no compiler, no GPU). | CI matrix on `windows-latest`, `macos-latest`, `ubuntu-latest`. |
| NFR-2 | Latency | Added algorithmic latency ≤ 40 ms in `--no-mute --no-enhance` mode and ≤ 120 ms in the default mode (its 80 ms look-ahead keeps word starts; the neural cleanup's 40 ms comes out of it). Meetings already have 100–300 ms. | `test_latency_bound`, `test_mute_latency_bound`, `test_default_latency_unchanged_by_enhancer`. |
| NFR-3 | Performance | Real-time factor ≤ 0.25 on a CI runner for the signal processing (≤ 5 % of one core on a laptop), ≤ 0.7 with the neural cleanup (25 % on a MacBook Air), so it runs alongside the meeting app. | `test_realtime_factor`, `clackless selftest`. |
| NFR-4 | Effectiveness | Typing while the typist is silent: ≥ 7 dB(A) quieter on each real keyboard recording, ≥ 18 dB(A) on the keyboard model; typing while talking: ≥ 5 dB(A) on the model; room tone changed by ≤ 2 dB. Default mode with the neural cleanup: real typing ≥ 40 dB(A) quieter while the typist is silent, and while they talk (typing as loud as the voice) on average ≥ 3 dB(A) more removed and ≥ 4 dB better voice SI-SDR than without it. | `tests/test_effectiveness.py`, `tests/test_enhance.py` (incl. real recordings in `tests/data`), `clackless selftest`, `scripts/evaluate.py`. |
| NFR-5 | Transparency | Speech with no typing passes essentially unchanged: SI-SDR ≥ 20 dB (≥ 15 dB for very deep ~80 Hz voices, the hardest case); speech under typing ≥ 18 dB; bypass is a pure delay (identical within float round-off). | `test_real_voice_is_transparent`, `test_deep_voice_is_preserved`, `test_bypass_is_exact_delay`. |
| NFR-6 | Determinism | Output must not depend on the audio buffer size the OS chooses. | `test_block_size_independence`. |
| NFR-7 | Robustness | Never crash on xruns, silence, clipping, NaN/Inf input, or device sample-rate mismatches; recover from buffer under/overflow by padding/dropping and report it. | Unit tests for edge inputs; ring-buffer tests. |
| NFR-8 | Privacy | 100 % local processing. No network access, no telemetry, nothing written to disk in live modes. | Code review: no network imports; documented. |
| NFR-9 | Usability | One command to start (`clackless listen`), sensible defaults, auto-detection of the virtual cable, actionable error messages. | Manual + CLI tests. |
| NFR-10 | Licensing / distribution | MIT licensed; installable with `pipx install clackless`; standalone binaries for the three OSes attached to each GitHub release. | Release workflow. |
| NFR-11 | Maintainability | Algorithm isolated from I/O (`dsp.py` has no audio-device code); ≥ 85 % of DSP code covered by tests; typed public API. | CI. |

## Out of scope (v1)

* Shipping our own virtual audio driver (signed kernel/driver work per OS). We
  rely on well-known free cables: BlackHole (macOS), VB-CABLE (Windows), and
  PipeWire/PulseAudio null sinks (Linux).
* GPU inference or downloading models at run time: the bundled models (Silero
  VAD, DeepFilterNet3) run on the CPU and ship inside the package.
* A graphical user interface (planned; the CLI is the stable core).
