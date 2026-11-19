# How Clackless works

## The signal path

```
                    listen mode (you hear the other person)
 Teams / Zoom / Meet ──► virtual cable ──► Clackless ──► your headphones

                    mic mode (others hear you)
 your microphone ──► Clackless ──► virtual cable ──► Teams / Zoom / Meet
```

Meeting apps only let you pick a speaker and a microphone device, so a
*virtual audio cable* (BlackHole on macOS, VB-CABLE on Windows, a PipeWire /
PulseAudio null sink on Linux) is how audio gets in and out of Clackless.
Input and output are two independent PortAudio streams joined by a 30 ms
jitter buffer that absorbs scheduling jitter and the clock drift between two
devices (it pads on underrun, drops on overrun, and counts both).

## DSP and neural networks

Clackless started as pure signal processing: a keystroke has a very specific,
physical signature that can be detected directly, which is small, fast (about
2 % of one CPU core), explainable and testable. That still powers `--no-mute
--no-enhance`. Measured on real keyboards, though, two problems needed more:
telling a voice from typing reliably (solved by a small neural voice detector,
below), and removing typing that overlaps a voice (solved by a neural speech
enhancer, below). Both are small ONNX models bundled with the package and run
on the CPU with onnxruntime, so there is still no GPU, download or account.

## What a keystroke looks like

* **Broadband**: the impact excites everything from ~100 Hz to well above
  10 kHz, with a few resonances from the key cap, switch and case.
* **Abrupt**: it rises in well under a millisecond.
* **Short**: most of the energy is gone within 10–30 ms.
* **Sparse**: even fast typists produce keystrokes ≥ 50 ms apart (press and
  release are two clicks).

Speech is different: a vowel lasts 50–300 ms, a word ending fades over tens of
milliseconds, and voiced speech is *periodic* (a glottal pulse every 2.5–14 ms).

## Default: mute between speech

The default for `listen` and `mic` is the simplest effective idea: **when the
other person isn't speaking, there's nothing to protect, so silence the line**
(typing, room tone, everything). When they speak, their voice passes through
untouched.

* **Speech detection**: [Silero VAD](https://github.com/snakers4/silero-vad)
  v5 (MIT, 2.3 MB ONNX model bundled in `src/clackless/models/`, run with
  onnxruntime on 32 ms chunks at 16 kHz). Speech starts at probability ≥ 0.6
  for two chunks or ≥ 0.8 for one (a single 0.5–0.8 blip is a key's "ping"),
  ends below 0.35, then a 300 ms hold. Measured: real typing < 0.5 % of the
  time, voices 60–85 %, still ~60–80 % under typing as loud as the voice.
  A hand-made pitch detector (`VoiceDetector`) is the fallback without
  onnxruntime; it was replaced as the default because real keyboards produce
  sustained pure tones (key springs ringing) that fool pitch tracking.
* **Look-ahead**: frames wait 80 ms before the mute decision, so the line opens
  *before* each word starts (word-start loss ≤ 0.5 dB in the first 200 ms).
* **Mute**: everything is turned down to 40 dB below the room level, with an
  8 ms fade-in and 24 ms fade-out so the switching itself is inaudible.
* **During speech**: the per-keystroke logic below is not used (it cost natural
  voices noticeable fidelity, down to ~15 dB SI-SDR, for only ~2 dB(A) less
  real typing under speech). The neural enhancer does that job instead.

## Neural voice cleanup: typing under speech (`src/clackless/enhance.py`)

[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) (Schröter et al.,
MIT / Apache-2.0) is a speech enhancer: every 10 ms it predicts gains for 32
ERB bands (what is voice, what is not) and, below 5 kHz, a 5-tap complex
filter across the last frames that rebuilds the voice's fine structure. It
treats typing like any other non-voice sound.

* **Streaming**: the official weights are exported by
  `scripts/export_deepfilternet.py` into a single ONNX graph that takes 480
  samples (10 ms at 48 kHz) plus the network's recurrent state and returns 10 ms
  of output plus the new state. STFT, features, the three networks and
  overlap-add are all inside the graph. The export matches the official
  offline enhancer within 38.6 dB SNR, and onnxruntime matches PyTorch to 2e-8.
* **Delay**: 40 ms (a 10 ms framing buffer, a 20 ms window and 2 frames of
  look-ahead). It sits in front of the mute stage, but the voice detector hears
  the *raw* input, so the mute stage's own look-ahead shrinks by those 40 ms:
  the total stays ~116 ms.
* **Other sample rates** are resampled to 48 kHz and back with a stateful
  polyphase filter (16 taps each side, Kaiser β = 8) whose group delay is a
  whole number of samples, so the delay stays exact and output is identical for
  any block size.
* **Cost**: 13 % of one core on a MacBook Air (whole default pipeline: 25 %).
* **Measured** (real keyboards as loud as natural voices, while the voice
  talks): typing 7.5 dB(A) quieter, voice 8 → 17 dB SI-SDR; typing right after
  speech stops (while the line is still open) 40–52 dB(A) quieter; clean voices
  unchanged (24–35 dB SI-SDR either way). Running the enhancer twice was tried
  and adds nothing (6.9 vs 6.7 dB(A)).

The algorithm below is the `--no-mute --no-enhance` mode (36 ms delay).

## Keystroke suppression (`--no-mute`, `src/clackless/dsp.py`)

Audio is analysed with an STFT (16 ms frames, 75 % overlap) and a 20 ms
look-ahead. Total added latency is 36 ms. The frame length is the same at
every sample rate: 16 ms averages over at least one glottal pulse of a deep
voice, so the pulses do not look like individual clicks (with 10 ms frames, a
buzzy low voice lost up to 9 dB of fidelity).

1. **Reference "what would be here without a click".** Per frequency bin:
   the median of the last 120 ms of *already cleaned* frames (so earlier
   clicks do not pollute it), the cleaned frames just before the current
   analysis window (a word ending is never louder than 10 ms earlier), and the
   level 10–20 ms in the future (a word onset is still there later).
2. **Decay test.** On frequency-smoothed spectra (so random noise fluctuation
   does not count), a bin is click-like if it falls > 8 dB below its peak
   within the look-ahead. Click-like bins are compared only with the past;
   sustained bins may also use the future level.
3. **Detection.** A keystroke event starts when the frame exceeds the
   reference by ≥ 4 dB in at least two of the 1–2.5 / 2.5–5 / 5–8 / 8–12 kHz
   bands. Two bands, because clicks are broadband and most consonants are not.
   The 8–12 kHz band matters while the other person is talking: their voice
   fills 1–5 kHz, but a click still sticks out above 5 kHz.
4. **Voice guards.**
   * *Pitch masking*: the raw input one pitch period earlier is part of the
     detection reference. A deep voice is a train of click-like glottal pulses,
     and each one is masked by the previous one, while a keystroke is not
     preceded by a similar burst 2.5–14 ms earlier. The raw (uncleaned)
     input is used on purpose: with the cleaned history, a single wrongly
     suppressed pulse would unmask the next one and the error would cascade.
   * *Live re-triggering*: an event can only be extended by frames that stick
     out above the live reference, never the frozen pre-event one, so an
     event that starts on a consonant cannot chain through the following vowel.
   * *Burst cap*: anything that sticks out continuously for > 50 ms is not a
     keystroke; the detector stands down until the signal settles.
5. **Suppression.** During an event each bin is clamped to the spectrum
   captured just before it started (+1.6 dB tolerance). Speech harmonics that
   were already there survive; only click energy above them is removed. The
   event is held 25 ms for the click's tail. After its first 20 ms, only bins
   that *stay* faded are clamped, which protects a voice that begins right after
   a keystroke.
6. **Typing gate (nobody speaking).** A small voice-activity detector
   (`src/clackless/vad.py`) looks for stable pitch in the 150–1000 Hz band
   (the harmonics of any voice, above desk and chassis booms) for ≥ 80 ms,
   ≥ 9 dB above the background and above −70 dBFS. Measured on real
   recordings, typing triggers it on < 1 % of frames and voices on 30–60 %.
   While no voice has been confirmed for 2 s, every event that sticks out
   ≥ 3 dB above the room floor is clamped to that floor (+4 dB) for 60 ms,
   except bins that stay up through the look-ahead (the start of a word).
   This is what handles dense, reverberant real typing, where "louder than
   both the past and the future" is ambiguous because the next click or the
   room echo is always close. A voice that starts is protected by that
   sustain test and by a fast path in the detector (30 ms of very clean,
   steady pitch).
7. **Optional stationary noise reduction**: a decision-directed Wiener filter
   against a minimum-statistics noise floor (1.5 s memory), with a gain floor
   set by the `--noise` strength.

The output does not depend on how the OS splits audio into buffers (tested),
and with suppression disabled the input is reconstructed exactly.

## How it was tested

Nothing here is tuned by ear alone. `src/clackless/evaluate.py` uses *shadow
filtering*: the gains computed on a mixture are replayed on each clean
component separately, so "how much typing was removed" and "how much speech
was damaged" are measured independently.

* **Keyboard**: a physical model with press + release clicks (broadband
  impulse, 3–6 damped resonances, a bottom-out thump, spacebar rattle) typed in
  word-like bursts at 4–9 keys/s, over pink room tone at −70, −55 and −45 dBFS.
* **Real keyboards**: three real typing recordings (laptop, desktop,
  mechanical; public domain / CC BY, see `tests/data/README.md`). These are
  the numbers to trust: 8–11 dB(A) removal while the typist is silent.
* **Voices**: a source-filter speech synthesizer (random and deliberately deep
  80 Hz pitch) plus the operating system's real text-to-speech voice (macOS
  `say`, Windows SAPI, `espeak-ng` on Linux CI).
* **Metrics**: loudness reduction in dB(A) (the A-weighting curve models how
  loud things sound), SI-SDR for speech fidelity, real-time factor, latency.
* `scripts/evaluate.py` runs the grid (3 sample rates × 3 seeds) and is the
  tool to use before and after any DSP change.

Several design decisions above came straight out of that grid and CI. For
example, the pitch masking and the live re-triggering were added after the grid
showed a low-pitched voice losing whole vowel endings. The 8–12 kHz band was
added after a spectrogram showed clicks surviving under speech. The 16 ms frame
length came from the first CI run: the Windows and espeak-ng voices (lower and
buzzier than the macOS one) failed the speech-fidelity checks with 10 ms frames.
The typing gate came from the first test on real keyboard recordings, which
showed only 5–8 dB of removal against 20+ on the model: real fast typing is
dense and reverberant, and the per-keystroke logic kept finding a nearby
click or echo to "protect".

## Known limitations

* On real keyboards, clicks end up about half as loud (8–11 dB(A)), not
  silent. Use `clackless record` + `clackless selftest --typing` to measure
  yours, and please share results in an issue.
* Typing *while the typist talks* is reduced, not removed: ~7.5 dB(A) on real
  keyboards with the neural cleanup (default), 2–5 dB(A) without it.
* Vocal fry (~65 Hz) is the hardest case for false detections (≥ 15 dB SI-SDR;
  deep 80 Hz voices and typical voices ≥ 20 dB).
* 8 kHz telephone-band audio has fewer cues (≈ 15 dB(A) removal).
