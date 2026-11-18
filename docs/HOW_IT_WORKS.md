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

## Why DSP and not a neural network

Neural denoisers (RNNoise, DeepFilterNet, NVIDIA Broadcast, Krisp) are good,
but they need model files, sometimes a GPU, and they are trained mainly for
*stationary* noise. A keystroke has a very specific, physical signature that
can be detected directly, which keeps Clackless small (NumPy + SciPy), fast
(about 2 % of one CPU core), explainable, and testable.

## What a keystroke looks like

* **Broadband**: the impact excites everything from ~100 Hz to well above
  10 kHz, with a few resonances from the key cap, switch and case.
* **Abrupt**: it rises in well under a millisecond.
* **Short**: most of the energy is gone within 10–30 ms.
* **Sparse**: even fast typists produce keystrokes ≥ 50 ms apart (press and
  release are two clicks).

Speech is different: a vowel lasts 50–300 ms, a word ending fades over tens of
milliseconds, and voiced speech is *periodic* (a glottal pulse every 2.5–14 ms).

## The algorithm (`src/clackless/dsp.py`)

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
6. **Optional stationary noise reduction**: a decision-directed Wiener filter
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

## Known limitations

* The keyboard is a physical *model*. Real keyboards vary (laptop
  scissor keys, mechanical switches, thocky keyboards). Use
  `clackless record` + `clackless selftest --typing` to measure yours, and
  please share results in an issue.
* Typing *while the typist talks* is only reduced (≈ 7–13 dB(A)), not removed:
  where the voice is louder than the click, the click is left alone on purpose.
* Vocal fry (~65 Hz) is the hardest case for false detections (≥ 15 dB SI-SDR;
  deep 80 Hz voices and typical voices ≥ 20 dB).
* 8 kHz telephone-band audio has fewer cues (≈ 15 dB(A) removal).
