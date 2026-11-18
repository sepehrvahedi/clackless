"""NFR-4/NFR-5: does it actually remove typing without hurting speech?

Gains are computed on the mixture and replayed on each clean component
("shadow filtering"), so removal and speech damage are measured separately.
"""

import numpy as np
import pytest

from clackless import metrics, synth
from clackless.dsp import Settings
from clackless.evaluate import format_report, run_benchmark, shadow_process

DUR = 8.0


@pytest.mark.parametrize("sr", [16000, 48000])
def test_benchmark_passes(sr):
    checks = run_benchmark(sample_rate=sr, duration=DUR, use_tts=False)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + format_report(checks)


def test_benchmark_passes_with_real_voice(tts_speech):
    tts_speech(48000)  # skip early when the OS has no voice
    checks = run_benchmark(sample_rate=48000, duration=DUR, use_tts=True)
    assert any("TTS" in c.name for c in checks)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + format_report(checks)


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("room_db", [-70, -55, -45])
def test_typing_removed_across_rooms_and_typists(seed, room_db):
    """Different typing patterns and room-noise levels."""
    sr = 48000
    typing, _ = synth.keyboard(sr, DUR, seed=seed, keys_per_second=4 + seed % 5)
    room = synth.stationary_noise(sr, DUR, seed=seed) * 10 ** (room_db / 20)
    _, (typing_out, room_out) = shadow_process(typing + room, [typing, room], sr)
    removed = metrics.reduction_dba(typing, typing_out, sr)
    # Clicks can only be pushed down to the room tone; louder rooms limit the ceiling.
    assert removed >= {-70: 18.0, -55: 17.0, -45: 14.0}[room_db]
    assert metrics.reduction_db(room, room_out) <= 2.0


@pytest.mark.parametrize("sr", [8000, 22050, 32000, 96000])
def test_works_at_other_sample_rates(sr):
    typing, _ = synth.keyboard(sr, DUR, seed=7)
    room = synth.stationary_noise(sr, DUR, seed=7) * 10 ** (-60 / 20)
    _, (typing_out,) = shadow_process(typing + room, [typing], sr)
    # 8 kHz is telephone-band audio (nothing above 4 kHz): fewer cues, less removal.
    assert metrics.reduction_dba(typing, typing_out, sr) >= (12.0 if sr == 8000 else 16.0)
    speech = synth.speech(sr, DUR, seed=7)
    _, (speech_out,) = shadow_process(speech, [speech], sr)
    assert metrics.si_sdr(speech_out, speech) >= 20.0


@pytest.mark.parametrize("sr", [16000, 48000])
def test_real_voice_is_transparent(sr, tts_speech):
    voice = tts_speech(sr)
    _, (out,) = shadow_process(voice, [voice], sr)
    assert metrics.si_sdr(out, voice) >= 20.0


@pytest.mark.parametrize("ratio_db", [-6, 0, 6])
def test_typing_while_talking_real_voice(ratio_db, tts_speech):
    sr = 48000
    voice = tts_speech(sr)
    typing, _ = synth.keyboard(sr, voice.size / sr, seed=1)
    mix, kb = synth.mix(voice, typing, ratio_db)
    ref = voice[: mix.size]
    out, (voice_out, kb_out) = shadow_process(mix, [ref, kb], sr)
    assert metrics.si_sdr(voice_out, ref) >= 18.0
    assert metrics.reduction_dba(kb, kb_out, sr) >= 4.0
    # Overall the output must be closer to the clean voice than the input was.
    assert metrics.si_sdr(out, ref) > metrics.si_sdr(mix, ref)


@pytest.mark.parametrize("color", ["pink", "white", "hum"])
def test_stationary_noise_reduction(color, tts_speech):
    """FR-2: background-noise mode for the outgoing microphone."""
    sr = 48000
    voice = tts_speech(sr)
    noise = synth.stationary_noise(sr, voice.size / sr, seed=4, color=color)[: voice.size]
    noise *= np.sqrt(np.mean(voice**2)) / 10 ** (10 / 20)  # 10 dB SNR
    mix = voice + noise
    out, (voice_out, noise_out) = shadow_process(mix, [voice, noise], sr, Settings(noise_reduction=0.6))
    skip = 2 * sr  # the noise tracker needs ~1.5 s to learn the floor
    assert metrics.reduction_db(noise[skip:], noise_out[skip:]) >= 4.0
    assert metrics.si_sdr(voice_out, voice) >= 12.0
    assert metrics.si_sdr(out, voice) > metrics.si_sdr(mix, voice) + 1.0


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("f0,floor", [(80, 20.0), (65, 15.0)])
def test_deep_voice_is_preserved(sr, seed, f0, floor):
    """Low voices are the hardest case: each glottal pulse looks a bit like a
    click. Deep voices (80 Hz) must pass cleanly, vocal fry (65 Hz) nearly so."""
    voice = synth.speech(sr, DUR, seed=seed, f0=f0)
    _, (out,) = shadow_process(voice, [voice], sr)
    assert metrics.si_sdr(out, voice) >= floor


def test_noise_reduction_off_by_default_leaves_room_tone():
    sr = 48000
    room = synth.stationary_noise(sr, 4.0, seed=9) * 0.01
    _, (out,) = shadow_process(room, [room], sr)
    assert abs(metrics.reduction_db(room, out)) < 1.5
