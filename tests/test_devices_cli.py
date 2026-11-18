"""Device selection, setup guide and command-line interface."""

import sys

import numpy as np
import pytest
import soundfile as sf

from clackless import cli, devices, setup_guide, synth
from clackless.devices import Device, DeviceError, find_device, is_virtual


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BlackHole 2ch", True),
        ("CABLE Output (VB-Audio Virtual Cable)", True),
        ("CABLE Input (VB-Audio Virtual Cable)", True),
        ("Loopback Audio", True),
        ("Monitor of Clackless-Meeting-Output", True),
        ("MacBook Air Speakers", False),
        ("Realtek High Definition Audio", False),
        ("Cablecam USB mic", False),
    ],
)
def test_virtual_cable_detection(name, expected):
    assert is_virtual(name) is expected


FAKE = [
    Device(0, "MacBook Air Microphone", "Core Audio", 1, 0, 48000),
    Device(1, "MacBook Air Speakers", "Core Audio", 0, 2, 48000),
    Device(2, "BlackHole 2ch", "Core Audio", 2, 2, 48000),
    Device(3, "BlackHole 16ch", "Core Audio", 16, 16, 48000),
]


class _FakeDefault:
    device = (0, 1)


class _FakeSd:
    default = _FakeDefault()


@pytest.fixture
def fake_sd(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: _FakeSd)


def test_find_by_index_and_name(fake_sd):
    assert find_device("2", "input", devices=FAKE).name == "BlackHole 2ch"
    assert find_device("16ch", "output", devices=FAKE).index == 3
    assert find_device("blackhole 2CH", "output", devices=FAKE).index == 2


def test_auto_selection(fake_sd):
    assert find_device(None, "input", want_virtual=True, devices=FAKE).name == "BlackHole 2ch"
    assert find_device(None, "output", want_virtual=False, devices=FAKE).name == "MacBook Air Speakers"
    assert find_device(None, "input", want_virtual=False, devices=FAKE).name == "MacBook Air Microphone"


def test_selection_errors_are_actionable(fake_sd):
    with pytest.raises(DeviceError, match="clackless devices"):
        find_device("nonexistent", "input", devices=FAKE)
    with pytest.raises(DeviceError, match="no input channels"):
        find_device("1", "input", devices=FAKE)
    with pytest.raises(DeviceError, match="clackless setup"):
        find_device(None, "input", want_virtual=True, devices=FAKE[:2])


def test_virtual_default_output_is_skipped(fake_sd, monkeypatch):
    class D:
        device = (2, 2)  # user set the system default to the cable

    monkeypatch.setattr(_FakeSd, "default", D())
    assert find_device(None, "output", want_virtual=False, devices=FAKE).name == "MacBook Air Speakers"


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_setup_guide_mentions_both_modes(platform):
    text = setup_guide.guide(platform)
    assert "clackless listen" in text and "clackless mic" in text


def test_cli_setup_prints_guide(capsys):
    assert cli.main(["setup"]) == 0
    assert "clackless listen" in capsys.readouterr().out


def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_cli_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "selftest" in capsys.readouterr().out


def test_cli_file_roundtrip_stereo(tmp_path):
    sr = 44100
    typing, _ = synth.keyboard(sr, 3.0, seed=6)
    speech = synth.speech(sr, 3.0, seed=6)
    stereo = np.stack([speech + typing, 0.5 * (speech + typing)], axis=1)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(str(src), stereo, sr, subtype="PCM_16")
    assert cli.main(["file", str(src), str(dst)]) == 0
    out, out_sr = sf.read(str(dst), always_2d=True)
    info = sf.info(str(dst))
    assert out_sr == sr and out.shape == stereo.shape and info.subtype == "PCM_16"
    assert np.sum(out**2) < np.sum(stereo**2)  # something was removed


def test_cli_file_errors(tmp_path, capsys):
    assert cli.main(["file", str(tmp_path / "missing.wav"), str(tmp_path / "o.wav")]) == 2
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros(1000), 16000)
    assert cli.main(["file", str(src), str(tmp_path / "o.wav"), "--strength", "3"]) == 2
    assert "keystroke_strength" in capsys.readouterr().err


def test_cli_selftest_and_save(tmp_path, capsys):
    rc = cli.main(
        ["selftest", "--no-tts", "--duration", "6", "--samplerate", "16000", "--save", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "RESULT: PASS" in out
    assert (tmp_path / "typing_before.wav").exists() and (tmp_path / "typing_after.wav").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_cli_setup_apply_without_pactl(monkeypatch, capsys):
    monkeypatch.setattr(setup_guide.shutil, "which", lambda _n: None)
    assert cli.main(["setup", "--apply"]) == 1
    assert "pactl" in capsys.readouterr().err


def test_cli_setup_apply_rejected_off_linux(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cli.main(["setup", "--apply"]) == 1


class _FakePipeline:
    """Stands in for realtime.Pipeline: runs for a few status polls, then stops."""

    instances = []

    def __init__(self, inp, out, sr, settings, buffer_ms, allow_same_device):
        from clackless.realtime import LiveStatus

        self.settings = settings
        self.polls = 0
        self._status = LiveStatus(True, False, -30.0, 3, 0, 0, 62.0, 1.0, 0.0)
        _FakePipeline.instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    @property
    def running(self):
        self.polls += 1
        return self.polls < 3

    def status(self):
        return self._status


def test_cli_listen_runs_and_reports_device_unplug(monkeypatch, capsys):
    import clackless.realtime as rt

    monkeypatch.setattr(cli, "_resolve_live", lambda args, mode: (FAKE[2], FAKE[1], False))
    monkeypatch.setattr(devices, "pick_samplerate", lambda i, o, p: 48000)
    monkeypatch.setattr(rt, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    rc = cli.main(["listen", "--noise", "0.2"])
    out = capsys.readouterr()
    assert rc == 3  # stream stopped underneath us
    assert "BlackHole 2ch" in out.out and "keystrokes removed" in out.out
    assert _FakePipeline.instances[-1].settings.noise_reduction == 0.2
    assert _FakePipeline.instances[-1].stopped


def test_cli_live_reports_missing_cable(monkeypatch, capsys):
    def boom(args, mode):
        raise DeviceError(
            "No virtual audio cable found for input. Install one first - run `clackless setup`."
        )

    monkeypatch.setattr(cli, "_resolve_live", boom)
    assert cli.main(["mic"]) == 2
    assert "clackless setup" in capsys.readouterr().err


def test_resolve_live_picks_cable_for_each_mode(fake_sd, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(devices, "list_devices", lambda: FAKE)
    args = cli.build_parser().parse_args(["listen"])
    inp, out, _ = cli._resolve_live(args, "listen")
    assert (inp.name, out.name) == ("BlackHole 2ch", "MacBook Air Speakers")
    args = cli.build_parser().parse_args(["mic", "--output", "16ch"])
    inp, out, _ = cli._resolve_live(args, "mic")
    assert (inp.name, out.name) == ("MacBook Air Microphone", "BlackHole 16ch")


def test_linux_apply_creates_missing_devices_only(monkeypatch):
    calls = []

    class R:
        def __init__(self, stdout="", rc=0):
            self.stdout, self.returncode, self.stderr = stdout, rc, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:3] == ["pactl", "list", "short"]:
            return R("1\tclackless_in\tmodule-null-sink\n" if cmd[3] == "sinks" else "")
        return R()

    monkeypatch.setattr(setup_guide.shutil, "which", lambda _n: "/usr/bin/pactl")
    monkeypatch.setattr(setup_guide.subprocess, "run", fake_run)
    log = setup_guide.linux_apply()
    assert log[0].startswith("exists") and all(line.startswith("created") for line in log[1:])
    loads = [c for c in calls if c[1] == "load-module"]
    assert len(loads) == 2 and "module-remap-source" in loads[1]


def test_cli_selftest_with_recordings(tmp_path, capsys):
    """Real recordings (here: generated, written at another sample rate) replace the models."""
    typing, _ = synth.keyboard(44100, 5.0, seed=21)
    room = synth.stationary_noise(44100, 5.0, seed=21) * 10 ** (-60 / 20)
    sf.write(str(tmp_path / "typing.wav"), typing + room, 44100)
    sf.write(str(tmp_path / "voice.wav"), synth.speech(22050, 3.0, seed=21), 22050)
    rc = cli.main(
        [
            "selftest",
            "--no-tts",
            "--typing",
            str(tmp_path / "typing.wav"),
            "--speech",
            str(tmp_path / "voice.wav"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "your recording" in out and "your voice" in out


def test_cli_selftest_bad_recording(tmp_path, capsys):
    (tmp_path / "bad.wav").write_text("not audio")
    assert cli.main(["selftest", "--typing", str(tmp_path / "bad.wav")]) == 2
