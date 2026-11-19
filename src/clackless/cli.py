"""Command-line interface: `clackless listen | mic | file | selftest | devices | setup`."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .dsp import Settings

EPILOG = """\
quick start:
  clackless setup        how to install a virtual audio cable on this OS
  clackless selftest     prove it works on this machine (no call needed)
  clackless listen       clean the meeting audio you hear
  clackless mic          clean the microphone audio you send
  clackless file a.wav b.wav
  clackless record my_typing.wav  then  clackless selftest --typing my_typing.wav
"""


def _settings(args) -> Settings:
    mute = getattr(args, "mute", False)
    enhance = getattr(args, "enhance", True)
    noise = args.noise
    if noise is None:  # the neural cleanup already removes steady noise
        noise = 0.0 if enhance else getattr(args, "noise_without_enhance", 0.0)
    return Settings(
        keystroke_strength=args.strength,
        noise_reduction=noise,
        mute_between_speech=mute,
        mute_depth_db=getattr(args, "mute_depth", 40.0),
        enhance_speech=enhance,
        only_while_typing=getattr(args, "only_while_typing", True),
    ).validate()


# --------------------------------------------------------------------- devices
def cmd_devices(args) -> int:
    from .devices import list_devices

    for d in list_devices():
        print(d.describe())
    return 0


# ----------------------------------------------------------------------- setup
def cmd_setup(args) -> int:
    from . import setup_guide

    if args.apply or args.remove:
        if not sys.platform.startswith("linux"):
            print("--apply/--remove are only needed on Linux. Follow the steps below instead.\n")
            print(setup_guide.guide())
            return 1
        try:
            lines = setup_guide.linux_remove() if args.remove else setup_guide.linux_apply()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(lines))
        return 0
    print(setup_guide.guide())
    return 0


# ------------------------------------------------------------------------ live
def _resolve_live(args, mode: str):
    from . import setup_guide
    from .devices import find_device, list_devices

    devices = list_devices()
    allow_same = False
    linux = sys.platform.startswith("linux")
    inp_spec, out_spec = args.input, args.output

    if mode == "listen":
        if linux and inp_spec is None:
            os.environ["PULSE_SOURCE"] = f"{setup_guide.LINUX_IN_SINK}.monitor"
            inp_spec, allow_same = "pulse", True
        inp = find_device(inp_spec, "input", want_virtual=True, devices=devices)
        out = find_device(out_spec, "output", want_virtual=False, devices=devices)
    else:
        if linux and out_spec is None:
            os.environ["PULSE_SINK"] = setup_guide.LINUX_MIC_SINK
            out_spec, allow_same = "pulse", True
        inp = find_device(inp_spec, "input", want_virtual=False, devices=devices)
        out = find_device(out_spec, "output", want_virtual=True, devices=devices)
    return inp, out, allow_same


def _render(st) -> None:
    state = "BYPASS " if st.bypass else ("VOICE   " if st.speaking else "QUIET   ")
    if st.enhancing and not st.bypass:
        state += "+AI "
    level = max(-90.0, st.input_dbfs)
    bar = "#" * int((level + 90) / 6)
    if st.gated:
        work = (
            f"{'TYPING - cleaning' if st.typing else 'no typing - untouched'} | keystrokes {st.keystrokes:5d}"
        )
    elif st.muted:
        work = f"silenced {st.muted_fraction * 100:3.0f}%"
    else:
        work = f"keystrokes removed {st.keystrokes:5d}"
    line = (
        f"\r{state} | in {level:5.1f} dBFS {bar:<15} | {work} "
        f"| strength {st.keystroke_strength:.1f} noise {st.noise_reduction:.1f} "
        f"| +{st.latency_ms:.0f} ms delay | dropouts {st.underruns + st.overruns}"
    )
    sys.stdout.write(line.ljust(140)[:140])
    sys.stdout.flush()


def _keyboard_loop(pipeline, stop: threading.Event) -> None:
    """Read simple commands from stdin while audio runs (works on every OS)."""
    for raw in sys.stdin:
        cmd = raw.strip().lower()
        s = pipeline.settings
        if cmd in ("q", "quit", "exit"):
            stop.set()
            return
        if cmd == "b":
            pipeline.set_bypass(not s.bypass)
        elif cmd in ("+", "-"):
            pipeline.adjust(keystroke_strength=s.keystroke_strength + (0.1 if cmd == "+" else -0.1))
        elif cmd in ("n+", "n-"):
            pipeline.adjust(noise_reduction=s.noise_reduction + (0.1 if cmd == "n+" else -0.1))
    stop.set()


def cmd_live(args, mode: str) -> int:
    from .devices import DeviceError, pick_samplerate
    from .realtime import Pipeline

    try:
        inp, out, allow_same = _resolve_live(args, mode)
        sr = pick_samplerate(inp, out, args.samplerate)
    except DeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if inp.index == out.index and not allow_same:
        print("error: input and output are the same device; that would loop audio.", file=sys.stderr)
        return 2

    what = "meeting audio you hear" if mode == "listen" else "microphone audio you send"
    print(f"Clackless {__version__} - cleaning the {what}")
    print(f"  from: {inp.name}  ->  to: {out.name}  @ {sr} Hz")
    settings = _settings(args)
    print("  commands: b+Enter bypass on/off | +/- strength | n+/n- noise reduction | q quit | Ctrl+C\n")

    stop = threading.Event()
    try:
        pipeline = Pipeline(inp, out, sr, settings, buffer_ms=args.buffer_ms, allow_same_device=allow_same)
        pipeline.start()
    except Exception as exc:  # PortAudio errors are not typed
        print(f"error: could not open audio devices: {exc}", file=sys.stderr)
        return 2

    if settings.enhance_speech and not pipeline.status().enhancing:
        print("  note: neural voice cleanup unavailable (onnxruntime missing); typing under speech stays.\n")
    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=_keyboard_loop, args=(pipeline, stop), daemon=True).start()
    try:
        while not stop.is_set():
            if not pipeline.running:
                print("\naudio stream stopped (device unplugged?)", file=sys.stderr)
                return 3
            _render(pipeline.status())
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print()
    return 0


# ------------------------------------------------------------------------ file
def cmd_file(args) -> int:
    import numpy as np
    import soundfile as sf

    from .dsp import Suppressor

    src, dst = Path(args.input), Path(args.output)
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 2
    info = sf.info(str(src))
    data, sr = sf.read(str(src), always_2d=True, dtype="float64")
    settings = _settings(args)
    keystrokes, muted = 0, 0.0
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        sup = Suppressor(sr, settings)
        lat = sup.latency_samples
        y = np.concatenate([sup.process(data[:, ch]), sup.flush()])
        out[:, ch] = y[lat : lat + data.shape[0]]
        keystrokes = max(keystrokes, sup.stats.keystrokes)
        muted = max(muted, sup.stats.muted_fraction)
    subtype = info.subtype if info.subtype in sf.available_subtypes(dst.suffix[1:].upper() or "WAV") else None
    sf.write(str(dst), np.clip(out, -1.0, 1.0), sr, subtype=subtype)
    what = (
        f"silenced {muted * 100:.0f}% of the time (no speech)"
        if settings.mute_between_speech
        else (f"keystrokes removed: {keystrokes}")
    )
    print(f"wrote {dst} ({data.shape[0] / sr:.1f} s, {data.shape[1]} ch, {sr} Hz); {what}")
    return 0


# -------------------------------------------------------------------- selftest
def cmd_selftest(args) -> int:
    from .evaluate import format_report, load_mono, run_benchmark

    saved = {} if args.save else None
    print(f"Clackless {__version__} self-test @ {args.samplerate} Hz")
    try:
        typing = load_mono(args.typing, args.samplerate) if args.typing else None
        speech = load_mono(args.speech, args.samplerate) if args.speech else None
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: could not read recording: {exc}", file=sys.stderr)
        return 2
    if typing is not None:
        print(f"  using your typing recording: {args.typing} ({typing.size / args.samplerate:.1f} s)")
    checks = run_benchmark(
        typing_recording=typing,
        speech_recording=speech,
        sample_rate=args.samplerate,
        duration=args.duration,
        use_tts=not args.no_tts,
        settings=_settings(args),
        progress=lambda m: print(f"  ... {m}"),
        save=saved,
    )
    print()
    print(format_report(checks))
    if saved:
        import numpy as np
        import soundfile as sf

        outdir = Path(args.save)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, sig in saved.items():
            sf.write(str(outdir / f"{name}.wav"), np.clip(sig, -1, 1), args.samplerate)
        print(f"\nListen for yourself: {outdir}/typing_before.wav vs typing_after.wav")
    ok = all(c.passed for c in checks)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------- record
def cmd_record(args) -> int:
    import numpy as np
    import soundfile as sf

    from .devices import DeviceError, find_device

    try:
        dev = find_device(args.input, "input", want_virtual=False)
        import sounddevice as sd

        print(f"Recording {args.seconds:.0f} s from {dev.name!r} - type normally (stay quiet) ...")
        data = sd.rec(
            int(args.seconds * args.samplerate),
            samplerate=args.samplerate,
            channels=1,
            device=dev.index,
            dtype="float32",
        )
        sd.wait()
    except DeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # PortAudio errors are untyped
        print(f"error: could not record: {exc}", file=sys.stderr)
        return 2
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    sf.write(str(args.output), data, args.samplerate)
    print(f"wrote {args.output} (peak {20 * np.log10(peak + 1e-12):.1f} dBFS)")
    if peak < 1e-4:
        print("warning: the recording is silent - check microphone permission for your terminal.")
    print(f"Now run:  clackless selftest --typing {args.output} --save demo")
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clackless",
        description="Remove keyboard typing sounds (and background noise) from calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument("--version", action="version", version=f"clackless {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    def tuning(sp, noise_default: float = 0.0):
        sp.add_argument("--strength", type=float, default=1.0, help="keystroke suppression 0..1 (default 1)")
        sp.add_argument(
            "--noise",
            type=float,
            default=None,
            help="extra stationary noise reduction 0..1 (default 0"
            + (f"; {noise_default} with --no-enhance)" if noise_default else ")"),
        )

    def no_enhance(sp):
        sp.add_argument(
            "--no-enhance",
            dest="enhance",
            action="store_false",
            help="turn off the neural voice cleanup (DeepFilterNet3) that removes typing and noise "
            "under speech (saves CPU, and 40 ms of delay with --no-mute)",
        )
        sp.add_argument(
            "--always-on",
            dest="only_while_typing",
            action="store_false",
            help="clean all the time; by default the audio passes untouched unless typing is heard",
        )

    def live(name, help_, noise_default):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--input", "-i", help="input device index or name (default: auto)")
        sp.add_argument("--output", "-o", help="output device index or name (default: auto)")
        sp.add_argument("--samplerate", type=int, default=None, help="force a sample rate")
        sp.add_argument("--buffer-ms", type=float, default=30.0, help="jitter buffer (default 30 ms)")
        sp.add_argument(
            "--no-mute",
            dest="mute",
            action="store_false",
            help="don't silence the stream between speech (lower delay: ~76 ms instead of ~116 ms, "
            "but typing while nobody talks is only partly removed)",
        )
        no_enhance(sp)
        sp.add_argument(
            "--mute-depth", type=float, default=40.0, help="how far to turn it down between speech, in dB"
        )
        tuning(sp, noise_default)
        sp.set_defaults(noise_without_enhance=noise_default)
        return sp

    live("listen", "clean incoming meeting audio (virtual cable -> your speakers)", 0.0)
    live("mic", "clean your outgoing microphone (mic -> virtual cable)", 0.5)

    sp = sub.add_parser("file", help="clean an audio file")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--mute", action="store_true", help="silence everything between speech")
    no_enhance(sp)
    tuning(sp, 0.0)

    sp = sub.add_parser("selftest", help="measure suppression on generated speech + typing")
    sp.add_argument("--samplerate", type=int, default=48000)
    sp.add_argument("--duration", type=float, default=10.0)
    sp.add_argument("--no-tts", action="store_true", help="skip the OS text-to-speech voice")
    sp.add_argument("--save", metavar="DIR", help="write before/after WAV files to DIR")
    sp.add_argument("--typing", metavar="FILE", help="use a real typing recording (see `clackless record`)")
    sp.add_argument(
        "--speech", metavar="FILE", help="use a real speech recording instead of generated speech"
    )
    tuning(sp, 0.0)

    sp = sub.add_parser("record", help="record your microphone to a file (e.g. your own typing)")
    sp.add_argument("output", help="WAV file to write")
    sp.add_argument("--seconds", type=float, default=15.0)
    sp.add_argument("--input", "-i", help="input device index or name (default: system microphone)")
    sp.add_argument("--samplerate", type=int, default=48000)

    sub.add_parser("devices", help="list audio devices")

    sp = sub.add_parser("setup", help="show how to set up virtual audio devices on this OS")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="Linux: create the virtual devices with pactl")
    g.add_argument("--remove", action="store_true", help="Linux: remove them again")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("listen", "mic"):
            return cmd_live(args, args.command)
        if args.command == "file":
            return cmd_file(args)
        if args.command == "selftest":
            return cmd_selftest(args)
        if args.command == "record":
            return cmd_record(args)
        if args.command == "devices":
            return cmd_devices(args)
        if args.command == "setup":
            return cmd_setup(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
