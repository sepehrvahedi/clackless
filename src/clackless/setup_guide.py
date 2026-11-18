"""Per-OS instructions for routing meeting audio through Clackless."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import List, Tuple

__all__ = ["guide", "linux_apply", "linux_remove", "LINUX_IN_SINK", "LINUX_MIC_SINK", "LINUX_MIC_SOURCE"]

LINUX_IN_SINK = "clackless_in"
LINUX_MIC_SINK = "clackless_mic"
LINUX_MIC_SOURCE = "clackless_mic_source"

_MAC = """\
macOS setup
===========
Clackless needs a free "virtual audio cable" so the meeting app can hand its
audio to Clackless (and Clackless can hand cleaned audio back).

1. Install BlackHole (free, open source):
       brew install blackhole-2ch
   or download it from https://existential.audio/blackhole/
   (if you also want the microphone mode, install a second one: brew install blackhole-16ch)

HEAR THE OTHER PERSON WITHOUT THEIR TYPING  (`clackless listen`)
2. In Teams/Zoom/Meet: Settings -> Devices -> Speaker = "BlackHole 2ch".
3. Run:  clackless listen
   It reads "BlackHole 2ch" and plays the cleaned audio on your normal
   speakers/headphones. Press b + Enter to hear the difference (bypass).

STOP SENDING YOUR OWN TYPING  (`clackless mic`)
4. Run:  clackless mic --output "BlackHole 16ch"
5. In the meeting app: Microphone = "BlackHole 16ch".

Use different cables for the two modes, otherwise audio would loop.
Do not select "Microsoft Teams Audio" - that is Teams' own screen-share device.
"""

_WIN = """\
Windows setup
=============
Clackless needs a free "virtual audio cable" so the meeting app can hand its
audio to Clackless (and Clackless can hand cleaned audio back).

1. Install VB-CABLE (free) from https://vb-audio.com/Cable/ and reboot.
   (For both modes at once you need two cables, e.g. VB-CABLE A+B, or Voicemeeter.)

HEAR THE OTHER PERSON WITHOUT THEIR TYPING  (`clackless listen`)
2. In Teams/Zoom/Meet: Settings -> Devices -> Speaker = "CABLE Input (VB-Audio Virtual Cable)".
3. Run:  clackless listen
   It reads "CABLE Output" and plays the cleaned audio on your normal
   speakers/headphones. Type b + Enter to hear the difference (bypass).

STOP SENDING YOUR OWN TYPING  (`clackless mic`)
4. Run:  clackless mic
   It reads your microphone and writes to "CABLE Input".
5. In the meeting app: Microphone = "CABLE Output (VB-Audio Virtual Cable)".
"""

_LINUX = """\
Linux setup (PipeWire or PulseAudio)
====================================
No driver needed - Clackless can create the virtual devices with pactl:

1. Run:  clackless setup --apply
   This creates (until next reboot/login; re-run any time):
     - "Clackless-Meeting-Output"  a sink for the meeting app's speaker
     - "Clackless-Microphone"      a source for the meeting app's microphone
   Remove them with:  clackless setup --remove

HEAR THE OTHER PERSON WITHOUT THEIR TYPING  (`clackless listen`)
2. In the meeting app (or pavucontrol): Speaker = "Clackless-Meeting-Output".
3. Run:  clackless listen

STOP SENDING YOUR OWN TYPING  (`clackless mic`)
4. Run:  clackless mic
5. In the meeting app: Microphone = "Clackless-Microphone".

Requires the PortAudio library (Debian/Ubuntu: sudo apt install libportaudio2)
and the ALSA "pulse" device (pipewire-alsa or pulseaudio's ALSA plugin).
"""


def guide(platform: str = sys.platform) -> str:
    if platform == "darwin":
        return _MAC
    if platform.startswith("win"):
        return _WIN
    return _LINUX


def _linux_commands() -> List[Tuple[str, List[str]]]:
    return [
        (
            "meeting output sink",
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={LINUX_IN_SINK}",
                "sink_properties=device.description=Clackless-Meeting-Output",
            ],
        ),
        (
            "microphone sink",
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={LINUX_MIC_SINK}",
                "sink_properties=device.description=Clackless-Mic-Sink",
            ],
        ),
        (
            "microphone source",
            [
                "pactl",
                "load-module",
                "module-remap-source",
                f"master={LINUX_MIC_SINK}.monitor",
                f"source_name={LINUX_MIC_SOURCE}",
                "source_properties=device.description=Clackless-Microphone",
            ],
        ),
    ]


def _pactl_list(kind: str) -> str:
    return subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True).stdout


def linux_apply() -> List[str]:
    """Create the null sinks / remap source. Returns human-readable log lines."""
    if not shutil.which("pactl"):
        raise RuntimeError("pactl not found. Install pulseaudio-utils (works with PipeWire too).")
    log = []
    existing = _pactl_list("sinks") + _pactl_list("sources")
    for label, cmd in _linux_commands():
        name = next(a.split("=", 1)[1] for a in cmd if a.startswith(("sink_name=", "source_name=")))
        if name in existing:
            log.append(f"exists   {label} ({name})")
            continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create {label}: {res.stderr.strip()}")
        log.append(f"created  {label} ({name})")
    return log


def linux_remove() -> List[str]:
    if not shutil.which("pactl"):
        raise RuntimeError("pactl not found.")
    log = []
    for line in _pactl_list("modules").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and "clackless" in parts[2]:
            subprocess.run(["pactl", "unload-module", parts[0]], capture_output=True)
            log.append(f"removed  module {parts[0]} ({parts[1]})")
    return log or ["nothing to remove"]
