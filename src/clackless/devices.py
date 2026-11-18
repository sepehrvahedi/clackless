"""Audio device discovery and selection (thin layer over PortAudio/sounddevice)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = ["Device", "list_devices", "find_device", "is_virtual", "pick_samplerate", "DeviceError"]

# Names of common free "virtual audio cable" drivers on each platform.
VIRTUAL_PATTERNS = [
    r"blackhole",
    r"soundflower",
    r"loopback",
    r"vb-audio",
    r"\bcable\b",  # VB-CABLE: "CABLE Input" / "CABLE Output"
    r"voicemeeter",
    r"clackless",  # Linux null sinks created by `clackless setup --apply`
    r"monitor of",
    r"null",
]


class DeviceError(RuntimeError):
    """A device could not be found or opened; the message is user-facing."""


@dataclass
class Device:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def virtual(self) -> bool:
        return is_virtual(self.name)

    def describe(self) -> str:
        kinds = []
        if self.max_input_channels:
            kinds.append(f"{self.max_input_channels} in")
        if self.max_output_channels:
            kinds.append(f"{self.max_output_channels} out")
        tag = "  [virtual cable]" if self.virtual else ""
        return f"{self.index:3d}  {self.name}  ({self.hostapi}; {', '.join(kinds)}){tag}"


def is_virtual(name: str) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in VIRTUAL_PATTERNS)


def _sd():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing (Linux without libportaudio2)
        raise DeviceError(
            "PortAudio could not be loaded. On Debian/Ubuntu run: sudo apt install libportaudio2"
        ) from exc
    return sd


def list_devices() -> List[Device]:
    sd = _sd()
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            Device(
                index=i,
                name=d["name"],
                hostapi=apis[d["hostapi"]]["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
            )
        )
    return out


def _preferred_api_rank(dev: Device) -> int:
    # On Windows the same device shows up once per host API. WASAPI has the
    # lowest latency that works without exclusive mode; MME is the fallback.
    if sys.platform.startswith("win"):
        order = ["Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS"]
        return order.index(dev.hostapi) if dev.hostapi in order else len(order)
    return 0


def find_device(
    spec: Optional[str],
    kind: str,
    *,
    want_virtual: Optional[bool] = None,
    devices: Optional[Sequence[Device]] = None,
) -> Device:
    """Resolve a device by index, by (case-insensitive) name substring, or automatically.

    kind is "input" or "output". With no `spec`, a virtual cable is picked when
    `want_virtual` is True, otherwise the system default (if it is not a
    virtual cable) or the first physical device.
    """
    devices = list(devices) if devices is not None else list_devices()

    def capable(d: Device) -> bool:
        return (d.max_input_channels if kind == "input" else d.max_output_channels) > 0

    candidates = sorted([d for d in devices if capable(d)], key=_preferred_api_rank)
    if not candidates:
        raise DeviceError(f"No {kind} devices found.")

    if spec is not None and str(spec).strip() != "":
        spec = str(spec).strip()
        if spec.isdigit():
            idx = int(spec)
            for d in devices:
                if d.index == idx:
                    if not capable(d):
                        raise DeviceError(f"Device {idx} ({d.name}) has no {kind} channels.")
                    return d
            raise DeviceError(f"No device with index {idx}. Run `clackless devices`.")
        matches = [d for d in candidates if spec.lower() in d.name.lower()]
        if not matches:
            raise DeviceError(f"No {kind} device matching {spec!r}. Run `clackless devices`.")
        exact = [d for d in matches if d.name.lower() == spec.lower()]
        return (exact or matches)[0]

    if want_virtual:
        virtual = [d for d in candidates if d.virtual]
        if not virtual:
            raise DeviceError(
                f"No virtual audio cable found for {kind}. Install one first - run `clackless setup`."
            )
        return virtual[0]

    sd = _sd()
    try:
        default_idx = sd.default.device[0 if kind == "input" else 1]
    except Exception:  # pragma: no cover - depends on host
        default_idx = -1
    for d in candidates:
        if d.index == default_idx and not d.virtual:
            return d
    physical = [d for d in candidates if not d.virtual]
    if physical:
        return physical[0]
    raise DeviceError(f"Could not find a physical {kind} device; pass one with --{kind}.")


def pick_samplerate(inp: Device, out: Device, preferred: Optional[int] = None) -> int:
    """Pick a sample rate both devices accept (48 kHz preferred)."""
    sd = _sd()
    rates = [preferred] if preferred else []
    rates += [48000, 44100, int(inp.default_samplerate), int(out.default_samplerate), 32000, 16000]
    tried = []
    for r in rates:
        if not r or r in tried:
            continue
        tried.append(r)
        try:
            sd.check_input_settings(device=inp.index, samplerate=r, channels=1)
            sd.check_output_settings(device=out.index, samplerate=r, channels=1)
            return r
        except Exception:
            continue
    raise DeviceError(
        f"{inp.name!r} and {out.name!r} have no sample rate in common (tried {tried}). "
        "Set both to 48000 Hz in your OS sound settings."
    )
