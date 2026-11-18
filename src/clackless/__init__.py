"""Clackless - remove keyboard typing sounds from live calls."""

from .dsp import Settings, Suppressor, process_array

__version__ = "0.1.0"
__all__ = ["Settings", "Suppressor", "process_array", "__version__"]
