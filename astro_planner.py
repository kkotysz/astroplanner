"""Compatibility launcher for AstroPlanner.

The `MainWindow` implementation lives in `astroplanner.main_window`. This module
keeps `python astro_planner.py` and older imports working during the package split.
"""
from __future__ import annotations

import sys

from astroplanner import main_window as _main_window
from astroplanner.main_window import MainWindow, Site, Target, main


def __getattr__(name: str):
    return getattr(_main_window, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_main_window)))


if __name__ == "__main__":
    sys.exit(main())
