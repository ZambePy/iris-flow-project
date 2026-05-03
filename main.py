"""
IrisFlow — Entry point

Executa direto: calibração (Fases 0 → 1 → 2) + teclado virtual.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))

from loguru import logger
from iris_tracker import IrisTracker
from calibration import CalibrationSession


def main() -> None:
    tracker = IrisTracker(camera_index=0)
    session = CalibrationSession(tracker)
    session.run()


if __name__ == "__main__":
    main()
