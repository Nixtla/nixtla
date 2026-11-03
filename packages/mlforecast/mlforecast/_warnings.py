"""Helpers for warnings that should point to the user's call site."""

import os
import sys
from pathlib import Path
from types import FrameType
from typing import Optional


_PACKAGE_DIR = str(Path(__file__).parent) + os.sep


def _user_warning_stacklevel() -> int:
    frame: Optional[FrameType] = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level += 1
        frame = frame.f_back
    return level
