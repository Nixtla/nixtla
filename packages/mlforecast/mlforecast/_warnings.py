"""Helpers for warnings that should point to the user's call site."""

import concurrent.futures.thread
import os
import sys
import threading
import warnings
from pathlib import Path
from types import FrameType
from typing import Optional


_PACKAGE_DIR = str(Path(__file__).parent) + os.sep
_THREAD_ENTRY_FILES = {threading.__file__, concurrent.futures.thread.__file__}


def _user_warning_stacklevel() -> int:
    frame: Optional[FrameType] = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level += 1
        frame = frame.f_back
    # package thread pool workers have no user frame, so point to the worker
    if frame is not None and frame.f_code.co_filename in _THREAD_ENTRY_FILES:
        level -= 1
    return level


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    warnings.warn(message, category, stacklevel=_user_warning_stacklevel())
