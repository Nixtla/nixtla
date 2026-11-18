"""Helpers for warnings that should point to the user's call site."""

import os
import site
import sys
import sysconfig
import warnings
from pathlib import Path
from types import FrameType
from typing import Optional


_PACKAGE_DIR = str(Path(__file__).parent) + os.sep
# installed libraries and the stdlib: optuna trials, dask/ray workers, thread pools
# ponytail: user code installed in site-packages gets the package frame too
_LIBRARY_DIRS = tuple(
    os.path.join(d, "")
    for d in {
        *site.getsitepackages(),
        site.getusersitepackages(),
        sysconfig.get_paths()["stdlib"],
    }
)


def _user_warning_stacklevel() -> int:
    frame: Optional[FrameType] = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level += 1
        frame = frame.f_back
    # called back by a library rather than the user, so point to the package frame
    if frame is not None and frame.f_code.co_filename.startswith(_LIBRARY_DIRS):
        level -= 1
    return level


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    warnings.warn(message, category, stacklevel=_user_warning_stacklevel())
