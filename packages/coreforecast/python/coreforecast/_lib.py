import ctypes
import platform
import sys

import numpy as np

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


if platform.system() in ("Windows", "Microsoft"):
    _prefix = "Release"
    _extension = "dll"
else:
    _prefix = ""
    _extension = "so"

_LIB = ctypes.CDLL(
    str(files("coreforecast") / "lib" / _prefix / f"libcoreforecast.{_extension}")
)

_indptr_dtype = np.int32
_indptr_t = ctypes.c_int32
