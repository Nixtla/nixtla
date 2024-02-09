import ctypes

import numpy as np

from ._lib import _LIB
from .differences import num_seas_diffs
from .utils import _data_as_void_ptr, _ensure_float, _float_arr_to_prefix


_LIB.Float32_Period.restype = ctypes.c_int
_LIB.Float64_Period.restype = ctypes.c_int


def find_season_length(x: np.ndarray, max_season_length: int) -> int:
    """Find the length of the seasonal period of the time series.
    Returns 0 if no seasonality is found.

    Args:
        x (np.ndarray): Array with the time series.

    Returns:
        int: Season period."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    period = getattr(_LIB, f"{prefix}_Period")(
        _data_as_void_ptr(x), ctypes.c_size_t(x.size), ctypes.c_int(max_season_length)
    )
    return num_seas_diffs(x, period, 1) * period
