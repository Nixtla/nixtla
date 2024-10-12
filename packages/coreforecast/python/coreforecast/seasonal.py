import numpy as np

from ._lib import seasonal as _seasonal
from .differences import num_seas_diffs


def find_season_length(x: np.ndarray, max_season_length: int) -> int:
    """Find the length of the seasonal period of the time series.
    Returns 0 if no seasonality is found.

    Args:
        x (np.ndarray): Array with the time series.

    Returns:
        int: Season period."""
    period = _seasonal.period(x, max_season_length)
    return num_seas_diffs(x, period, 1) * period
