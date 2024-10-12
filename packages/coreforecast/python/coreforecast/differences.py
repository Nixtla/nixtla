import numpy as np

from ._lib import differences as _diffs


def num_diffs(x: np.ndarray, max_d: int = 1) -> int:
    """Find the optimal number of differences

    Args:
        x (np.ndarray): Array with the time series.
        max_d (int, optional): Maximum number of differences to consider. Defaults to 1.

    Returns:
        int: Optimal number of differences."""
    return _diffs.num_diffs(x, max_d)


def num_seas_diffs(x: np.ndarray, season_length: int, max_d: int = 1) -> int:
    """Find the optimal number of seasonal differences

    Args:
        x (np.ndarray): Array with the time series.
        season_length (int): Length of the seasonal pattern.
        max_d (int, optional): Maximum number of differences to consider. Defaults to 1.

    Returns:
        int: Optimal number of seasonal differences."""
    return _diffs.num_seas_diffs(x, season_length, max_d)


def diff(x: np.ndarray, d: int) -> np.ndarray:
    """Subtract previous values of the series

    Args:
        x (np.ndarray): Array with the time series.
        d (int): Lag to subtract

    Returns:
        np.ndarray: Differenced time series."""
    return _diffs.diff(x, d)
