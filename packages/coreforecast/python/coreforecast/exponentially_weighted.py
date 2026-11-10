__all__ = ["exponentially_weighted_mean"]

import numpy as np

from ._lib import exponentially_weighted as _ew


def exponentially_weighted_mean(
    x: np.ndarray, alpha: float, skipna: bool = False
) -> np.ndarray:
    """Compute the exponentially weighted mean of the input array.

    Args:
        x (np.ndarray): Input array.
        alpha (float): Weight parameter.
        skipna (bool): If True, exclude NaN values from calculations using
            forward-fill behavior. When False (default), any NaN value causes
            the result to be NaN, maintaining backwards compatibility. When True,
            the last valid value is forward-filled through NaN values (matching
            pandas default behavior).

    Returns:
        np.ndarray: Array with the exponentially weighted mean.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        >>> # Default behavior: NaN propagates
        >>> exponentially_weighted_mean(x, alpha=0.5)
        array([1., 1.5, nan, nan, nan])
        >>> # With skipna=True: forward-fill through NaN
        >>> exponentially_weighted_mean(x, alpha=0.5, skipna=True)
        array([1., 1.5, 1.5, 2.75, 3.875])
    """
    return _ew.exponentially_weighted_mean(x, alpha, skipna)
