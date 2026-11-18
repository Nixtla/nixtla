"""Coverage utilities for prediction intervals."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval


def _nanmean(arr: NDArray[np.floating], axis: Any = None) -> Any:
    """``np.nanmean`` without the empty-slice warning.

    A cell with no evaluable window is not an error -- NaN is the truthful
    answer for "coverage of nothing" -- so the warning is noise.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _coverage_indicator(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Cell-wise coverage indicator, NaN where the cell is not evaluable.

    A NaN bound or a NaN truth means "this window does not exist for this
    series", or "the method could not produce an interval here" (an online
    method's warm-up). Such a cell is **excluded** from every coverage average
    rather than scored as a miss -- comparing against NaN yields False, which
    would silently depress coverage in exactly the ragged panels the package is
    built for.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (n_series, n_samples, horizon)
        ``1.0`` where ``lower <= truth <= upper``, ``0.0`` where it is missed,
        and ``np.nan`` where either bound or the truth is NaN.

    Raises
    ------
    ValueError
        If ``interval`` is not 4-D with last axis 2, or if
        ``truth.shape != interval.shape[:-1]``.
    """
    if interval.ndim != 4 or interval.shape[-1] != 2:
        raise ValueError(
            f"interval must have shape (n_series, n_samples, horizon, 2); got {interval.shape}."
        )
    if truth.shape != interval.shape[:-1]:
        raise ValueError(
            "truth.shape must equal interval.shape[:-1]; "
            f"got truth {truth.shape}, interval {interval.shape}."
        )
    lower = interval[..., 0]
    upper = interval[..., 1]
    evaluable = ~(np.isnan(lower) | np.isnan(upper) | np.isnan(truth))
    with np.errstate(invalid="ignore"):
        covered = (lower <= truth) & (truth <= upper)
    return np.where(evaluable, covered.astype(np.float64), np.nan)


def marginal_coverage(interval: Interval, truth: Forecast) -> float:
    """
    Overall fraction of cells covered by their interval.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    float
        Mean of the cell-wise coverage indicator over all axes. Cells that are
        not evaluable (NaN bound or NaN truth) are excluded from the mean, not
        counted as misses; see :func:`_coverage_indicator`. NaN if no cell is
        evaluable.

    Raises
    ------
    ValueError
        If shapes are inconsistent (see :func:`_coverage_indicator`).
    """
    indicator = _coverage_indicator(interval, truth)
    return float(_nanmean(indicator))


def coverage_by_horizon(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across series and samples, per horizon step.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (horizon,)
        Coverage per horizon step, over evaluable cells only. A step with no
        evaluable cell is NaN.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(_nanmean(indicator, axis=(0, 1)), dtype=np.float64)


def coverage_by_series(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across samples and horizon, per series.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (n_series,)
        Coverage per series, over evaluable cells only. A series with no
        evaluable cell is NaN.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(_nanmean(indicator, axis=(1, 2)), dtype=np.float64)


def coverage_per_cell(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across samples, per (series, horizon).

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (n_series, horizon)
        Coverage per (series, horizon) cell, over evaluable samples only. A
        cell with no evaluable sample is NaN.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(_nanmean(indicator, axis=1), dtype=np.float64)


def rolling_coverage(
    interval: Interval,
    truth: Forecast,
    window: int,
) -> NDArray[np.floating]:
    """
    Rolling coverage averaged across the samples axis.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    window : int
        Rolling window size in samples. Must satisfy
        ``1 <= window <= n_samples``.

    Returns
    -------
    NDArray, shape (n_series, n_samples - window + 1, horizon)
        Rolling-mean coverage. Output index ``t`` corresponds to coverage
        over samples ``[t, t + window)`` along the sample axis, counting
        evaluable cells only. A window with none is NaN.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if ``window`` is out of range.
    """
    indicator = _coverage_indicator(interval, truth)
    n_samples = indicator.shape[1]

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if window > n_samples:
        raise ValueError(f"window must be <= n_samples ({n_samples}), got {window}.")

    # sliding_window_view on the sample axis (axis=1) produces shape
    # (n_series, n_samples - window + 1, horizon, window); average over the
    # last axis.
    windows = np.lib.stride_tricks.sliding_window_view(
        indicator.astype(np.float64), window_shape=window, axis=1
    )
    return np.asarray(_nanmean(windows, axis=-1), dtype=np.float64)
