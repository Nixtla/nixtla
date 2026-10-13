"""Conditional (subgroup) coverage diagnostics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval
from .coverage import _coverage_indicator, _nanmean


def coverage_by_magnitude_bin(
    interval: Interval,
    truth: Forecast,
    n_bins: int = 10,
    by: Literal["truth", "midpoint"] = "truth",
) -> dict[str, NDArray[Any]]:
    """
    Coverage by magnitude bin.

    Bins all cells by either ``|truth|`` or by ``|interval midpoint|``,
    then computes coverage and mean width per bin. Useful for catching the
    case where intervals are well-calibrated on average but fail on extreme
    values.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    n_bins : int, default 10
        Number of equal-frequency (quantile) bins. Must be ``>= 2``.
    by : {"truth", "midpoint"}, default "truth"
        Whether to bin by the absolute truth value or by the absolute
        interval midpoint.

    Returns
    -------
    dict with keys:
        ``"bin_edges"`` : NDArray, shape (n_bins + 1,)
            Monotonically increasing bin edges.
        ``"coverage_per_bin"`` : NDArray, shape (n_bins,)
            Coverage per bin, over evaluable cells only. Empty bins receive
            ``np.nan``.
        ``"mean_width_per_bin"`` : NDArray, shape (n_bins,)
            Mean interval width per bin. Empty bins receive ``np.nan``.
        ``"n_per_bin"`` : NDArray of int, shape (n_bins,)
            Number of **evaluable** cells in each bin -- cells with a NaN bound
            or NaN truth are excluded, since they contribute to neither mean.

    Raises
    ------
    ValueError
        If shapes are inconsistent, if ``n_bins < 2``, or if ``by`` is not one
        of the supported options.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
    if by not in ("truth", "midpoint"):
        raise ValueError(f'by must be "truth" or "midpoint", got {by!r}.')

    indicator = _coverage_indicator(interval, truth)
    widths = interval[..., 1].astype(np.float64) - interval[..., 0].astype(np.float64)

    if by == "truth":
        binning_var = np.abs(truth.astype(np.float64))
    else:
        midpoint = (interval[..., 0].astype(np.float64) + interval[..., 1].astype(np.float64)) / 2.0
        binning_var = np.abs(midpoint)

    flat_var = binning_var.reshape(-1)
    flat_indicator = indicator.reshape(-1).astype(np.float64)
    flat_widths = widths.reshape(-1)

    # Equal-frequency bin edges. Quantile edges may not be unique if many
    # values tie; np.unique handles that but we ensure n_bins via clipping.
    quantile_levels = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(flat_var, quantile_levels)

    # Assign each cell to a bin. ``np.digitize`` with the inner edges returns
    # indices in ``[0, n_bins]``; clip the right tail (max-valued cells) to
    # the last bin.
    bin_idx = np.digitize(flat_var, bin_edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    coverage_per_bin = np.full(n_bins, np.nan, dtype=np.float64)
    mean_width_per_bin = np.full(n_bins, np.nan, dtype=np.float64)
    n_per_bin = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        # Count evaluable cells, not cells: a NaN bound or truth contributes to
        # neither mean, so reporting it in n_per_bin would misstate the sample
        # the bin's coverage was computed from.
        count = int(np.count_nonzero(mask & ~np.isnan(flat_indicator)))
        n_per_bin[b] = count
        if count > 0:
            coverage_per_bin[b] = float(_nanmean(flat_indicator[mask]))
            mean_width_per_bin[b] = float(_nanmean(flat_widths[mask]))

    return {
        "bin_edges": bin_edges,
        "coverage_per_bin": coverage_per_bin,
        "mean_width_per_bin": mean_width_per_bin,
        "n_per_bin": n_per_bin,
    }


def coverage_by_group(
    interval: Interval,
    truth: Forecast,
    groups: NDArray[Any],
) -> dict[str, NDArray[Any]]:
    """
    Coverage by user-supplied group label.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    groups : NDArray, shape (n_series, n_samples, horizon)
        Group label for each cell. Any dtype that supports :func:`np.unique`
        (int, str, etc.).

    Returns
    -------
    dict with keys:
        ``"groups"`` : NDArray, shape (n_groups,)
            Unique group values, sorted.
        ``"coverage_per_group"`` : NDArray, shape (n_groups,)
            Over evaluable cells only; NaN for a group with none.
        ``"mean_width_per_group"`` : NDArray, shape (n_groups,)
        ``"n_per_group"`` : NDArray of int, shape (n_groups,)
            Number of **evaluable** cells per group -- cells with a NaN bound
            or NaN truth are excluded, since they contribute to neither mean.

    Raises
    ------
    ValueError
        If shapes are inconsistent, in particular if
        ``groups.shape != truth.shape``.
    """
    if groups.shape != truth.shape:
        raise ValueError(
            f"groups.shape must equal truth.shape; got groups {groups.shape}, truth {truth.shape}."
        )

    indicator = _coverage_indicator(interval, truth)
    widths = interval[..., 1].astype(np.float64) - interval[..., 0].astype(np.float64)

    flat_groups = groups.reshape(-1)
    flat_indicator = indicator.reshape(-1).astype(np.float64)
    flat_widths = widths.reshape(-1)

    unique_groups, inverse = np.unique(flat_groups, return_inverse=True)
    n_groups = unique_groups.shape[0]

    coverage_per_group = np.empty(n_groups, dtype=np.float64)
    mean_width_per_group = np.empty(n_groups, dtype=np.float64)
    n_per_group = np.empty(n_groups, dtype=np.int64)

    for g in range(n_groups):
        mask = inverse == g
        # Evaluable cells only -- see the note in coverage_by_magnitude_bin.
        count = int(np.count_nonzero(mask & ~np.isnan(flat_indicator)))
        n_per_group[g] = count
        coverage_per_group[g] = float(_nanmean(flat_indicator[mask])) if count else np.nan
        mean_width_per_group[g] = float(_nanmean(flat_widths[mask])) if count else np.nan

    return {
        "groups": unique_groups,
        "coverage_per_group": coverage_per_group,
        "mean_width_per_group": mean_width_per_group,
        "n_per_group": n_per_group,
    }
