"""Public DataFrame-free helpers for long ↔ panel conversion.

This module is the pure-numpy boundary between long-format forecasting data
(one row per ``(series, timestamp)``) and the ``(n_series, T)`` panels the
conformal methods consume. It imports **only** numpy — no pandas, no adapters, no
library types — so it is embeddable in the Nixtla orchestrators exactly like
the methods' array surface, and a future move to ``utilsforecast`` stays trivial.

Ragged panels use **left-NaN padding**: a series that starts late has a
contiguous run of leading NaN. Interior gaps (a series missing an interior
timestamp) are illegal and raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PanelIndex:
    """Inversion metadata for a ``(n_series, T)`` panel.

    Attributes
    ----------
    series_ids : tuple
        Series identifiers in panel-row order (axis 0). Sorted ascending.
    start_offsets : NDArray[np.integer], shape (n_series,)
        Column index of each series' first non-NaN observation.
    lengths : NDArray[np.integer], shape (n_series,)
        Per-series effective (non-NaN) observation count.
    timestamps : tuple or None
        Length-``T`` time labels for the columns, or ``None`` when the panel
        was built positionally.
    """

    series_ids: tuple[object, ...]
    start_offsets: NDArray[np.integer]
    lengths: NDArray[np.integer]
    timestamps: tuple[object, ...] | None


def long_to_panel(
    values: Sequence[float] | NDArray[np.floating],
    series_ids: Sequence[object],
    time_index: Sequence[object],
    *,
    dtype: type[np.floating] = np.float64,
) -> tuple[NDArray[np.floating], PanelIndex]:
    """Pivot long-format observations into a left-NaN-padded panel.

    Parameters
    ----------
    values : sequence of float, shape (n_obs,)
        Observation values.
    series_ids : sequence, shape (n_obs,)
        Series identifier for each observation.
    time_index : sequence, shape (n_obs,)
        Time label for each observation. Labels must be mutually comparable
        (e.g. all timestamps, or all ints).
    dtype : numpy float dtype, default float64
        Output panel dtype.

    Returns
    -------
    panel : NDArray, shape (n_series, T)
        Left-NaN-padded panel; ``T`` is the number of distinct timestamps.
    index : PanelIndex
        Inversion metadata.

    Raises
    ------
    ValueError
        If the input lengths mismatch, or a series has an interior gap
        (a missing interior timestamp, which would produce an interior NaN).
    """
    vals = np.asarray(values, dtype=dtype)
    sids = np.asarray(series_ids, dtype=object)
    times = np.asarray(time_index, dtype=object)
    if not (vals.shape[0] == sids.shape[0] == times.shape[0]):
        raise ValueError(
            f"values, series_ids, time_index must be the same length; got "
            f"{vals.shape[0]}, {sids.shape[0]}, {times.shape[0]}."
        )

    unique_series = tuple(sorted(set(sids.tolist())))
    unique_times = sorted(set(times.tolist()))
    series_pos = {s: i for i, s in enumerate(unique_series)}
    time_pos = {t: j for j, t in enumerate(unique_times)}

    n_series = len(unique_series)
    t_len = len(unique_times)
    panel = np.full((n_series, t_len), np.nan, dtype=dtype)
    for v, s, t in zip(vals.tolist(), sids.tolist(), times.tolist(), strict=True):
        panel[series_pos[s], time_pos[t]] = v

    start_offsets = np.empty(n_series, dtype=np.int64)
    lengths = np.empty(n_series, dtype=np.int64)
    for i in range(n_series):
        valid = ~np.isnan(panel[i])
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            raise ValueError(f"Series {unique_series[i]!r} has no observations.")
        start, end = int(idx[0]), int(idx[-1])
        if not valid[start:].all():
            raise ValueError(
                f"Series {unique_series[i]!r} has an interior or trailing gap; "
                "only a contiguous leading NaN prefix (left padding) is allowed."
            )
        start_offsets[i] = start
        lengths[i] = end - start + 1

    index = PanelIndex(
        series_ids=unique_series,
        start_offsets=start_offsets,
        lengths=lengths,
        timestamps=tuple(unique_times),
    )
    return panel, index


def panel_to_long(
    panel: NDArray[np.floating],
    index: PanelIndex,
) -> tuple[list[object], list[object], NDArray[np.floating]]:
    """Flatten a panel back to long-format non-NaN rows.

    Parameters
    ----------
    panel : NDArray, shape (n_series, T)
    index : PanelIndex
        Metadata from :func:`long_to_panel`. When ``index.timestamps`` is
        ``None``, column positions ``0..T-1`` are used as time labels.

    Returns
    -------
    row_series_ids : list, shape (n_rows,)
    row_timestamps : list, shape (n_rows,)
    row_values : NDArray, shape (n_rows,)
        Non-NaN cells in ``(series, time)`` order.
    """
    n_series, t_len = panel.shape
    times: tuple[object, ...] = (
        index.timestamps if index.timestamps is not None else tuple(range(t_len))
    )
    row_series: list[object] = []
    row_times: list[object] = []
    row_vals: list[float] = []
    for i in range(n_series):
        for j in range(t_len):
            v = panel[i, j]
            if not np.isnan(v):
                row_series.append(index.series_ids[i])
                row_times.append(times[j])
                row_vals.append(float(v))
    return row_series, row_times, np.asarray(row_vals, dtype=panel.dtype)


def cv_frames_to_arrays(
    cubes: Sequence[NDArray[np.floating]],
    *,
    aligned: bool,
) -> list[NDArray[np.floating]]:
    """Assemble calendar-cutoff CV cubes into per-series or aligned arrays.

    Parameters
    ----------
    cubes : sequence of NDArray, each shape (n_series, n_windows, horizon)
        CV outputs sharing a NaN pattern (e.g. ``[predictions, truths]``). A
        ``(series, window)`` cell is valid iff it is non-NaN across the whole
        horizon in **every** cube.
    aligned : bool
        ``True`` returns the cubes unchanged (NaN cells preserved for the
        ragged calibration path). ``False`` returns dense per-series arrays: each
        series keeps only its valid windows, right-aligned, truncated to the
        minimum valid-window count across series.

    Returns
    -------
    list of NDArray
        One array per input cube, in the same order.

    Raises
    ------
    ValueError
        If ``cubes`` is empty, shapes disagree, or (``aligned=False``) any
        series has zero valid windows.
    """
    if not cubes:
        raise ValueError("cubes must contain at least one array.")
    ref = cubes[0]
    for c in cubes:
        if c.shape != ref.shape:
            msg = f"all cubes must share a shape; got {ref.shape} and {c.shape}."
            raise ValueError(msg)
    if aligned:
        return [np.asarray(c) for c in cubes]

    n_series, n_windows, _ = ref.shape
    # valid[(s, w)] iff non-NaN across horizon in every cube.
    valid = np.ones((n_series, n_windows), dtype=bool)
    for c in cubes:
        valid &= ~np.isnan(c).any(axis=2)

    per_series_counts = valid.sum(axis=1)
    if (per_series_counts == 0).any():
        bad = np.flatnonzero(per_series_counts == 0).tolist()
        raise ValueError(f"Series {bad} have zero valid CV windows.")
    w_min = int(per_series_counts.min())

    out: list[NDArray[np.floating]] = []
    for c in cubes:
        dense = np.empty((n_series, w_min, ref.shape[2]), dtype=c.dtype)
        for s in range(n_series):
            valid_idx = np.flatnonzero(valid[s])[-w_min:]  # most-recent w_min
            dense[s] = c[s, valid_idx, :]
        out.append(dense)
    return out


def exog_to_long(
    exog: NDArray[np.floating],
    series_ids: Sequence[object],
) -> tuple[list[object], list[int], list[int], NDArray[np.floating]]:
    """Flatten a 4-D exogenous array to long-format rows.

    Parameters
    ----------
    exog : NDArray, shape (n_series, n_samples, horizon, n_features)
    series_ids : sequence, shape (n_series,)

    Returns
    -------
    row_series_ids : list, shape (n_rows,)
    row_sample_idx : list of int, shape (n_rows,)
    row_step_idx : list of int, shape (n_rows,)
        Zero-based horizon step.
    features : NDArray, shape (n_rows, n_features)
        Rows in ``(series, sample, step)`` order.

    Raises
    ------
    ValueError
        If ``exog`` is not 4-D or its leading axis mismatches ``series_ids``.
    """
    arr = np.asarray(exog, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(
            f"exog must be 4-D (n_series, n_samples, horizon, n_features), got shape {arr.shape}"
        )
    n_series, n_samples, horizon, n_features = arr.shape
    if len(series_ids) != n_series:
        raise ValueError(f"series_ids length {len(series_ids)} != exog leading axis {n_series}.")
    r_sids: list[object] = []
    r_samp: list[int] = []
    r_step: list[int] = []
    rows: list[NDArray[np.floating]] = []
    for s in range(n_series):
        for smp in range(n_samples):
            for h in range(horizon):
                r_sids.append(series_ids[s])
                r_samp.append(smp)
                r_step.append(h)
                rows.append(arr[s, smp, h, :])
    features = np.asarray(rows, dtype=np.float64).reshape(-1, n_features)
    return r_sids, r_samp, r_step, features


def level_to_quantiles(level: int) -> tuple[float, float]:
    """Convert a symmetric Nixtla interval level to a (lo, hi) quantile pair.

    Parameters
    ----------
    level : int
        Interval level in (0, 100), e.g. ``80``.

    Returns
    -------
    tuple of float
        ``((100 - level) / 200, 1 - (100 - level) / 200)`` — e.g. ``(0.1, 0.9)``.

    Raises
    ------
    ValueError
        If ``level`` is not in (0, 100).
    """
    if not 0 < level < 100:
        raise ValueError(f"level must be in (0, 100), got {level}.")
    tail = (100 - level) / 200.0
    return (tail, 1.0 - tail)


def quantiles_to_level(q_lo: float, q_hi: float) -> int:
    """Convert a symmetric (lo, hi) quantile pair to a Nixtla interval level.

    Parameters
    ----------
    q_lo, q_hi : float
        Quantiles in (0, 1) symmetric about 0.5 (``q_lo + q_hi == 1``).

    Returns
    -------
    int
        The interval level, e.g. ``(0.1, 0.9) -> 80``.

    Raises
    ------
    ValueError
        If the pair is out of range or not symmetric about 0.5.
    """
    if not (0.0 < q_lo < 0.5 < q_hi < 1.0):
        msg = f"expected 0 < q_lo < 0.5 < q_hi < 1, got ({q_lo}, {q_hi})."
        raise ValueError(msg)
    if not np.isclose(q_lo + q_hi, 1.0):
        msg = f"quantiles must be symmetric about 0.5, got ({q_lo}, {q_hi})."
        raise ValueError(msg)
    return int(round((1.0 - 2.0 * q_lo) * 100.0))
