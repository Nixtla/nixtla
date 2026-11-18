"""NaN-aware exact-order-statistic quantile helpers shared by all methods.

Implements the ``k = ceil((1 - alpha) * (n + 1))`` order statistic
(``method="higher"`` semantics), matching the conformal prediction
literature — never linear interpolation.

A NaN cell in the scores array means "this calibration window does not
exist for this series"; NaN cells are excluded from both the effective
sample count and the order statistic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import InsufficientCalibrationSamples

# Float noise tolerances. ``_ceil`` absorbs noise below 1e-9 in rank units;
# ``_NOISE_TOL`` is absolute on normalised cumulative weight, so genuine
# non-uniform weights landing within 1e-9 below the target may pick one rank
# lower (a weighted-coverage shortfall <= 1e-9). The two only differ for alphas
# given to >= 10 significant digits. The effective-sample-size floor reuses
# ``_NOISE_TOL``: flat weights other than 1 give (Σw)² / Σw² a hair below n.
_NOISE_DECIMALS = 9
_NOISE_TOL = 10.0**-_NOISE_DECIMALS


def _ceil(x: ArrayLike) -> NDArray[np.int64]:
    """Element-wise ``ceil`` that ignores float noise below 1e-9.

    ``1 / (1 - 0.8)`` is ``5.000000000000001``; this returns 5, not 6. An alpha
    given to more than 9 significant digits may lose that last bit.
    """
    return np.ceil(np.round(np.asarray(x, dtype=np.float64), _NOISE_DECIMALS)).astype(np.int64)


def conformal_quantile(
    scores: NDArray[np.floating],
    alpha: float,
    *,
    context: str = "",
) -> NDArray[np.floating]:
    """
    Per-(series, horizon) finite-sample conformal quantile of nonconformity scores.

    For each ``(series, horizon)`` cell, returns the ``k``-th smallest of the
    ``n`` effective (non-NaN) scores along the window axis, where
    ``k = min(ceil((1 - alpha) * (n + 1)), n)``.

    Parameters
    ----------
    scores : NDArray, shape (n_series, n_windows, horizon)
        Nonconformity scores. NaN marks a window that does not exist for
        that series (ragged panels).
    alpha : float
        Miscoverage level in (0, 1).
    context : str, default ""
        Extra text appended to the InsufficientCalibrationSamples message
        (e.g. ``", named alpha_lo"``), so callers can name their own
        parameters.

    Returns
    -------
    NDArray, shape (n_series, horizon)
        The conformal score quantile per (series, horizon) pair. Same float
        dtype as ``scores``.

    Raises
    ------
    ValueError
        If ``scores`` is not 3-D or ``alpha`` is outside (0, 1).
    InsufficientCalibrationSamples
        If any series has fewer than ``ceil(1 / alpha)`` effective samples
        at any horizon. The exception carries the offending series indices.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    arr = np.asarray(scores)
    if arr.ndim != 3:
        raise ValueError(
            f"scores must be 3-D (n_series, n_windows, horizon), got shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    # Effective (non-NaN) window count per (series, horizon).
    n_eff = np.sum(~np.isnan(arr), axis=1)  # (n_series, horizon)

    min_samples = int(_ceil(1.0 / alpha))
    bad = np.flatnonzero(n_eff.min(axis=1) < min_samples)
    if bad.size:
        indices = [int(i) for i in bad]
        raise InsufficientCalibrationSamples(
            f"Series {indices} have fewer than ceil(1/alpha) = {min_samples} "
            f"effective calibration samples for alpha={alpha}{context}.",
            series_indices=indices,
        )

    # k-th order statistic; np.sort places NaN last, so index k-1 is NaN-safe.
    k = _ceil((1.0 - alpha) * (n_eff + 1))
    k = np.minimum(k, n_eff)  # quantile level capped at 1.0
    sorted_scores = np.sort(arr, axis=1)
    idx = (k - 1)[:, np.newaxis, :]  # (n_series, 1, horizon)
    result: NDArray[np.floating] = np.take_along_axis(sorted_scores, idx, axis=1)[:, 0, :]
    return result


# Saturation sentinel for quantile levels at the [0, 1] boundary. 1e300 (not
# float64-max) keeps downstream width / Winkler arithmetic inside float64 range.
_SATURATION_SENTINEL: float = 1e300


def _saturation_value(dtype: np.dtype) -> float:
    """Large-but-finite sentinel for ``dtype``, leaving arithmetic headroom.

    ``1e300`` overflows ``float32`` (max ~3.4e38); for narrower dtypes use
    ``finfo(dtype).max / 1e8`` so interval-width and Winkler arithmetic on the
    sentinel stays finite.
    """
    return float(min(_SATURATION_SENTINEL, float(np.finfo(dtype).max) / 1e8))


def _linear_quantile_from_sorted(
    sorted_scores: NDArray[np.floating],
    n_valid: NDArray[np.integer],
    levels: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Batched linear-interpolation quantile of a pre-sorted, NaN-padded panel.

    The linear-interpolation quantile of every ``(series, horizon)`` cell, in one
    gather instead of a Python loop.

    The arithmetic is fixed rather than delegated (see
    ``numpy.lib._function_base_impl._quantile`` / ``_get_gamma`` / ``_lerp``):
    the virtual index and gamma are evaluated in ``float64`` whatever the input
    dtype, the interpolation runs in the *array* dtype, and it uses numpy's
    two-branch form -- ``a + (b - a) * t`` for ``t < 0.5``, ``b - (b - a) *
    (1 - t)`` otherwise. That last detail is not incidental: the two-branch form
    and a plain ``a + (b - a) * t`` differ in the last bit. Only the final cast
    back to ``sorted_scores.dtype`` narrows, so ``float32`` input keeps
    ``float32`` output without ``float32`` rounding inside the interpolation.

    numpy computed gamma in the array dtype through 2.4 and switched to
    ``float64`` in 2.5, which moves ``float32`` results by about an ulp. This
    helper uses the 2.5 formulation on every version -- the more accurate of the
    two -- so its output does not shift when numpy is upgraded. It is therefore
    bit-identical to ``np.quantile`` for ``float64`` on every supported numpy,
    and for ``float32`` from numpy 2.5 on.

    Parameters
    ----------
    sorted_scores : NDArray, shape (n_series, t, horizon)
        Scores sorted ascending along axis 1 with NaN last (what ``np.sort``
        produces), so each cell's valid entries occupy positions
        ``0 .. n_valid - 1``.
    n_valid : NDArray, shape (n_series, horizon)
        Per-cell count of non-NaN entries.
    levels : NDArray, shape (..., n_series, horizon)
        Quantile level per cell, in [0, 1]. Leading batch dimensions (e.g. an
        AgACI expert axis) are allowed and are preserved in the output.

    Returns
    -------
    NDArray, shape (..., n_series, horizon)
        Same float dtype as ``sorted_scores``. Saturation, in this precedence:
        ``level >= 1`` -> ``+sentinel``; ``level <= 0`` -> ``-sentinel``;
        an empty cell -> ``+sentinel``.
    """
    dtype = sorted_scores.dtype
    q = np.asarray(levels, dtype=np.float64)
    counts = np.asarray(n_valid)
    safe_counts = np.maximum(counts, 1)

    virtual = q * (safe_counts.astype(np.float64) - 1.0)
    previous = np.floor(virtual)
    gamma = virtual - previous

    lo_idx = np.clip(previous, 0.0, None).astype(np.intp)
    lo_idx = np.minimum(lo_idx, safe_counts - 1)
    hi_idx = np.minimum(lo_idx + 1, safe_counts - 1)

    # Broadcast the source by index arithmetic, not by materializing copies:
    # take_along_axis broadcasts arr against indices.
    batch_ndim = q.ndim - 2
    src = sorted_scores.reshape((1,) * batch_ndim + sorted_scores.shape)
    lo = np.take_along_axis(src, lo_idx[..., np.newaxis, :], axis=-2)[..., 0, :]
    hi = np.take_along_axis(src, hi_idx[..., np.newaxis, :], axis=-2)[..., 0, :]

    # The interpolation itself runs in the array dtype: numpy passes gamma to
    # `_lerp` as a weak Python float, so `diff * gamma` promotes to the array
    # dtype. Note `1 - gamma` is evaluated in float64 *before* that narrowing.
    diff = hi - lo
    out = np.where(
        gamma < 0.5,
        lo + diff * gamma.astype(dtype, copy=False),
        hi - diff * (1.0 - gamma).astype(dtype, copy=False),
    )

    # Saturation, applied in reverse precedence so the first rule wins.
    sentinel = dtype.type(_saturation_value(dtype))
    out = np.where(np.broadcast_to(counts == 0, out.shape), sentinel, out)
    out = np.where(q <= 0, -sentinel, out)
    result: NDArray[np.floating] = np.where(q >= 1, sentinel, out).astype(dtype, copy=False)
    return result


def masked_quantile(
    scores: NDArray[np.floating],
    levels: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Per-(series, horizon) NaN-aware empirical quantile at per-cell levels.

    NaN cells are excluded by sorting them to the end and counting the valid
    entries per cell, so no explicit mask array is needed.

    Parameters
    ----------
    scores : NDArray, shape (n_series, t, horizon)
        Observed-score panel; time axis is axis 1. NaN cells are excluded.
    levels : NDArray, shape (..., n_series, horizon)
        Quantile level per cell, in [0, 1]. Leading batch dimensions (e.g. an
        AgACI expert axis) are allowed and are preserved in the output.

    Returns
    -------
    NDArray, shape (..., n_series, horizon)
        Per-cell linear quantile of the non-NaN entries -- see
        :func:`_linear_quantile_from_sorted` for the exact arithmetic and how it
        relates to ``np.quantile`` (identical for ``float64`` on every supported
        numpy; for ``float32`` from numpy 2.5 on). Same float dtype as
        ``scores``. Saturation: ``level >= 1`` -> ``+_SATURATION_SENTINEL``;
        ``level <= 0`` -> ``-_SATURATION_SENTINEL``; an all-NaN cell ->
        ``+_SATURATION_SENTINEL``.
    """
    arr = np.asarray(scores)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    sorted_scores = np.sort(arr, axis=1)  # NaN sorts last
    n_valid = np.sum(~np.isnan(arr), axis=1)  # (n_series, horizon)
    return _linear_quantile_from_sorted(sorted_scores, n_valid, np.asarray(levels))


class ExpandingSortedScores:
    """
    Sorted-prefix state for an online window loop.

    Maintains the chronological prefix ``scores[:, :t, :]`` in sorted order
    (NaN last) together with per-cell non-NaN counts, so a per-cell quantile at
    arbitrary per-cell levels costs one batched gather
    (:func:`_linear_quantile_from_sorted`) instead of a
    ``(n_series x horizon)`` Python loop over ``np.quantile``. Results are
    bit-identical to ``masked_quantile(scores[:, :t, :], levels)``: sorting the
    same multiset yields the same ordering, and NaN-last sort ordering is the
    same masking rule.

    ``push`` inserts one step into the already-sorted buffer with a vectorized
    single-element insertion (an insertion index per cell from
    ``(buffer < x).sum(axis=1)``, then one shift-and-gather), so a full loop
    costs ``O(n^2)`` array work rather than the ``O(n^2 log n)`` of re-sorting
    the prefix at every step. Each cell keeps its valid values packed ascending
    at the front with NaN after, which is exactly the layout ``np.sort``
    produces — hence the bit-identical results.

    Parameters
    ----------
    n_series, horizon : int
        Panel shape of each pushed step.
    dtype : np.dtype
        Float dtype of the scores; preserved by :meth:`quantile`.
    capacity : int or None
        Expected number of pushes. Preallocates the buffer to avoid
        reallocation; it grows automatically if exceeded.
    """

    def __init__(
        self,
        n_series: int,
        horizon: int,
        dtype: np.dtype,
        capacity: int | None = None,
    ) -> None:
        size = max(1, capacity if capacity is not None else 8)
        # Per cell: valid values sorted ascending in [0, n_valid), NaN after.
        self._sorted: NDArray[np.floating] = np.full((n_series, size, horizon), np.nan, dtype=dtype)
        self._n_valid: NDArray[np.integer] = np.zeros((n_series, horizon), dtype=np.int64)
        self._n_pushed: int = 0

    @property
    def n_pushed(self) -> int:
        """Number of steps pushed so far."""
        return self._n_pushed

    def push(self, step_scores: NDArray[np.floating]) -> None:
        """
        Append one window's scores, keeping every cell sorted.

        Parameters
        ----------
        step_scores : NDArray, shape (n_series, horizon)
            NaN marks cells for which this window does not exist; those cells
            are left untouched.
        """
        x = np.asarray(step_scores)
        upper = self._n_pushed  # occupied upper bound across cells
        if upper + 1 > self._sorted.shape[1]:
            grown = np.full(
                (
                    self._sorted.shape[0],
                    max(2 * self._sorted.shape[1], upper + 1),
                    self._sorted.shape[2],
                ),
                np.nan,
                dtype=self._sorted.dtype,
            )
            grown[:, :upper, :] = self._sorted[:, :upper, :]
            self._sorted = grown

        window = self._sorted[:, : upper + 1, :]
        step = x[:, np.newaxis, :]
        # NaN never compares less-than, so trailing NaN never inflate the index.
        position = (window[:, :upper, :] < step).sum(axis=1)  # (n_series, horizon)
        # Cells with no new value get an out-of-range target: nothing moves.
        target = np.where(np.isnan(x), upper + 1, position)[:, np.newaxis, :]
        slot = np.arange(upper + 1)[np.newaxis, :, np.newaxis]

        # Each output slot takes one of three values: the entry already there
        # (before the insertion point), the new score (at it), or the entry one
        # slot left (after it). `target` varies per cell, so the choice is
        # per-element -- but the "one slot left" operand does not, and building it
        # as a contiguous slice copy avoids materializing a full-buffer index
        # array for take_along_axis. That gather was the whole cost of the loop:
        # dropping it measured 1.761 s -> 1.121 s at 200x400x12, bit-identical.
        shifted = np.empty_like(window)
        shifted[:, 1:, :] = window[:, :upper, :]
        shifted[:, :1, :] = window[:, :1, :]  # slot 0 is never "after" the target
        self._sorted[:, : upper + 1, :] = np.where(
            slot < target, window, np.where(slot == target, step, shifted)
        )

        self._n_valid = self._n_valid + ~np.isnan(x)
        self._n_pushed += 1

    def quantile(self, levels: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Per-cell quantile of the current prefix.

        Parameters
        ----------
        levels : NDArray, shape (..., n_series, horizon)
            Quantile level per cell, in [0, 1]. Leading batch dimensions are
            allowed and preserved.

        Returns
        -------
        NDArray, shape (..., n_series, horizon)
            Same dtype as the pushed scores; identical to
            ``masked_quantile(prefix, levels)``. An empty prefix saturates high.
        """
        # A length-1 floor keeps the gather well-defined on an empty prefix; the
        # slot is NaN, and n_valid == 0 saturates it anyway.
        end = max(self._n_pushed, 1)
        return _linear_quantile_from_sorted(
            self._sorted[:, :end, :], self._n_valid, np.asarray(levels)
        )


def weighted_quantile_from_weights(
    scores: NDArray[np.floating],
    alpha: float,
    weights: NDArray[np.floating],
    *,
    context: str = "",
) -> NDArray[np.floating]:
    """
    Per-(series, horizon) NaN-aware exact weighted conformal quantile.

    For each cell, over the valid (non-NaN) scores with their supplied
    non-negative weights: ``W = sum(w)``,
    ``target = clip((1 - alpha) * (W + 1) / W, 0, 1)``, and the returned value
    is the smallest observed score whose normalized cumulative weight reaches
    ``target`` (exact weighted order statistic).

    Weights, ``W`` and the cumulative-weight comparison are computed in
    ``float64`` regardless of the score dtype; only the returned order
    statistic carries the score dtype. NaN cells are keyed to ``+inf`` for the
    sort and given zero weight, so they can never be selected.

    Parameters
    ----------
    scores : NDArray, shape (n_series, n_windows, horizon)
        Nonconformity scores. NaN marks a missing window.
    alpha : float
        Miscoverage level in (0, 1).
    weights : NDArray, shape (n_series, n_windows, horizon)
        Non-negative, finite weights aligned with ``scores``.
    context : str, default ""
        Extra text appended to the InsufficientCalibrationSamples message
        (e.g. ``", rho=0.99"``), so callers can name their own parameters.

    Returns
    -------
    NDArray, shape (n_series, horizon)
        Same float dtype as ``scores``.

    Raises
    ------
    ValueError
        If ``scores`` is not 3-D, shapes disagree, ``alpha`` is outside (0, 1),
        or any weight is negative or non-finite.
    InsufficientCalibrationSamples
        If any series' weights sum to zero at a horizon that has valid scores,
        or its effective sample size (Sum(w)^2 / Sum(w^2)) is below
        ``ceil(1 / alpha)`` at any horizon. Carries the offending indices.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    arr = np.asarray(scores)
    if arr.ndim != 3:
        raise ValueError(
            f"scores must be 3-D (n_series, n_windows, horizon), got shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != arr.shape:
        raise ValueError(
            f"scores and weights must have the same shape, got {arr.shape} and {w.shape}"
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be non-negative and finite.")

    min_samples = int(_ceil(1.0 / alpha))
    sentinel = _saturation_value(arr.dtype)

    valid = ~np.isnan(arr)
    w = np.where(valid, w, 0.0)
    m = valid.sum(axis=1)  # (n_series, horizon)
    nonempty = m > 0
    big_w = w.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(nonempty, big_w * big_w / (w * w).sum(axis=1), 0.0)
        target = np.clip(np.where(nonempty, (1.0 - alpha) * (big_w + 1.0) / big_w, 0.0), 0.0, 1.0)
    ess_min = ess.min(axis=1)

    # A cell with valid scores but zero total weight carries no information, and
    # the ESS check below cannot see it: big_w == 0 makes ess NaN, and a NaN
    # comparison is False. Caught first so the message names the real problem.
    zeroed = np.flatnonzero((nonempty & (big_w == 0.0)).any(axis=1))
    if zeroed.size:
        indices = [int(i) for i in zeroed]
        raise InsufficientCalibrationSamples(
            f"Series {indices} have calibration weights summing to zero at some "
            f"horizon, so no score can be selected for alpha={alpha}{context}.",
            series_indices=indices,
        )

    keyed = np.where(valid, arr, np.inf)
    order = np.argsort(keyed, axis=1)
    sorted_scores = np.take_along_axis(keyed, order, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cumw = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1) / big_w[:, None, :]

    # A saturated target (1.0) gets no tolerance, so it still reaches the largest score.
    thr = np.where(target >= 1.0, target, target - _NOISE_TOL)
    reached = cumw >= thr[:, np.newaxis, :]
    idx = np.argmax(reached, axis=1)
    idx = np.where(reached.any(axis=1), idx, np.maximum(m, 1) - 1)
    out = np.take_along_axis(sorted_scores, idx[:, np.newaxis, :], axis=1)[:, 0, :]
    out = np.where(nonempty, out, arr.dtype.type(sentinel)).astype(arr.dtype, copy=False)

    bad = np.flatnonzero(ess_min < min_samples - _NOISE_TOL)
    if bad.size:
        indices = [int(i) for i in bad]
        raise InsufficientCalibrationSamples(
            f"Series {indices} have effective sample size below the required "
            f"ceil(1/alpha) = {min_samples} for alpha={alpha}{context}.",
            series_indices=indices,
        )
    result: NDArray[np.floating] = out
    return result


def weighted_conformal_quantile(
    scores: NDArray[np.floating],
    alpha: float,
    rho: float,
) -> NDArray[np.floating]:
    """
    Per-(series, horizon) NaN-aware exponentially-weighted conformal quantile.

    For each cell, over the valid (non-NaN) scores ``v_0..v_{m-1}`` in
    chronological order: ``w_j = rho ** (m - 1 - j)``, ``W = sum(w)``,
    ``target = clip((1 - alpha) * (W + 1) / W, 0, 1)``, and the returned value
    is the smallest observed score whose normalized cumulative weight reaches
    ``target`` (exact weighted order statistic, ``method="higher"`` analogue).

    Weights, ``W`` and the cumulative-weight comparison are computed in
    ``float64`` regardless of the score dtype (as the per-cell implementation
    this replaced did); only the returned order statistic carries the score
    dtype. NaN cells are handled by keying them to ``+inf`` for the sort and
    giving them zero weight, so they can never be selected.

    Parameters
    ----------
    scores : NDArray, shape (n_series, n_windows, horizon)
        Chronological on axis 1 (oldest first). NaN marks a missing window.
    alpha : float
        Miscoverage level in (0, 1).
    rho : float
        Decay in (0, 1]. ``rho == 1`` recovers the unweighted conformal quantile.

    Returns
    -------
    NDArray, shape (n_series, horizon)
        Same float dtype as ``scores``.

    Raises
    ------
    ValueError
        If ``scores`` is not 3-D, ``alpha`` is outside (0, 1), or ``rho`` is
        outside (0, 1].
    InsufficientCalibrationSamples
        If any series' effective sample size (Sum(w)^2 / Sum(w^2)) is below
        ``ceil(1 / alpha)`` at any horizon. Carries the offending series indices.
    """
    if not 0 < rho <= 1:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    arr = np.asarray(scores)
    if arr.ndim != 3:
        raise ValueError(
            f"scores must be 3-D (n_series, n_windows, horizon), got shape {arr.shape}"
        )
    valid = ~np.isnan(arr)
    m = valid.sum(axis=1)
    rank = np.cumsum(valid, axis=1) - 1
    n_windows = arr.shape[1]
    pow_table = np.power(np.float64(rho), np.arange(n_windows + 1, dtype=np.float64))
    weights = np.where(valid, pow_table[m[:, np.newaxis, :] - 1 - rank], 0.0)
    return weighted_quantile_from_weights(scores, alpha, weights, context=f", rho={rho}")


def weighted_quantile_per_test_point(
    scores: NDArray[np.floating],
    alpha: float,
    weights: NDArray[np.floating],
    test_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Weighted conformal quantile with a per-test-point normalization.

    Tibshirani et al. (2019) normalize calibration weights by
    ``sum(w) + w(x_test)``, so the quantile depends on the test point. This is
    equivalent to keeping the cumulative weights normalized by ``W = sum(w)``
    and moving the test weight into the target:
    ``target_j = (1 - alpha) * (W + w_test_j) / W``.

    The calibration scores are sorted once and reused for every test point.
    Selection is then a single vectorized step: the normalized cumulative
    weights are compared against every test point's target at once, forming a
    ``(n_series, n_windows, n_samples, horizon)`` boolean array whose
    ``argmax`` along the window axis gives each cell's order statistic. There
    is no Python loop over the ``(series x horizon)`` grid.

    Parameters
    ----------
    scores : NDArray, shape (n_series, n_windows, horizon)
        Calibration scores. NaN marks a missing window.
    alpha : float
        Miscoverage level in (0, 1).
    weights : NDArray, shape (n_series, n_windows, horizon)
        Non-negative, finite calibration weights.
    test_weights : NDArray, shape (n_series, n_samples, horizon)
        Non-negative, finite weights for the test points.

    Returns
    -------
    NDArray, shape (n_series, n_samples, horizon)
        Same float dtype as ``scores``.

    Raises
    ------
    ValueError
        If shapes disagree, ``alpha`` is outside (0, 1), or any weight is
        negative or non-finite.
    InsufficientCalibrationSamples
        If any series' weights sum to zero, or its effective sample size
        (Sum(w)^2 / Sum(w^2)) is below ``ceil(1 / alpha)``, at any horizon.
        Carries the offending series indices.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    arr = np.asarray(scores)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    tw = np.asarray(test_weights, dtype=np.float64)
    if arr.ndim != 3 or w.shape != arr.shape:
        raise ValueError(
            f"scores and weights must be 3-D and matching, got {arr.shape} and {w.shape}"
        )
    if tw.ndim != 3 or tw.shape[0] != arr.shape[0] or tw.shape[2] != arr.shape[2]:
        raise ValueError(
            "test_weights must have shape (n_series, n_samples, horizon) matching "
            f"scores on axes 0 and 2; got {tw.shape} against {arr.shape}"
        )
    finite = np.all(np.isfinite(w)) and np.all(np.isfinite(tw))
    if not finite or np.any(w < 0.0) or np.any(tw < 0.0):
        raise ValueError("weights must be non-negative and finite.")

    valid = ~np.isnan(arr)
    w = np.where(valid, w, 0.0)
    m = valid.sum(axis=1)
    nonempty = m > 0
    big_w = w.sum(axis=1)  # (n_series, horizon)
    sentinel = _saturation_value(arr.dtype)

    # The same two guards weighted_quantile_from_weights applies, for the same
    # reason: without them a cell whose weights carry no information falls
    # through to the ``idx = m - 1`` fallback below and silently returns the
    # largest calibration score. Zero total weight is caught first, since
    # big_w == 0 makes ess NaN and a NaN comparison is False.
    zeroed = np.flatnonzero((nonempty & (big_w == 0.0)).any(axis=1))
    if zeroed.size:
        indices = [int(i) for i in zeroed]
        raise InsufficientCalibrationSamples(
            f"Series {indices} have calibration weights summing to zero at some "
            f"horizon, so no score can be selected for alpha={alpha}.",
            series_indices=indices,
        )
    min_samples = int(_ceil(1.0 / alpha))
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(nonempty, big_w * big_w / (w * w).sum(axis=1), 0.0)
    bad = np.flatnonzero(ess.min(axis=1) < min_samples - _NOISE_TOL)
    if bad.size:
        indices = [int(i) for i in bad]
        raise InsufficientCalibrationSamples(
            f"Series {indices} have effective sample size below the required "
            f"ceil(1/alpha) = {min_samples} for alpha={alpha}.",
            series_indices=indices,
        )

    keyed = np.where(valid, arr, np.inf)
    order = np.argsort(keyed, axis=1)
    sorted_scores = np.take_along_axis(keyed, order, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cumw = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1) / big_w[:, None, :]
        # (n_series, n_samples, horizon)
        target = np.clip((1.0 - alpha) * (big_w[:, None, :] + tw) / big_w[:, None, :], 0.0, 1.0)

    # cumw is non-decreasing on axis 1, so a mask-argmax is searchsorted(left).
    thr = np.where(target >= 1.0, target, target - _NOISE_TOL)  # saturated: no tolerance
    reached = cumw[:, :, None, :] >= thr[:, None, :, :]
    idx = np.argmax(reached, axis=1)
    idx = np.where(reached.any(axis=1), idx, np.maximum(m, 1)[:, None, :] - 1)
    out = np.take_along_axis(sorted_scores[:, :, None, :], idx[:, None, :, :], axis=1)[:, 0]
    out = np.where(nonempty[:, None, :], out, arr.dtype.type(sentinel))
    result: NDArray[np.floating] = out.astype(arr.dtype, copy=False)
    return result
