"""Scoring rules for prediction intervals and quantile forecasts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval
from .coverage import _nanmean, marginal_coverage


def winkler_score(
    interval: Interval,
    truth: Forecast,
    alpha: float,
) -> NDArray[np.floating]:
    """
    Per-cell Winkler interval score.

    The Winkler / interval score (Winkler 1972) is the standard proper
    scoring rule for prediction intervals. Lower is better. The score equals
    the interval width plus a penalty for misses, scaled by ``1 / alpha`` so
    that misses are more costly at smaller ``alpha``.

    .. math::

        W_\\alpha(C, y) = (u - l)
            + \\frac{2}{\\alpha} (l - y) \\mathbb{1}\\{y < l\\}
            + \\frac{2}{\\alpha} (y - u) \\mathbb{1}\\{y > u\\}

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    alpha : float
        Target miscoverage level in ``(0, 1)``.

    Returns
    -------
    NDArray, shape (n_series, n_samples, horizon)
        Per-cell Winkler scores. Non-negative, and NaN wherever a bound or the
        truth is NaN -- a cell that does not exist is not scored. Aggregate
        with :func:`numpy.nanmean`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if ``alpha`` is not in ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if interval.ndim != 4 or interval.shape[-1] != 2:
        raise ValueError(
            f"interval must have shape (n_series, n_samples, horizon, 2); got {interval.shape}."
        )
    if truth.shape != interval.shape[:-1]:
        raise ValueError(
            "truth.shape must equal interval.shape[:-1]; "
            f"got truth {truth.shape}, interval {interval.shape}."
        )

    lower = interval[..., 0].astype(np.float64)
    upper = interval[..., 1].astype(np.float64)
    truth_f = truth.astype(np.float64)

    # Winkler is a proper scoring rule, so a cell with no observed target has no
    # score -- and it must be masked explicitly. Both penalties compare against
    # the truth, and `NaN < lower` is False, so a missing truth would otherwise
    # come back as the bare width: the *best* score that width can achieve. That
    # flatters the mean in proportion to how gappy the panel is, which makes the
    # metric incomparable across panels. Mirrors `_coverage_indicator`.
    evaluable = ~(np.isnan(lower) | np.isnan(upper) | np.isnan(truth_f))
    width = upper - lower
    with np.errstate(invalid="ignore"):
        below = np.where(truth_f < lower, (lower - truth_f) * (2.0 / alpha), 0.0)
        above = np.where(truth_f > upper, (truth_f - upper) * (2.0 / alpha), 0.0)
    return np.asarray(np.where(evaluable, width + below + above, np.nan), dtype=np.float64)


def mean_interval_width(interval: Interval) -> NDArray[np.floating]:
    """
    Mean interval width per (series, horizon), averaged over samples.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)

    Returns
    -------
    NDArray, shape (n_series, horizon)
        Mean width per cell, over samples whose bounds are both non-NaN. A
        cell with no such sample is NaN.

        Note this is a wider set than :func:`winkler_score` and the coverage
        metrics average over: a width needs no observed target, so a sample
        whose truth is NaN still contributes here. Compare
        ``Report.n_evaluated_cells`` when the two need to agree.

    Raises
    ------
    ValueError
        If ``interval`` does not have shape ``(n_series, n_samples, horizon, 2)``.
    """
    if interval.ndim != 4 or interval.shape[-1] != 2:
        raise ValueError(
            f"interval must have shape (n_series, n_samples, horizon, 2); got {interval.shape}."
        )
    widths = interval[..., 1].astype(np.float64) - interval[..., 0].astype(np.float64)
    return np.asarray(_nanmean(widths, axis=1), dtype=np.float64)


def pinball_loss(
    prediction: Forecast,
    truth: Forecast,
    quantile: float,
) -> NDArray[np.floating]:
    """
    Per-cell pinball loss at a single quantile level.

    .. math::

        \\rho_q(y, \\hat y) = \\max\\bigl(q (y - \\hat y),\\;
            (q - 1)(y - \\hat y)\\bigr)

    Used when the user has access to a point or quantile prediction and wants
    to evaluate calibration at a specific level. At ``quantile = 0.5`` this
    reduces to ``|y - y_hat| / 2``.

    Parameters
    ----------
    prediction : Forecast, shape (n_series, n_samples, horizon)
        Single-quantile forecast.
    truth : Forecast, shape (n_series, n_samples, horizon)
    quantile : float
        Quantile level in ``(0, 1)``.

    Returns
    -------
    NDArray, shape (n_series, n_samples, horizon)
        Per-cell pinball losses. Non-negative.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if ``quantile`` is not in ``(0, 1)``.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
    if prediction.shape != truth.shape:
        raise ValueError(
            "prediction.shape must equal truth.shape; "
            f"got prediction {prediction.shape}, truth {truth.shape}."
        )

    diff = truth.astype(np.float64) - prediction.astype(np.float64)
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def coverage_width_summary(
    interval: Interval,
    truth: Forecast,
    alpha: float,
) -> dict[str, float]:
    """
    One-line summary of coverage and interval quality.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    alpha : float
        Target miscoverage level in ``(0, 1)``.

    Returns
    -------
    dict
        Keys:

        * ``"marginal_coverage"`` : float
        * ``"target_coverage"`` : float (``1 - alpha``)
        * ``"mean_width"`` : float, averaged over cells with two finite bounds
        * ``"mean_winkler"`` : float, averaged over evaluable cells

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if ``alpha`` is not in ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    cov = marginal_coverage(interval, truth)
    widths = interval[..., 1].astype(np.float64) - interval[..., 0].astype(np.float64)
    mean_width = float(_nanmean(widths))
    mean_winkler = float(_nanmean(winkler_score(interval, truth, alpha)))
    return {
        "marginal_coverage": cov,
        "target_coverage": 1.0 - alpha,
        "mean_width": mean_width,
        "mean_winkler": mean_winkler,
    }
