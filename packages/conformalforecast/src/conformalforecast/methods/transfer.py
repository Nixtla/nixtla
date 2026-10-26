"""Transfer support: target-domain adaptation of a source-fitted method.

``adapt()`` exists because several transfer strategies have no target-domain
``(prediction, truth)`` pairs to calibrate on — they modify a source-fitted
quantile using target features or target history alone. It lives on the
fitted side and returns a new fitted object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ..base import CalibrationResult, Exog, Forecast, Series
from ..weighting.base import WeightFunction
from .base import ConformalForecaster

_ESTIMATORS = ("mad", "std", "iqr")
_MAD_TO_SIGMA = 1.4826


def _scale_1d(values: NDArray[np.floating], estimator: str) -> float:
    """
    Scale of one series, computed on its first differences.

    First differences make the estimate robust to level and to slow trend,
    which raw values are not.

    Parameters
    ----------
    values : NDArray, shape (n,)
        Finite values, NaN already removed.
    estimator : {"mad", "std", "iqr"}

    Returns
    -------
    float
        Non-negative scale. Zero signals a degenerate (constant) series.
    """
    d = np.diff(values)
    if d.size == 0:
        return 0.0
    if estimator == "std":
        return float(np.std(d))
    if estimator == "iqr":
        q25, q75 = np.quantile(d, [0.25, 0.75])
        return float(q75 - q25)
    med = np.median(d)
    return float(np.median(np.abs(d - med)) * _MAD_TO_SIGMA)


def _validate_estimator(estimator: str) -> None:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")


def _finalize(scales: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the scales, raising ``ValueError`` if any is zero or non-finite."""
    bad = np.flatnonzero(~(scales > 0.0))
    if bad.size:
        indices = [int(i) for i in bad]
        raise ValueError(
            f"Series {indices} have a degenerate (zero or non-finite) scale. A "
            "constant series carries no scale information; drop it or choose a "
            "different scale_estimator."
        )
    return scales


def compute_series_scales(series: Series, estimator: str) -> NDArray[np.floating]:
    """
    Per-series scale from a 2-D panel of observations.

    Parameters
    ----------
    series : Series, shape (n_series, T)
        Left-NaN-padded panel. Padding is ignored.
    estimator : {"mad", "std", "iqr"}

    Returns
    -------
    NDArray, shape (n_series,)

    Raises
    ------
    ValueError
        If ``estimator`` is unknown, ``series`` is not 2-D, any series has
        fewer than 2 observations, or any scale is degenerate.
    """
    _validate_estimator(estimator)
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"series must be 2-D (n_series, T), got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("series must have at least 2 observations per series.")
    scales = np.array([_scale_1d(row[~np.isnan(row)], estimator) for row in arr], dtype=np.float64)
    return _finalize(scales)


def scales_from_truths(truths: Forecast, estimator: str) -> NDArray[np.floating]:
    """
    Per-series scale from a 3-D calibration truth array.

    Values are flattened in ``(window, horizon)`` order per series and NaN
    windows dropped, so this works at ``horizon == 1`` where there is no
    within-window difference to take.

    Parameters
    ----------
    truths : Forecast, shape (n_series, n_windows, horizon)
    estimator : {"mad", "std", "iqr"}

    Returns
    -------
    NDArray, shape (n_series,)

    Raises
    ------
    ValueError
        If ``estimator`` is unknown, ``truths`` is not 3-D, or any scale is
        degenerate. A series with fewer than 2 usable (non-NaN) observations
        after flattening yields a scale of zero, reported as degenerate with
        its series index.
    """
    _validate_estimator(estimator)
    arr = np.asarray(truths, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"truths must be 3-D (n_series, n_windows, horizon), got shape {arr.shape}"
        )
    flat = arr.reshape(arr.shape[0], -1)
    scales = np.array([_scale_1d(row[~np.isnan(row)], estimator) for row in flat], dtype=np.float64)
    return _finalize(scales)


class SupportsTransfer(ConformalForecaster, ABC):
    """
    A fitted method that can be adapted to a target domain after source calibration.

    ``adapt()`` exists because several transfer strategies have no target-domain
    ``(prediction, truth)`` pairs to calibrate on: density-ratio weighting needs
    only target features, scale alignment only target history. Methods with
    target labels use ``recalibrate()`` on
    :class:`~conformalforecast.methods.base.ConformalForecaster` instead.

    This is an intermediate ABC rather than a standalone mixin because
    ``adapt`` reads ``scores_`` and ``n_calibration_samples_``: a mixin would
    have to re-declare fields ``ConformalForecaster`` already owns.
    Implementors provide ``_scale_estimator`` (the configured estimator, read
    from ``method_``) and ``_recompute_quantile()``, which rebuilds fitted
    state from the stored source scores plus the current weights and scale
    ratio.

    ``adapt`` returns a **new** object and leaves the receiver untouched, and it
    always recomputes from stored source state -- so it is idempotent and never
    compounds onto a previously adapted quantile.

    Attributes
    ----------
    weighting_ : WeightFunction
        The weight function in force: ``UniformWeights`` after ``fit``, or the
        one bound by the ``adapt`` call that produced this object.
    scores_ : NDArray, shape (n_series, n_windows, horizon)
        Source-domain nonconformity scores, retained so ``adapt`` can recompute.
    X_calibration_ : NDArray or None, shape (n_series, n_windows, horizon, n_features)
        Source-domain features handed to ``fit``, retained for the weighting
        function's ``fit_source``.
    source_scales_, target_scales_ : NDArray or None, shape (n_series,)
        Per-series scales of the source and target domains. ``source_scales_``
        is ``None`` when no scale estimator is configured; ``target_scales_``
        until ``adapt(target_series=...)``.
    scale_ratio_ : NDArray or None, shape (n_series,)
        The multiplier applied to the base quantile.
    """

    weighting_: WeightFunction
    #: Shape (n_series, n_windows, horizon) — retained source scores.
    scores_: NDArray[np.floating]
    #: Shape (n_series, n_windows, horizon, n_features) — retained source features.
    X_calibration_: NDArray[np.floating] | None
    source_scales_: NDArray[np.floating] | None
    target_scales_: NDArray[np.floating] | None = None
    scale_ratio_: NDArray[np.floating] | None = None

    @property
    @abstractmethod
    def _scale_estimator(self) -> str | None:
        """The configured scale estimator, or None when scale alignment is off."""
        ...

    @abstractmethod
    def _recompute_quantile(self) -> None:
        """Rebuild fitted state from ``scores_``, the current weights and scales."""
        ...

    def adapt(
        self,
        *,
        weighting: WeightFunction | None = None,
        target_X: Exog | None = None,
        target_series: Series | None = None,
    ) -> Self:
        """
        Adapt this source-fitted object to a target domain.

        Parameters
        ----------
        weighting : WeightFunction or None
            Weight function to bind, e.g.
            :class:`~conformalforecast.weighting.DensityRatio`. A copy is fitted
            against the retained source features and score shape and kept as
            the new object's ``weighting_``; the instance passed in is not
            modified. None keeps the current weighting.
        target_X : Exog or None, shape (n_series, n_samples, horizon, n_features)
            Target-domain features. Fits the weighting function.
        target_series : Series or None, shape (n_series, T)
            Target-domain observations. Sets the scale ratio.

        Returns
        -------
        SupportsTransfer
            A new object of the same class. ``calibration_.diagnostics`` carries
            ``"path": "adapt"``, ``"scale_ratio"`` and ``"weight_summary"``
            (min / median / max of calibration weights).

        Raises
        ------
        ValueError
            If all three arguments are None; if ``target_series`` is given
            without a ``scale_estimator`` configured; or if ``target_X`` is
            given but the bound weighting function does not declare
            ``CONSUMES_TARGET_X`` (``UniformWeights`` and
            ``PrecomputedWeights`` both ignore target features, so passing
            them would silently discard it).
        """
        if weighting is None and target_X is None and target_series is None:
            raise ValueError(
                "adapt() requires at least one of weighting, target_X or target_series."
            )
        # A copy, always: fit_source / fit_target mutate the weighting, and the
        # receiver (or the caller's instance) must not see that.
        active = (self.weighting_ if weighting is None else weighting).model_copy(deep=True)

        if target_X is not None and not type(active).CONSUMES_TARGET_X:
            raise ValueError(
                "target_X was supplied but "
                f"{type(active).__name__} does not consume target "
                "features, so it would be silently discarded. Pass "
                "adapt(weighting=DensityRatio(...)) to use target features for "
                "reweighting."
            )
        estimator = self._scale_estimator
        if target_series is not None and estimator is None:
            raise ValueError(
                "target_series was supplied but scale_estimator is None. "
                "Construct the method with scale_estimator='mad', 'std' or 'iqr'."
            )

        target_scales = self.target_scales_
        if target_series is not None:
            assert estimator is not None  # checked above
            target_scales = compute_series_scales(target_series, estimator)

        if weighting is not None:
            n_series, n_windows, horizon = self.scores_.shape
            active.fit_source(
                self.X_calibration_,
                (n_series, n_windows, horizon),
                valid=~np.isnan(self.scores_),
            )
        if target_X is not None:
            active.fit_target(target_X)

        # Shallow: the arrays shared with the receiver are only ever rebound,
        # never written into, by _recompute_quantile.
        adapted = self.model_copy(update={"weighting_": active, "target_scales_": target_scales})
        adapted._recompute_quantile()

        # Summarize over real windows only. A ragged panel's padded cells carry
        # weight 0, which the weighted quantile ignores, so including them
        # would report a minimum of 0 for a perfectly healthy weighting.
        weights = active.calibration_weights()[~np.isnan(self.scores_)]
        adapted.calibration_ = CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_.copy(),
            diagnostics={
                "path": "adapt",
                "scale_ratio": None
                if adapted.scale_ratio_ is None
                else np.asarray(adapted.scale_ratio_).copy(),
                "weight_summary": {
                    "min": float(np.min(weights)),
                    "median": float(np.median(weights)),
                    "max": float(np.max(weights)),
                },
            },
        )
        return adapted
