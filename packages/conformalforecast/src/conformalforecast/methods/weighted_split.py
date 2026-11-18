"""Weighted split conformal prediction under covariate shift."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from .._quantiles import weighted_quantile_from_weights, weighted_quantile_per_test_point
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    Interval,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.absolute import AbsoluteResidual
from ..weighting.base import UniformWeights
from .base import ConformalMethod
from .transfer import SupportsTransfer, scales_from_truths


def _weighted_quantiles(
    scores: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    alpha: float,
    pooling: str,
    scale_estimator: str | None,
    source_scales: NDArray[np.floating] | None,
    target_scales: NDArray[np.floating] | None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating] | None,
    NDArray[np.floating],
]:
    """
    The weighted conformal quantile, before and after the scale ratio.

    Shared by ``fit`` and ``adapt`` so both build the quantile one way, always
    from the stored source scores -- never from a previously adapted quantile,
    which is what makes ``adapt`` idempotent.

    Parameters
    ----------
    scores, weights : NDArray, shape (n_series, n_windows, horizon)
    alpha : float
    pooling : {"per_series", "global"}
    scale_estimator : str or None
    source_scales, target_scales : NDArray or None, shape (n_series,)

    Returns
    -------
    base_quantile : NDArray, shape (n_series | 1, horizon)
    scale_ratio : NDArray or None, shape (n_series,)
    score_quantile : NDArray, shape (n_series, horizon)

    Raises
    ------
    ValueError
        If ``pooling="per_series"`` and the target scales do not match the
        source series count.
    """
    if (
        pooling == "per_series"
        and target_scales is not None
        and source_scales is not None
        and target_scales.shape[0] != source_scales.shape[0]
    ):
        raise ValueError(
            f"target_series n_series {target_scales.shape[0]} does not "
            f"match the fitted n_series {source_scales.shape[0]}. Use "
            "pooling='global' to serve a target panel with a different "
            "series count."
        )
    if pooling == "global":
        assert source_scales is not None  # pooling="global" requires a scale_estimator
        # source_scales is float64; cast back so float32 scores are not upcast.
        norm = (scores / source_scales[:, np.newaxis, np.newaxis]).astype(scores.dtype, copy=False)
        n_series, n_windows, horizon = norm.shape
        pooled = norm.reshape(1, n_series * n_windows, horizon)
        pooled_w = weights.reshape(1, n_series * n_windows, horizon)
        base_quantile = weighted_quantile_from_weights(pooled, alpha, pooled_w)
    else:
        base_quantile = weighted_quantile_from_weights(scores, alpha, weights)

    scale_ratio: NDArray[np.floating] | None = None
    if scale_estimator is not None and target_scales is not None:
        assert source_scales is not None
        # Under global pooling the base quantile is in normalized units, so the
        # target scale alone is the multiplier.
        scale_ratio = target_scales if pooling == "global" else target_scales / source_scales
    elif pooling == "global":
        assert source_scales is not None
        # Un-adapted global pooling returns to source units.
        scale_ratio = source_scales

    if scale_ratio is None:
        return base_quantile, None, base_quantile
    score_quantile = (base_quantile * scale_ratio[:, np.newaxis]).astype(
        base_quantile.dtype, copy=False
    )
    return base_quantile, scale_ratio, score_quantile


class WeightedSplit(ConformalMethod["WeightedSplitForecaster"]):
    """
    Split conformal prediction with per-calibration-point weights.

    Implements weighted conformal prediction under covariate shift (Tibshirani
    et al., 2019): calibration scores are weighted by an estimate of the
    density ratio ``d_target(x) / d_source(x)``, and the normalization includes
    the test point's own weight.

    Until ``adapt`` binds a weight function this is exactly
    :class:`~conformalforecast.methods.split.SplitConformal`: ``fit`` binds
    ``UniformWeights``, which recovers the unweighted quantile.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict lifecycle. ``fit`` returns a
    :class:`WeightedSplitForecaster`, which offers ``adapt`` (see
    :class:`~conformalforecast.methods.transfer.SupportsTransfer`).
    Construction is keyword-only: ``WeightedSplit(alpha=0.1)``.

    Parameters
    ----------
    alpha : float
        Miscoverage level in (0, 1). Target coverage is ``1 - alpha``.
    seed : int, default 0
        Inert (kept for a uniform constructor signature).
    pooling : {"per_series", "global"}, default "per_series"
        ``"global"`` pools scale-normalized calibration scores across the
        series axis into one quantile per horizon step, which is what lets a
        target series absent from the source panel be served — but only once
        ``adapt(target_series=...)`` has widened that pooled quantile to one
        row per target series. Until then ``interval`` still requires the
        source panel's series count (or a single-series source, whose
        length-1 quantile broadcasts). It **requires** ``scale_estimator``,
        because pooling residuals across series of different scales is
        unsound without normalizing first.
    scale_estimator : {"mad", "std", "iqr"} or None
        Enables scale alignment through ``adapt(target_series=...)``. Required
        when ``pooling="global"``.

    Notes
    -----
    A custom ``score`` is deliberately narrow: the weighted per-test-point path
    builds the interval as ``prediction +/- q``, correct only for a symmetric
    residual score, so ``score=`` must be an :class:`AbsoluteResidual` (or a
    subclass) -- anything else is a ``pydantic.ValidationError`` at
    construction.

    References
    ----------
    Tibshirani, R. J., Barber, R. F., Candes, E. J., & Ramdas, A. (2019).
    Conformal prediction under covariate shift. NeurIPS 32.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    SCORE_TYPE: ClassVar[type[ScoreFunction]] = AbsoluteResidual
    IS_ONLINE: ClassVar[bool] = False

    pooling: Literal["per_series", "global"] = Field(
        default="per_series",
        description=(
            "How calibration scores are grouped before the quantile is taken. "
            "'per_series' gives each series its own quantile, the right choice "
            "whenever the target panel is the source panel. 'global' pools "
            "scale-normalized scores across series into one quantile per horizon "
            "step, which is what lets a target series absent from the source panel "
            "be served -- once adapt(target_series=...) has widened that pooled "
            "quantile to one row per target series; before that, interval still "
            "requires the source series count. It requires scale_estimator, since "
            "pooling residuals across series of different scales is unsound."
        ),
    )
    scale_estimator: Literal["mad", "std", "iqr"] | None = Field(
        default=None,
        description=(
            "Per-series scale statistic used to align a source-fitted quantile to "
            "a target domain, computed on first differences so it is robust to "
            "level and slow trend. 'mad' is the robust default choice, 'std' the "
            "classic one, 'iqr' the most outlier-tolerant. None disables scale "
            "alignment, which makes adapt(target_series=...) an error."
        ),
    )

    @model_validator(mode="after")
    def _check_pooling(self) -> WeightedSplit:
        """``pooling="global"`` is unsound without normalizing scales first."""
        if self.pooling == "global" and self.scale_estimator is None:
            raise ValueError(
                "pooling='global' requires a scale_estimator. Pooling residuals "
                "across series of different scales is unsound; normalize first."
            )
        return self

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> WeightedSplitForecaster:
        """
        Fit the weighted conformal quantile on source-domain arrays.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            NaN cells mark windows that do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Source-domain features, retained as ``X_calibration_`` for the
            weight function ``adapt`` later binds.

        Returns
        -------
        WeightedSplitForecaster
            Bound to ``UniformWeights``; ``calibration_.diagnostics`` carries
            ``"score_quantile"`` and ``"pooling"``.

        Raises
        ------
        ValueError
            If array shapes are invalid or mismatched.
        InsufficientCalibrationSamples
            If any series' effective sample size is below ``ceil(1 / alpha)``.
        """
        preds = as_forecast(predictions, "predictions")
        obs = as_forecast(truths, "truths")
        if preds.shape != obs.shape:
            raise ValueError(
                f"predictions and truths must have the same shape, "
                f"got {preds.shape} and {obs.shape}"
            )

        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        # NaN cells propagate through scoring — never imputed.
        scores = score_fn.score(preds, obs)

        source_scales = (
            scales_from_truths(obs, self.scale_estimator)
            if self.scale_estimator is not None
            else None
        )
        valid = ~np.isnan(scores)
        n_calibration_samples = valid.sum(axis=1).min(axis=1)
        # scores.shape is tuple[int, ...] to mypy; unpack so fit_source's
        # tuple[int, int, int] parameter is satisfied without a cast.
        n_series, n_windows, horizon = scores.shape
        weighting = UniformWeights()
        weighting.fit_source(X, (n_series, n_windows, horizon), valid=valid)

        base_quantile, scale_ratio, score_quantile = _weighted_quantiles(
            scores,
            weighting.calibration_weights(),
            alpha=self.alpha,
            pooling=self.pooling,
            scale_estimator=self.scale_estimator,
            source_scales=source_scales,
            target_scales=None,
        )
        return WeightedSplitForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_calibration_samples,
            calibration_=CalibrationResult(
                n_calibration_samples=n_calibration_samples.copy(),
                diagnostics={
                    "score_quantile": score_quantile.copy(),
                    "pooling": self.pooling,
                },
            ),
            weighting_=weighting,
            scores_=scores,
            X_calibration_=X,
            source_scales_=source_scales,
            scale_ratio_=scale_ratio,
            base_quantile_=base_quantile,
            score_quantile_=score_quantile,
        )


class WeightedSplitForecaster(SupportsTransfer):
    """
    A fitted :class:`WeightedSplit`.

    Attributes
    ----------
    method_ : WeightedSplit
    score_quantile_ : NDArray, shape (n_series, horizon)
        The fitted per-cell quantile, in target units. Under
        ``pooling="global"`` these rows come from the scale ratio
        (``target_scales_`` once adapted, else the ``source_scales_``
        fallback), so ``n_series`` here follows the target panel after
        ``adapt(target_series=...)``, not the source panel ``fit`` ran on.
    base_quantile_ : NDArray, shape (n_series, horizon)
        The quantile before the scale ratio is applied. Under
        ``pooling="global"`` the leading axis is 1 (pooled across series),
        broadcast against the scale ratio to produce ``score_quantile_``.
    """

    method_: WeightedSplit
    base_quantile_: NDArray[np.floating]
    score_quantile_: NDArray[np.floating]

    @property
    def _scale_estimator(self) -> str | None:
        return self.method_.scale_estimator

    def _recompute_quantile(self) -> None:
        """
        Rebuild the quantile from stored source scores, weights and scales.

        Always computed from ``scores_``, never from a previously adapted
        quantile, which is what makes ``adapt`` idempotent.
        """
        self.base_quantile_, self.scale_ratio_, self.score_quantile_ = _weighted_quantiles(
            self.scores_,
            self.weighting_.calibration_weights(),
            alpha=self.method_.alpha,
            pooling=self.method_.pooling,
            scale_estimator=self.method_.scale_estimator,
            source_scales=self.source_scales_,
            target_scales=self.target_scales_,
        )

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Apply the weighted quantile to new predictions.

        Under ``pooling="per_series"``, when the weighting function is adapted
        and ``X`` is supplied, the normalization includes each test point's own
        weight, per Tibshirani et al. (2019). Otherwise — including every
        ``pooling="global"`` call, where the quantile is pooled across the
        series axis and so has no per-series score panel to reweight — the
        fitted per-cell quantile is applied directly.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)

        Raises
        ------
        ValueError
            If the predictions shape does not match the fitted state.
        """
        pooling = self.method_.pooling
        preds = as_forecast(predictions, "predictions")
        n_series, _, horizon = preds.shape
        if horizon != self.score_quantile_.shape[1]:
            raise ValueError(
                f"predictions horizon {horizon} does not match fitted horizon "
                f"{self.score_quantile_.shape[1]}"
            )
        if pooling == "per_series" and n_series != self.score_quantile_.shape[0]:
            raise ValueError(
                f"predictions n_series {n_series} does not match fitted n_series "
                f"{self.score_quantile_.shape[0]}. Use pooling='global' to serve a "
                "target panel with a different series count."
            )

        if self.weighting_.is_adapted_ and X is not None and pooling == "per_series":
            tw = self.weighting_.test_weights(X)
            if tw.shape[1] != preds.shape[1]:
                raise ValueError(
                    f"X yielded {tw.shape[1]} test weights per series but predictions "
                    f"carry {preds.shape[1]} samples. X must be aligned with the "
                    "predictions it reweights, shape "
                    "(n_series, n_samples, horizon, n_features)."
                )
            q = weighted_quantile_per_test_point(
                self.scores_, self.method_.alpha, self.weighting_.calibration_weights(), tw
            )
            if self.scale_ratio_ is not None:
                # scale_ratio_ is float64; cast back so float32 scores are not upcast.
                q = (q * self.scale_ratio_[:, np.newaxis, np.newaxis]).astype(q.dtype, copy=False)
            lower = preds - q
            upper = preds + q
            return np.stack([lower, upper], axis=-1)

        q_cells = self.score_quantile_
        if pooling == "global" and q_cells.shape[0] == 1:
            # Not dead code: a single-series source fitted under global pooling and
            # adapted with only `weighting` (no target_series) leaves scale_ratio_ at
            # source_scales_, length 1 -- this is what still lets it serve a wider
            # target panel.
            q_cells = np.broadcast_to(q_cells, (n_series, horizon))
        if n_series != q_cells.shape[0]:
            raise ValueError(
                f"predictions n_series {n_series} does not match the fitted quantile's "
                f"n_series {q_cells.shape[0]}. Under pooling='global' the quantile is "
                "widened to a target panel by adapt(target_series=...)."
            )
        result: Interval = self.score_fn_.invert(preds, q_cells)
        return result
