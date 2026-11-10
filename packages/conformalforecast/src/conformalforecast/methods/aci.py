"""Adaptive Conformal Inference (ACI) for time series (Gibbs & Candès 2021)."""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from .._quantiles import ExpandingSortedScores, _ceil, masked_quantile
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    InsufficientCalibrationSamples,
    Interval,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.absolute import AbsoluteResidual
from .base import ConformalForecaster, ConformalMethod


class AdaptiveConformalInference(ConformalMethod["ACIForecaster"]):
    """
    Online Adaptive Conformal Inference on panel time series.

    Maintains a time-varying miscoverage rate ``alpha_t`` per
    ``(series, horizon)`` cell. Each realized step nudges it toward the target
    ``alpha`` by ``alpha_t <- alpha_t + gamma * (alpha - 1{miss})``, which
    guarantees long-run coverage of ``1 - alpha`` without exchangeability.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict/update lifecycle and the ``calibrate`` / ``calibrate_cv``
    split. ``fit`` returns an :class:`ACIForecaster`, whose ``update`` advances
    ``alpha_t_`` in place.

    Any adapter works — no special capabilities required. Bind one with
    ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``. Construction is keyword-only:
    ``AdaptiveConformalInference(alpha=0.1)``.

    Parameters
    ----------
    alpha : float
        Miscoverage in (0, 1). Target coverage is ``1 - alpha``.
    gamma : float, default 0.05
        Learning rate for the ``alpha_t`` recursion. Gibbs & Candès recommend
        ``[0.005, 0.05]``; values > 0.1 warn.
    seed : int, default 0
        Unused (kept for a uniform constructor signature); ACI is
        deterministic by construction.
    score : ScoreFunction or None, default None
        Defaults to :class:`AbsoluteResidual`.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    IS_ONLINE: ClassVar[bool] = True

    gamma: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "Learning rate for the alpha_t <- alpha_t + gamma * (alpha - 1{miss}) "
            "update: each realized outcome nudges the working miscoverage rate "
            "toward alpha, which is what lets coverage adapt online without "
            "assuming exchangeability. Gibbs & Candes recommend [0.005, 0.05]: "
            "a larger value reacts to a real coverage miss faster, at the cost "
            "of a noisier alpha_t (and so a noisier interval width) step to "
            "step; a smaller value is smoother but adapts more slowly, so "
            "intervals stay miscalibrated for longer after a genuine shift."
        ),
    )

    @field_validator("gamma")
    @classmethod
    def _warn_on_large_gamma(cls, value: float) -> float:
        if value > 0.1:
            warnings.warn(
                f"gamma={value} is unusually large; Gibbs & Candes recommend "
                "[0.005, 0.05]. Updates will be very noisy.",
                stacklevel=2,
            )
        return value

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> ACIForecaster:
        """
        Roll ``alpha_t`` forward over the calibration windows.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            Chronological on axis 1 (oldest first). NaN cells mark windows
            that do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None
            Ignored by this method.

        Returns
        -------
        ACIForecaster
            ``calibration_.diagnostics`` carries ``"gamma"`` and a copy of
            ``alpha_t_``.

        Raises
        ------
        ValueError
            If array shapes are invalid or mismatched.
        InsufficientCalibrationSamples
            If any series has fewer than ``ceil(1 / alpha)`` effective
            (non-NaN) calibration windows.
        """
        preds = as_forecast(predictions, "predictions")
        obs = as_forecast(truths, "truths")
        if preds.shape != obs.shape:
            raise ValueError(
                f"predictions and truths must have the same shape, got "
                f"{preds.shape} and {obs.shape}"
            )
        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        scores = score_fn.score(preds, obs)  # NaN propagates

        n_eff = np.sum(~np.isnan(scores), axis=1)  # (n_series, horizon)
        min_samples = int(_ceil(1.0 / self.alpha))
        bad = np.flatnonzero(n_eff.min(axis=1) < min_samples)
        if bad.size:
            idx = [int(i) for i in bad]
            raise InsufficientCalibrationSamples(
                f"Series {idx} have fewer than ceil(1/alpha) = {min_samples} "
                f"effective calibration samples for alpha={self.alpha}.",
                series_indices=idx,
            )

        n_series, _, horizon = scores.shape
        alpha_t = self._run_loop(
            scores, np.full((n_series, horizon), self.alpha, dtype=scores.dtype)
        )
        n_calibration_samples = n_eff.min(axis=1)
        return ACIForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_calibration_samples,
            calibration_=CalibrationResult(
                n_calibration_samples=n_calibration_samples.copy(),
                diagnostics={"gamma": self.gamma, "alpha_t": alpha_t.copy()},
            ),
            alpha_t_=alpha_t,
            scores_=scores,
            n_observations_=scores.shape[1],
        )

    def _run_loop(
        self,
        scores: NDArray[np.floating],
        alpha_t: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Roll ``alpha_t`` forward over the window axis (NaN-masking contract).

        The growing history lives in a single :class:`ExpandingSortedScores`
        rather than being re-quantiled from scratch each step, and the per-cell
        quantile is one batched gather. Output is bit-identical to the naive
        loop this replaced.
        """
        n_series, n_windows, horizon = scores.shape
        has_past = np.zeros((n_series, horizon), dtype=bool)
        prefix = ExpandingSortedScores(n_series, horizon, scores.dtype, capacity=n_windows)
        for t in range(n_windows):
            current = scores[:, t, :]  # (n_series, horizon)
            valid = ~np.isnan(current)
            missed = np.zeros((n_series, horizon), dtype=scores.dtype)
            if t > 0:
                levels = np.clip(1.0 - alpha_t, 0.0, 1.0)
                threshold = prefix.quantile(levels)
                seen = valid & has_past
                missed = np.where(seen, (current > threshold).astype(scores.dtype), missed)
            # A cell's first valid score has no past -> treat as a miss
            # (deliberate initial widening, matching methods/aci.py).
            first_valid = valid & ~has_past
            missed = np.where(first_valid, scores.dtype.type(1.0), missed)
            step = self.gamma * (self.alpha - missed)
            alpha_t = np.where(valid, alpha_t + step, alpha_t)
            has_past = has_past | valid
            # Push AFTER the threshold: the prefix must stay scores[:, :t, :].
            prefix.push(current)
        return alpha_t


class ACIForecaster(ConformalForecaster):
    """
    A fitted :class:`AdaptiveConformalInference`. ``update`` mutates it in place.

    Attributes
    ----------
    method_ : AdaptiveConformalInference
    alpha_t_ : NDArray, shape (n_series, horizon)
        The current time-varying miscoverage rate.
    n_observations_ : int
        Number of windows seen so far (grows by one per :meth:`update`).
    scores_ : NDArray, shape (n_series, n_windows, horizon)
        Nonconformity scores retained for the online ``update`` step.
    """

    method_: AdaptiveConformalInference
    alpha_t_: NDArray[np.floating]
    n_observations_: int
    scores_: NDArray[np.floating]

    def _threshold(self) -> NDArray[np.floating]:
        """Score threshold per (series, horizon) at the current ``alpha_t_``."""
        return masked_quantile(self.scores_, np.clip(1.0 - self.alpha_t_, 0.0, 1.0))

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Apply the current ``alpha_t_`` threshold to new predictions.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
        X : Exog or None
            Ignored by this method.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)

        Raises
        ------
        ValueError
            If predictions shape (n_series, horizon) does not match fitted state.
        """
        preds = as_forecast(predictions, "predictions")
        pred_shape = (preds.shape[0], preds.shape[2])
        if pred_shape != self.alpha_t_.shape:
            raise ValueError(
                f"predictions shape (n_series, horizon) = {pred_shape} does not "
                f"match fitted alpha_t_ shape {self.alpha_t_.shape}"
            )
        result: Interval = self.score_fn_.invert(preds, self._threshold())
        return result

    def _update(
        self,
        prediction: Forecast,
        truth: Forecast,
        *,
        X: Exog | None = None,
    ) -> None:
        """
        Roll ``alpha_t_`` forward with a new ``(prediction, truth)`` pair.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)
        X : Exog or None
            Ignored by this method.

        Raises
        ------
        ValueError
            If ``prediction``/``truth`` do not have shape ``(n_series, 1, horizon)``.
        """
        n_series, _, horizon = self.scores_.shape
        p = as_forecast(prediction, "prediction")
        t = as_forecast(truth, "truth")
        expected = (n_series, 1, horizon)
        if p.shape != expected or t.shape != expected:
            raise ValueError(
                f"prediction and truth must have shape {expected}, got {p.shape} and {t.shape}"
            )
        new_score = self.score_fn_.score(p, t)  # (n_series, 1, horizon)
        threshold = self._threshold()  # against pre-append history
        missed = (new_score[:, 0, :] > threshold).astype(self.scores_.dtype)
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
        method = self.method_
        self.alpha_t_ = self.alpha_t_ + method.gamma * (method.alpha - missed)
