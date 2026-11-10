"""Nonexchangeable Conformal Prediction (NexCP) for time series (Barber et al. 2023)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from .._quantiles import weighted_conformal_quantile
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    Interval,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.absolute import AbsoluteResidual
from .base import ConformalForecaster, ConformalMethod


class NonexchangeableConformalPrediction(ConformalMethod["NexCPForecaster"]):
    """
    Recency-weighted (nonexchangeable) conformal prediction on panel time series.

    Each calibration window gets an exponential weight ``rho ** (age)``; the
    newest window has weight 1 and older windows decay by ``rho``. The
    conformal threshold is the exact weighted order statistic of the scores
    (see :func:`weighted_conformal_quantile`). ``rho == 1`` recovers split CP.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict/update lifecycle and the ``calibrate`` / ``calibrate_cv``
    split. ``fit`` returns a :class:`NexCPForecaster`, whose ``update`` appends
    to the score history in place.

    Any adapter works — no special capabilities required. Bind one with
    ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``.

    Parameters
    ----------
    alpha : float
        Miscoverage level in (0, 1). The target coverage is ``1 - alpha``.
    rho : float, default 0.99
        Decay factor in (0, 1]. Smaller values weight recent windows more.
        ``rho == 1`` recovers the unweighted (split conformal) quantile.
    seed : int, default 0
        Unused (kept for a uniform constructor signature); NexCP is
        deterministic by construction.
    score : ScoreFunction or None, default None
        Defaults to :class:`AbsoluteResidual`.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    IS_ONLINE: ClassVar[bool] = True

    rho: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description=(
            "Geometric decay applied to older calibration windows: the newest "
            "window gets weight 1 and a window aged k steps gets weight "
            "rho ** k. 1.0 means no decay and recovers unweighted split "
            "conformal exactly; a smaller value weights recent windows more "
            "heavily, which is what makes the method valid under distribution "
            "shift (non-exchangeable data) at the cost of a noisier quantile "
            "estimate from fewer effective samples."
        ),
    )

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> NexCPForecaster:
        """
        Fit the weighted conformal quantile on calibration arrays.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            Chronological on axis 1 (oldest first). NaN cells mark windows that
            do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None
            Ignored by this method.

        Returns
        -------
        NexCPForecaster
            ``calibration_.diagnostics`` carries ``"rho"`` and a copy of
            ``score_quantile_``.

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
                f"predictions and truths must have the same shape, got "
                f"{preds.shape} and {obs.shape}"
            )
        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        scores = score_fn.score(preds, obs)  # NaN propagates

        score_quantile = weighted_conformal_quantile(scores, self.alpha, self.rho)
        n_calibration_samples = np.sum(~np.isnan(scores), axis=1).min(axis=1)
        return NexCPForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_calibration_samples,
            calibration_=CalibrationResult(
                n_calibration_samples=n_calibration_samples.copy(),
                diagnostics={"rho": self.rho, "score_quantile": score_quantile.copy()},
            ),
            score_quantile_=score_quantile,
            scores_=scores,
            n_observations_=scores.shape[1],
        )


class NexCPForecaster(ConformalForecaster):
    """
    A fitted :class:`NonexchangeableConformalPrediction`. ``update`` mutates it in place.

    Attributes
    ----------
    method_ : NonexchangeableConformalPrediction
    score_quantile_ : NDArray, shape (n_series, horizon)
        The current per-(series, horizon) weighted conformal quantile.
    n_observations_ : int
        Number of windows seen so far (grows by one per :meth:`update`).
    scores_ : NDArray, shape (n_series, n_windows, horizon)
        Nonconformity scores retained for the online ``update`` step.
    """

    method_: NonexchangeableConformalPrediction
    score_quantile_: NDArray[np.floating]
    n_observations_: int
    scores_: NDArray[np.floating]

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Apply the fitted weighted quantile to new predictions.

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
        if pred_shape != self.score_quantile_.shape:
            raise ValueError(
                f"predictions shape (n_series, horizon) = {pred_shape} does not "
                f"match fitted score_quantile_ shape {self.score_quantile_.shape}"
            )
        result: Interval = self.score_fn_.invert(preds, self.score_quantile_)
        return result

    def _update(
        self,
        prediction: Forecast,
        truth: Forecast,
        *,
        X: Exog | None = None,
    ) -> None:
        """
        Append a new score and recompute the weighted quantile.

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
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
        self.score_quantile_ = weighted_conformal_quantile(
            self.scores_, self.method_.alpha, self.method_.rho
        )
