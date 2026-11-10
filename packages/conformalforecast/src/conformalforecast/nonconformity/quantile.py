"""Quantile nonconformity score for Conformalized Quantile Regression."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval, ScoreFunction


class QuantileScore(ScoreFunction):
    """
    Nonconformity score for Conformalized Quantile Regression (CQR).

    Unlike :class:`AbsoluteResidual`, the ``prediction`` input is a pair of
    quantile forecasts ``(q_lo, q_hi)`` rather than a point forecast. The
    score measures how far truth falls outside the predicted quantile
    interval:

    .. math::

        E_i = \\max\\{q_{\\text{lo}}(X_i) - Y_i,\\; Y_i - q_{\\text{hi}}(X_i)\\}

    Negative scores mean truth is inside the predicted interval; positive
    scores mean it falls outside (above the upper bound or below the lower
    bound). :meth:`invert` widens the predicted interval symmetrically by
    the score threshold.
    """

    CONSUMES_QUANTILES: ClassVar[bool] = True

    kind: Literal["quantile"] = "quantile"

    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute the CQR nonconformity scores.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon, 2)
            Quantile forecasts. The last axis is ``(q_lo, q_hi)``.
        truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)

        Raises
        ------
        ValueError
            If ``prediction`` is not 4-D with a trailing axis of size 2, or
            if ``truth`` does not match ``prediction.shape[:-1]``.
        """
        pred = np.asarray(prediction)
        tru = np.asarray(truth)
        if pred.ndim != 4 or pred.shape[-1] != 2:
            raise ValueError(
                "QuantileScore.score expects prediction of shape "
                "(n_series, n_samples, horizon, 2); got shape "
                f"{pred.shape}."
            )
        if tru.shape != pred.shape[:-1]:
            raise ValueError(
                "QuantileScore.score expects truth.shape == prediction.shape[:-1]; "
                f"got truth shape {tru.shape} and prediction shape {pred.shape}."
            )

        q_lo = pred[..., 0]
        q_hi = pred[..., 1]
        return np.maximum(q_lo - tru, tru - q_hi)

    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Widen the predicted quantile interval by ``score_threshold``.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon, 2)
            Quantile forecasts. The last axis is ``(q_lo, q_hi)``.
        score_threshold : NDArray
            Either scalar, shape ``(n_series,)``, or shape
            ``(n_series, horizon)``. Broadcast against ``prediction`` along
            its sample and horizon axes.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
            Last axis is ``(lower, upper)`` with
            ``lower = q_lo - threshold`` and ``upper = q_hi + threshold``.

        Raises
        ------
        ValueError
            If ``prediction`` is not 4-D with a trailing axis of size 2.
        """
        pred = np.asarray(prediction)
        if pred.ndim != 4 or pred.shape[-1] != 2:
            raise ValueError(
                "QuantileScore.invert expects prediction of shape "
                "(n_series, n_samples, horizon, 2); got shape "
                f"{pred.shape}."
            )

        q = np.asarray(score_threshold)
        # Reshape so q broadcasts against (n_series, n_samples, horizon).
        if q.ndim == 0:
            pass  # scalar broadcasts naturally
        elif q.ndim == 1:
            # (n_series,) -> (n_series, 1, 1)
            q = q[:, np.newaxis, np.newaxis]
        elif q.ndim == 2:
            # (n_series, horizon) -> (n_series, 1, horizon)
            q = q[:, np.newaxis, :]

        lower = pred[..., 0] - q
        upper = pred[..., 1] + q
        return np.stack([lower, upper], axis=-1)
