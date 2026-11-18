"""Absolute residual nonconformity score."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval, ScoreFunction


class AbsoluteResidual(ScoreFunction):
    """
    Nonconformity score defined as the absolute residual |truth - prediction|.

    This is the simplest and most widely used score function in conformal
    prediction. It produces symmetric intervals around the point forecast.
    """

    kind: Literal["absolute"] = "absolute"

    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute absolute residual scores.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)
        """
        return np.abs(truth - prediction)

    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Produce a symmetric interval around the point forecast.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        score_threshold : NDArray
            Either scalar, shape (n_series,), or shape (n_series, horizon).
            Broadcast against ``prediction`` along its last two axes.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
            Last axis is (lower, upper).
        """
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

        lower = prediction - q
        upper = prediction + q
        return np.stack([lower, upper], axis=-1)
