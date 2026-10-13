"""The fitted form shared by split conformal and CQR: a fixed per-cell offset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import Exog, Forecast, Interval, as_forecast
from .base import ConformalForecaster

__all__ = ["FixedOffsetForecaster"]


def _as_quantile_predictions(arr: object) -> NDArray[np.floating]:
    """Coerce to a 4-D float array with trailing axis 2, preserving float32."""
    out = np.asarray(arr)
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    if out.ndim != 4 or out.shape[-1] != 2:
        raise ValueError(
            "quantile predictions must be 4-D (n_series, n_windows | n_samples, "
            f"horizon, 2), got shape {out.shape}"
        )
    return out


class FixedOffsetForecaster(ConformalForecaster):
    """
    A fitted split-conformal or CQR method.

    Once fitted, both methods are the same thing: a per-(series, horizon)
    conformal quantile that never changes, inverted through the score. Only
    the input differs — point forecasts for
    :class:`~conformalforecast.methods.split.SplitConformal`, a quantile pair
    for :class:`~conformalforecast.methods.cqr.ConformalizedQuantileRegression`
    (whose method sets ``CONSUMES_QUANTILES``).

    Attributes
    ----------
    score_quantile_ : NDArray
        The fitted conformal quantile. Shape ``(n_series, horizon)``, or
        ``(n_series, horizon, 2)`` — last axis ``(lower_offset,
        upper_offset)`` — for asymmetric split conformal.
    """

    score_quantile_: NDArray[np.floating]

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Apply the fitted quantile to new predictions.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
            Point forecasts — or, for CQR, quantile forecasts of shape
            ``(n_series, n_samples, horizon, 2)``, last axis ``(q_lo, q_hi)``.
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)
            Ignored.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)

        Raises
        ------
        ValueError
            If the input is malformed, or its ``(n_series, horizon)`` does not
            match the fitted ``score_quantile_``.
        """
        if type(self.method_).CONSUMES_QUANTILES:
            inputs = _as_quantile_predictions(predictions)
        else:
            inputs = as_forecast(predictions, "predictions")
        got = (inputs.shape[0], inputs.shape[2])
        expected = self.score_quantile_.shape[:2]
        if got != expected:
            raise ValueError(
                f"predictions shape (n_series, horizon) = {got} does not match the "
                f"fitted score_quantile_ shape {expected}"
            )
        result: Interval = self.score_fn_.invert(inputs, self.score_quantile_)
        return result
