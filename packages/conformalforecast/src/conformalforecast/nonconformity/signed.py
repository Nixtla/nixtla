"""Signed residual nonconformity score for asymmetric intervals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval, ScoreFunction


class SignedResidual(ScoreFunction):
    """
    Nonconformity score defined as the signed residual ``truth - prediction``.

    Unlike :class:`AbsoluteResidual`, this score preserves directional
    information. Positive scores indicate truth was above prediction;
    negative scores indicate truth was below. Used by methods that produce
    asymmetric intervals around the point forecast (e.g. SPCI).
    """

    kind: Literal["signed"] = "signed"

    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute signed residual scores.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)
            Signed residuals (truth - prediction). Positive when truth exceeds
            prediction, negative when truth is below.
        """
        return np.asarray(truth) - np.asarray(prediction)

    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Produce an asymmetric interval around the point forecast.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        score_threshold : NDArray, shape (n_series, horizon, 2) or \
                (n_series, n_samples, horizon, 2)
            For each cell, the last axis is ``(lower_offset, upper_offset)``.
            The interval becomes ``[prediction + lower_offset, prediction +
            upper_offset]``. Note that ``lower_offset`` is typically negative
            and ``upper_offset`` positive for non-degenerate intervals.

            The 3-D form is one offset pair per ``(series, horizon)`` cell,
            broadcast over every sample -- what :meth:`interval` produces. The
            4-D form carries a distinct pair per sample, which is what a method
            whose offsets vary window-to-window needs; see
            :meth:`~conformalforecast.methods.base.ConformalMethod.calibration_intervals`.
            NaN offsets propagate to NaN bounds, marking a window as not
            evaluable.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
            Last axis is ``(lower, upper)``.

        Raises
        ------
        ValueError
            If ``score_threshold`` is not 3-D or 4-D with trailing axis of size 2.
        """
        q = np.asarray(score_threshold)
        if q.ndim not in (3, 4) or q.shape[-1] != 2:
            raise ValueError(
                "SignedResidual.invert expects score_threshold of shape "
                "(n_series, horizon, 2) or (n_series, n_samples, horizon, 2); "
                f"got shape {q.shape}."
            )
        # Reshape so q broadcasts against (n_series, n_samples, horizon).
        # 3-D: (n_series, horizon, 2) -> (n_series, 1, horizon, 2).
        # 4-D already carries the sample axis.
        q_b = q[:, np.newaxis, :, :] if q.ndim == 3 else q
        lower = prediction + q_b[..., 0]
        upper = prediction + q_b[..., 1]
        return np.stack([lower, upper], axis=-1)
