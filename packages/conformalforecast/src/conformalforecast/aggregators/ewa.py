"""Exponentially Weighted Average aggregator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from .base import OnlineAggregator


class EWA(OnlineAggregator):
    """
    Exponentially Weighted Average aggregator (Vovk 1990).

    Weights are proportional to ``exp(-eta * cumulative_loss)``. Larger ``eta``
    converges faster to the best expert but is noisier.

    Call :meth:`initialize` with the panel shape before :meth:`weights` or
    :meth:`update`.
    """

    eta: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Sharpness of the softmax over experts: weights are proportional to "
            "exp(-eta * cumulative_loss), so this controls how strongly weight "
            "concentrates on whichever expert has accumulated the least loss so "
            "far. No interior value is special, but the limits are informative: "
            "eta -> 0 recovers an unweighted (uniform) average across experts, "
            "and eta -> infinity recovers winner-take-all on the single best "
            "expert. A larger eta reacts faster once one expert starts winning, "
            "at the cost of noisier weights that can swing between experts as "
            "their losses fluctuate; a smaller eta gives smoother, more stable "
            "weights but is slower to move away from an expert that has stopped "
            "performing well. The knob exists to trade adaptation speed against "
            "weight stability when combining several online experts."
        ),
    )

    def weights(self) -> NDArray[np.floating]:
        """
        Return current softmax weights over experts.

        Returns
        -------
        NDArray, shape (n_experts, n_series, horizon)
            Numerically stable softmax along axis 0. Sums to 1 along axis 0.

        Raises
        ------
        CalibrationError
            If called before :meth:`initialize`.
        """
        cumulative = self._require_initialized()
        neg_loss = -self.eta * cumulative
        neg_loss_max = neg_loss.max(axis=0, keepdims=True)
        exp_w = np.exp(neg_loss - neg_loss_max)
        denom = exp_w.sum(axis=0, keepdims=True)
        weights: NDArray[np.floating] = exp_w / denom
        return weights
