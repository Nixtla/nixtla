"""Abstract base class for quantile regressors used by SPCI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .._model import ArrayModel


class QuantileRegressor(ArrayModel, ABC):
    """
    Quantile regression model used by SPCI.

    Trained on autoregressive residual data: given the last ``w`` residuals,
    predict the quantile distribution of the next residual. Subclasses can
    wrap any quantile-regression library (QRF, LightGBM quantile, neural
    quantile networks, etc.).
    """

    # No fields are declared here, so there is nothing yet to forbid or allow.
    # Left at pydantic's default ("ignore") rather than "forbid": "forbid" would
    # only matter to a caller filling in a schema, and no schema points here --
    # SequentialPredictiveConformalInference's ``regressor`` is typed as the
    # concrete QRFQuantileRegressor (an ABC-typed field would render an empty
    # nested schema), so this ABC's remaining job is the ``predict_quantiles``
    # default plus the ``fit`` / ``predict_quantile`` contract for a future
    # backend, which must subclass QRFQuantileRegressor to be usable there.

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> None:
        """
        Fit the regressor.

        Parameters
        ----------
        X : NDArray, shape (n_samples, w)
            Training features. Each row is a window of ``w`` past residuals.
        y : NDArray, shape (n_samples,)
            Training targets (the next residual after each window).
        """

    @abstractmethod
    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: float,
    ) -> NDArray[np.floating]:
        """
        Predict the ``q``-quantile of the conditional distribution.

        Parameters
        ----------
        X : NDArray, shape (n_samples, w)
        q : float in (0, 1)

        Returns
        -------
        NDArray, shape (n_samples,)
        """

    def predict_quantiles(
        self,
        X: NDArray[np.floating],
        qs: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Predict several quantiles at once.

        SPCI asks for ``2 * beta_grid_size`` quantiles per ``(series, horizon)``
        cell while searching for the width-minimizing ``beta``. Most quantile
        libraries answer a whole list in one traversal of the fitted model, which
        is dramatically cheaper than one call per level -- for
        ``quantile-forest``, 42 separate calls measured 52.2 ms against 1.23 ms
        batched. That loop dominated SPCI's ``interval()`` outright: the
        measured ~53 ms per cell was essentially all quantile calls.

        The default implementation loops over :meth:`predict_quantile`, so a
        custom regressor only has to implement the scalar method. Override this
        when the backend can do better.

        Parameters
        ----------
        X : NDArray, shape (n_samples, w)
        qs : NDArray, shape (n_quantiles,)
            Levels in ``(0, 1)``.

        Returns
        -------
        NDArray, shape (n_samples, n_quantiles)
            Column ``j`` is the ``qs[j]``-quantile.
        """
        levels = np.asarray(qs, dtype=np.float64).reshape(-1)
        columns = [np.asarray(self.predict_quantile(X, float(q))).reshape(-1) for q in levels]
        return np.stack(columns, axis=1)
