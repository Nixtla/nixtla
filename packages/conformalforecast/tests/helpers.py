"""Shared test helpers.

Quantile-capable callable adapters used by the CQR orchestration-layer tests,
plus the mock quantile regressor the SPCI tests fit instead of a real forest. The
production :class:`CallableAdapter` is point-only by design; these variants add
``SupportsQuantiles`` (and, for the CV path, ``SupportsCrossValidationQuantiles``)
without touching production code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr
from pydantic.json_schema import SkipJsonSchema

from conformalforecast.base import Forecast, ForecasterAdapter, Series
from conformalforecast.capabilities import (
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
)
from conformalforecast.quantile_regressors.base import QuantileRegressor
from conformalforecast.quantile_regressors.qrf import QRFQuantileRegressor


class QuantileCallableAdapter(ForecasterAdapter, SupportsQuantiles):
    """Quantile-capable callable adapter for testing CQR."""

    predict_fn: SkipJsonSchema[Callable[[NDArray[np.floating]], NDArray[np.floating]]]
    predict_quantiles_fn: SkipJsonSchema[
        Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
    ]

    def predict(self, history: Series) -> Forecast:
        history = self._validate_history(history)
        raw = np.asarray(self.predict_fn(history), dtype=np.float64)
        if raw.shape != (self.n_series, self.horizon):
            raise ValueError(
                f"predict_fn must return shape ({self.n_series}, {self.horizon}), got {raw.shape}"
            )
        return raw[:, np.newaxis, :]

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        history = self._validate_history(history)
        q_arr = np.asarray(quantiles, dtype=np.float64)
        raw = np.asarray(self.predict_quantiles_fn(history, q_arr), dtype=np.float64)
        n_q = q_arr.shape[0]
        if raw.shape != (self.n_series, n_q, self.horizon):
            raise ValueError(
                f"predict_quantiles_fn must return shape "
                f"({self.n_series}, {n_q}, {self.horizon}), got {raw.shape}"
            )
        return raw


class QuantileCVCallableAdapter(QuantileCallableAdapter, SupportsCrossValidationQuantiles):
    """Variant that also implements ``SupportsCrossValidationQuantiles``.

    The CV implementation replays a pre-baked list of calibration histories
    through ``predict_quantiles_fn`` and zips the results with a pre-baked
    truths panel, so the CV path can be tested against the loop path with
    identical inputs.
    """

    cv_histories: list[NDArray[np.floating]]
    cv_truths: NDArray[np.floating]

    def cross_validate_quantiles(
        self,
        n_windows: int,
        step_size: int,
        quantiles: NDArray[np.floating],
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        if n_windows > len(self.cv_histories):
            raise ValueError(
                f"n_windows={n_windows} exceeds bound histories ({len(self.cv_histories)})"
            )
        raw = np.stack(
            [self.predict_quantiles(h, quantiles) for h in self.cv_histories[:n_windows]],
            axis=0,
        )  # (n_windows, n_series, n_q, horizon)
        # -> (n_series, n_windows, horizon, n_q): quantiles on the LAST axis.
        preds = np.transpose(raw, (1, 0, 3, 2))
        truths = self.cv_truths[:, :n_windows, :]
        return preds, truths


class MockQuantileRegressor(QRFQuantileRegressor):
    """Empirical-quantile regressor; lets SPCI run without ``quantile-forest``.

    Subclasses :class:`QRFQuantileRegressor` because SPCI's ``regressor`` field
    is typed as that concrete class (so the QRF knobs show up in its JSON
    schema), which rejects a plain :class:`QuantileRegressor` subclass. Neither
    ``fit`` nor the two predict methods touch the forest, so no extra is needed:
    ``predict_quantiles`` is taken from the ABC's loop-over-``predict_quantile``
    default rather than QRF's forest-backed override.
    """

    predict_quantiles = QuantileRegressor.predict_quantiles

    _y: NDArray[np.floating] | None = PrivateAttr(default=None)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit first.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)), dtype=np.float64)


class ConditionalMockQuantileRegressor(QRFQuantileRegressor):
    """Mock whose quantiles shift with the query window, as a real one does.

    :class:`MockQuantileRegressor` ignores ``X``, so it returns the same offsets
    for every window — under it a per-window replay and a broadcast of the final
    query are indistinguishable. SPCI's whole premise is conditioning on the
    recent residual window, so anything asserting that its
    ``calibration_intervals`` actually *moves* needs a regressor that does.

    Fits an AR-style mean of the feature row and keeps the empirical
    distribution of the residual around it.
    """

    predict_quantiles = QuantileRegressor.predict_quantiles

    _resid: NDArray[np.floating] | None = PrivateAttr(default=None)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        centre = np.asarray(X, dtype=np.float64).mean(axis=1)
        self._resid = np.asarray(y, dtype=np.float64) - centre

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        if self._resid is None:
            raise RuntimeError("ConditionalMockQuantileRegressor must be fit first.")
        centre = np.asarray(X, dtype=np.float64).mean(axis=1)
        return centre + float(np.quantile(self._resid, q))
