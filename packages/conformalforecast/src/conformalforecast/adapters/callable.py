"""Adapter that wraps a plain callable into a ForecasterAdapter."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from ..base import Forecast, ForecasterAdapter, Series


class CallableAdapter(ForecasterAdapter):
    """
    Wrap a user-supplied function into a :class:`ForecasterAdapter`.

    The callable must accept a history array and return a point forecast.
    No capability mixins are provided — the callable has no concept of
    refit, quantiles, or bootstrap.

    Parameters
    ----------
    predict_fn : callable
        ``fn(history) -> forecast`` where *history* has shape
        ``(n_series, T)`` and the return value has shape
        ``(n_series, horizon)``. May optionally accept an ``X`` keyword to
        receive exogenous features; see :meth:`predict`.
    horizon : int
        Number of steps to forecast.
    n_series : int
        Number of series in the panel.

    Notes
    -----
    ``predict_fn`` is hidden from ``model_json_schema()``: pydantic cannot
    render a JsonSchema for a callable, and no schema-filled call could supply
    a Python function anyway -- it has to be passed programmatically. For the
    same reason this adapter does not round-trip through
    ``model_validate(model_dump())``.
    """

    predict_fn: SkipJsonSchema[Callable[..., NDArray[np.floating]]] = Field(
        description=(
            "Called as predict_fn(history) -> (n_series, horizon), where history "
            "has shape (n_series, T). May accept an X keyword to receive "
            "exogenous features of shape (n_series, 1, horizon, n_features). "
            "The simplest adapter and the test workhorse."
        ),
    )

    def predict(self, history: Series, *, X: Forecast | None = None) -> Forecast:
        """
        Produce a point forecast by calling the wrapped function.

        If the wrapped callable accepts an ``X`` keyword, the exogenous array
        is forwarded to it; otherwise ``X`` is ignored.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        X : NDArray or None, shape (n_series, 1, horizon, n_features)

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        ValueError
            If ``history`` is not 2-D, has the wrong leading axis, contains
            NaN, or if the callable returns an unexpected shape.
        """
        history = self._validate_history(history)

        if X is not None and "X" in inspect.signature(self.predict_fn).parameters:
            raw = self.predict_fn(history, X=X)
        else:
            raw = self.predict_fn(history)
        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != (self.n_series, self.horizon):
            raise ValueError(
                f"predict_fn must return shape ({self.n_series}, {self.horizon}), got {raw.shape}"
            )

        # (n_series, horizon) -> (n_series, 1, horizon)
        return raw[:, np.newaxis, :]
