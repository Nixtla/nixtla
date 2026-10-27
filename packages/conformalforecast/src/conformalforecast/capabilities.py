"""
Capability mixins for forecaster adapters.

Each mixin declares one optional capability via an abstract method. Adapters
inherit from the mixins they implement; conformal methods declare which mixins
they require via `REQUIRED_CAPABILITIES`.

Inheriting from a mixin without overriding its method is a programming error
caught at adapter import time (the abstract method makes the class
uninstantiable).

Each mixin is an ``ArrayModel`` as well as an ``ABC``: an adapter inherits
``ForecasterAdapter`` and several mixins at once, and pydantic requires every
base of a model to be a model. They declare no fields, so they add nothing to
an adapter's schema, and ``isinstance`` still works -- which is what
``ForecasterAdapter.supports`` gates on. Model config (including
``extra="forbid"``) comes from ``ForecasterAdapter``; the mixins set none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ._model import ArrayModel
from .base import Forecast, Series


class SupportsRefit(ArrayModel, ABC):
    """Adapter can retrain its underlying model on new data."""

    @abstractmethod
    def refit(self, history: Series) -> None:
        """
        Refit the underlying model on `history`.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        """


class SupportsQuantiles(ArrayModel, ABC):
    """Adapter can produce quantile forecasts directly (needed for CQR)."""

    @abstractmethod
    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        """
        Predict requested quantiles.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        quantiles : NDArray, shape (n_quantiles,)
            Values in (0, 1).

        Returns
        -------
        Forecast, shape (n_series, n_quantiles, horizon)
            Quantile forecasts. Note the middle axis indexes quantiles, not samples.
        """


class SupportsBootstrap(ArrayModel, ABC):
    """Adapter can produce bootstrap ensemble forecasts (needed for EnbPI)."""

    @abstractmethod
    def bootstrap_predict(
        self,
        history: Series,
        n_bootstraps: int,
        seed: int = 0,
    ) -> Forecast:
        """
        Produce `n_bootstraps` predictions from bootstrap resamples of training data.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        n_bootstraps : int
        seed : int, default 0
            Never ``None``: resampling must be reproducible given the same
            fitted state and inputs.

        Returns
        -------
        Forecast, shape (n_series, n_bootstraps, horizon)
        """


class SupportsCrossValidation(ArrayModel, ABC):
    """Adapter can run rolling-origin cross-validation natively."""

    @abstractmethod
    def cross_validate(
        self,
        n_windows: int,
        step_size: int,
        refit: bool | int = False,
        *,
        aligned: bool = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation on the adapter's training data.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows.
        step_size : int
            Number of time steps between successive windows.
        refit : bool or int
            Whether (or how often) to refit the model between windows.
        aligned : bool, default False
            ``False`` returns dense per-series arrays (window axis = k steps
            before each series' own end). ``True`` returns calendar-aligned
            cutoffs with NaN cells for late-starting series.

        Returns
        -------
        predictions : Forecast, shape (n_series, n_windows, horizon)
        truths : Forecast, shape (n_series, n_windows, horizon)
        """


class SupportsCrossValidationQuantiles(ArrayModel, ABC):
    """Adapter can produce quantile forecasts via cross-validation.

    This is the fast calibration path for CQR (and any future quantile-based
    conformal method). Adapters with this capability avoid the per-sample loop
    of ``predict_quantiles``, instead leveraging the underlying library's
    cross-validation API to produce all calibration samples in a single call.
    """

    @abstractmethod
    def cross_validate_quantiles(
        self,
        n_windows: int,
        step_size: int,
        quantiles: NDArray[np.floating],
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation producing quantile forecasts.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows.
        step_size : int
            Number of time steps between successive windows.
        quantiles : NDArray, shape (n_quantiles,)
            Values in (0, 1).
        refit : bool or int, default False
            Same semantics as :meth:`SupportsCrossValidation.cross_validate`.

        Returns
        -------
        quantile_predictions : Forecast, shape (n_series, n_windows, horizon, n_quantiles)
            Quantiles are on the **last axis**, matching :class:`QuantileScore`'s
            expected input. This differs from :meth:`SupportsQuantiles.predict_quantiles`,
            where quantiles are on axis 1 (because ``predict_quantiles`` produces a
            single sample per call). The intentional axis difference avoids an
            extra transpose in the CQR fast path.
        truths : Forecast, shape (n_series, n_windows, horizon)
        """
