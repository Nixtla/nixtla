"""
Base abstractions for conformalforecast.

This module holds the shared contracts: the type aliases every array crossing a
module boundary is described with, `ForecasterAdapter`, `ScoreFunction`, the
result models and the exception hierarchy.

The methods themselves live in `methods/`, each a pair of classes: a
declarative :class:`~conformalforecast.methods.base.ConformalMethod` whose
``fit`` / ``calibrate`` / ``calibrate_cv`` return a fitted
:class:`~conformalforecast.methods.base.ConformalForecaster`, which carries
``interval`` / ``update`` / ``predict``. That array surface is what
statsforecast, mlforecast and neuralforecast embed, since they own their own
CV/predict orchestration.

`ForecasterAdapter` converts framework-specific objects to/from numpy arrays so
that methods never see framework types. `ScoreFunction` maps
(prediction, truth) pairs to nonconformity scores.

Capability flags are expressed as mixins (see `capabilities.py`). Methods declare
what mixins their adapter must provide via ``REQUIRED_CAPABILITIES``, checked
where the forecaster is bound — at the top of ``calibrate`` and
``calibrate_cv``, never in ``predict`` — via the adapter's
:meth:`ForecasterAdapter.supports` hook. Score compatibility is checked at
construction: the score is a config field on the method, validated against the
method's ``SCORE_TYPE``.

All array shapes are panel-aware: `n_series` is the leading axis everywhere.
Single-series users pass `n_series=1`.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from conformalforecast._model import ArrayModel

# -----------------------------------------------------------------------------
# Type aliases. These are the only shapes that cross module boundaries.
# -----------------------------------------------------------------------------

# A panel of time series. Shape: (n_series, T).
# Series may have different effective lengths via NaN-padding on the left.
Series = NDArray[np.floating]

# Predictions or truths over multiple horizons across a panel.
# Shape: (n_series, n_samples, horizon). All three axes always present.
Forecast = NDArray[np.floating]

# Prediction intervals. Shape: (n_series, n_samples, horizon, 2).
# Last axis is (lower, upper).
Interval = NDArray[np.floating]

# Exogenous features aligned to each predicted step.
# Shape: (n_series, n_windows | n_samples, horizon, n_features).
Exog = NDArray[np.floating]


def as_forecast(arr: object, name: str) -> NDArray[np.floating]:
    """
    Coerce input to a 3-D float array, preserving ``float32``.

    Parameters
    ----------
    arr : array-like
        Anything ``np.asarray``-coercible with shape
        ``(n_series, n_windows | n_samples, horizon)``.
    name : str
        Argument name, used in the error message.

    Returns
    -------
    NDArray[np.floating], shape (n_series, n_windows | n_samples, horizon)
        Float array. ``float32`` input stays ``float32``; non-float dtypes
        are cast to ``float64``.

    Raises
    ------
    ValueError
        If the coerced array is not 3-D.
    """
    out = np.asarray(arr)
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    if out.ndim != 3:
        raise ValueError(
            f"{name} must be 3-D (n_series, n_windows | n_samples, horizon), got shape {out.shape}"
        )
    return out


# -----------------------------------------------------------------------------
# ForecasterAdapter — abstract base
# -----------------------------------------------------------------------------


class ForecasterAdapter(ArrayModel, ABC):
    """
    Uniform interface over any forecasting library.

    Concrete adapters live in `adapters/`. They convert framework-specific
    objects to/from numpy arrays so that ConformalMethod implementations
    never see framework types.

    Capabilities are declared via mixins in `capabilities.py`. An adapter that
    can produce quantiles inherits `SupportsQuantiles`. An adapter that can
    refit on demand inherits `SupportsRefit`. Etc. Whether an adapter provides
    a capability is queried via :meth:`supports`.

    The minimum contract is `predict()`. Everything else is opt-in via mixins.
    """

    # extra="forbid" (pydantic's default is "ignore"): an adapter is
    # constructed by hand with up to nine keyword arguments, so a typo in one
    # of them -- ``targetcol=``, ``exog_feature=`` -- must fail at
    # construction rather than silently leave the default in place and
    # mis-read the training frame much later. Concrete adapters inherit this.
    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(ge=1, description="Forecast horizon this adapter produces per call.")
    n_series: int = Field(
        ge=1, description="Number of series in the panel, the leading array axis."
    )

    def supports(self, capability: type) -> bool:
        """
        Return True if this adapter provides ``capability`` (a mixin type).

        The default is an ``isinstance`` check. Adapters whose capability is
        runtime-conditional (e.g. a NeuralForecast adapter whose quantile
        support depends on the fitted loss) override this. Keeping the check
        here means the runtime capability gate lives where the forecaster is
        bound -- the first thing ``calibrate`` and ``calibrate_cv`` do -- never
        inside ``predict``.

        Parameters
        ----------
        capability : type
            A capability mixin class (e.g. ``SupportsCrossValidation``).

        Returns
        -------
        bool
        """
        return isinstance(self, capability)

    @abstractmethod
    def predict(self, history: Series, *, X: Exog | None = None) -> Forecast:
        """
        Produce a point forecast for the next ``self.horizon`` steps per series.

        Parameters
        ----------
        history : Series, shape (n_series, T)
            Past observations for each series, ending immediately before the
            forecast window. T may be any value >= the model's required minimum.
        X : Exog or None, shape (n_series, 1, horizon, n_features)
            Future exogenous features aligned to the forecast window. Adapters
            that do not support exogenous features raise if ``X`` is not None.

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)
            Point forecast over the horizon for each series. The middle axis
            is 1 because this call produces one forecast per series.
        """

    def predict_batch(self, histories: list[Series], *, X: Exog | None = None) -> Forecast:
        """
        Vectorized prediction over multiple history windows.

        Parameters
        ----------
        histories : list of Series, each shape (n_series, T)
        X : Exog or None, shape (n_series, len(histories), horizon, n_features)
            Per-window future exogenous features; window ``w`` receives
            ``X[:, w:w+1, :, :]``.

        Returns
        -------
        Forecast, shape (n_series, len(histories), horizon)

        Default implementation calls `predict` in a loop and stacks results.
        Adapters where the underlying library supports batch inference should
        override this for performance.
        """
        if X is None:
            out = [self.predict(h) for h in histories]
        else:
            X_arr = np.asarray(X)
            if X_arr.shape[1] != len(histories):
                raise ValueError(
                    f"X window axis {X_arr.shape[1]} != number of histories {len(histories)}."
                )
            out = [self.predict(h, X=X_arr[:, w : w + 1, :, :]) for w, h in enumerate(histories)]
        return np.concatenate(out, axis=1)

    def _validate_history(
        self, history: Series, *, allow_ragged: bool = False
    ) -> NDArray[np.floating]:
        """
        Validate and normalise a history array.

        Casts to ``float64`` and checks that the array is 2-D with a leading
        axis matching ``self.n_series``. Concrete adapters call this at the top
        of every method that takes a history.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        allow_ragged : bool, default False
            If ``False``, any NaN raises. If ``True``, a contiguous **leading**
            NaN prefix per series is allowed (left-NaN padding for late-starting
            series); interior or trailing NaN raises, naming the offending series.

        Returns
        -------
        NDArray[np.floating], shape (n_series, T)
            The same data cast to ``float64``.

        Raises
        ------
        ValueError
            If ``history`` is not 2-D, has the wrong leading axis, or violates
            the NaN policy selected by ``allow_ragged``.
        """
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"history must be 2-D (n_series, T), got shape {arr.shape}")
        if arr.shape[0] != self.n_series:
            raise ValueError(f"history leading axis must be {self.n_series}, got {arr.shape[0]}")
        if not allow_ragged:
            if np.isnan(arr).any():
                raise ValueError("history contains NaN values.")
            return arr

        # Ragged: leading NaN prefix allowed; interior/trailing NaN rejected.
        bad: list[int] = []
        for i in range(arr.shape[0]):
            valid = ~np.isnan(arr[i])
            idx = np.flatnonzero(valid)
            if idx.size == 0 or not valid[int(idx[0]) :].all():
                bad.append(i)
        if bad:
            raise ValueError(
                f"history has interior or trailing NaN for series {bad}; only a "
                "contiguous leading NaN prefix (left padding) is allowed."
            )
        return arr

    def _ragged_ranges(
        self, history: NDArray[np.floating]
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        """
        Per-series effective range of a validated (possibly ragged) history.

        Parameters
        ----------
        history : NDArray, shape (n_series, T)
            An array already passed through ``_validate_history(..., allow_ragged=True)``.

        Returns
        -------
        start_offsets : NDArray[np.integer], shape (n_series,)
            Column index of each series' first non-NaN value.
        lengths : NDArray[np.integer], shape (n_series,)
            Per-series non-NaN observation count.
        """
        n_series, t_len = history.shape
        starts = np.empty(n_series, dtype=np.int64)
        lengths = np.empty(n_series, dtype=np.int64)
        for i in range(n_series):
            idx = np.flatnonzero(~np.isnan(history[i]))
            starts[i] = int(idx[0]) if idx.size else t_len
            lengths[i] = int(idx.size)
        return starts, lengths


# -----------------------------------------------------------------------------
# ScoreFunction
# -----------------------------------------------------------------------------


class ScoreFunction(ArrayModel, ABC):
    """
    Maps (predictions, truths) to nonconformity scores.

    Stateless except for parameters fitted in `fit`. All operations are
    panel-aware: the leading axis is `n_series`.

    Every concrete subclass declares its own ``kind`` tag as a ``Literal`` with
    a default, e.g. ``kind: Literal["absolute"] = "absolute"``. The tag is what
    lets a method's ``score`` field survive ``model_validate(model_dump())``: a
    dumped score is a dict, and the tag names the class to rebuild it as.
    Concrete subclasses are registered by tag when the class is defined, and a
    subclass that reuses another class's tag -- typically by inheriting it from
    a built-in without declaring its own -- raises ``TypeError`` there, since it
    would otherwise reload silently as its parent.
    """

    #: Tag -> concrete class. Populated by ``__pydantic_init_subclass__``.
    _registry: ClassVar[dict[str, type[ScoreFunction]]] = {}

    #: True when ``score`` / ``invert`` take a quantile cube, shape
    #: (n_series, n_samples, horizon, 2), rather than a point forecast. A method
    #: accepts a score only when this matches its own ``CONSUMES_QUANTILES``.
    CONSUMES_QUANTILES: ClassVar[bool] = False

    kind: str = Field(
        description=(
            "Tag naming the score function. Each concrete score fixes it to a "
            "single literal value, so a serialized score can be rebuilt as the "
            "right class."
        )
    )

    # extra="forbid" (pydantic's default is "ignore"): a custom ScoreFunction
    # subclass is a supported extension point (passed as a method's `score=`),
    # constructed directly by users rather than filled in from a schema --
    # unlike QuantileRegressor's ABC, whose only pydantic-facing use is
    # SkipJsonSchema-hidden. Forbidding extras means a typo in a future field
    # (e.g. a normalised-residual score's scale) surfaces at construction,
    # matching CalibrationResult/PredictionResult's reasoning below.
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a concrete subclass under its ``kind`` tag.

        Raises
        ------
        TypeError
            If a concrete subclass declares no ``kind`` default, or reuses the
            tag of a different registered class.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        kind = cls.model_fields["kind"].default
        if not isinstance(kind, str):
            raise TypeError(
                f"{cls.__qualname__} must declare its own score tag, e.g. "
                f'kind: Literal["my_score"] = "my_score".'
            )
        existing = ScoreFunction._registry.get(kind)
        # Redefining the same class (a module reload, a class defined in a test
        # body) replaces the entry; a *different* class claiming the tag is the
        # silent-reload bug the tag exists to prevent.
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise TypeError(
                f"{cls.__qualname__} reuses the score tag {kind!r} of "
                f"{existing.__qualname__}. Declare its own tag, e.g. "
                f'kind: Literal["my_score"] = "my_score".'
            )
        ScoreFunction._registry[kind] = cls

    @classmethod
    def from_kind(cls, value: dict[str, Any]) -> ScoreFunction:
        """
        Rebuild a serialized score as the class its ``kind`` tag names.

        Parameters
        ----------
        value : dict
            A dumped score, carrying a ``kind`` key.

        Returns
        -------
        ScoreFunction

        Raises
        ------
        ValueError
            If ``kind`` is missing or names no registered score. The module
            defining a custom score must be imported before loading it.
        """
        kind = value.get("kind")
        target = ScoreFunction._registry.get(kind) if isinstance(kind, str) else None
        if target is None:
            raise ValueError(
                f"unknown score kind {kind!r}; registered kinds are "
                f"{sorted(ScoreFunction._registry)}. Import the module defining a "
                "custom score before loading it."
            )
        return target.model_validate(value)

    @abstractmethod
    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute nonconformity scores.

        Parameters
        ----------
        prediction, truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)
        """

    @abstractmethod
    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Given a point forecast and a score threshold, return the prediction interval.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        score_threshold : NDArray
            Either scalar, or shape (n_series,), or shape (n_series, horizon).
            Will be broadcast against `prediction` along its last two axes.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
        """

    def fit(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Fit any internal parameters of the score function (e.g. scale
        estimates for normalised residuals). Default: no-op.

        Called by the method's ``fit`` **before** the first call to :meth:`score` on the
        calibration data, so implementations can assume that ``prediction`` and
        ``truth`` are the full calibration arrays.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        truth : Forecast, shape (n_series, n_samples, horizon)
        """
        return None


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------


class CalibrationResult(ArrayModel):
    """Snapshot taken at fit, carried on the fitted object as ``calibration_``.

    Every fitted :class:`~conformalforecast.methods.base.ConformalForecaster`
    holds one, built by its method's ``fit`` and stamped with a ``"path"`` by
    the calibrate entry points (``recalibrate`` and ``adapt`` build their own).

    Attributes
    ----------
    n_calibration_samples : NDArray[np.integer], shape (n_series,)
        Per-series effective (non-NaN) calibration window counts.
    diagnostics : dict
        Method-specific diagnostics (quantile levels, learning rates,
        per-method quantile arrays, and the orchestration-layer ``"path"`` marker).
        Arrays stored here are defensively copied.

    Notes
    -----
    ``extra="forbid"``: results are *returned to* callers rather than
    constructed by them, so the only writers are this package's ``fit``
    implementations — a mistyped keyword there is a bug worth failing on, and
    forbidding extras cannot break a user call that does not exist.
    """

    model_config = ConfigDict(extra="forbid")

    n_calibration_samples: NDArray[np.integer]
    diagnostics: dict[str, object]


class PredictionResult(ArrayModel):
    """Returned by `ConformalForecaster.predict`.

    Attributes
    ----------
    point : Forecast, shape (n_series, n_samples, horizon)
        Point forecasts.
    interval : Interval, shape (n_series, n_samples, horizon, 2)
        Prediction intervals; the last axis is ``(lower, upper)``.
    alpha : float
        The miscoverage level the intervals target.

    Notes
    -----
    ``extra="forbid"`` for the same reason as :class:`CalibrationResult`.
    """

    model_config = ConfigDict(extra="forbid")

    point: Forecast  # shape (n_series, n_samples, horizon)
    interval: Interval  # shape (n_series, n_samples, horizon, 2)
    alpha: float


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class ConformalForecastError(Exception):
    """Base exception for the package."""


class CalibrationError(ConformalForecastError):
    """Raised when predict() is called before calibrate(), or calibration data is invalid."""


class UnsupportedCapability(ConformalForecastError):
    """Raised when a method requires an adapter capability that isn't supported."""


class InsufficientCalibrationSamples(CalibrationError):
    """Raised when one or more series have too few effective calibration samples.

    "Effective" means non-NaN calibration windows. Each series needs at least
    ``ceil(1 / alpha)`` effective samples for the finite-sample-corrected
    conformal quantile to be well-defined.

    Attributes
    ----------
    series_indices : list[int]
        Indices (along the leading ``n_series`` axis) of the offending series.
    """

    def __init__(self, message: str, series_indices: list[int]) -> None:
        super().__init__(message)
        self.series_indices = series_indices
