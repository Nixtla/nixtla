"""The two contracts every method implements: a declarative method and its fitted forecaster.

A :class:`ConformalMethod` is **config only**: ``alpha``, ``seed``, the
nonconformity ``score`` and per-method knobs, all pydantic fields, so a whole
method round-trips through ``model_dump()``. It holds no fitted state and is
never mutated by fitting. Its three entry points return a new
:class:`ConformalForecaster`:

- the **array surface** — ``fit(predictions, truths)`` — pure
  numpy-in calibration math, the surface the Nixtla libraries embed;
- the **forecaster surface** — ``calibrate`` / ``calibrate_cv`` — written once
  here, dispatching through the adapter.

The returned :class:`ConformalForecaster` carries every piece of calibration
state plus a deep copy of the config it was fitted under (``method_``), and
owns what uses that state: ``interval`` and ``update`` (array surface),
``predict`` and ``recalibrate`` (forecaster surface).

Array-surface signatures may only accept numpy-coercible arrays and
plain-Python config — never adapters, DataFrames, or library types.

Shapes
------
- ``Series``: ``(n_series, T)``.
- ``Forecast``: ``(n_series, n_windows | n_samples, horizon)``, always 3-D.
- ``Interval``: ``(n_series, n_samples, horizon, 2)``, last axis (lower, upper).
- ``Exog``: ``(n_series, n_windows | n_samples, horizon, n_features)`` —
  features aligned to each predicted step.

A method that ``CONSUMES_QUANTILES`` (CQR) takes a quantile cube in place of
point forecasts: ``(n_series, n_windows | n_samples, horizon, 2)``, last axis
``(q_lo, q_hi)``.

A NaN cell in ``predictions``/``truths`` at ``fit`` time means "this window
does not exist for this series" (ragged panels via left-NaN padding). NaN
propagates through scoring — never imputed — and per-(series, horizon)
quantiles use the masked helpers in ``_quantiles.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    SkipValidation,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from .._model import ArrayModel
from ..base import (
    CalibrationError,
    CalibrationResult,
    Exog,
    Forecast,
    ForecasterAdapter,
    Interval,
    PredictionResult,
    ScoreFunction,
    Series,
    UnsupportedCapability,
)
from ..nonconformity import ScoreField
from ..nonconformity._field import BUILTIN_KINDS, score_options

if TYPE_CHECKING:
    # Annotation-only: typing.Self is 3.11+, and typing_extensions is not a
    # declared dependency, so it is never imported at runtime.
    from typing_extensions import Self

__all__ = ["ConformalForecaster", "ConformalMethod"]


class ConformalForecaster(ArrayModel, ABC):
    """
    A fitted conformal method: calibration state plus the interval it implies.

    Returned by :meth:`ConformalMethod.fit`, :meth:`ConformalMethod.calibrate`
    and :meth:`ConformalMethod.calibrate_cv`; never constructed by hand.
    Subclasses implement ``interval`` and, when the method is online,
    ``_update``. ``update``, ``predict`` and ``recalibrate`` are concrete here
    and never overridden; ``calibration_intervals`` is overridden only when the
    interval is conditional on query-time state (SPCI).

    Every subclass field is required: the object exists only once fitted, so
    there is no "not yet fitted" state. ``forecaster_`` and the retained
    calibration arrays are the exception — a bare ``fit`` never sets them.

    Online methods mutate this object in place through ``update``. Take a
    snapshot with ``model_copy(deep=True)``; nothing a fitted object mutates is
    shared with its method or with another fitted object.

    Attributes
    ----------
    method_ : ConformalMethod
        Deep copy of the config this object was fitted under. Fitted code reads
        config through it.
    score_fn_ : ScoreFunction
        The score this object was fitted under; ``interval`` and the online
        update invert with it.
    n_calibration_samples_ : NDArray, shape (n_series,)
        Effective calibration samples per series.
    calibration_ : CalibrationResult
        Snapshot taken at fit. ``diagnostics["path"]`` names the entry point.
    forecaster_ : ForecasterAdapter or None
        The adapter bound by ``calibrate`` / ``calibrate_cv``; ``None`` after a
        bare ``fit``.
    predictions_calibration_ : NDArray or None, shape (n_series, n_windows, horizon)
        Calibration forecasts retained by ``calibrate`` / ``calibrate_cv`` for
        in-sample diagnostics (a defensive copy). ``None`` after a bare
        ``fit``.
    truths_calibration_ : NDArray or None, shape (n_series, n_windows, horizon)
        Realized values aligned with ``predictions_calibration_``; same
        lifecycle.
    quantile_predictions_calibration_ : NDArray or None, shape (n_series, n_windows, horizon, 2)
        Retained quantile predictions, quantile-consuming methods (CQR) only.
    """

    model_config = ConfigDict(extra="forbid")

    # SkipValidation: pydantic would check against the *parametrized* class
    # ConformalMethod[Any], which no concrete method subclasses. Concrete
    # fitted classes narrow this to their method's class, which is validated.
    method_: SkipValidation[ConformalMethod[Any]]
    score_fn_: ScoreFunction
    #: Shape (n_series,) — effective calibration samples per series.
    n_calibration_samples_: NDArray[np.integer]
    calibration_: CalibrationResult
    forecaster_: ForecasterAdapter | None = None
    #: Shape (n_series, n_windows, horizon) — retained calibration forecasts.
    predictions_calibration_: NDArray[np.floating] | None = None
    #: Shape (n_series, n_windows, horizon) — realized values aligned with them.
    truths_calibration_: NDArray[np.floating] | None = None
    #: Shape (n_series, n_windows, horizon, 2) — quantile methods (CQR) only.
    quantile_predictions_calibration_: NDArray[np.floating] | None = None

    # ------------------------------------------------------------------
    # Array surface
    # ------------------------------------------------------------------

    @abstractmethod
    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Produce calibrated intervals for new predictions.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
            A method that ``CONSUMES_QUANTILES`` takes quantile forecasts
            instead, shape ``(n_series, n_samples, horizon, 2)``.
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
            Last axis is (lower, upper).
        """
        ...

    def update(self, prediction: Forecast, truth: Forecast, *, X: Exog | None = None) -> None:
        """
        Online update step, mutating this object in place.

        Concrete here and never overridden: it delegates to the :meth:`_update`
        algorithm hook, which is a no-op for offline methods.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
            The point forecast that produced the interval being reported on —
            typically the ``point`` field of the most recent
            :class:`PredictionResult`.
        truth : Forecast, shape (n_series, 1, horizon)
            The realized values for the same horizon.
        X : Exog or None, shape (n_series, 1, horizon, n_features)
            Exogenous features. Methods that do not use them ignore ``X``.
        """
        self._update(prediction, truth, X=X)

    def _update(self, prediction: Forecast, truth: Forecast, *, X: Exog | None = None) -> None:
        """
        Online-update algorithm hook. No-op default; online methods
        (``IS_ONLINE = True``) override this, not :meth:`update`.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)
        X : Exog or None, shape (n_series, 1, horizon, n_features)
        """
        return None

    # ------------------------------------------------------------------
    # Forecaster surface
    # ------------------------------------------------------------------

    def _bound_forecaster(self, entry_point: str) -> ForecasterAdapter:
        """
        The forecaster bound at calibration.

        Parameters
        ----------
        entry_point : str
            Name of the calling method, for the error message.

        Returns
        -------
        ForecasterAdapter

        Raises
        ------
        CalibrationError
            If no forecaster is bound (the object came from a bare ``fit``).
        """
        if self.forecaster_ is None:
            raise CalibrationError(
                f"{type(self).__name__}.{entry_point}() needs a forecaster. Fit with "
                "method.calibrate(forecaster, histories, truths) or "
                "method.calibrate_cv(forecaster, n_windows), or call "
                "interval(predictions) to use the array surface."
            )
        return self.forecaster_

    def predict(self, history: Series, *, X: Exog | None = None) -> PredictionResult:
        """
        Produce a calibrated point forecast and prediction interval.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        X : Exog or None, shape (n_series, 1, horizon, n_features)
            Exogenous features, forwarded to ``forecaster_.predict`` and
            :meth:`interval`. Methods that do not use them ignore ``X``.
            Not supported for quantile-consuming methods (CQR).

        Returns
        -------
        PredictionResult
            ``point`` has shape ``(n_series, 1, horizon)``.
            ``interval`` has shape ``(n_series, 1, horizon, 2)``.

        Raises
        ------
        CalibrationError
            If no forecaster is bound (the object came from a bare ``fit``).
        NotImplementedError
            If ``X`` is not None and the method is quantile-consuming.
        """
        forecaster = self._bound_forecaster("predict")
        method = self.method_

        if type(method).CONSUMES_QUANTILES:
            if X is not None:
                raise NotImplementedError(
                    "Exogenous X with quantile-consuming methods (CQR) is not supported in v0.1."
                )
            from ..capabilities import SupportsQuantiles

            assert isinstance(forecaster, SupportsQuantiles)  # checked when bound
            q_pred = forecaster.predict_quantiles(history, method._calibration_quantiles())
            # (n_series, 2, horizon) -> (n_series, horizon, 2) -> (n_series, 1, horizon, 2)
            qp = np.transpose(q_pred, (0, 2, 1))[:, np.newaxis, :, :]
            point: Forecast = (qp[..., 0] + qp[..., 1]) / 2.0
            interval = self.interval(qp)
        else:
            point = forecaster.predict(history, X=X)
            interval = self.interval(point, X=X)

        return PredictionResult(point=point, interval=interval, alpha=method.alpha)

    def recalibrate(
        self,
        target_histories: list[Series] | None = None,
        target_truths: Forecast | None = None,
        *,
        X: Exog | None = None,
    ) -> Self:
        """
        Fit the same method on a target-domain panel, with the same forecaster.

        This is ``method_.calibrate(forecaster_, ...)`` pointed at a different
        panel: the forecaster is never refitted, so looping ``predict_batch``
        over target-domain histories and scoring against target truths *is*
        the frozen backtest transfer learning needs. The score carries over
        because it is config on ``method_``.

        Returns a **new** object; this one is left untouched. For online
        methods the new object's state comes from the target panel's own
        trajectory, so adaptation accumulated here through ``update()`` does
        not carry over. For a method supporting ``adapt`` the new object starts
        from uniform weights: target labels supersede the unlabelled proxy.

        Parameters
        ----------
        target_histories : list of Series, each shape (n_series, T)
            Target-domain histories, one per calibration window.
        target_truths : Forecast, shape (n_series, n_windows, horizon)
            Realized target-domain values.
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Target-domain features, forwarded to ``predict_batch`` and ``fit``.

        Returns
        -------
        ConformalForecaster
            Same class as ``self``; ``calibration_.diagnostics["path"]`` is
            ``"recalibrate"``.

        Raises
        ------
        CalibrationError
            If no forecaster is bound (the object came from a bare ``fit``).
        ValueError
            If either argument is missing, or the target panel's ``n_series``
            differs from the adapter's. Adapters fix ``n_series`` at
            construction; build a fresh adapter for a target panel with a
            different series count.
        NotImplementedError
            From ``calibrate`` if ``X`` is supplied for a quantile-consuming
            method.
        """
        if target_histories is None or target_truths is None:
            raise ValueError("recalibrate requires both target_histories and target_truths.")
        forecaster = self._bound_forecaster("recalibrate")
        expected = forecaster.n_series
        for i, hist in enumerate(target_histories):
            got = np.asarray(hist).shape[0]
            if got != expected:
                raise ValueError(
                    f"target_histories[{i}] has n_series={got} but the adapter was "
                    f"constructed with n_series={expected}. Adapters fix n_series at "
                    "construction; build a fresh adapter for this target panel."
                )
        fitted: Self = self.method_.calibrate(forecaster, target_histories, target_truths, X=X)
        fitted.calibration_.diagnostics["path"] = "recalibrate"
        return fitted

    # ------------------------------------------------------------------
    # In-sample replay
    # ------------------------------------------------------------------

    def calibration_intervals(self, predictions: Forecast, *, X: Exog | None = None) -> Interval:
        """
        The intervals this object's own fit implies at each calibration window.

        Aligned window-for-window with the arrays ``fit`` was given: entry ``t``
        is the interval the fitted state implies *at* window ``t``. Windows the
        method cannot evaluate are ``(NaN, NaN)``, which
        :func:`conformalforecast.diagnostics.evaluate_calibration` excludes from
        its averages rather than scoring as misses.

        The default broadcasts :meth:`interval` over every window. That is exact
        whenever the offsets ``interval`` applies are a constant of the whole
        calibration set, which holds for every method shipped here except SPCI:
        Split, CQR and WeightedSplit take a fixed quantile, and ACI, NexCP and
        AgACI take a quantile of the full score history at their final adapted
        level.

        A method whose ``interval`` is instead conditional on **query-time
        state** must override this. SPCI is the shipped example -- it
        conditions on the most recent residual window, so broadcasting its
        final answer over past windows measures nothing.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            The point forecasts ``fit`` was calibrated on (the quantile cube,
            shape ``(n_series, n_windows, horizon, 2)``, for a method that
            ``CONSUMES_QUANTILES``).
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)

        Returns
        -------
        Interval, shape (n_series, n_windows, horizon, 2)
        """
        return self.interval(predictions, X=X)

    def _calibration_intervals(self) -> Interval:
        """
        Replay :meth:`calibration_intervals` over the stored calibration arrays.

        Used by :func:`conformalforecast.diagnostics.evaluate_calibration`. For
        a quantile-consuming method the stored quantile predictions are passed
        in place of point forecasts, exactly as :meth:`predict` does.

        Returns
        -------
        Interval, shape (n_series, n_windows, horizon, 2)
        """
        if type(self.method_).CONSUMES_QUANTILES:
            qp = self.quantile_predictions_calibration_
            assert qp is not None  # stored by every calibrate* path
            return self.calibration_intervals(qp)
        preds = self.predictions_calibration_
        assert preds is not None  # stored by every calibrate* path
        return self.calibration_intervals(preds)

    # ------------------------------------------------------------------
    # Calibration-data retention (set by the calibrate entry points)
    # ------------------------------------------------------------------

    def _store_calibration_point(
        self, preds: NDArray[np.floating], truths: NDArray[np.floating]
    ) -> None:
        """Retain point calibration data for in-sample diagnostics (defensive copy)."""
        self.predictions_calibration_ = preds.copy()
        self.truths_calibration_ = truths.copy()

    def _store_calibration_quantile(
        self, qp: NDArray[np.floating], truths: NDArray[np.floating]
    ) -> None:
        """Retain quantile calibration data for in-sample diagnostics (defensive copy)."""
        qp_arr = np.asarray(qp, dtype=np.float64)
        self.quantile_predictions_calibration_ = qp_arr.copy()
        # The natural point-forecast equivalent is the midpoint of the pair.
        self.predictions_calibration_ = (qp_arr[..., 0] + qp_arr[..., 1]) / 2.0
        self.truths_calibration_ = truths.copy()


#: The fitted class a method returns.
F = TypeVar("F", bound=ConformalForecaster)


class ConformalMethod(ArrayModel, ABC, Generic[F]):
    """
    A conformal prediction method: declarative config plus the fitting math.

    Subclasses implement ``_default_score`` and ``fit``, and declare
    ``REQUIRED_CAPABILITIES`` / ``CONSUMES_QUANTILES`` / ``SCORE_TYPE`` /
    ``IS_ONLINE``. ``calibrate`` and ``calibrate_cv`` are concrete here and are
    never overridden. The type parameter ``F`` is the fitted class ``fit``
    returns, so ``SplitConformal(...).calibrate(...)`` is typed precisely.

    Lifecycle:
        1. Construct with ``alpha`` and, optionally, ``score`` and per-method
           config. There is no forecaster at construction.
        2. ``fitted = calibrate(forecaster, histories, truths)`` (any adapter)
           **or** ``fitted = calibrate_cv(forecaster, n_windows)`` (native
           cross-validation) — or ``fitted = fit(predictions, truths)``
           directly on arrays. Each call returns a **new**
           :class:`ConformalForecaster` and leaves the method untouched.
        3. ``fitted.predict(history)`` repeatedly (or
           ``fitted.interval(predictions)`` on the array surface).
        4. Online methods (``IS_ONLINE = True``): ``fitted.update(prediction,
           truth)`` after each realized step, mutating ``fitted`` in place.

    The score is **config**: ``score`` is a constructor field, ``None`` meaning
    :meth:`_default_score`. Compatibility with the method is checked at
    construction against ``SCORE_TYPE`` and raises ``pydantic.ValidationError``.
    ``fit`` deep-copies the score before fitting it, so the fitted object never
    shares a score with the method or with another fitted object.

    The forecaster is **bound at calibration**: capabilities are checked where
    it is bound, and it is stored on the returned object as ``forecaster_``.
    ``predict`` never raises ``UnsupportedCapability``.

    This is a pydantic model, so construction is **keyword-only**
    (``SplitConformal(alpha=0.1)``, never ``SplitConformal(0.1)``) and an
    out-of-range ``alpha`` raises ``pydantic.ValidationError`` (a subclass of
    ``ValueError``). ``model_json_schema()`` describes the config only:
    ``REQUIRED_CAPABILITIES`` / ``CONSUMES_QUANTILES`` / ``SCORE_TYPE`` /
    ``IS_ONLINE`` are ``ClassVar`` class metadata rather than fields.

    Parameters
    ----------
    alpha : float
        Miscoverage level in (0, 1). Target coverage is ``1 - alpha``.
    seed : int, default 0
        Determinism seed. Never ``None``.
    score : ScoreFunction or None, default None
        Nonconformity score. ``None`` takes the method's default. Must be an
        instance of the method's ``SCORE_TYPE``.

    Raises
    ------
    pydantic.ValidationError
        If ``alpha`` is not in (0, 1), if ``seed`` is negative, if ``score``
        does not fit the method, or if an unknown keyword is passed.

    Invariants
    ----------
    - Identical config + identical arrays -> identical output.
    - A method has no fitted state; config fields carry no trailing underscore.
    - All shape conventions are panel-aware (``n_series`` leading).
    """

    # extra="forbid" turns a typo'd or invented keyword into a ValidationError
    # instead of a silently ignored argument, which matters most for a caller
    # filling in model_json_schema() rather than reading the signature.
    model_config = ConfigDict(extra="forbid")

    # Class metadata, not fields: ClassVar keeps these off the schema, off the
    # constructor, and readable on the class itself (mlforecast_shim.py reads
    # CONSUMES_QUANTILES with no instance in hand). Subclasses override them.
    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    #: True when ``fit``'s ``predictions`` (and the fitted ``interval``'s) are a
    #: quantile cube, shape (n_series, n_windows | n_samples, horizon, 2).
    CONSUMES_QUANTILES: ClassVar[bool] = False
    #: The class every configured ``score`` must be an instance of. Only ever
    #: used in ``isinstance``, so the abstract base is a legitimate value here.
    SCORE_TYPE: ClassVar[type[ScoreFunction]] = ScoreFunction  # type: ignore[type-abstract]
    IS_ONLINE: ClassVar[bool] = False

    alpha: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Miscoverage level, strictly between 0 and 1. The intervals target "
            "a coverage of 1 - alpha, so alpha=0.1 asks for 90% coverage. "
            "Smaller alpha means wider intervals, and needs more calibration "
            "windows: at least ceil(1 / alpha) per series."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seed for any randomness a method uses, so that the same config on "
            "the same arrays gives the same intervals. Deterministic methods "
            "such as split conformal ignore it; it exists so every method has "
            "the same knob."
        ),
    )
    score: ScoreField = Field(
        default=None,
        description=(
            "Nonconformity score, given by its kind tag: 'absolute' (|truth - "
            "prediction|, symmetric intervals), 'signed' (truth - prediction, "
            "asymmetric intervals) or 'quantile' (for quantile forecasts, CQR "
            "only). Leave it null to use the method's default, which is almost "
            "always the right choice."
        ),
    )

    @classmethod
    def _accepts_score(cls, score_type: type[ScoreFunction]) -> bool:
        """
        Whether this method can be configured with a score of ``score_type``.

        The score must subclass ``SCORE_TYPE`` and consume the same prediction
        shape as the method: a quantile cube for a method that
        ``CONSUMES_QUANTILES``, a point forecast otherwise.

        Parameters
        ----------
        score_type : type[ScoreFunction]
            The score class to check.

        Returns
        -------
        accepted : bool
        """
        return (
            issubclass(score_type, cls.SCORE_TYPE)
            and score_type.CONSUMES_QUANTILES == cls.CONSUMES_QUANTILES
        )

    @model_validator(mode="after")
    def _check_score_type(self) -> Self:
        """Reject a score the method cannot invert, at construction."""
        cls = type(self)
        if self.score is None or cls._accepts_score(type(self.score)):
            return self
        got = type(self.score).__name__
        if not isinstance(self.score, cls.SCORE_TYPE):
            raise ValueError(
                f"{cls.__name__} requires a score that is a {cls.SCORE_TYPE.__name__} "
                f"(or a subclass of it), got {got}."
            )
        expected = "a quantile forecast" if cls.CONSUMES_QUANTILES else "a point forecast"
        raise ValueError(f"{cls.__name__} consumes {expected}, which {got} cannot score.")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Advertise only the built-in score kinds this method accepts."""
        json_schema = handler.resolve_ref_schema(
            super().__get_pydantic_json_schema__(core_schema, handler)
        )
        kinds = [k for k in BUILTIN_KINDS if cls._accepts_score(ScoreFunction._registry[k])]
        json_schema["properties"]["score"]["anyOf"] = score_options(kinds)
        return json_schema

    # ------------------------------------------------------------------
    # Array surface — implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _default_score(self) -> ScoreFunction:
        """Each method declares its default score function."""
        ...

    @abstractmethod
    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> F:
        """
        Fit the conformal correction on calibration arrays.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            Calibration-window forecasts. NaN cells mark windows that do
            not exist for a series. A method that ``CONSUMES_QUANTILES``
            takes the quantile cube instead, shape
            ``(n_series, n_windows, horizon, 2)``.
        truths : Forecast, shape (n_series, n_windows, horizon)
            Realized values aligned with ``predictions``.
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Exogenous features. Methods that do not use them ignore ``X``.

        Returns
        -------
        ConformalForecaster
            A new fitted object. The method itself is not modified.
        """
        ...

    def _bind_score(self) -> ScoreFunction:
        """
        The score to fit under: the configured one or the default, deep-copied.

        ``ScoreFunction.fit`` may set internal parameters, so fitting the
        configured instance would mutate the user's config and alias one score
        across every fitted object. Every ``fit`` calls this exactly once.

        Returns
        -------
        ScoreFunction
        """
        resolved = self.score if self.score is not None else self._default_score()
        return resolved.model_copy(deep=True)

    def _snapshot(self) -> Self:
        """
        A deep copy of this config, for the fitted object's ``method_``.

        Returns
        -------
        ConformalMethod
            Independent of ``self``: later edits to the method never reach a
            fitted object.
        """
        return self.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Capability / input introspection
    # ------------------------------------------------------------------

    def _check_capabilities(self, forecaster: ForecasterAdapter) -> None:
        """
        Gate an adapter against ``REQUIRED_CAPABILITIES``.

        Called where the forecaster is bound — never from ``predict``.

        Parameters
        ----------
        forecaster : ForecasterAdapter

        Raises
        ------
        UnsupportedCapability
            If ``forecaster`` is missing a required capability.
        """
        for cap in self.REQUIRED_CAPABILITIES:
            if not forecaster.supports(cap):
                raise UnsupportedCapability(
                    f"{type(self).__name__} requires an adapter implementing "
                    f"{cap.__name__}. Got {type(forecaster).__name__}."
                )

    def _calibration_quantiles(self) -> NDArray[np.floating]:
        """
        Quantile levels to query the forecaster at (quantile methods only).

        The base implementation raises; quantile-consuming methods (CQR)
        override it to return the levels derived from their construction.

        Returns
        -------
        NDArray, shape (2,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare calibration quantile levels."
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        forecaster: ForecasterAdapter,
        histories: list[Series] | None = None,
        truths: Forecast | None = None,
        *,
        X: Exog | None = None,
    ) -> F:
        """
        Fit the conformal correction on an explicit calibration set.

        Loops over ``histories`` via ``forecaster.predict_batch`` (or
        ``predict_quantiles`` when the method ``CONSUMES_QUANTILES``) and hands
        the resulting arrays to :meth:`fit`. Works with any adapter.

        Parameters
        ----------
        forecaster : ForecasterAdapter
            Adapter providing every mixin in ``REQUIRED_CAPABILITIES``.
            Retained on the returned object as ``forecaster_``.
        histories : list of Series
            Each shape ``(n_series, T)``.
        truths : Forecast, shape (n_series, len(histories), horizon)
            Realized values aligned with each forecast window.
        X : Exog or None, shape (n_series, len(histories), horizon, n_features)
            Exogenous features, forwarded to ``forecaster.predict_batch`` and
            :meth:`fit`. Methods that do not use them ignore ``X``. Not
            supported for quantile-consuming methods (CQR).

        Returns
        -------
        ConformalForecaster
            A new fitted object with ``forecaster_`` bound and
            ``calibration_.diagnostics["path"] == "loop"``.

        Raises
        ------
        UnsupportedCapability
            If ``forecaster`` is missing a required capability.
        ValueError
            If either ``histories`` or ``truths`` is missing.
        NotImplementedError
            If ``X`` is not None and the method is quantile-consuming.
        InsufficientCalibrationSamples
            If any series has too few effective calibration samples (raised by
            :meth:`fit`, per series).
        """
        self._check_capabilities(forecaster)
        if histories is None or truths is None:
            raise ValueError("calibrate requires both histories and truths.")
        truths_arr = np.asarray(truths, dtype=np.float64)

        if self.CONSUMES_QUANTILES:
            if X is not None:
                raise NotImplementedError(
                    "Exogenous X with quantile-consuming methods (CQR) is not supported in v0.1."
                )
            qp = self._collect_quantile_predictions_loop(forecaster, histories)
            fitted = self.fit(qp, truths_arr)
            fitted._store_calibration_quantile(qp, truths_arr)
        else:
            preds = forecaster.predict_batch(histories, X=X)
            fitted = self.fit(preds, truths_arr, X=X)
            fitted._store_calibration_point(np.asarray(preds, dtype=np.float64), truths_arr)

        fitted.calibration_.diagnostics["path"] = "loop"
        fitted.forecaster_ = forecaster
        return fitted

    def calibrate_cv(
        self,
        forecaster: ForecasterAdapter,
        n_windows: int,
        *,
        step_size: int = 1,
        refit: bool | int = False,
        aligned: bool = False,
        X: Exog | None = None,
    ) -> F:
        """
        Fit the conformal correction via the adapter's native cross-validation.

        Delegates to ``forecaster.cross_validate`` (or
        ``cross_validate_quantiles`` when the method ``CONSUMES_QUANTILES``) to
        produce all calibration windows in a single library-native call, then
        hands the arrays to :meth:`fit`. For libraries with a real CV
        implementation this is dramatically faster than :meth:`calibrate`.

        Parameters
        ----------
        forecaster : ForecasterAdapter
            Adapter providing every mixin in ``REQUIRED_CAPABILITIES`` plus the
            cross-validation capability the chosen path needs. Retained on the
            returned object as ``forecaster_``.
        n_windows : int
            Number of cross-validation windows.
        step_size : int, default 1
            Step size between successive CV windows.
        refit : bool or int, default False
            Whether (or how often) to refit the model between windows.
        aligned : bool, default False
            Forwarded to the point-CV path (``forecaster.cross_validate``)
            only. ``False`` returns dense per-series arrays (window axis = k
            steps before each series' own end); ``True`` returns
            calendar-aligned cutoffs with NaN cells for series that do not
            reach every cutoff. The quantile-CV path (CQR) ignores this flag:
            it always densifies, so ragged panels calibrate but the aligned
            cube is not available there.
        X : Exog or None
            Exogenous features. Not supported: exogenous cross-validation is
            out of scope in v0.1. Passing a non-None value always raises. Use
            :meth:`calibrate` with an explicit loop instead.

        Returns
        -------
        ConformalForecaster
            A new fitted object with ``forecaster_`` bound and
            ``calibration_.diagnostics["path"] == "cross_validation"``.

        Raises
        ------
        NotImplementedError
            If ``X`` is not None.
        UnsupportedCapability
            If the adapter does not provide the required cross-validation
            capability.
        InsufficientCalibrationSamples
            If any series has too few effective calibration samples (raised by
            :meth:`fit`, per series).
        """
        self._check_capabilities(forecaster)
        if X is not None:
            raise NotImplementedError(
                "exogenous cross-validation is not supported in v0.1. Use "
                "calibrate(forecaster, histories, truths, X=...) with the explicit loop path."
            )
        from ..capabilities import (
            SupportsCrossValidation,
            SupportsCrossValidationQuantiles,
        )

        if self.CONSUMES_QUANTILES:
            if not forecaster.supports(SupportsCrossValidationQuantiles):
                raise UnsupportedCapability(
                    "calibrate_cv requires an adapter implementing "
                    f"SupportsCrossValidationQuantiles. Got "
                    f"{type(forecaster).__name__}. Pass explicit "
                    "(histories, truths) to calibrate() instead."
                )
            cv_q = forecaster
            assert isinstance(cv_q, SupportsCrossValidationQuantiles)  # checked above
            qp, truths = cv_q.cross_validate_quantiles(
                n_windows=n_windows,
                step_size=step_size,
                quantiles=self._calibration_quantiles(),
                refit=refit,
            )
            truths_arr = np.asarray(truths, dtype=np.float64)
            fitted = self.fit(qp, truths_arr)
            fitted._store_calibration_quantile(qp, truths_arr)
        else:
            if not forecaster.supports(SupportsCrossValidation):
                raise UnsupportedCapability(
                    "calibrate_cv requires an adapter implementing "
                    f"SupportsCrossValidation. Got {type(forecaster).__name__}. "
                    "Pass explicit (histories, truths) to calibrate() instead."
                )
            cv = forecaster
            assert isinstance(cv, SupportsCrossValidation)  # checked above
            preds, truths = cv.cross_validate(
                n_windows=n_windows, step_size=step_size, refit=refit, aligned=aligned
            )
            truths_arr = np.asarray(truths, dtype=np.float64)
            fitted = self.fit(preds, truths_arr)
            fitted._store_calibration_point(np.asarray(preds, dtype=np.float64), truths_arr)

        fitted.calibration_.diagnostics["path"] = "cross_validation"
        fitted.forecaster_ = forecaster
        return fitted

    def _collect_quantile_predictions_loop(
        self, forecaster: ForecasterAdapter, histories: list[Series]
    ) -> NDArray[np.floating]:
        """
        Query the forecaster's quantile predictions for each calibration window.

        Parameters
        ----------
        forecaster : ForecasterAdapter
            The adapter bound by the calling entry point; must implement
            :class:`SupportsQuantiles`.
        histories : list of Series, each shape (n_series, T)

        Returns
        -------
        NDArray, shape (n_series, n_windows, horizon, 2)
            Quantiles on the last axis — the layout ``QuantileScore`` expects.
        """
        from ..capabilities import SupportsQuantiles

        quantiles = self._calibration_quantiles()
        assert isinstance(forecaster, SupportsQuantiles)  # checked when bound
        raw = np.stack(
            [forecaster.predict_quantiles(h, quantiles) for h in histories],
            axis=0,
        )  # (n_windows, n_series, 2, horizon)
        return np.transpose(raw, (1, 0, 3, 2))


# ConformalForecaster is defined first so it can bound F, which leaves its
# ``method_: ConformalMethod[Any]`` annotation a forward reference until now.
ConformalForecaster.model_rebuild()
