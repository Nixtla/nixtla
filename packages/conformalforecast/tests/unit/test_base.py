"""Contract tests for the ``ConformalMethod`` / ``ConformalForecaster`` orchestration.

These exercise calibrate / calibrate_cv dispatch, the ``supports()`` capability
gate and the ``CalibrationResult`` carried on the fitted object, through the two
simplest concrete methods (``SplitConformal`` offline, ``AdaptiveConformalInference`` online),
plus the adapter and exception contracts that still live in ``base.py``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from pydantic import PydanticInvalidForJsonSchema, ValidationError

from conformalforecast._model import ArrayModel
from conformalforecast.adapters import CallableAdapter
from conformalforecast.base import (
    CalibrationError,
    CalibrationResult,
    InsufficientCalibrationSamples,
    PredictionResult,
    ScoreFunction,
    UnsupportedCapability,
)
from conformalforecast.capabilities import SupportsCrossValidation, SupportsQuantiles
from conformalforecast.methods import (
    ACIForecaster,
    AdaptiveConformalInference,
    FixedOffsetForecaster,
    SplitConformal,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal method subclasses + adapters
# ---------------------------------------------------------------------------


class _NeedsQuantiles(SplitConformal):
    """A method declaring a quantile capability requirement (for the gate test)."""

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = (SupportsQuantiles,)


def _point_adapter(n_series: int = 2, horizon: int = 3) -> CallableAdapter:
    """A point-only adapter: forecast = last value repeated over the horizon."""
    return CallableAdapter(
        predict_fn=lambda h: np.repeat(h[:, -1:], horizon, axis=1),
        horizon=horizon,
        n_series=n_series,
    )


class _CVAdapter(CallableAdapter, SupportsCrossValidation):
    """Point adapter that also replays a pre-baked CV panel."""

    def __init__(self, predict_fn, horizon, n_series, preds, truths):  # type: ignore[no-untyped-def]
        super().__init__(predict_fn=predict_fn, horizon=horizon, n_series=n_series)
        self._preds = np.asarray(preds, dtype=np.float64)
        self._truths = np.asarray(truths, dtype=np.float64)

    def cross_validate(  # type: ignore[no-untyped-def]
        self, n_windows, step_size, refit=False, *, aligned=False
    ):
        return self._preds[:, :n_windows, :], self._truths[:, :n_windows, :]


def _make_histories_truths(n_series, horizon, n_windows, seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n_windows)]
    truths = rng.normal(size=(n_series, n_windows, horizon))
    return histories, truths


# ---------------------------------------------------------------------------
# supports() hook
# ---------------------------------------------------------------------------


class TestSupportsHook:
    def test_point_adapter_lacks_cv(self) -> None:
        adapter = _point_adapter()
        assert adapter.supports(SupportsCrossValidation) is False
        assert adapter.supports(SupportsQuantiles) is False

    def test_cv_adapter_supports_cv(self) -> None:
        adapter = _CVAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], 3, axis=1),
            horizon=3,
            n_series=2,
            preds=np.zeros((2, 5, 3)),
            truths=np.zeros((2, 5, 3)),
        )
        assert adapter.supports(SupportsCrossValidation) is True

    def test_capability_gate_fires_when_the_forecaster_is_bound(self) -> None:
        # _NeedsQuantiles requires SupportsQuantiles; a point adapter lacks it.
        # Construction is unconditional now — the gate is at calibrate time.
        method = _NeedsQuantiles(alpha=0.1)
        histories, truths = _make_histories_truths(2, 3, 40)
        with pytest.raises(UnsupportedCapability, match="SupportsQuantiles"):
            method.calibrate(_point_adapter(), histories, truths)


# ---------------------------------------------------------------------------
# calibrate (explicit loop path)
# ---------------------------------------------------------------------------


class TestCalibrateLoop:
    def test_loop_path_returns_a_fitted_object(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate(adapter, histories, truths)

        assert isinstance(fitted, FixedOffsetForecaster)
        assert fitted.forecaster_ is adapter
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert isinstance(result.n_calibration_samples, np.ndarray)
        assert result.n_calibration_samples.shape == (n_series,)
        assert result.diagnostics["path"] == "loop"

    def test_stores_calibration_data(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = SplitConformal(alpha=0.1).calibrate(adapter, histories, truths)
        assert fitted.predictions_calibration_.shape == (n_series, 40, horizon)
        assert fitted.truths_calibration_.shape == (n_series, 40, horizon)

    def test_missing_args_raise(self) -> None:
        adapter = _point_adapter()
        method = SplitConformal(alpha=0.1)
        with pytest.raises(ValueError):
            method.calibrate(adapter, None, None)
        with pytest.raises(ValueError):
            method.calibrate(adapter, _make_histories_truths(2, 3, 40)[0], None)

    def test_insufficient_samples_from_fit(self) -> None:
        # alpha=0.1 needs ceil(1/0.1)=10; give 5 windows.
        histories, truths = _make_histories_truths(2, 3, 5)
        adapter = _point_adapter()
        method = SplitConformal(alpha=0.1)
        with pytest.raises(InsufficientCalibrationSamples):
            method.calibrate(adapter, histories, truths)


# ---------------------------------------------------------------------------
# calibrate_cv (native cross-validation)
# ---------------------------------------------------------------------------


class TestCalibrateCV:
    def _cv_adapter(self, n_series=2, horizon=3, n_windows=40, seed=1):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(seed)
        preds = rng.normal(size=(n_series, n_windows, horizon))
        truths = preds + rng.normal(size=(n_series, n_windows, horizon))
        return _CVAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], horizon, axis=1),
            horizon=horizon,
            n_series=n_series,
            preds=preds,
            truths=truths,
        )

    def test_cv_path_dispatches(self) -> None:
        adapter = self._cv_adapter()
        fitted = SplitConformal(alpha=0.1).calibrate_cv(adapter, n_windows=40)
        assert isinstance(fitted, FixedOffsetForecaster)
        assert fitted.forecaster_ is adapter
        assert fitted.calibration_.diagnostics["path"] == "cross_validation"
        assert fitted.calibration_.n_calibration_samples.shape == (2,)

    def test_cv_on_non_cv_adapter_raises_unsupported(self) -> None:
        adapter = _point_adapter()
        method = SplitConformal(alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate_cv(adapter, n_windows=40)


# ---------------------------------------------------------------------------
# predict / update delegation
# ---------------------------------------------------------------------------


class TestPredictUpdate:
    def test_predict_after_a_bare_fit_raises(self) -> None:
        rng = np.random.default_rng(0)
        fitted = SplitConformal(alpha=0.1).fit(
            rng.normal(size=(2, 40, 3)), rng.normal(size=(2, 40, 3))
        )
        with pytest.raises(CalibrationError, match="calibrate"):
            fitted.predict(np.zeros((2, 20)))

    def test_predict_shape(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = SplitConformal(alpha=0.1).calibrate(adapter, histories, truths)
        result = fitted.predict(np.zeros((n_series, 20)))
        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert (result.interval[..., 0] <= result.interval[..., 1]).all()

    def test_update_extends_online_history(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = AdaptiveConformalInference(alpha=0.1).calibrate(adapter, histories, truths)
        assert isinstance(fitted, ACIForecaster)
        before = fitted.scores_.shape[1]
        fitted.update(np.zeros((n_series, 1, horizon)), np.full((n_series, 1, horizon), 5.0))
        assert fitted.scores_.shape[1] == before + 1

    def test_update_noop_for_offline(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = SplitConformal(alpha=0.1).calibrate(adapter, histories, truths)
        snapshot = fitted.model_copy(deep=True)
        assert (
            fitted.update(np.zeros((n_series, 1, horizon)), np.zeros((n_series, 1, horizon)))
            is None
        )
        assert fitted == snapshot

    def test_a_method_has_no_update_or_predict(self) -> None:
        """Using a method before fit is unrepresentable rather than guarded."""
        method = AdaptiveConformalInference(alpha=0.1)
        for name in ("update", "predict", "interval", "recalibrate"):
            assert not hasattr(method, name)


class TestCalibrationIntervals:
    def test_returns_interval_shape(self) -> None:
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = SplitConformal(alpha=0.1).calibrate(adapter, histories, truths)
        intervals = fitted._calibration_intervals()
        assert intervals.shape == (n_series, 40, horizon, 2)

    def test_default_broadcasts_interval(self) -> None:
        """The base default is exactly ``interval`` applied to every window."""
        n_series, horizon = 2, 3
        histories, truths = _make_histories_truths(n_series, horizon, 40)
        adapter = _point_adapter(n_series, horizon)
        fitted = SplitConformal(alpha=0.1).calibrate(adapter, histories, truths)
        preds = fitted.predictions_calibration_
        np.testing.assert_array_equal(fitted.calibration_intervals(preds), fitted.interval(preds))


# ---------------------------------------------------------------------------
# Exception rename
# ---------------------------------------------------------------------------


class TestExceptionRename:
    def test_conformalforecasterror_importable(self) -> None:
        from conformalforecast.base import ConformalForecastError

        assert issubclass(CalibrationError, ConformalForecastError)
        assert issubclass(UnsupportedCapability, ConformalForecastError)

    def test_old_name_gone(self) -> None:
        from conformalforecast import base

        assert not hasattr(base, "ConformalTSError")


# ---------------------------------------------------------------------------
# Ragged left-NaN history validation
# ---------------------------------------------------------------------------


class TestRaggedHistory:
    def _adapter(self, n_series: int) -> CallableAdapter:
        return CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], 2)), horizon=2, n_series=n_series
        )

    def test_strict_default_rejects_any_nan(self) -> None:
        adapter = self._adapter(1)
        arr = np.array([[1.0, np.nan, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            adapter._validate_history(arr)

    def test_ragged_allows_leading_nan(self) -> None:
        adapter = self._adapter(2)
        arr = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]])
        out = adapter._validate_history(arr, allow_ragged=True)
        assert out.shape == (2, 3)
        starts, lengths = adapter._ragged_ranges(out)
        np.testing.assert_array_equal(starts, [0, 1])
        np.testing.assert_array_equal(lengths, [3, 2])

    def test_ragged_rejects_interior_nan(self) -> None:
        adapter = self._adapter(1)
        arr = np.array([[1.0, np.nan, 3.0]])
        with pytest.raises(ValueError, match="interior|trailing"):
            adapter._validate_history(arr, allow_ragged=True)

    def test_ragged_rejects_trailing_nan(self) -> None:
        adapter = self._adapter(1)
        arr = np.array([[1.0, 2.0, np.nan]])
        with pytest.raises(ValueError, match="interior|trailing"):
            adapter._validate_history(arr, allow_ragged=True)

    def test_ragged_names_offending_series(self) -> None:
        adapter = self._adapter(2)
        arr = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
        with pytest.raises(ValueError, match="1"):
            adapter._validate_history(arr, allow_ragged=True)


# ---------------------------------------------------------------------------
# Exogenous X threading
# ---------------------------------------------------------------------------


class TestExogThreading:
    def test_calibrate_cv_with_x_raises(self) -> None:
        # Build any method on a CV-capable stub; a non-None X must be rejected.
        adapter = CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], 2)), horizon=2, n_series=1
        )
        method = SplitConformal(alpha=0.1)
        with pytest.raises(NotImplementedError, match="exogenous"):
            method.calibrate_cv(adapter, 5, X=np.ones((1, 5, 2, 1)))


class TestResultTypes:
    """The two result types are models, so equality must not raise on arrays."""

    def test_results_compare_without_raising(self) -> None:
        a = CalibrationResult(
            n_calibration_samples=np.array([20, 20]),
            diagnostics={"score_quantile": np.ones((2, 3))},
        )
        b = CalibrationResult(
            n_calibration_samples=np.array([20, 20]),
            diagnostics={"score_quantile": np.ones((2, 3))},
        )
        c = CalibrationResult(
            n_calibration_samples=np.array([20, 19]),
            diagnostics={"score_quantile": np.ones((2, 3))},
        )
        assert a == b
        assert not (a == c)  # noqa: SIM201

    def test_prediction_result_fields_are_not_copied_or_upcast(self) -> None:
        point = np.zeros((2, 1, 3), dtype=np.float32)
        interval = np.zeros((2, 1, 3, 2), dtype=np.float32)
        result = PredictionResult(point=point, interval=interval, alpha=0.1)
        assert result.point is point
        assert result.interval.dtype == np.float32

    def test_results_reject_unknown_fields(self) -> None:
        """Both forbid extras, so a typo in a producer surfaces at construction."""
        with pytest.raises(ValidationError, match="typo"):
            PredictionResult(
                point=np.zeros((1, 1, 1)),
                interval=np.zeros((1, 1, 1, 2)),
                alpha=0.1,
                typo=1,
            )
        with pytest.raises(ValidationError, match="typo"):
            CalibrationResult(n_calibration_samples=np.array([5]), diagnostics={}, typo=1)

    @pytest.mark.parametrize("cls", [CalibrationResult, PredictionResult])
    def test_results_have_no_json_schema(self, cls: type[ArrayModel]) -> None:
        """Deliberately the one asymmetry with the six methods, and a hard one.

        A bare ``NDArray`` field reaches pydantic as an ``IsInstanceSchema``,
        which has no JSON schema — and a result type is all array payload, so
        unlike a method it has nothing to hide behind ``SkipJsonSchema``. This
        is why the two are absent from the round-trip table in
        ``test_pydantic_conversion.py``; give the arrays a serializer and this
        test fails, which is the moment to revisit that exclusion rather than
        silently invalidate it.
        """
        with pytest.raises(PydanticInvalidForJsonSchema):
            cls.model_json_schema()


class TestScoreFunctionIsAbstract:
    """Becoming an ``ArrayModel`` must not undo the ABC's construction guard."""

    def test_score_function_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            ScoreFunction()  # type: ignore[abstract]
