"""Tests for the ConformalMethod / ConformalForecaster contract."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import (
    CalibrationError,
    CalibrationResult,
    Exog,
    Forecast,
    Interval,
    ScoreFunction,
    UnsupportedCapability,
    as_forecast,
)
from conformalforecast.capabilities import SupportsQuantiles
from conformalforecast.methods.base import ConformalForecaster, ConformalMethod
from conformalforecast.methods.split import SplitConformal
from conformalforecast.nonconformity.absolute import AbsoluteResidual


def _adapter() -> CallableAdapter:
    """Minimal adapter: these contract tests never forecast."""
    return CallableAdapter(predict_fn=lambda h: h[:, -1:], horizon=1, n_series=1)


class _DummyForecaster(ConformalForecaster):
    """Minimal fitted class for contract tests: a fixed +/- 1 band."""

    method_: _DummyMethod

    def interval(self, predictions: Forecast, *, X: Exog | None = None) -> Interval:
        preds = np.asarray(predictions)
        return np.stack([preds - 1.0, preds + 1.0], axis=-1)


class _DummyMethod(ConformalMethod[_DummyForecaster]):
    """Minimal concrete method for contract tests."""

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def fit(
        self, predictions: Forecast, truths: Forecast, *, X: Exog | None = None
    ) -> _DummyForecaster:
        preds = np.asarray(predictions)
        score_fn = self._bind_score()
        score_fn.fit(preds, np.asarray(truths))
        n = np.full(preds.shape[0], preds.shape[1])
        return _DummyForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n,
            calibration_=CalibrationResult(n_calibration_samples=n.copy(), diagnostics={}),
        )


_DummyForecaster.model_rebuild()


class _CountingScore(AbsoluteResidual):
    """A stateful score: ``fit`` records how often it was called."""

    kind: Literal["counting_test"] = "counting_test"
    n_fits: int = 0

    def fit(self, prediction: Forecast, truth: Forecast) -> None:
        self.n_fits += 1


def _arrays(n_series: int = 1, n_windows: int = 5, horizon: int = 1) -> tuple[Any, Any]:
    return np.zeros((n_series, n_windows, horizon)), np.zeros((n_series, n_windows, horizon))


class TestConstructor:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _DummyMethod(alpha=alpha)

    def test_defaults(self) -> None:
        method = _DummyMethod(alpha=0.1)
        assert method.alpha == 0.1
        assert method.seed == 0
        assert method.score is None
        assert method.IS_ONLINE is False
        assert method.CONSUMES_QUANTILES is False
        assert method.SCORE_TYPE is ScoreFunction
        assert method.REQUIRED_CAPABILITIES == ()

    def test_custom_score_is_config(self) -> None:
        score = AbsoluteResidual()
        method = _DummyMethod(alpha=0.1, score=score)
        assert method.score is score
        fitted = method.fit(*_arrays())
        assert isinstance(fitted.score_fn_, AbsoluteResidual)
        # Deep-copied at fit: never shared with the method's config.
        assert fitted.score_fn_ is not score

    def test_score_type_is_checked_at_construction(self) -> None:
        from conformalforecast.nonconformity.signed import SignedResidual

        class _NeedsSignedResidual(_DummyMethod):
            SCORE_TYPE: ClassVar[type[ScoreFunction]] = SignedResidual

        with pytest.raises(ValidationError, match="SignedResidual"):
            _NeedsSignedResidual(alpha=0.1, score=AbsoluteResidual())
        _NeedsSignedResidual(alpha=0.1, score=SignedResidual())

    def test_score_must_consume_the_methods_prediction_shape(self) -> None:
        from conformalforecast.nonconformity.quantile import QuantileScore

        # QuantileScore passes the ScoreFunction gate but scores a quantile cube.
        with pytest.raises(ValidationError, match="consumes a point forecast"):
            _DummyMethod(alpha=0.1, score=QuantileScore())

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ConformalMethod(alpha=0.1)  # type: ignore[abstract]


class TestFitReturnsANewObject:
    def test_fit_returns_a_fitted_object_and_leaves_the_method_untouched(self) -> None:
        method = _DummyMethod(alpha=0.1)
        before = method.model_dump()
        fitted = method.fit(*_arrays())
        assert isinstance(fitted, _DummyForecaster)
        assert fitted is not method
        assert method.model_dump() == before
        assert method == type(method).model_validate(method.model_dump())

    def test_method_is_a_deep_copy_on_the_fitted_object(self) -> None:
        method = _DummyMethod(alpha=0.1)
        fitted = method.fit(*_arrays())
        assert fitted.method_ == method
        assert fitted.method_ is not method
        method.alpha = 0.3  # validate_assignment=False: a plain write
        assert fitted.method_.alpha == 0.1

    def test_two_fits_are_independent(self) -> None:
        method = _DummyMethod(alpha=0.1, score=_CountingScore())
        first = method.fit(*_arrays())
        second = method.fit(*_arrays(n_series=2))
        assert first is not second
        assert first.score_fn_ is not second.score_fn_
        assert first.n_calibration_samples_.shape == (1,)
        assert second.n_calibration_samples_.shape == (2,)
        assert first.score_fn_.n_fits == 1  # type: ignore[attr-defined]
        assert second.score_fn_.n_fits == 1  # type: ignore[attr-defined]

    def test_a_stateful_configured_score_is_not_mutated_by_calibrate(self) -> None:
        score = _CountingScore()
        method = _DummyMethod(alpha=0.1, score=score)
        histories = [np.arange(10.0).reshape(1, 10) for _ in range(5)]
        fitted = method.calibrate(_adapter(), histories, np.zeros((1, 5, 1)))
        assert fitted.score_fn_.n_fits == 1  # type: ignore[attr-defined]
        assert score.n_fits == 0
        assert method.score.n_fits == 0  # type: ignore[union-attr]


class TestUpdateDefault:
    def test_update_is_noop_for_an_offline_fitted_object(self) -> None:
        pred = np.zeros((1, 1, 3))
        truth = np.ones((1, 1, 3))
        fitted = _DummyMethod(alpha=0.1).fit(pred, truth)
        snapshot = fitted.model_copy(deep=True)
        assert fitted.update(pred, truth) is None
        assert fitted == snapshot


class TestCalibrationQuantiles:
    def test_default_raises(self) -> None:
        method = _DummyMethod(alpha=0.1)
        with pytest.raises(NotImplementedError, match="quantile levels"):
            method._calibration_quantiles()


class TestAsForecast:
    def test_float32_preserved(self) -> None:
        out = as_forecast(np.zeros((2, 3, 4), dtype=np.float32), "predictions")
        assert out.dtype == np.float32

    def test_int_upcast_to_float64(self) -> None:
        out = as_forecast(np.zeros((2, 3, 4), dtype=np.int64), "predictions")
        assert out.dtype == np.float64

    def test_list_coerced(self) -> None:
        out = as_forecast([[[1.0, 2.0]]], "predictions")
        assert out.shape == (1, 1, 2)

    def test_wrong_ndim_raises_with_name(self) -> None:
        with pytest.raises(ValueError, match="truths"):
            as_forecast(np.zeros((2, 3)), "truths")


class TestForecasterBinding:
    """The forecaster is bound at calibrate time, onto the returned object."""

    def test_predict_on_an_array_only_fitted_object_raises(self) -> None:
        """fit() takes arrays only, so predict() has no forecaster to call."""
        rng = np.random.default_rng(0)
        preds = rng.normal(size=(2, 30, 3))
        truths = preds + rng.normal(scale=0.1, size=(2, 30, 3))
        fitted = _DummyMethod(alpha=0.1).fit(preds, truths)

        assert fitted.forecaster_ is None
        with pytest.raises(CalibrationError, match="needs a forecaster"):
            fitted.predict(rng.normal(size=(2, 50)))

    def test_capability_failure_raises_at_calibrate_not_construction(self) -> None:
        """CQR needs SupportsQuantiles; a point-only adapter fails when bound."""
        from conformalforecast.methods import ConformalizedQuantileRegression

        method = ConformalizedQuantileRegression(alpha=0.1)  # constructs fine
        point_only = CallableAdapter(predict_fn=lambda h: h[:, -1:], horizon=1, n_series=1)

        with pytest.raises(UnsupportedCapability, match="SupportsQuantiles"):
            method.calibrate_cv(point_only, n_windows=20)

    def test_each_calibrate_returns_its_own_binding(self) -> None:
        method = _DummyMethod(alpha=0.1)
        first, second = _adapter(), _adapter()
        histories = [np.arange(10.0).reshape(1, 10) for _ in range(5)]
        truths = np.zeros((1, 5, 1))

        a = method.calibrate(first, histories, truths)
        b = method.calibrate(second, histories, truths)
        assert a.forecaster_ is first
        assert b.forecaster_ is second
        assert a.calibration_.diagnostics["path"] == "loop"

    def test_a_failed_calibrate_leaves_nothing_behind(self) -> None:
        """The gate runs first; an earlier fitted object is untouched."""
        from tests.helpers import QuantileCallableAdapter

        class _NeedsQuantiles(_DummyMethod):
            REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = (SupportsQuantiles,)

        method = _NeedsQuantiles(alpha=0.1)
        first = QuantileCallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            predict_quantiles_fn=lambda h, q: np.zeros((1, q.shape[0], 1)),
            horizon=1,
            n_series=1,
        )
        histories = [np.arange(10.0).reshape(1, 10) for _ in range(5)]
        truths = np.zeros((1, 5, 1))

        fitted = method.calibrate(first, histories, truths)
        before = method.model_dump()
        with pytest.raises(UnsupportedCapability, match="SupportsQuantiles"):
            method.calibrate(_adapter(), histories, truths)
        assert fitted.forecaster_ is first
        assert method.model_dump() == before


class TestRecalibrate:
    def _source(self) -> tuple[CallableAdapter, list[NDArray[np.floating]], Any]:
        adapter = CallableAdapter(predict_fn=lambda h: h[:, -1:], horizon=1, n_series=2)
        rng = np.random.default_rng(0)
        return adapter, [rng.normal(size=(2, 10)) for _ in range(30)], rng

    def test_recalibrate_returns_a_new_object_and_leaves_the_receiver(self) -> None:
        adapter, histories, rng = self._source()
        fitted = _DummyMethod(alpha=0.1).calibrate(adapter, histories, rng.normal(size=(2, 30, 1)))
        snapshot = fitted.model_copy(deep=True)
        target = fitted.recalibrate(
            [rng.normal(size=(2, 10)) for _ in range(30)], rng.normal(size=(2, 30, 1))
        )
        assert isinstance(target, _DummyForecaster)
        assert target is not fitted
        assert target.forecaster_ is adapter
        assert target.calibration_.diagnostics["path"] == "recalibrate"
        assert fitted == snapshot

    def test_recalibrate_on_an_array_only_fitted_object_raises(self) -> None:
        fitted = _DummyMethod(alpha=0.1).fit(*_arrays())
        with pytest.raises(CalibrationError, match="needs a forecaster"):
            fitted.recalibrate([np.zeros((1, 5))], np.zeros((1, 1, 1)))

    def test_drifted_online_state_does_not_survive_recalibrate(self) -> None:
        """The recalibrated object's online state comes from the target panel,
        not from whatever update() accumulated on the receiver."""
        from conformalforecast.methods import AdaptiveConformalInference

        adapter, histories, rng = self._source()
        fitted = AdaptiveConformalInference(alpha=0.1, gamma=0.05).calibrate(
            adapter, histories, rng.normal(size=(2, 30, 1))
        )

        after_calibrate = fitted.alpha_t_.copy()
        for _ in range(8):
            fitted.update(np.zeros((2, 1, 1)), np.full((2, 1, 1), 1000.0))
        drifted = fitted.alpha_t_.copy()
        assert not np.array_equal(drifted, after_calibrate)  # update() actually moved it

        target = fitted.recalibrate(
            [rng.normal(size=(2, 10)) for _ in range(30)], rng.normal(size=(2, 30, 1))
        )
        assert not np.array_equal(target.alpha_t_, drifted)
        np.testing.assert_array_equal(fitted.alpha_t_, drifted)  # receiver untouched


def _calibration_arrays() -> tuple[np.ndarray, np.ndarray]:
    """A dense 2-series calibration cube with skewed (asymmetric) residuals."""
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(2, 60, 3))
    truths = preds + rng.gamma(shape=2.0, scale=1.0, size=(2, 60, 3))
    return preds, truths


class TestScoreBoundFromConfig:
    def test_default_score_is_resolved_at_fit(self) -> None:
        from conformalforecast.nonconformity.signed import SignedResidual

        method = SplitConformal(alpha=0.1)
        assert method.score is None  # config stays as given

        preds, truths = _calibration_arrays()
        assert isinstance(method.fit(preds, truths).score_fn_, AbsoluteResidual)

        signed = SplitConformal(alpha=0.1, score=SignedResidual())
        assert isinstance(signed.fit(preds, truths).score_fn_, SignedResidual)

    def test_interval_uses_the_configured_score(self) -> None:
        """
        ``interval`` inverts with ``score_fn_``, not with the default: fitting
        under an asymmetric score gives an asymmetric interval where the
        default ``AbsoluteResidual`` gives a symmetric one.

        (``SignedResidual`` is not usable here — its ``invert`` needs a
        two-sided ``(n_series, horizon, 2)`` threshold, which only SPCI and
        asymmetric split produce. A minimal asymmetric score exercises the
        same wiring.)
        """

        class _Lopsided(AbsoluteResidual):
            """Absolute score whose inversion widens upward twice as far."""

            kind: Literal["lopsided"] = "lopsided"

            def invert(self, prediction, score_threshold):  # type: ignore[no-untyped-def]
                iv = super().invert(prediction, score_threshold)
                iv[..., 1] = prediction + 2.0 * (iv[..., 1] - prediction)
                return iv

        preds, truths = _calibration_arrays()
        new = preds[:, :1, :]

        symmetric = SplitConformal(alpha=0.1).fit(preds, truths).interval(new)
        assert np.allclose(new - symmetric[..., 0], symmetric[..., 1] - new)

        lopsided = SplitConformal(alpha=0.1, score=_Lopsided()).fit(preds, truths).interval(new)
        lower_width = new - lopsided[..., 0]
        upper_width = lopsided[..., 1] - new
        assert not np.allclose(lower_width, upper_width)
        np.testing.assert_allclose(upper_width, 2.0 * lower_width)


class TestPydanticModel:
    """The pydantic conversion: keyword-only construction and a config-only schema."""

    def test_construction_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            SplitConformal(0.1)  # type: ignore[misc]  # models take no positional args
        assert SplitConformal(alpha=0.1).alpha == 0.1

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_is_validated_declaratively(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SplitConformal(alpha=bad)

    def test_schema_describes_config_only(self) -> None:
        schema = SplitConformal.model_json_schema()
        assert set(schema["properties"]) == {
            "alpha",
            "seed",
            "score",
            "symmetric",
            "alpha_lo",
            "alpha_hi",
        }
        assert schema["properties"]["alpha"]["exclusiveMinimum"] == 0.0
        assert schema["properties"]["alpha"]["exclusiveMaximum"] == 1.0
        for name in ("alpha", "seed", "symmetric", "alpha_lo", "alpha_hi"):
            assert schema["properties"][name]["description"]
        assert not [name for name in schema["properties"] if name.endswith("_")]

    def test_class_metadata_is_not_a_field(self) -> None:
        for name in ("IS_ONLINE", "CONSUMES_QUANTILES", "SCORE_TYPE", "REQUIRED_CAPABILITIES"):
            assert name not in SplitConformal.model_fields
        assert SplitConformal.IS_ONLINE is False  # readable off the class
        assert SplitConformal.CONSUMES_QUANTILES is False

    def test_a_method_has_no_fitted_state(self) -> None:
        assert not [name for name in SplitConformal.model_fields if name.endswith("_")]

    def test_fitted_state_lives_on_the_returned_object(self) -> None:
        preds, truths = _calibration_arrays()
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        assert fitted.score_quantile_.shape == (2, 3)
        assert fitted.n_calibration_samples_.shape == (2,)
        assert fitted.predictions_calibration_ is None  # bare fit stores none

        calibrated = SplitConformal(alpha=0.1).calibrate(
            _adapter(), [np.zeros((1, 5))] * 20, np.zeros((1, 20, 1))
        )
        assert calibrated.predictions_calibration_ is not None
        assert calibrated.truths_calibration_ is not None
