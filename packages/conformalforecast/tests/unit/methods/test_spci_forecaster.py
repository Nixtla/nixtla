"""Orchestration-layer plumbing tests for SequentialPredictiveConformalInference.

Statistical correctness (coverage, beta optimization, seed determinism, exog)
lives in ``tests/unit/methods/test_spci.py``. These tests check the
orchestration surface: calibrate / calibrate_cv dispatch, online ``update``
delegation, and a coverage smoke. The shared ``MockQuantileRegressor`` from
``tests/helpers.py`` keeps the suite runnable without ``quantile-forest``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import (
    CalibrationError,
    CalibrationResult,
    UnsupportedCapability,
)
from conformalforecast.capabilities import SupportsCrossValidation
from conformalforecast.methods.spci import SequentialPredictiveConformalInference, SPCIForecaster
from conformalforecast.nonconformity.absolute import AbsoluteResidual
from tests.helpers import MockQuantileRegressor


def _mean_adapter(n_series: int = 2, horizon: int = 3) -> CallableAdapter:
    return CallableAdapter(
        predict_fn=lambda h: np.repeat(h[:, -1:], horizon, axis=1),
        horizon=horizon,
        n_series=n_series,
    )


class _CVAdapter(CallableAdapter, SupportsCrossValidation):
    def __init__(self, predict_fn, horizon, n_series, preds, truths):  # type: ignore[no-untyped-def]
        super().__init__(predict_fn=predict_fn, horizon=horizon, n_series=n_series)
        self._preds = np.asarray(preds, dtype=np.float64)
        self._truths = np.asarray(truths, dtype=np.float64)

    def cross_validate(  # type: ignore[no-untyped-def]
        self, n_windows, step_size, refit=False, *, aligned=False
    ):
        return self._preds[:, :n_windows, :], self._truths[:, :n_windows, :]


def _histories_truths(n_series, horizon, n_windows, seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n_windows)]
    truths = rng.normal(size=(n_series, n_windows, horizon))
    return histories, truths


def _method(alpha=0.1, window_size=10, **kw):  # type: ignore[no-untyped-def]
    return SequentialPredictiveConformalInference(
        alpha=alpha, window_size=window_size, regressor=MockQuantileRegressor(), **kw
    )


class TestDelegation:
    def test_config_on_the_method(self) -> None:
        method = _method(window_size=7)
        assert "forecaster_" not in type(method).model_fields  # bound at calibrate time
        assert method.window_size == 7
        assert method.IS_ONLINE is True

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            _method(window_size=0)

    def test_non_signed_score_rejected_at_construction(self) -> None:
        """The score is config, so construction is where an incompatible one fails."""
        with pytest.raises(ValidationError, match="SignedResidual"):
            _method(score=AbsoluteResidual())


class TestCalibrateDispatch:
    def test_loop_path(self) -> None:
        histories, truths = _histories_truths(2, 3, 40)
        method = _method()
        adapter = _mean_adapter()
        fitted = method.calibrate(adapter, histories, truths)
        assert isinstance(fitted, SPCIForecaster)
        assert fitted.forecaster_ is adapter
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (2,)
        assert result.diagnostics["path"] == "loop"

    def test_cv_path(self) -> None:
        rng = np.random.default_rng(1)
        preds = rng.normal(size=(2, 40, 3))
        truths = preds + rng.normal(size=(2, 40, 3))
        adapter = _CVAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], 3, axis=1),
            horizon=3,
            n_series=2,
            preds=preds,
            truths=truths,
        )
        method = _method()
        fitted = method.calibrate_cv(adapter, n_windows=40)
        assert fitted.calibration_.diagnostics["path"] == "cross_validation"
        assert fitted.forecaster_ is adapter

    def test_cv_on_non_cv_adapter_raises_unsupported(self) -> None:
        method = _method()
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate_cv(_mean_adapter(), n_windows=20)


class TestPredictUpdate:
    def test_predict_after_a_bare_fit_raises(self) -> None:
        rng = np.random.default_rng(0)
        preds = rng.normal(size=(2, 40, 3))
        fitted = _method().fit(preds, preds + rng.normal(size=(2, 40, 3)))
        with pytest.raises(CalibrationError, match="calibrate"):
            fitted.predict(np.zeros((2, 20)))

    def test_predict_shape(self) -> None:
        histories, truths = _histories_truths(2, 3, 40)
        fitted = _method().calibrate(_mean_adapter(), histories, truths)
        result = fitted.predict(np.zeros((2, 20)))
        assert result.interval.shape == (2, 1, 3, 2)
        assert (result.interval[..., 0] <= result.interval[..., 1]).all()

    def test_update_delegates(self) -> None:
        histories, truths = _histories_truths(2, 3, 40)
        fitted = _method().calibrate(_mean_adapter(), histories, truths)
        before = fitted.residuals_.shape[1]
        fitted.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        assert fitted.residuals_.shape[1] == before + 1

    def test_recalibrate_returns_a_new_object(self) -> None:
        histories, truths = _histories_truths(2, 3, 40)
        fitted = _method().calibrate(_mean_adapter(), histories, truths)
        fitted.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        th, tt = _histories_truths(2, 3, 40, seed=1)
        again = fitted.recalibrate(th, tt)
        assert isinstance(again, SPCIForecaster)
        assert again is not fitted
        assert again.calibration_.diagnostics["path"] == "recalibrate"
        assert again.residuals_.shape[1] == 40
        assert fitted.residuals_.shape[1] == 41


class TestCoverageSmoke:
    def test_marginal_coverage(self) -> None:
        rng = np.random.default_rng(42)
        n_series, horizon, alpha = 2, 2, 0.1
        adapter = CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], horizon)),
            horizon=horizon,
            n_series=n_series,
        )
        histories = [rng.normal(size=(n_series, 20)) for _ in range(300)]
        truths = rng.normal(size=(n_series, 300, horizon))
        fitted = _method(window_size=20).calibrate(adapter, histories, truths)
        test_truths = rng.normal(size=(n_series, 300, horizon))
        result = fitted.predict(np.zeros((n_series, 20)))
        lo, hi = result.interval[..., 0], result.interval[..., 1]
        covered = (test_truths >= lo) & (test_truths <= hi)
        assert abs(covered.mean() - (1 - alpha)) < 0.05


class TestEvaluateCalibration:
    def test_evaluate_calibration_runs(self) -> None:
        from conformalforecast.diagnostics import evaluate_calibration

        histories, truths = _histories_truths(2, 3, 40)
        fitted = _method().calibrate(_mean_adapter(), histories, truths)
        report = evaluate_calibration(fitted)
        assert report is not None


def test_spci_end_to_end_with_exog() -> None:
    """SPCI + CallableAdapter + X flows end to end to shaped intervals."""
    import numpy as np

    from conformalforecast.adapters import CallableAdapter
    from conformalforecast.methods import SequentialPredictiveConformalInference as SPCI

    rng = np.random.default_rng(0)
    n_series, horizon, n_windows, n_feat = 2, 3, 40, 2

    def fn(h, X=None):  # callable that accepts (and here ignores) X
        return np.repeat(h[:, -1:], horizon, axis=1)

    adapter = CallableAdapter(predict_fn=fn, horizon=horizon, n_series=n_series)
    method = SPCI(alpha=0.1, window_size=10, regressor=MockQuantileRegressor())

    histories = [rng.normal(size=(n_series, 20)) for _ in range(n_windows)]
    truths = rng.normal(size=(n_series, n_windows, horizon))
    X_cal = rng.normal(size=(n_series, n_windows, horizon, n_feat))
    fitted = method.calibrate(adapter, histories, truths, X=X_cal)

    X_new = rng.normal(size=(n_series, 1, horizon, n_feat))
    result = fitted.predict(rng.normal(size=(n_series, 20)), X=X_new)
    assert result.interval.shape == (n_series, 1, horizon, 2)
    assert (result.interval[..., 0] <= result.interval[..., 1]).all()
