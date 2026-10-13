"""Orchestration-layer plumbing tests for NonexchangeableConformalPrediction.

Statistical correctness lives in ``tests/unit/methods/test_nexcp.py``. These
tests check the orchestration surface: calibrate / calibrate_cv dispatch,
online ``update`` delegation, and a coverage smoke.
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
from conformalforecast.methods.nexcp import NexCPForecaster, NonexchangeableConformalPrediction


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


class TestDelegation:
    def test_config_on_the_method(self) -> None:
        method = NonexchangeableConformalPrediction(alpha=0.1, rho=0.95)
        assert "forecaster_" not in type(method).model_fields  # bound at calibrate time
        assert method.rho == 0.95
        assert method.IS_ONLINE is True

    def test_invalid_rho(self) -> None:
        with pytest.raises(ValidationError, match="rho"):
            NonexchangeableConformalPrediction(alpha=0.1, rho=1.5)


class TestCalibrateDispatch:
    def test_loop_path(self) -> None:
        histories, truths = _histories_truths(2, 3, 60)
        method = NonexchangeableConformalPrediction(alpha=0.1)
        adapter = _mean_adapter()
        fitted = method.calibrate(adapter, histories, truths)
        assert isinstance(fitted, NexCPForecaster)
        assert fitted.forecaster_ is adapter
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (2,)
        assert result.diagnostics["path"] == "loop"

    def test_cv_path(self) -> None:
        rng = np.random.default_rng(1)
        preds = rng.normal(size=(2, 60, 3))
        truths = preds + rng.normal(size=(2, 60, 3))
        adapter = _CVAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], 3, axis=1),
            horizon=3,
            n_series=2,
            preds=preds,
            truths=truths,
        )
        method = NonexchangeableConformalPrediction(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=60)
        assert fitted.forecaster_ is adapter
        assert fitted.calibration_.diagnostics["path"] == "cross_validation"

    def test_cv_on_non_cv_adapter_raises_unsupported(self) -> None:
        method = NonexchangeableConformalPrediction(alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate_cv(_mean_adapter(), n_windows=20)


class TestPredictUpdate:
    def test_predict_after_bare_fit_raises(self) -> None:
        histories, truths = _histories_truths(2, 3, 60)
        fitted = NonexchangeableConformalPrediction(alpha=0.1).fit(truths, truths)
        assert fitted.forecaster_ is None
        with pytest.raises(CalibrationError, match="calibrate"):
            fitted.predict(np.zeros((2, 20)))

    def test_predict_shape(self) -> None:
        histories, truths = _histories_truths(2, 3, 60)
        method = NonexchangeableConformalPrediction(alpha=0.1)
        fitted = method.calibrate(_mean_adapter(), histories, truths)
        result = fitted.predict(np.zeros((2, 20)))
        assert result.interval.shape == (2, 1, 3, 2)
        assert (result.interval[..., 0] <= result.interval[..., 1]).all()

    def test_update_delegates(self) -> None:
        histories, truths = _histories_truths(2, 3, 60)
        method = NonexchangeableConformalPrediction(alpha=0.1)
        fitted = method.calibrate(_mean_adapter(), histories, truths)
        before = fitted.n_observations_
        fitted.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        assert fitted.n_observations_ == before + 1

    def test_recalibrate_returns_a_new_object(self) -> None:
        histories, truths = _histories_truths(2, 3, 60)
        adapter = _mean_adapter()
        fitted = NonexchangeableConformalPrediction(alpha=0.1).calibrate(adapter, histories, truths)
        fitted.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        th, tt = _histories_truths(2, 3, 60, seed=1)
        again = fitted.recalibrate(th, tt)
        assert isinstance(again, NexCPForecaster)
        assert again is not fitted
        assert again.forecaster_ is adapter
        assert again.calibration_.diagnostics["path"] == "recalibrate"
        assert again.n_observations_ == 60  # update() history does not carry over
        assert fitted.n_observations_ == 61  # receiver untouched


class TestCoverageSmoke:
    def test_marginal_coverage(self) -> None:
        rng = np.random.default_rng(42)
        n_series, horizon, alpha = 3, 4, 0.1
        adapter = CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], horizon)),
            horizon=horizon,
            n_series=n_series,
        )
        histories = [rng.normal(size=(n_series, 20)) for _ in range(500)]
        truths = rng.normal(size=(n_series, 500, horizon))
        method = NonexchangeableConformalPrediction(alpha=alpha, rho=1.0)
        fitted = method.calibrate(adapter, histories, truths)
        test_truths = rng.normal(size=(n_series, 500, horizon))
        result = fitted.predict(np.zeros((n_series, 20)))
        lo, hi = result.interval[..., 0], result.interval[..., 1]
        covered = (test_truths >= lo) & (test_truths <= hi)
        assert abs(covered.mean() - (1 - alpha)) < 0.05


class TestEvaluateCalibration:
    def test_evaluate_calibration_runs(self) -> None:
        from conformalforecast.diagnostics import evaluate_calibration

        histories, truths = _histories_truths(2, 3, 60)
        method = NonexchangeableConformalPrediction(alpha=0.1)
        fitted = method.calibrate(_mean_adapter(), histories, truths)
        report = evaluate_calibration(fitted)
        assert report is not None
