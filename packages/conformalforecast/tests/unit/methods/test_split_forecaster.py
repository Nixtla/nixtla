"""Orchestration-layer plumbing tests for SplitConformal.

Statistical correctness (coverage, NaN masking, exact quantiles) is covered by
the array-surface tests in ``tests/unit/methods/test_split.py``. These tests
check that the orchestration surface wires the adapter to the calibration math
correctly: calibrate / calibrate_cv dispatch, the returned
``FixedOffsetForecaster`` and its ``CalibrationResult``, and one end-to-end
coverage smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import (
    CalibrationError,
    CalibrationResult,
    InsufficientCalibrationSamples,
    UnsupportedCapability,
)
from conformalforecast.capabilities import SupportsCrossValidation
from conformalforecast.methods.fixed_offset import FixedOffsetForecaster
from conformalforecast.methods.split import SplitConformal


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
        method = SplitConformal(alpha=0.2)
        assert "forecaster_" not in SplitConformal.model_fields  # bound on the fitted object
        assert method.alpha == 0.2
        assert method.IS_ONLINE is False

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            SplitConformal(alpha=1.5)


class TestCalibrateDispatch:
    def test_loop_path(self) -> None:
        histories, truths = _histories_truths(2, 3, 40)
        method = SplitConformal(alpha=0.1)
        adapter = _mean_adapter()
        fitted = method.calibrate(adapter, histories, truths)
        assert isinstance(fitted, FixedOffsetForecaster)
        assert fitted.forecaster_ is adapter
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (2,)
        assert result.diagnostics["path"] == "loop"
        assert method == SplitConformal(alpha=0.1)  # the method is never mutated

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
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=40)
        assert fitted.calibration_.diagnostics["path"] == "cross_validation"
        assert fitted.forecaster_ is adapter

    def test_cv_on_non_cv_adapter_raises_unsupported(self) -> None:
        method = SplitConformal(alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate_cv(_mean_adapter(), n_windows=20)

    def test_missing_args_raise(self) -> None:
        method = SplitConformal(alpha=0.1)
        with pytest.raises(ValueError):
            method.calibrate(_mean_adapter())

    def test_insufficient_samples(self) -> None:
        histories, truths = _histories_truths(2, 3, 5)
        method = SplitConformal(alpha=0.1)
        with pytest.raises(InsufficientCalibrationSamples):
            method.calibrate(_mean_adapter(), histories, truths)


class TestPredict:
    def test_predict_after_bare_fit_raises(self) -> None:
        rng = np.random.default_rng(0)
        preds = rng.normal(size=(2, 40, 3))
        fitted = SplitConformal(alpha=0.1).fit(preds, preds + rng.normal(size=(2, 40, 3)))
        assert fitted.forecaster_ is None
        with pytest.raises(CalibrationError, match="calibrate"):
            fitted.predict(np.zeros((2, 20)))

    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_predict_shape(self, n_series: int, horizon: int) -> None:
        histories, truths = _histories_truths(n_series, horizon, 40)
        fitted = SplitConformal(alpha=0.1).calibrate(
            _mean_adapter(n_series, horizon), histories, truths
        )
        result = fitted.predict(np.zeros((n_series, 20)))
        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert (result.interval[..., 0] <= result.interval[..., 1]).all()


class TestCoverageSmoke:
    def test_marginal_coverage(self) -> None:
        # Loose smoke check; rigorous coverage lives in test_split.py.
        rng = np.random.default_rng(42)
        n_series, horizon, alpha = 3, 4, 0.1
        adapter = CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], horizon)),
            horizon=horizon,
            n_series=n_series,
        )
        histories = [rng.normal(size=(n_series, 20)) for _ in range(500)]
        truths = rng.normal(size=(n_series, 500, horizon))
        fitted = SplitConformal(alpha=alpha).calibrate(adapter, histories, truths)

        test_truths = rng.normal(size=(n_series, 500, horizon))
        result = fitted.predict(np.zeros((n_series, 20)))
        lo, hi = result.interval[..., 0], result.interval[..., 1]
        covered = (test_truths >= lo) & (test_truths <= hi)
        assert abs(covered.mean() - (1 - alpha)) < 0.03


class TestEvaluateCalibration:
    def test_evaluate_calibration_runs(self) -> None:
        from conformalforecast.diagnostics import evaluate_calibration

        histories, truths = _histories_truths(2, 3, 60)
        fitted = SplitConformal(alpha=0.1).calibrate(_mean_adapter(), histories, truths)
        report = evaluate_calibration(fitted)
        assert report is not None


def test_asymmetric_config_survives_calibrate_and_predict() -> None:
    """The asymmetric interval must reach predict(), not just interval()."""
    m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
    histories, truths = _histories_truths(2, 3, 40)
    fitted = m.calibrate(_mean_adapter(), histories, truths)
    assert fitted.score_quantile_.shape == (2, 3, 2)
    out = fitted.predict(np.zeros((2, 20)))
    assert out.interval.shape == (2, 1, 3, 2)
    assert (out.interval[..., 0] <= out.interval[..., 1]).all()
