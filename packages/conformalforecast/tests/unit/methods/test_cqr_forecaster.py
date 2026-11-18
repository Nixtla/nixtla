"""Orchestration-layer plumbing tests for ConformalizedQuantileRegression.

Statistical correctness (coverage, exact quantiles) lives in
``tests/unit/methods/test_cqr.py``. These tests check the orchestration
surface: the ``SupportsQuantiles`` construction gate, quantile-level derivation
(``symmetric`` / ``alpha_lo`` / ``alpha_hi``), calibrate / calibrate_cv dispatch
(via ``SupportsCrossValidationQuantiles``), and a coverage smoke.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import norm

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import (
    CalibrationError,
    CalibrationResult,
    UnsupportedCapability,
)
from conformalforecast.methods.cqr import ConformalizedQuantileRegression
from conformalforecast.methods.fixed_offset import FixedOffsetForecaster
from conformalforecast.nonconformity.quantile import QuantileScore
from tests.helpers import QuantileCallableAdapter, QuantileCVCallableAdapter


def _const_quantile_adapter(n_series: int, horizon: int) -> QuantileCallableAdapter:
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        z = norm.ppf(quantiles)
        last = history[:, -1]
        out = last[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis]
        return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

    return QuantileCallableAdapter(
        predict_fn=predict_fn,
        predict_quantiles_fn=predict_quantiles_fn,
        horizon=horizon,
        n_series=n_series,
    )


def _cv_adapter(n_series, horizon, cv_histories, cv_truths):  # type: ignore[no-untyped-def]
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        z = norm.ppf(quantiles)
        last = history[:, -1]
        out = last[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis]
        return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

    return QuantileCVCallableAdapter(
        predict_fn=predict_fn,
        predict_quantiles_fn=predict_quantiles_fn,
        horizon=horizon,
        n_series=n_series,
        cv_histories=cv_histories,
        cv_truths=cv_truths,
    )


class TestConstruction:
    def test_contract_attributes(self) -> None:
        method = ConformalizedQuantileRegression(alpha=0.1)
        assert method.IS_ONLINE is False
        assert method.CONSUMES_QUANTILES is True

    def test_capability_gate_point_only_adapter(self) -> None:
        """The gate fires where the forecaster is bound, not at construction."""
        adapter = CallableAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], 1, axis=1), horizon=1, n_series=1
        )
        method = ConformalizedQuantileRegression(alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsQuantiles"):
            method.calibrate(adapter, [np.ones((1, 10))] * 20, np.zeros((1, 20, 1)))

    def test_default_symmetric_quantiles(self) -> None:
        method = ConformalizedQuantileRegression(alpha=0.2)
        assert method.symmetric is True
        np.testing.assert_array_almost_equal(method.quantile_levels, [0.1, 0.9])

    def test_symmetric_leaves_the_level_fields_unset(self) -> None:
        method = ConformalizedQuantileRegression(alpha=0.1)
        assert method.alpha_lo is None
        assert method.alpha_hi is None
        assert np.allclose(method.quantile_levels, [0.05, 0.95])

    def test_asymmetric_quantiles(self) -> None:
        method = ConformalizedQuantileRegression(
            alpha=0.2, symmetric=False, alpha_lo=0.05, alpha_hi=0.85
        )
        np.testing.assert_array_almost_equal(method.quantile_levels, [0.05, 0.85])

    def test_valid_asymmetric_levels_are_accepted(self) -> None:
        method = ConformalizedQuantileRegression(
            alpha=0.1, symmetric=False, alpha_lo=0.02, alpha_hi=0.92
        )
        assert np.allclose(method.quantile_levels, [0.02, 0.92])

    def test_symmetric_with_alpha_lo_raises(self) -> None:
        with pytest.raises(ValidationError, match="symmetric=True"):
            ConformalizedQuantileRegression(alpha=0.2, symmetric=True, alpha_lo=0.1)

    def test_symmetric_rejects_explicit_levels(self) -> None:
        with pytest.raises(ValidationError, match="cannot be specified when symmetric=True"):
            ConformalizedQuantileRegression(alpha=0.1, alpha_lo=0.02, alpha_hi=0.92)

    def test_symmetric_consistent_levels_are_normalized_to_none(self) -> None:
        """The acceptance check is ``np.isclose``, not equality.

        A tolerated pair is not stored: the levels are derived from ``alpha``,
        so the stored config is the same as if the pair had been omitted.
        """
        method = ConformalizedQuantileRegression(
            alpha=0.1, symmetric=True, alpha_lo=0.0500001, alpha_hi=0.9499999
        )
        assert method == ConformalizedQuantileRegression(alpha=0.1)
        assert np.allclose(method.quantile_levels, [0.05, 0.95])

    def test_asymmetric_missing_hi_raises(self) -> None:
        with pytest.raises(ValidationError, match="symmetric=False"):
            ConformalizedQuantileRegression(alpha=0.2, symmetric=False, alpha_lo=0.05)

    def test_asymmetric_requires_both_levels(self) -> None:
        with pytest.raises(ValidationError, match="must both be provided"):
            ConformalizedQuantileRegression(alpha=0.1, symmetric=False, alpha_lo=0.02)

    def test_asymmetric_levels_must_straddle_the_median(self) -> None:
        with pytest.raises(ValidationError, match="0 < alpha_lo < 0.5 < alpha_hi < 1"):
            ConformalizedQuantileRegression(alpha=0.1, symmetric=False, alpha_lo=0.6, alpha_hi=0.7)

    def test_asymmetric_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="must equal alpha"):
            ConformalizedQuantileRegression(alpha=0.1, symmetric=False, alpha_lo=0.1, alpha_hi=0.85)

    def test_asymmetric_levels_must_sum_to_alpha(self) -> None:
        with pytest.raises(ValidationError, match="must equal alpha"):
            ConformalizedQuantileRegression(
                alpha=0.1, symmetric=False, alpha_lo=0.02, alpha_hi=0.80
            )

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValidationError, match="alpha"):
            ConformalizedQuantileRegression(alpha=alpha)

    @pytest.mark.parametrize("level", [0.0, -0.1, 1.0, 1.5])
    def test_out_of_range_level_names_the_level(self, level: float) -> None:
        """A bad level is reported against its own field, not masked."""
        with pytest.raises(ValidationError, match="alpha_lo"):
            ConformalizedQuantileRegression(
                alpha=0.1, symmetric=False, alpha_lo=level, alpha_hi=0.95
            )

    def test_construction_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            ConformalizedQuantileRegression(0.1)  # type: ignore[misc]


class TestCalibrateDispatch:
    def test_loop_path(self) -> None:
        rng = np.random.default_rng(0)
        adapter = _const_quantile_adapter(2, 3)
        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        method = ConformalizedQuantileRegression(alpha=0.1)
        fitted = method.calibrate(adapter, histories, truths)
        assert isinstance(fitted, FixedOffsetForecaster)
        assert fitted.forecaster_ is adapter
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (2,)
        assert result.diagnostics["path"] == "loop"
        assert fitted.quantile_predictions_calibration_.shape == (2, 50, 3, 2)

    def test_cv_path(self) -> None:
        rng = np.random.default_rng(7)
        n_cal = 60
        histories = [rng.standard_normal((2, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((2, n_cal, 3))
        adapter = _cv_adapter(2, 3, histories, truths)
        method = ConformalizedQuantileRegression(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=n_cal)
        assert fitted.calibration_.diagnostics["path"] == "cross_validation"
        assert fitted.forecaster_ is adapter

    def test_cv_on_non_cv_adapter_raises_unsupported(self) -> None:
        # Loop-capable (SupportsQuantiles) but not SupportsCrossValidationQuantiles.
        method = ConformalizedQuantileRegression(alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidationQuantiles"):
            method.calibrate_cv(_const_quantile_adapter(1, 1), n_windows=20)

    def test_missing_args_raise(self) -> None:
        method = ConformalizedQuantileRegression(alpha=0.1)
        with pytest.raises(ValueError):
            method.calibrate(_const_quantile_adapter(1, 1))


class TestPredict:
    def test_predict_after_bare_fit_raises(self) -> None:
        rng = np.random.default_rng(0)
        truths = rng.standard_normal((1, 30, 1))
        qp = np.stack([truths - 1.0, truths + 1.0], axis=-1)
        fitted = ConformalizedQuantileRegression(alpha=0.1).fit(qp, truths)
        with pytest.raises(CalibrationError, match="calibrate"):
            fitted.predict(np.ones((1, 10)))

    def test_predict_shape(self) -> None:
        rng = np.random.default_rng(0)
        adapter = _const_quantile_adapter(2, 3)
        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        fitted = ConformalizedQuantileRegression(alpha=0.1).calibrate(adapter, histories, truths)
        result = fitted.predict(rng.standard_normal((2, 30)))
        assert result.point.shape == (2, 1, 3)
        assert result.interval.shape == (2, 1, 3, 2)
        assert (result.interval[..., 0] <= result.interval[..., 1]).all()


class TestCoverageSmoke:
    def test_marginal_coverage(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon, alpha = 1, 1, 0.1
        noise_std = 1.0
        T = 30
        signal = rng.standard_normal(n_series) * 5.0

        def predict_fn(history: np.ndarray) -> np.ndarray:
            return np.broadcast_to(signal, (n_series, horizon)).copy()

        def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
            z = norm.ppf(quantiles)
            out = signal[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis] * noise_std
            return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

        adapter = QuantileCallableAdapter(
            predict_fn=predict_fn,
            predict_quantiles_fn=predict_quantiles_fn,
            horizon=horizon,
            n_series=n_series,
        )
        histories = [
            signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T)) for _ in range(500)
        ]
        truths = np.stack(
            [
                signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
                for _ in range(500)
            ],
            axis=1,
        )
        fitted = ConformalizedQuantileRegression(alpha=alpha).calibrate(adapter, histories, truths)

        covered = 0
        total = 0
        for _ in range(500):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            iv = fitted.predict(history).interval
            truth_3d = truth[:, np.newaxis, :]
            covered += ((truth_3d >= iv[..., 0]) & (truth_3d <= iv[..., 1])).sum()
            total += truth_3d.size
        assert abs(covered / total - (1 - alpha)) < 0.03


class TestEvaluateCalibration:
    def test_evaluate_calibration_runs(self) -> None:
        from conformalforecast.diagnostics import evaluate_calibration

        rng = np.random.default_rng(0)
        adapter = _const_quantile_adapter(2, 3)
        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        fitted = ConformalizedQuantileRegression(alpha=0.1).calibrate(adapter, histories, truths)
        report = evaluate_calibration(fitted)
        assert report is not None


class _TaggedQuantileScore(QuantileScore):
    """A QuantileScore subclass, so the fitted score's type identifies its origin."""

    kind: Literal["cqr_forecaster_test_tagged"] = "cqr_forecaster_test_tagged"


class TestScoreForwarding:
    def test_calibrate_cv_uses_the_configured_score(self) -> None:
        """The quantile-CV path fits under the configured score — the least-covered site."""
        rng = np.random.default_rng(7)
        n_cal = 60
        histories = [rng.standard_normal((2, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((2, n_cal, 3))
        adapter = _cv_adapter(2, 3, histories, truths)
        score = _TaggedQuantileScore()
        method = ConformalizedQuantileRegression(alpha=0.1, score=score)

        fitted = method.calibrate_cv(adapter, n_windows=n_cal)
        assert type(fitted.score_fn_) is _TaggedQuantileScore
        assert fitted.score_fn_ is not score  # a copy: the config is never fitted
        assert method.score is score

    def test_incompatible_score_rejected_at_construction(self) -> None:
        from conformalforecast.nonconformity.absolute import AbsoluteResidual

        with pytest.raises(ValidationError, match="QuantileScore"):
            ConformalizedQuantileRegression(alpha=0.1, score=AbsoluteResidual())
