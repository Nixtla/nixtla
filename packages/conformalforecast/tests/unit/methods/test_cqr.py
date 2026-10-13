"""Tests for ConformalizedQuantileRegression's array surface.

Quantile-prediction input, coverage, shapes, CONSUMES_QUANTILES.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.base import CalibrationResult
from conformalforecast.methods.cqr import ConformalizedQuantileRegression
from conformalforecast.methods.fixed_offset import FixedOffsetForecaster


def _quantile_data(n_series, n_windows, horizon, z=1.0, seed=0):
    """Return ``(quantile_predictions (.., 2), truths)`` for a constant band.

    The band ``[-z, +z]`` is deliberately narrow so the conformal step must
    widen it to reach the nominal level.
    """
    rng = np.random.default_rng(seed)
    truths = rng.normal(size=(n_series, n_windows, horizon))
    q_lo = np.full_like(truths, -z)
    q_hi = np.full_like(truths, z)
    qp = np.stack([q_lo, q_hi], axis=-1)
    return qp, truths


def _cqr(alpha: float = 0.1) -> ConformalizedQuantileRegression:
    return ConformalizedQuantileRegression(alpha=alpha)


class TestContract:
    def test_consumes_quantiles(self):
        assert ConformalizedQuantileRegression.CONSUMES_QUANTILES is True

    def test_quantile_levels_derived_from_alpha(self):
        np.testing.assert_allclose(_cqr(0.1).quantile_levels, [0.05, 0.95])

    def test_quantile_levels_follow_alpha_through_model_copy(self):
        """``model_copy`` skips validation, so the levels must not be stored."""
        copied = _cqr(0.2).model_copy(update={"alpha": 0.1})
        np.testing.assert_allclose(copied.quantile_levels, [0.05, 0.95])

    def test_symmetric_dump_reloads_at_another_alpha(self):
        dumped = _cqr(0.2).model_dump()
        assert dumped["alpha_lo"] is None
        assert dumped["alpha_hi"] is None
        reloaded = ConformalizedQuantileRegression.model_validate({**dumped, "alpha": 0.1})
        np.testing.assert_allclose(reloaded.quantile_levels, [0.05, 0.95])

    @pytest.mark.parametrize("via", ["model_copy", "assignment"])
    def test_asymmetric_without_levels_raises(self, via):
        """Validation-skipping paths leave the levels unset; the error says so."""
        if via == "model_copy":
            m = _cqr(0.1).model_copy(update={"symmetric": False})
        else:
            m = _cqr(0.1)
            m.symmetric = False
        with pytest.raises(ValueError, match="skipped validation"):
            _ = m.quantile_levels

    def test_is_offline(self):
        assert _cqr().IS_ONLINE is False

    def test_point_forecasts_rejected(self):
        """``fit`` takes the 4-D quantile cube positionally; 3-D is refused."""
        _, truths = _quantile_data(2, 50, 3)
        with pytest.raises(ValueError, match="4-D"):
            _cqr().fit(truths, truths)

    def test_update_is_noop(self):
        qp, truths = _quantile_data(2, 50, 3)
        fitted = _cqr().fit(qp, truths)
        assert fitted.update(np.zeros((2, 1, 3)), np.zeros((2, 1, 3))) is None


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series, horizon):
        qp, truths = _quantile_data(n_series, 60, horizon)
        fitted = _cqr().fit(qp, truths)
        assert isinstance(fitted, FixedOffsetForecaster)
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (n_series,)
        assert fitted.score_quantile_.shape == (n_series, horizon)
        iv = fitted.interval(qp)
        assert iv.shape == (n_series, 60, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_float32_preserved(self):
        qp, truths = _quantile_data(2, 60, 3)
        fitted = _cqr().fit(qp.astype(np.float32), truths.astype(np.float32))
        iv = fitted.interval(qp.astype(np.float32))
        assert iv.dtype == np.float32


class TestCoverage:
    def test_empirical_coverage_matches_alpha(self):
        # tol 0.03 matches tests/unit/methods/test_cqr_forecaster.py. Uses the
        # exact order-statistic conformal_quantile, never an interpolated one.
        qp_cal, truths_cal = _quantile_data(3, 500, 4, seed=1)
        qp_test, truths_test = _quantile_data(3, 500, 4, seed=2)
        fitted = _cqr().fit(qp_cal, truths_cal)
        iv = fitted.interval(qp_test)
        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.03


class TestDeterminism:
    def test_same_input_same_output(self):
        qp, truths = _quantile_data(3, 80, 4)
        k1 = _cqr().fit(qp, truths)
        k2 = _cqr().fit(qp, truths)
        np.testing.assert_array_equal(k1.score_quantile_, k2.score_quantile_)
        np.testing.assert_array_equal(k1.interval(qp), k2.interval(qp))


class TestLifecycle:
    def test_interval_rejects_point_forecasts(self):
        qp, truths = _quantile_data(2, 50, 3)
        fitted = _cqr().fit(qp, truths)
        with pytest.raises(ValueError, match="4-D"):
            fitted.interval(truths)

    def test_fit_leaves_method_untouched(self):
        qp, truths = _quantile_data(2, 50, 3)
        method = _cqr()
        fitted = method.fit(qp, truths)
        assert fitted.method_ is not method
        assert method == _cqr()

    def test_result_defensively_copied(self):
        qp, truths = _quantile_data(1, 50, 3)
        fitted = _cqr().fit(qp, truths)
        snap = fitted.calibration_.diagnostics["score_quantile"]
        assert isinstance(snap, np.ndarray)
        snap[:] = -999.0
        assert not np.any(fitted.score_quantile_ == -999.0)


class TestScoreGate:
    """CQR narrows the score to QuantileScore (or a subclass) at construction."""

    def test_point_score_rejected_at_construction(self):
        from conformalforecast.nonconformity.absolute import AbsoluteResidual

        with pytest.raises(ValidationError, match="QuantileScore"):
            ConformalizedQuantileRegression(alpha=0.1, score=AbsoluteResidual())

    def test_configured_score_is_copied_not_shared(self):
        """The fitted object fits a copy, so the configured score stays config."""
        from conformalforecast.nonconformity.quantile import QuantileScore

        qp, truths = _quantile_data(2, 50, 3)
        good = QuantileScore()
        method = ConformalizedQuantileRegression(alpha=0.1, score=good)
        fitted = method.fit(qp, truths)
        assert isinstance(fitted.score_fn_, QuantileScore)
        assert fitted.score_fn_ is not good
        assert method.score is good


class TestNaNMasking:
    def test_ragged_equals_per_series_dense_fit(self):
        # Fitting a ragged panel must equal fitting each series separately
        # on its dense sub-array (spec: NaN-masking test requirement),
        # mirroring tests/unit/methods/test_split.py's TestNaNMasking.
        rng = np.random.default_rng(7)
        horizon = 3
        lengths = [40, 25]  # series 1 is shorter
        n_windows = max(lengths)

        truths = np.full((2, n_windows, horizon), np.nan)
        qp = np.full((2, n_windows, horizon, 2), np.nan)
        for i, n in enumerate(lengths):
            truths[i, -n:] = rng.normal(size=(n, horizon))
            qp[i, -n:, :, 0] = -1.0
            qp[i, -n:, :, 1] = 1.0

        ragged = _cqr().fit(qp, truths)
        np.testing.assert_array_equal(ragged.calibration_.n_calibration_samples, lengths)

        for i, n in enumerate(lengths):
            dense = _cqr().fit(qp[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_array_equal(ragged.score_quantile_[i : i + 1], dense.score_quantile_)

    def test_insufficient_series_named(self):
        from conformalforecast.base import InsufficientCalibrationSamples

        qp, truths = _quantile_data(2, 20, 2)
        # Series 1 keeps only 5 effective windows < ceil(1/0.1) = 10.
        truths[1, :15] = np.nan
        qp[1, :15] = np.nan

        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            _cqr().fit(qp, truths)
        assert excinfo.value.series_indices == [1]


class TestCalibrationIntervals:
    """CQR widens ``calibration_intervals`` exactly as it widens ``interval``."""

    def test_matches_interval_on_the_same_quantile_predictions(self) -> None:
        qp, truths = _quantile_data(2, 60, 3)
        fitted = ConformalizedQuantileRegression(alpha=0.1).fit(qp, truths)
        np.testing.assert_array_equal(fitted.calibration_intervals(qp), fitted.interval(qp))
