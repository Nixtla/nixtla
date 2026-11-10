"""Tests for diagnostics.reports.evaluate_calibration (Mode 1)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import CalibrationError, Forecast, ForecasterAdapter, Series
from conformalforecast.capabilities import SupportsQuantiles
from conformalforecast.diagnostics import Report, evaluate, evaluate_calibration
from conformalforecast.methods.aci import AdaptiveConformalInference
from conformalforecast.methods.agaci import AggregatedAdaptiveConformalInference
from conformalforecast.methods.cqr import ConformalizedQuantileRegression
from conformalforecast.methods.nexcp import NonexchangeableConformalPrediction
from conformalforecast.methods.spci import SequentialPredictiveConformalInference
from conformalforecast.methods.split import SplitConformal
from tests.helpers import ConditionalMockQuantileRegressor, MockQuantileRegressor

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


class _QuantileCallableAdapter(ForecasterAdapter, SupportsQuantiles):
    """Quantile adapter mirroring the const-Gaussian style used in CQR tests."""

    def predict(self, history: Series) -> Forecast:
        self._validate_history(history)
        return np.zeros((self.n_series, 1, self.horizon))

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        self._validate_history(history)
        z = norm.ppf(quantiles)
        # Centered at 0; quantile = z (broadcast over series & horizon).
        out = np.broadcast_to(
            z[np.newaxis, :, np.newaxis], (self.n_series, quantiles.shape[0], self.horizon)
        )
        return np.asarray(out, dtype=np.float64).copy()


# ===========================================================================
# Per-method end-to-end
# ===========================================================================


class TestEvaluateCalibrationSplit:
    def test_basic(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(alpha=0.1)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert isinstance(report, Report)
        assert report.method_name == "SplitConformal (calibration)"
        assert report.alpha == 0.1
        assert report.n_holdout_samples == 200
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert report.truths.shape == (n_series, 200, horizon)
        assert report.points.shape == (n_series, 200, horizon)
        # Split CP achieves close to nominal in-sample (no time-varying state).
        assert abs(report.marginal_coverage - 0.9) < 0.05
        # Nothing is unevaluable, so the summary omits the excluded-cells line.
        assert report.n_evaluated_cells == report.n_total_cells
        assert "evaluated cells:" not in report.summary()
        # SplitConformal exposes no method-specific state.
        assert report.method_state == {}


class TestEvaluateCalibrationCQR:
    def test_basic(self) -> None:
        n_series, horizon = 1, 2
        adapter = _QuantileCallableAdapter(n_series=n_series, horizon=horizon)
        method = ConformalizedQuantileRegression(alpha=0.1)
        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((n_series, 20)) for _ in range(200)]
        truths = rng.standard_normal((n_series, 200, horizon))
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert report.method_name == "ConformalizedQuantileRegression (calibration)"
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert abs(report.marginal_coverage - 0.9) < 0.05


class TestEvaluateCalibrationACI:
    def test_basic_and_state_populated(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert report.method_name == "AdaptiveConformalInference (calibration)"
        assert "alpha_t" in report.method_state
        assert "gamma" in report.method_state
        # In-sample coverage uses the post-calibration alpha_t_ applied to all
        # samples; should land within a loose band of nominal.
        assert 0.5 < report.marginal_coverage < 1.0


class TestEvaluateCalibrationAgACI:
    def test_basic_and_weights_in_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(alpha=0.1, gammas=(0.01, 0.05, 0.1))
        histories, truths = _make_iid(rng, n_series, horizon, 100)
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert report.method_name == "AggregatedAdaptiveConformalInference (calibration)"
        assert "weights_lower" in report.method_state
        assert "weights_upper" in report.method_state
        assert report.intervals.shape == (n_series, 100, horizon, 2)


class TestEvaluateCalibrationNexCP:
    def test_basic_and_ess_in_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(alpha=0.1, rho=0.99)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert report.method_name == "NonexchangeableConformalPrediction (calibration)"
        assert "effective_sample_size" in report.method_state
        assert abs(report.marginal_coverage - 0.9) < 0.06


class TestEvaluateCalibrationSPCI:
    """SPCI's Mode 1 used to report ~23% where the others reported ~90%.

    Its offsets are conditional on the preceding residual window, so applying
    the final query's answer to every calibration window measured nothing. The
    method now replays each window against its own history.
    """

    WINDOW_SIZE = 10
    N_WINDOWS = 60

    def _calibrated(self, n_series: int = 1, horizon: int = 2):
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            alpha=0.1,
            window_size=self.WINDOW_SIZE,
            regressor=ConditionalMockQuantileRegressor(),
            beta_grid_size=5,
        )
        # Not i.i.d.: a random walk gives the residual stream the level shifts
        # SPCI conditions on. Under i.i.d. noise every query window has roughly
        # the same mean, so the broadcast bug and its fix agree and none of the
        # assertions below can tell them apart.
        histories = [np.zeros((n_series, 20)) for _ in range(self.N_WINDOWS)]
        level = np.cumsum(rng.normal(size=(n_series, self.N_WINDOWS, horizon)) * 0.8, axis=1)
        truths = level + rng.normal(size=(n_series, self.N_WINDOWS, horizon)) * 0.3
        fitted = method.calibrate(adapter, histories, truths)
        return fitted, n_series, horizon

    def test_basic(self) -> None:
        fitted, n_series, horizon = self._calibrated()
        report = evaluate_calibration(fitted)
        assert report.method_name == "SequentialPredictiveConformalInference (calibration)"
        assert "n_regressors" in report.method_state
        assert report.intervals.shape == (n_series, self.N_WINDOWS, horizon, 2)

    def test_coverage_is_no_longer_an_artifact(self) -> None:
        """The regression assertion the old smoke test lacked.

        Broadcasting the final query's offsets over this fixture scores 0.575 --
        the ROADMAP's 23.4% in miniature. Replayed per window it lands near
        target. The band is generous because the point is to separate "a real
        measurement" from "an artifact", not to pin a value. It pins the *mock*,
        not real SPCI: with a quantile forest the replay queries the rows the
        forests were fit on and lands near 100%, an in-sample number that does
        not track the target.
        """
        fitted, _, _ = self._calibrated()
        report = evaluate_calibration(fitted)
        assert abs(report.marginal_coverage - 0.9) < 0.1

    def test_warm_up_windows_are_excluded_not_missed(self) -> None:
        fitted, n_series, horizon = self._calibrated()
        report = evaluate_calibration(fitted)
        assert report.n_total_cells == n_series * self.N_WINDOWS * horizon
        excluded = report.n_total_cells - report.n_evaluated_cells
        assert excluded == n_series * self.WINDOW_SIZE * horizon

    def test_intervals_move_along_the_sample_axis(self) -> None:
        """The broadcast placed every window's interval at the same offset.

        Measured *within* each (series, horizon) cell: across cells the offsets
        differ either way, so a global spread would pass without the fix. The
        threshold is not ``> 0`` for the same reason -- the broadcast leaves
        ~1e-15 of float noise from ``preds + offset - preds``, where the replay
        moves the offset by ~1.4 on this fixture.
        """
        fitted, _, _ = self._calibrated()
        iv = fitted._calibration_intervals()
        offsets = iv[..., 0] - fitted.predictions_calibration_
        per_cell_spread = np.nanstd(offsets, axis=1)  # (n_series, horizon)
        assert np.all(per_cell_spread > 0.1)

    def test_summary_reports_the_excluded_cells(self) -> None:
        fitted, _, _ = self._calibrated()
        assert "evaluated cells:" in evaluate_calibration(fitted).summary()

    def test_unconditional_mock_still_runs(self) -> None:
        """The plain mock ignores its features; only the warm-up should be NaN."""
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor(), beta_grid_size=5
        )
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        fitted = method.calibrate(adapter, histories, truths)

        report = evaluate_calibration(fitted)
        assert report.n_evaluated_cells == n_series * (60 - 10) * horizon


# ===========================================================================
# Error paths
# ===========================================================================


class TestEvaluateCalibrationErrors:
    def test_method_rather_than_fitted_raises(self) -> None:
        """Passing the method instead of what ``calibrate`` returned."""
        method = SplitConformal(alpha=0.1)
        with pytest.raises(CalibrationError, match="calibrate"):
            evaluate_calibration(method)  # type: ignore[arg-type]

    def test_array_surface_fit_raises(self) -> None:
        """The real path: a bare ``fit`` never populates the retention fields.

        They are declared fields defaulting to None, so ``hasattr`` is always
        True — the guard must test the value, not the attribute's existence.
        """
        rng = np.random.default_rng(0)
        preds = rng.normal(size=(1, 50, 1))
        fitted = SplitConformal(alpha=0.1).fit(preds, preds + rng.normal(size=(1, 50, 1)))
        assert fitted.predictions_calibration_ is None
        with pytest.raises(RuntimeError, match="array surface"):
            evaluate_calibration(fitted)

    def test_missing_truths_calibration_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(alpha=0.1)
        histories, truths = _make_iid(rng, 1, 1, 50)
        fitted = method.calibrate(adapter, histories, truths)
        fitted.truths_calibration_ = None  # half-populated
        with pytest.raises(RuntimeError, match="calibration forecasts"):
            evaluate_calibration(fitted)


# ===========================================================================
# Cross-mode consistency
# ===========================================================================


class TestCrossModeConsistency:
    """Mode 1 and Mode 2 produce the same Report structure."""

    def test_report_fields_match(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(alpha=0.1)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        fitted = method.calibrate(adapter, histories, truths)
        ho_h, ho_t = _make_iid(rng, n_series, horizon, 100)

        report_cal = evaluate_calibration(fitted)
        report_ho = evaluate(fitted, ho_h, ho_t)

        # Same field names with the same shapes.
        assert report_cal.coverage_by_horizon.shape == report_ho.coverage_by_horizon.shape
        assert report_cal.coverage_by_series.shape == report_ho.coverage_by_series.shape
        # method_name differs only by the " (calibration)" suffix.
        assert report_cal.method_name.endswith("(calibration)")
        assert report_ho.method_name == "SplitConformal"
