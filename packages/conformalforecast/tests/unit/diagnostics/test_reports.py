"""Tests for diagnostics.reports."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel, PydanticInvalidForJsonSchema, ValidationError

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import CalibrationError
from conformalforecast.diagnostics.reports import Report, evaluate
from conformalforecast.diagnostics.scoring import winkler_score
from conformalforecast.methods.aci import AdaptiveConformalInference
from conformalforecast.methods.split import SplitConformal


def _build_report() -> Report:
    return Report(
        method_name="SplitConformal",
        alpha=0.1,
        n_holdout_samples=5,
        marginal_coverage=0.9,
        coverage_by_horizon=np.array([0.9, 0.9]),
        coverage_by_series=np.array([0.9]),
        coverage_per_cell=np.array([[0.9, 0.9]]),
        mean_interval_width=1.0,
        mean_winkler_score=2.0,
    )


class TestReportIsAModel:
    """``Report`` became an ``ArrayModel``; a dataclass's contract must still hold."""

    def test_report_is_a_model_and_compares_without_raising(self) -> None:
        a = _build_report()
        b = _build_report()
        assert isinstance(a, BaseModel)
        assert a == b

    def test_report_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="typo"):
            Report(
                method_name="x",
                alpha=0.1,
                n_holdout_samples=1,
                marginal_coverage=0.9,
                coverage_by_horizon=np.zeros(1),
                coverage_by_series=np.zeros(1),
                coverage_per_cell=np.zeros((1, 1)),
                mean_interval_width=1.0,
                mean_winkler_score=1.0,
                typo=1,
            )

    def test_report_has_no_json_schema(self) -> None:
        """Same asymmetry as CalibrationResult/PredictionResult: an all-array

        result type has a bare ``NDArray`` field, which reaches pydantic as
        an ``IsInstanceSchema`` with no JSON schema. See
        ``test_results_have_no_json_schema`` in ``tests/unit/test_base.py``.
        """
        with pytest.raises(PydanticInvalidForJsonSchema):
            Report.model_json_schema()


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


class TestEvaluateSplit:
    def test_basic_split_conformal(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(alpha=0.1)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 200)
        fitted = method.calibrate(adapter, cal_h, cal_t)

        ho_h, ho_t = _make_iid(rng, n_series, horizon, 200)
        report = evaluate(fitted, ho_h, ho_t)

        assert isinstance(report, Report)
        assert report.method_name == "SplitConformal"
        assert report.alpha == 0.1
        assert report.n_holdout_samples == 200
        # Marginal coverage should be near 1 - alpha on iid data.
        assert abs(report.marginal_coverage - 0.9) < 0.06
        assert report.coverage_by_horizon.shape == (horizon,)
        assert report.coverage_by_series.shape == (n_series,)
        assert report.coverage_per_cell.shape == (n_series, horizon)
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert report.truths.shape == (n_series, 200, horizon)
        assert report.points.shape == (n_series, 200, horizon)
        # Method-specific state is empty for SplitConformal.
        assert report.method_state == {}

    def test_evaluate_aci_updates_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 100)
        fitted = method.calibrate(adapter, cal_h, cal_t)

        n_obs_before = fitted.n_observations_
        ho_h, ho_t = _make_iid(rng, n_series, horizon, 50)
        report = evaluate(fitted, ho_h, ho_t)

        assert fitted.n_observations_ == n_obs_before + 50
        # method_state should be populated for ACI.
        assert "alpha_t" in report.method_state
        assert "gamma" in report.method_state

    def test_update_online_false_does_not_call_update(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 100)
        fitted = method.calibrate(adapter, cal_h, cal_t)
        n_obs_before = fitted.n_observations_

        ho_h, ho_t = _make_iid(rng, n_series, horizon, 50)
        evaluate(fitted, ho_h, ho_t, update_online=False)
        assert fitted.n_observations_ == n_obs_before

    def test_evaluate_mutates_in_place_but_not_a_prior_snapshot(self) -> None:
        """``evaluate`` updates an online fitted object in place, as a deployment
        would; a ``model_copy(deep=True)`` taken beforehand is untouched."""
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        fitted = method.calibrate(adapter, *_make_iid(rng, n_series, horizon, 100))
        snapshot = fitted.model_copy(deep=True)
        alpha_t_before = snapshot.alpha_t_.copy()
        scores_before = snapshot.scores_.copy()

        ho_h, ho_t = _make_iid(rng, n_series, horizon, 30)
        ho_t[:] = 1e6  # every step misses, so alpha_t certainly moves
        evaluate(fitted, ho_h, ho_t)

        assert fitted.n_observations_ == snapshot.n_observations_ + 30
        assert not np.array_equal(fitted.alpha_t_, alpha_t_before)
        np.testing.assert_array_equal(snapshot.alpha_t_, alpha_t_before)
        np.testing.assert_array_equal(snapshot.scores_, scores_before)
        assert snapshot.n_observations_ == 100


class TestEvaluateErrors:
    def test_method_rather_than_fitted_raises(self) -> None:
        """Passing the method instead of what ``calibrate`` returned."""
        method = SplitConformal(alpha=0.1)
        ho_h = [np.zeros((1, 5))]
        ho_t = np.zeros((1, 1, 1))
        with pytest.raises(CalibrationError, match="calibrate"):
            evaluate(method, ho_h, ho_t)  # type: ignore[arg-type]

    def test_non_forecaster_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="ConformalForecaster"):
            evaluate(object(), [np.zeros((1, 5))], np.zeros((1, 1, 1)))  # type: ignore[arg-type]

    def test_shape_mismatch_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 50)
        fitted = method.calibrate(adapter, cal_h, cal_t)

        # 3 histories but holdout_truths claims 5 samples.
        ho_h = [np.zeros((1, 5)) for _ in range(3)]
        ho_t = np.zeros((1, 5, 1))
        with pytest.raises(ValueError, match="holdout_histories"):
            evaluate(fitted, ho_h, ho_t)

    def test_holdout_truths_wrong_ndim_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 50)
        fitted = method.calibrate(adapter, cal_h, cal_t)
        ho_h = [np.zeros((1, 5))]
        bad_truths = np.zeros((1, 1))  # 2-D
        with pytest.raises(ValueError, match="holdout_truths"):
            evaluate(fitted, ho_h, bad_truths)


class TestGappyHoldoutAggregates:
    """A holdout truth can be missing while the interval around it is fine.

    A series that stopped reporting mid-holdout still gets a prediction, so the
    bounds are finite and only the target is NaN. Coverage and Winkler must then
    describe exactly the cells ``n_evaluated_cells`` counts.
    """

    def _report_over_gappy_holdout(self) -> Report:
        rng = np.random.default_rng(0)
        n_series, horizon, n = 2, 3, 60
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate(adapter, *_make_iid(rng, n_series, horizon, 200))

        ho_h, ho_t = _make_iid(rng, n_series, horizon, n)
        ho_t[1, 40:, :] = np.nan  # series 1 stops reporting partway through
        return evaluate(fitted, ho_h, ho_t)

    def test_mean_winkler_matches_evaluated_cells(self) -> None:
        report = self._report_over_gappy_holdout()
        evaluable = ~np.isnan(report.truths)
        assert report.n_evaluated_cells == int(np.count_nonzero(evaluable))
        assert report.n_evaluated_cells < report.n_total_cells

        expected = winkler_score(report.intervals, report.truths, report.alpha)[evaluable].mean()
        assert report.mean_winkler_score == pytest.approx(float(expected))

    def test_summary_reports_the_excluded_cells(self) -> None:
        assert "evaluated cells:" in self._report_over_gappy_holdout().summary()

    def test_coverage_is_not_depressed_by_the_gap(self) -> None:
        """The gap must not read as 60 misses."""
        report = self._report_over_gappy_holdout()
        assert abs(report.marginal_coverage - 0.9) < 0.1


class TestReportSummary:
    def test_summary_contains_key_info(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 100)
        fitted = method.calibrate(adapter, cal_h, cal_t)
        ho_h, ho_t = _make_iid(rng, 1, 1, 50)
        report = evaluate(fitted, ho_h, ho_t)

        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "SplitConformal" in text
        assert f"{report.marginal_coverage:.3f}" in text
