"""Tests for NonexchangeableConformalPrediction's array surface: shapes, coverage,
determinism, recency weighting, NaN masking.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.base import (
    CalibrationResult,
    InsufficientCalibrationSamples,
)
from conformalforecast.methods.nexcp import NexCPForecaster, NonexchangeableConformalPrediction
from conformalforecast.methods.split import SplitConformal


def _make_data(n_series, n_windows, horizon, seed=0):
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_series, n_windows, horizon))
    truths = preds + rng.normal(size=(n_series, n_windows, horizon))
    return preds, truths


class TestConstructor:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(ValidationError, match="alpha"):
            NonexchangeableConformalPrediction(alpha=alpha)

    @pytest.mark.parametrize("rho", [0.0, -0.1, 1.5])
    def test_rho_out_of_range_raises(self, rho):
        with pytest.raises(ValidationError, match="rho"):
            NonexchangeableConformalPrediction(alpha=0.1, rho=rho)

    def test_rho_one_valid(self):
        m = NonexchangeableConformalPrediction(alpha=0.1, rho=1.0)
        assert m.rho == 1.0

    def test_defaults(self):
        m = NonexchangeableConformalPrediction(alpha=0.1)
        assert m.rho == 0.99
        assert m.IS_ONLINE is True
        assert m.score is None


def test_rho_bounds():
    assert NonexchangeableConformalPrediction(alpha=0.1, rho=1.0).rho == 1.0  # 1 is allowed
    for bad in (0.0, -0.5, 1.1):
        with pytest.raises(ValidationError):
            NonexchangeableConformalPrediction(alpha=0.1, rho=bad)


def test_rho_is_described_in_the_schema():
    props = NonexchangeableConformalPrediction.model_json_schema()["properties"]
    assert props["rho"]["exclusiveMinimum"] == 0.0
    assert props["rho"]["maximum"] == 1.0
    assert "decay" in props["rho"]["description"].lower()


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series, horizon):
        preds, truths = _make_data(n_series, 60, horizon)
        m = NonexchangeableConformalPrediction(alpha=0.1)
        f = m.fit(preds, truths)
        assert isinstance(f, NexCPForecaster)
        assert isinstance(f.calibration_, CalibrationResult)
        assert f.calibration_.n_calibration_samples.shape == (n_series,)
        assert f.score_quantile_.shape == (n_series, horizon)
        new_preds = _make_data(n_series, 7, horizon, seed=1)[0]
        iv = f.interval(new_preds)
        assert iv.shape == (n_series, 7, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_float32_preserved(self):
        preds, truths = _make_data(2, 60, 3)
        m = NonexchangeableConformalPrediction(alpha=0.1)
        f = m.fit(preds.astype(np.float32), truths.astype(np.float32))
        assert f.interval(preds.astype(np.float32)).dtype == np.float32


class TestCoverage:
    @pytest.mark.parametrize("rho", [1.0, 0.99])
    def test_empirical_coverage_matches_alpha(self, rho):
        # tol 0.04 matches methods/tests/test_nexcp_forecaster.py.
        rng = np.random.default_rng(42)
        n_series, n_cal, n_test, horizon = 3, 500, 500, 4
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))
        m = NonexchangeableConformalPrediction(alpha=0.1, rho=rho)
        f = m.fit(preds_cal, truths_cal)
        preds_test = rng.normal(size=(n_series, n_test, horizon))
        truths_test = preds_test + rng.normal(size=(n_series, n_test, horizon))
        iv = f.interval(preds_test)
        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.04


class TestDeterminism:
    def test_same_config_same_arrays_same_output(self):
        preds, truths = _make_data(3, 80, 4)
        m1 = NonexchangeableConformalPrediction(alpha=0.1)
        m2 = NonexchangeableConformalPrediction(alpha=0.1)
        f1 = m1.fit(preds, truths)
        f2 = m2.fit(preds, truths)
        np.testing.assert_array_equal(f1.score_quantile_, f2.score_quantile_)
        new = _make_data(3, 5, 4, seed=9)[0]
        np.testing.assert_array_equal(f1.interval(new), f2.interval(new))


class TestRecencyWeighting:
    def test_rho_one_equals_split_quantile(self):
        preds, truths = _make_data(3, 200, 2)
        nex = NonexchangeableConformalPrediction(alpha=0.1, rho=1.0)
        split = SplitConformal(alpha=0.1)
        f_nex = nex.fit(preds, truths)
        f_split = split.fit(preds, truths)
        np.testing.assert_allclose(f_nex.score_quantile_, f_split.score_quantile_)

    def test_update_recomputes_quantile(self):
        preds, truths = _make_data(2, 120, 3)
        m = NonexchangeableConformalPrediction(alpha=0.1, rho=0.9)
        f = m.fit(preds, truths)
        before = f.score_quantile_.copy()
        for _ in range(10):
            assert f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 4.0)) is None
        assert f.n_observations_ == 130
        assert not np.allclose(before, f.score_quantile_)

    def test_deep_copy_snapshot_is_independent_of_later_updates(self):
        preds, truths = _make_data(2, 120, 3)
        f = NonexchangeableConformalPrediction(alpha=0.1, rho=0.9).fit(preds, truths)
        q, scores, n_obs = f.score_quantile_.copy(), f.scores_.copy(), f.n_observations_
        snap = f.model_copy(deep=True)
        for _ in range(10):
            f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 4.0))
        np.testing.assert_array_equal(snap.score_quantile_, q)
        np.testing.assert_array_equal(snap.scores_, scores)
        assert snap.n_observations_ == n_obs
        assert not np.allclose(f.score_quantile_, q)
        assert f.n_observations_ == n_obs + 10


class TestLifecycle:
    def test_fit_returns_new_object_and_leaves_method_untouched(self):
        preds, truths = _make_data(2, 60, 3)
        m = NonexchangeableConformalPrediction(alpha=0.1, rho=0.9)
        dumped = m.model_dump()
        f1 = m.fit(preds, truths)
        f2 = m.fit(preds, truths)
        assert f1 is not f2
        assert f1.method_ is not m
        assert f1.method_.rho == 0.9
        assert m.model_dump() == dumped
        f1.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 4.0))
        assert f2.n_observations_ == 60  # the other fit is unaffected

    def test_result_defensively_copied(self):
        preds, truths = _make_data(1, 60, 3)
        m = NonexchangeableConformalPrediction(alpha=0.1)
        f = m.fit(preds, truths)
        snap = f.calibration_.diagnostics["score_quantile"]
        assert isinstance(snap, np.ndarray)
        snap[:] = -999.0
        assert not np.any(f.score_quantile_ == -999.0)


class TestNaNMasking:
    def test_ragged_equals_per_series_dense_fit(self):
        rng = np.random.default_rng(7)
        horizon = 3
        lengths = [60, 40]
        n_windows = max(lengths)
        preds = np.full((2, n_windows, horizon), np.nan)
        truths = np.full((2, n_windows, horizon), np.nan)
        for i, n in enumerate(lengths):
            preds[i, -n:] = rng.normal(size=(n, horizon))
            truths[i, -n:] = preds[i, -n:] + rng.normal(size=(n, horizon))
        ragged = NonexchangeableConformalPrediction(alpha=0.1, rho=0.95)
        f_ragged = ragged.fit(preds, truths)
        for i, n in enumerate(lengths):
            dense = NonexchangeableConformalPrediction(alpha=0.1, rho=0.95)
            f_dense = dense.fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_allclose(f_ragged.score_quantile_[i : i + 1], f_dense.score_quantile_)

    def test_insufficient_series_named(self):
        preds, truths = _make_data(2, 20, 2)
        preds[1, :16] = np.nan
        truths[1, :16] = np.nan  # 4 effective < ceil(1/0.1)=10
        m = NonexchangeableConformalPrediction(alpha=0.1, rho=1.0)
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            m.fit(preds, truths)
        assert excinfo.value.series_indices == [1]
