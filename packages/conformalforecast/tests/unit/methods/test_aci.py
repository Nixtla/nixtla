"""Tests for AdaptiveConformalInference's array surface: shapes, coverage,
determinism, NaN masking, online update.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.base import (
    CalibrationResult,
    InsufficientCalibrationSamples,
)
from conformalforecast.methods.aci import ACIForecaster, AdaptiveConformalInference


def _make_data(n_series, n_windows, horizon, seed=0):
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_series, n_windows, horizon))
    truths = preds + rng.normal(size=(n_series, n_windows, horizon))
    return preds, truths


class TestConstructor:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(ValidationError, match="alpha"):
            AdaptiveConformalInference(alpha=alpha)

    def test_gamma_must_be_positive(self):
        for bad in (0.0, -0.01):
            with pytest.raises(ValidationError, match="gamma"):
                AdaptiveConformalInference(alpha=0.1, gamma=bad)

    def test_large_gamma_warns_but_constructs(self):
        with pytest.warns(UserWarning, match="unusually large"):
            method = AdaptiveConformalInference(alpha=0.1, gamma=0.5)
        assert method.gamma == 0.5

    def test_recommended_gamma_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AdaptiveConformalInference(alpha=0.1, gamma=0.05)

    def test_defaults(self):
        m = AdaptiveConformalInference(alpha=0.1)
        assert m.gamma == 0.05
        assert m.IS_ONLINE is True
        assert m.score is None


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series, horizon):
        preds, truths = _make_data(n_series, 60, horizon)
        m = AdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds, truths)
        assert isinstance(f, ACIForecaster)
        assert isinstance(f.calibration_, CalibrationResult)
        assert f.calibration_.n_calibration_samples.shape == (n_series,)
        assert f.alpha_t_.shape == (n_series, horizon)
        new_preds = _make_data(n_series, 7, horizon, seed=1)[0]
        iv = f.interval(new_preds)
        assert iv.shape == (n_series, 7, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_float32_preserved(self):
        preds, truths = _make_data(2, 60, 3)
        m = AdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds.astype(np.float32), truths.astype(np.float32))
        assert f.interval(preds.astype(np.float32)).dtype == np.float32


class TestCoverage:
    def test_empirical_coverage_matches_alpha(self):
        # tol 0.04 matches methods/tests/test_aci_forecaster.py.
        rng = np.random.default_rng(42)
        n_series, n_cal, n_test, horizon = 3, 500, 500, 4
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))
        m = AdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds_cal, truths_cal)
        preds_test = rng.normal(size=(n_series, n_test, horizon))
        truths_test = preds_test + rng.normal(size=(n_series, n_test, horizon))
        iv = f.interval(preds_test)
        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.04


class TestDeterminism:
    def test_same_config_same_arrays_same_output(self):
        preds, truths = _make_data(3, 80, 4)
        m1 = AdaptiveConformalInference(alpha=0.1)
        m2 = AdaptiveConformalInference(alpha=0.1)
        f1 = m1.fit(preds, truths)
        f2 = m2.fit(preds, truths)
        np.testing.assert_array_equal(f1.alpha_t_, f2.alpha_t_)
        new = _make_data(3, 5, 4, seed=9)[0]
        np.testing.assert_array_equal(f1.interval(new), f2.interval(new))


class TestOnlineUpdate:
    def test_update_grows_history_and_moves_alpha_t(self):
        preds, truths = _make_data(2, 60, 3)
        m = AdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds, truths)
        before = f.alpha_t_.copy()
        p = np.zeros((2, 1, 3))
        t = np.full((2, 1, 3), 5.0)  # a big miss -> alpha_t should move
        assert f.update(p, t) is None  # in place
        assert f.scores_.shape[1] == 61
        assert f.n_observations_ == 61
        assert not np.array_equal(before, f.alpha_t_)

    def test_update_wrong_shape_raises(self):
        preds, truths = _make_data(2, 60, 3)
        f = AdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        with pytest.raises(ValueError):
            f.update(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)))

    def test_deep_copy_snapshot_is_independent_of_later_updates(self):
        preds, truths = _make_data(2, 60, 3)
        f = AdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        alpha_t, scores, n_obs = f.alpha_t_.copy(), f.scores_.copy(), f.n_observations_
        snap = f.model_copy(deep=True)
        f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        np.testing.assert_array_equal(snap.alpha_t_, alpha_t)
        np.testing.assert_array_equal(snap.scores_, scores)
        assert snap.n_observations_ == n_obs
        assert not np.array_equal(f.alpha_t_, alpha_t)
        assert f.n_observations_ == n_obs + 1


class TestLifecycle:
    def test_fit_returns_new_object_and_leaves_method_untouched(self):
        preds, truths = _make_data(2, 60, 3)
        m = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
        dumped = m.model_dump()
        f1 = m.fit(preds, truths)
        f2 = m.fit(preds, truths)
        assert f1 is not f2
        assert f1.method_ is not m
        assert f1.method_.gamma == 0.01
        assert m.model_dump() == dumped
        f1.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        assert f2.n_observations_ == 60  # the other fit is unaffected


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
        ragged = AdaptiveConformalInference(alpha=0.1)
        f_ragged = ragged.fit(preds, truths)
        for i, n in enumerate(lengths):
            dense = AdaptiveConformalInference(alpha=0.1)
            f_dense = dense.fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_allclose(f_ragged.alpha_t_[i : i + 1], f_dense.alpha_t_)

    def test_insufficient_series_named(self):
        preds, truths = _make_data(2, 20, 2)
        preds[1, :15] = np.nan
        truths[1, :15] = np.nan  # 5 effective < ceil(1/0.1)=10
        m = AdaptiveConformalInference(alpha=0.1)
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            m.fit(preds, truths)
        assert excinfo.value.series_indices == [1]

    def test_alpha_float_noise_does_not_raise_the_floor(self):
        # 1 - 0.8 == 0.19999999999999996; five windows meet ceil(1/0.2) = 5.
        preds, truths = _make_data(1, 5, 2)
        AdaptiveConformalInference(alpha=1 - 0.8).fit(preds, truths)
