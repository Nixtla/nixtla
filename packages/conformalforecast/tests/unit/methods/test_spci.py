"""Tests for SequentialPredictiveConformalInference's array surface: quantile-regressor
offsets, coverage, seed, exogenous X.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.base import (
    CalibrationResult,
    InsufficientCalibrationSamples,
)
from conformalforecast.methods.spci import SequentialPredictiveConformalInference, SPCIForecaster
from conformalforecast.nonconformity.absolute import AbsoluteResidual
from conformalforecast.quantile_regressors.base import QuantileRegressor
from tests.helpers import ConditionalMockQuantileRegressor
from tests.helpers import MockQuantileRegressor as _MockBase

try:
    importlib.import_module("quantile_forest")

    _HAS_QRF = True
except ImportError:
    _HAS_QRF = False


class MockQuantileRegressor(_MockBase):
    """Deterministic unconditional-quantile regressor (ignores X features).

    ``_MockBase`` subclasses ``QRFQuantileRegressor`` because that is the
    declared type of SPCI's ``regressor`` field; see ``tests/helpers.py``.
    """

    tag: str = "mock"
    n_features_: int | None = None

    def fit(self, X, y):
        super().fit(X, y)
        self.n_features_ = X.shape[1]


class ExogMock(_MockBase):
    """Quantile regressor whose output depends on the feature columns."""

    def predict_quantile(self, X, q):
        return super().predict_quantile(X, q) + 0.1 * np.asarray(X).sum(axis=1)


def _make_data(n_series, n_windows, horizon, seed=0):
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_series, n_windows, horizon))
    truths = preds + rng.normal(size=(n_series, n_windows, horizon))
    return preds, truths


class TestConstructor:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(ValidationError, match="alpha"):
            SequentialPredictiveConformalInference(alpha=alpha, regressor=MockQuantileRegressor())

    def test_window_size_validation(self):
        with pytest.raises(ValidationError, match="window_size"):
            SequentialPredictiveConformalInference(
                alpha=0.1, window_size=0, regressor=MockQuantileRegressor()
            )

    def test_beta_grid_validation(self):
        with pytest.raises(ValidationError, match="beta_grid_size"):
            SequentialPredictiveConformalInference(
                alpha=0.1, beta_grid_size=1, regressor=MockQuantileRegressor()
            )

    def test_refit_every_validation(self):
        with pytest.raises(ValidationError, match="refit_every"):
            SequentialPredictiveConformalInference(
                alpha=0.1, refit_every=0, regressor=MockQuantileRegressor()
            )

    def test_qrf_knobs_are_in_the_schema(self):
        props = SequentialPredictiveConformalInference.model_json_schema()["properties"]
        assert props["window_size"]["minimum"] == 1
        assert props["beta_grid_size"]["minimum"] == 2
        assert props["refit_every"]["minimum"] == 1
        assert "regressor" in props
        nested = SequentialPredictiveConformalInference.model_json_schema()["$defs"][
            "QRFQuantileRegressor"
        ]["properties"]
        assert {"n_estimators", "min_samples_leaf", "random_state"} <= set(nested)

    def test_construction_without_the_spci_extra(self):
        """The ImportError is deferred to fit, so both of these must work bare."""
        method = SequentialPredictiveConformalInference(alpha=0.1)
        assert method.regressor is not None
        assert SequentialPredictiveConformalInference.model_json_schema()["properties"]

    def test_is_online(self):
        assert (
            SequentialPredictiveConformalInference(
                alpha=0.1, regressor=MockQuantileRegressor()
            ).IS_ONLINE
            is True
        )


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series, horizon):
        preds, truths = _make_data(n_series, 60, horizon)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds, truths)
        assert isinstance(k, SPCIForecaster)
        result = k.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (n_series,)
        assert k.residuals_.shape == (n_series, 60, horizon)
        new_preds = _make_data(n_series, 7, horizon, seed=1)[0]
        iv = k.interval(new_preds)
        assert iv.shape == (n_series, 7, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_float32_preserved(self):
        preds, truths = _make_data(2, 60, 3)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds.astype(np.float32), truths.astype(np.float32))
        iv = k.interval(preds.astype(np.float32))
        assert iv.dtype == np.float32


class TestCoverage:
    def test_empirical_coverage_matches_alpha_mock(self):
        # Mock regressor path: tol 0.05 (deterministic).
        rng = np.random.default_rng(42)
        n_series, n_cal, n_test, horizon = 2, 300, 300, 3
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds_cal, truths_cal)
        preds_test = rng.normal(size=(n_series, n_test, horizon))
        truths_test = preds_test + rng.normal(size=(n_series, n_test, horizon))
        iv = k.interval(preds_test)
        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.05

    @pytest.mark.skipif(not _HAS_QRF, reason="quantile-forest not installed")
    def test_empirical_coverage_matches_alpha_qrf(self):
        # QRF path, online predict->update cycle (SPCI's coverage guarantee is
        # sequential). tol 0.08 absorbs QRF finite-sample variance, its internal
        # randomness (controlled here via seed), and beta-optimization
        # selection bias on iid data. beta_grid_size=3 keeps that bias mild;
        # refit_every=20 amortizes the refit cost.
        rng = np.random.default_rng(0)
        n_series, n_cal, n_holdout, horizon = 1, 400, 200, 1
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=30, beta_grid_size=3, refit_every=20, seed=0
        )
        k = k.fit(preds_cal, truths_cal)

        covered = []
        for _ in range(n_holdout):
            p = rng.normal(size=(n_series, 1, horizon))
            t = p + rng.normal(size=(n_series, 1, horizon))
            iv = k.interval(p)
            covered.append((t >= iv[..., 0]) & (t <= iv[..., 1]))
            k.update(p, t)
        coverage = np.mean(covered)
        assert abs(coverage - 0.9) < 0.08


class TestSeedDeterminism:
    def test_mock_deterministic(self):
        preds, truths = _make_data(2, 60, 3)
        k1 = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k2 = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k1 = k1.fit(preds, truths)
        k2 = k2.fit(preds, truths)
        np.testing.assert_array_equal(k1.interval(preds), k2.interval(preds))

    @pytest.mark.skipif(not _HAS_QRF, reason="quantile-forest not installed")
    def test_default_qrf_seed_deterministic(self):
        preds, truths = _make_data(1, 120, 1)
        k1 = SequentialPredictiveConformalInference(alpha=0.1, window_size=20, seed=0)
        k2 = SequentialPredictiveConformalInference(alpha=0.1, window_size=20, seed=0)
        k1 = k1.fit(preds, truths)
        k2 = k2.fit(preds, truths)
        np.testing.assert_array_equal(k1.interval(preds), k2.interval(preds))


class TestBetaOptimization:
    def test_offsets_can_be_asymmetric(self):
        # Skewed residuals -> width-minimizing offsets are asymmetric.
        rng = np.random.default_rng(1)
        n_series, n_cal, horizon = 1, 400, 1
        preds = np.zeros((n_series, n_cal, horizon))
        truths = rng.gamma(shape=2.0, scale=1.0, size=(n_series, n_cal, horizon))
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, beta_grid_size=21, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds, truths)
        iv = k.interval(np.zeros((n_series, 1, horizon)))
        lower_offset = iv[0, 0, 0, 0]
        upper_offset = iv[0, 0, 0, 1]
        assert not np.isclose(-lower_offset, upper_offset, atol=1e-3)


class TestExogenous:
    def test_feature_width_reflects_exog(self):
        preds, truths = _make_data(2, 60, 3)
        n_features = 2
        rng = np.random.default_rng(5)
        X = rng.normal(size=(2, 60, 3, n_features))
        w = 10
        k_x = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
        )
        k_x = k_x.fit(preds, truths, X=X)
        assert k_x.quantile_regressors_[(0, 0)].n_features_ == w + n_features

        k_none = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
        )
        k_none = k_none.fit(preds, truths)
        assert k_none.quantile_regressors_[(0, 0)].n_features_ == w

    def test_exog_changes_interval(self):
        preds, truths = _make_data(2, 60, 3)
        rng = np.random.default_rng(6)
        X = rng.normal(size=(2, 60, 3, 2))
        k_x = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=ExogMock()
        )
        k_x = k_x.fit(preds, truths, X=X)
        k_none = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=ExogMock()
        )
        k_none = k_none.fit(preds, truths)
        assert not np.allclose(k_x.interval(preds), k_none.interval(preds))


class TestRefitCadence:
    def test_refit_every_5(self):
        preds, truths = _make_data(1, 60, 1)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, refit_every=5, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds, truths)
        calls = {"n": 0}
        original = k._fit_all_regressors

        def counting():
            calls["n"] += 1
            original()

        k._fit_all_regressors = counting
        for _ in range(25):
            k.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
        assert calls["n"] == 5

    def test_refit_every_1(self):
        preds, truths = _make_data(1, 60, 1)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, refit_every=1, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds, truths)
        calls = {"n": 0}
        original = k._fit_all_regressors

        def counting():
            calls["n"] += 1
            original()

        k._fit_all_regressors = counting
        for _ in range(7):
            k.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
        assert calls["n"] == 7


class TestOnlineUpdate:
    def test_deep_copy_snapshot_is_independent_of_a_later_refit(self):
        """``model_copy(deep=True)`` must not alias the per-cell regressor dict."""
        preds, truths = _make_data(2, 60, 3)
        f = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, refit_every=1, regressor=MockQuantileRegressor()
        ).fit(preds, truths)
        snap = f.model_copy(deep=True)
        snap_regressors = dict(snap.quantile_regressors_)
        snap_residuals = snap.residuals_.copy()
        snap_interval = snap.interval(preds[:, :1, :])

        f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))  # refit_every=1: refits

        assert snap.quantile_regressors_ is not f.quantile_regressors_
        assert all(snap.quantile_regressors_[key] is reg for key, reg in snap_regressors.items())
        assert all(f.quantile_regressors_[key] is not reg for key, reg in snap_regressors.items())
        np.testing.assert_array_equal(snap.residuals_, snap_residuals)
        assert snap.n_observations_ == 60
        assert snap.steps_since_refit_ == 0
        np.testing.assert_array_equal(snap.interval(preds[:, :1, :]), snap_interval)

    def test_update_grows_residuals(self):
        preds, truths = _make_data(2, 60, 3)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        k = k.fit(preds, truths)
        k.update(np.zeros((2, 1, 3)), np.zeros((2, 1, 3)))
        assert k.residuals_.shape[1] == 61
        assert k.n_observations_ == 61
        assert (k.n_calibration_samples_ == 60).all()

    def test_steps_since_refit_is_a_public_fitted_field(self):
        preds, truths = _make_data(1, 60, 1)
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, refit_every=3, regressor=MockQuantileRegressor()
        ).fit(preds, truths)
        seen = []
        for _ in range(4):
            k.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
            seen.append(k.steps_since_refit_)
        assert seen == [1, 2, 0, 1]


class TestNaNMasking:
    def test_ragged_equals_per_series_dense_fit(self):
        rng = np.random.default_rng(7)
        horizon = 2
        w = 10
        lengths = [60, 40]
        n_windows = max(lengths)
        preds = np.full((2, n_windows, horizon), np.nan)
        truths = np.full((2, n_windows, horizon), np.nan)
        for i, n in enumerate(lengths):
            preds[i, -n:] = rng.normal(size=(n, horizon))
            truths[i, -n:] = preds[i, -n:] + rng.normal(size=(n, horizon))
        ragged = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
        )
        ragged = ragged.fit(preds, truths)
        zero = np.zeros((2, 1, horizon))
        iv_ragged = ragged.interval(zero)
        for i, n in enumerate(lengths):
            dense = SequentialPredictiveConformalInference(
                alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
            )
            dense = dense.fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            iv_dense = dense.interval(np.zeros((1, 1, horizon)))
            np.testing.assert_allclose(iv_ragged[i : i + 1], iv_dense)

    def test_insufficient_series_named(self):
        preds, truths = _make_data(2, 30, 2)
        preds[1, :26] = np.nan
        truths[1, :26] = np.nan  # 4 effective < window_size + 1 = 11
        k = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=MockQuantileRegressor()
        )
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            k.fit(preds, truths)
        assert excinfo.value.series_indices == [1]

    def test_alpha_float_noise_does_not_raise_the_floor(self):
        # 1 - 0.8 == 0.19999999999999996; five windows meet
        # max(window_size + 1, ceil(1/0.2)) = 5.
        preds, truths = _make_data(1, 5, 2)
        SequentialPredictiveConformalInference(
            alpha=1 - 0.8, window_size=4, regressor=MockQuantileRegressor()
        ).fit(preds, truths)


class TestRegressorTypeNarrowing:
    """`regressor` is typed as the concrete QRFQuantileRegressor.

    That is what puts the QRF knobs in SPCI's JSON schema (an ABC-typed field
    renders an empty nested schema), and it closes the duck-typed extension
    point: an object with only `fit` + `predict_quantile`, and even a plain
    `QuantileRegressor` subclass, is now rejected at construction.
    """

    def test_duck_typed_regressor_is_rejected(self) -> None:
        class DuckTyped:
            def fit(self, X, y):
                return self

            def predict_quantile(self, X, q):
                return X

        with pytest.raises(ValidationError):
            SequentialPredictiveConformalInference(alpha=0.1, regressor=DuckTyped())

    def test_non_qrf_quantile_regressor_subclass_is_rejected(self) -> None:
        class Abstract(QuantileRegressor):
            def fit(self, X, y):
                return self

            def predict_quantile(self, X, q):
                return X

        with pytest.raises(ValidationError):
            SequentialPredictiveConformalInference(alpha=0.1, regressor=Abstract())


def test_spci_rejects_a_non_signed_score_at_construction():
    with pytest.raises(ValidationError, match="SignedResidual"):
        SequentialPredictiveConformalInference(
            alpha=0.1, regressor=MockQuantileRegressor(), score=AbsoluteResidual()
        )


def test_spci_fits_a_copy_of_the_configured_signed_score():
    """The score is config; the fitted object holds its own copy of it."""
    from typing import Literal

    from conformalforecast.nonconformity.signed import SignedResidual

    class _Tagged(SignedResidual):
        kind: Literal["spci_tagged_signed"] = "spci_tagged_signed"

    score = _Tagged()
    method = SequentialPredictiveConformalInference(
        alpha=0.1, window_size=10, regressor=MockQuantileRegressor(), score=score
    )
    preds, truths = _make_data(2, 40, 2)
    fitted = method.fit(preds, truths)
    assert type(fitted.score_fn_) is _Tagged
    assert fitted.score_fn_ is not score
    assert method.score is score


class TestRegressorPrototype:
    def test_regressor_prototype_is_cloned_per_cell(self):
        proto = MockQuantileRegressor(tag="proto")
        method = SequentialPredictiveConformalInference(alpha=0.1, window_size=5, regressor=proto)
        preds, truths = _make_data(2, 40, 3)
        fitted = method.fit(preds, truths)

        assert len(fitted.quantile_regressors_) == 6  # 2 series x 3 horizons
        clones = list(fitted.quantile_regressors_.values())
        assert all(c is not proto for c in clones)
        assert all(c is not fitted.method_.regressor for c in clones)
        assert fitted.method_.regressor is not proto  # method_ is a deep copy
        assert len({id(c) for c in clones}) == 6  # all distinct
        assert all(c.tag == "proto" for c in clones)  # config carried over

    def test_prototype_is_never_mutated(self):
        proto = MockQuantileRegressor(tag="proto")
        method = SequentialPredictiveConformalInference(alpha=0.1, window_size=5, regressor=proto)
        preds, truths = _make_data(2, 40, 3)
        fitted = method.fit(preds, truths)
        fitted.update(preds[:, :1, :], truths[:, :1, :])  # refit_every=1: refits
        assert method.regressor is proto
        assert proto._y is None  # fitted state lives on the clones
        assert proto.n_features_ is None
        assert fitted.method_.regressor._y is None

    def test_refit_replaces_clones_with_fresh_ones(self):
        proto = MockQuantileRegressor(tag="proto")
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=5, refit_every=1, regressor=proto
        )
        preds, truths = _make_data(1, 40, 2)
        fitted = method.fit(preds, truths)
        # Hold the objects, not their ids: CPython reuses an id once the old
        # clone is collected, which made an id-set comparison flaky.
        before = list(fitted.quantile_regressors_.values())
        fitted.update(preds[:, :1, :], truths[:, :1, :])
        after = list(fitted.quantile_regressors_.values())
        assert all(a is not b for a in before for b in after)
        assert len({id(c) for c in after}) == 2
        assert proto._y is None

    def test_seed_threads_into_the_default_regressor(self):
        from conformalforecast.quantile_regressors.qrf import QRFQuantileRegressor

        method = SequentialPredictiveConformalInference(alpha=0.1, seed=7)
        assert isinstance(method.regressor, QRFQuantileRegressor)
        assert method.regressor.random_state == 7

    def test_explicit_regressor_random_state_wins_over_seed(self):
        from conformalforecast.quantile_regressors.qrf import QRFQuantileRegressor

        proto = QRFQuantileRegressor(random_state=99)
        method = SequentialPredictiveConformalInference(alpha=0.1, seed=7, regressor=proto)
        assert method.regressor is proto
        assert method.regressor.random_state == 99


class TestCalibrationIntervals:
    """Per-window in-sample replay: each window queried with its own history.

    ``interval()`` answers from the *last* ``window_size`` residuals and
    broadcasts, which is right for the next step and meaningless for past
    windows. This is the override that makes Mode-1 diagnostics real.
    """

    def _fit(self, n_series=2, n_windows=40, horizon=3, window_size=10, seed=0):
        preds, truths = _make_data(n_series, n_windows, horizon, seed=seed)
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=window_size, regressor=MockQuantileRegressor()
        )
        method = method.fit(preds, truths)
        return method, preds, truths

    def test_shape_and_warmup_nan_placement(self):
        method, preds, _ = self._fit()
        iv = method.calibration_intervals(preds)
        assert iv.shape == (2, 40, 3, 2)
        # Exactly the first window_size windows of each cell are unevaluable.
        unevaluable = np.isnan(iv).any(axis=-1)
        assert unevaluable[:, :10, :].all()
        assert not unevaluable[:, 10:, :].any()

    def test_offsets_move_along_the_sample_axis(self):
        """The defect: interval() broadcasts one offset pair over every window.

        Needs the conditioning mock and a drifting residual stream -- under the
        unconditional mock, or under i.i.d. residuals, every query window looks
        the same and the broadcast is indistinguishable from the replay.
        """
        rng = np.random.default_rng(0)
        level = np.cumsum(rng.normal(size=(2, 40, 3)) * 0.8, axis=1)
        preds = np.zeros((2, 40, 3))
        truths = level + rng.normal(size=(2, 40, 3)) * 0.3
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=10, regressor=ConditionalMockQuantileRegressor()
        )
        method = method.fit(preds, truths)

        offsets = method.calibration_intervals(preds)[..., 0] - preds
        # Per cell, not globally: cells differ from one another either way.
        assert np.all(np.nanstd(offsets, axis=1) > 0.1)
        # Whereas interval() is constant along the sample axis, by construction.
        broadcast = method.interval(preds)[..., 0] - preds
        assert np.allclose(broadcast, broadcast[:, :1, :])

    def test_deterministic(self):
        method, preds, _ = self._fit()
        np.testing.assert_array_equal(
            method.calibration_intervals(preds), method.calibration_intervals(preds)
        )

    def test_preserves_float32(self):
        preds, truths = _make_data(2, 30, 2)
        preds, truths = preds.astype(np.float32), truths.astype(np.float32)
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=5, regressor=MockQuantileRegressor()
        )
        method = method.fit(preds, truths)
        assert method.calibration_intervals(preds).dtype == np.float32

    def test_bounds_are_ordered(self):
        method, preds, _ = self._fit()
        iv = method.calibration_intervals(preds)
        ok = ~np.isnan(iv).any(axis=-1)
        assert np.all(iv[..., 0][ok] <= iv[..., 1][ok])

    def test_ragged_equals_per_series_dense(self):
        """A left-NaN-padded series replays exactly as its dense sub-array does.

        Compaction is per cell, so a series that starts late gets its warm-up
        boundary at its own tenth *valid* window, not at absolute window 10.
        """
        lengths, horizon, w = [40, 25], 2, 10
        n_windows = max(lengths)
        preds = np.full((2, n_windows, horizon), np.nan)
        truths = np.full_like(preds, np.nan)
        rng = np.random.default_rng(0)
        for i, n in enumerate(lengths):
            preds[i, -n:] = rng.normal(size=(n, horizon))
            truths[i, -n:] = preds[i, -n:] + rng.normal(size=(n, horizon))

        ragged = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
        )
        ragged = ragged.fit(preds, truths)
        iv = ragged.calibration_intervals(preds)

        for i, n in enumerate(lengths):
            dense = SequentialPredictiveConformalInference(
                alpha=0.1, window_size=w, regressor=MockQuantileRegressor()
            )
            dense = dense.fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            dense_iv = dense.calibration_intervals(preds[i : i + 1, -n:])
            np.testing.assert_allclose(iv[i, -n:], dense_iv[0], equal_nan=True)
            # Padding stays NaN; the warm-up sits at this series' own start.
            assert np.isnan(iv[i, : n_windows - n]).all()
            assert np.isnan(iv[i, n_windows - n : n_windows - n + w]).all()

    def test_replays_only_the_calibration_windows_after_update(self):
        """update() grows the residual buffer; the replay must still fit preds."""
        method, preds, truths = self._fit(n_windows=30)
        method.update(preds[:, -1:, :], truths[:, -1:, :])
        assert method.residuals_.shape[1] == 31
        iv = method.calibration_intervals(preds)
        assert iv.shape == (2, 30, 3, 2)

    def test_mismatched_predictions_raise(self):
        method, _, _ = self._fit(n_windows=30)
        with pytest.raises(ValueError, match="does not fit the calibration"):
            method.calibration_intervals(_make_data(2, 99, 3)[0])

    @pytest.mark.parametrize("bad_horizon", [2, 5])
    def test_mismatched_horizon_raises(self, bad_horizon):
        """The horizon axis must be validated, not just the series count.

        A too-large horizon otherwise returns silently all-NaN columns, and a
        too-small one an IndexError from the write into `offsets`, where the
        guard promises a ValueError naming both shapes.
        """
        method, _, _ = self._fit(horizon=3)
        with pytest.raises(ValueError, match="does not fit the calibration"):
            method.calibration_intervals(_make_data(2, 40, bad_horizon)[0])

    def test_all_windows_unevaluable_when_history_is_too_short(self):
        """window_size == n_windows leaves no window with a full history."""
        preds, truths = _make_data(1, 21, 1)
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=20, regressor=MockQuantileRegressor()
        )
        method = method.fit(preds, truths)
        iv = method.calibration_intervals(preds)
        assert np.isnan(iv[:, :20]).all()
        assert not np.isnan(iv[:, 20]).any()

    def test_exogenous_rows_are_threaded(self):
        """With exog fitted, each replay row carries that window's features."""
        preds, truths = _make_data(1, 30, 2)
        X = np.random.default_rng(1).normal(size=(*preds.shape, 2))
        method = SequentialPredictiveConformalInference(
            alpha=0.1, window_size=5, regressor=ExogMock()
        )
        method = method.fit(preds, truths, X=X)
        iv = method.calibration_intervals(preds)
        assert iv.shape == (1, 30, 2, 2)
        offsets = iv[..., 0] - preds
        assert np.all(np.nanstd(offsets, axis=1) > 0.0)
