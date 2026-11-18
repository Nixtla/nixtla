"""Tests for AggregatedAdaptiveConformalInference's array surface: multi-expert
aggregation, shapes, coverage, determinism, NaN masking, online update.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.aggregators.base import OnlineAggregator
from conformalforecast.aggregators.ewa import EWA
from conformalforecast.base import (
    CalibrationResult,
    InsufficientCalibrationSamples,
)
from conformalforecast.methods.aci import AdaptiveConformalInference
from conformalforecast.methods.agaci import AgACIForecaster, AggregatedAdaptiveConformalInference


def _make_data(n_series, n_windows, horizon, seed=0):
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_series, n_windows, horizon))
    truths = preds + rng.normal(size=(n_series, n_windows, horizon))
    return preds, truths


class _UniformAggregator(EWA):
    """Ignores losses; always returns uniform weights.

    Subclasses EWA rather than OnlineAggregator because the ``aggregator`` field
    is typed as the concrete EWA (see ``test_non_ewa_aggregator_is_rejected``).
    """

    def weights(self):
        self._require_initialized()
        return np.full((self.n_experts, self.n_series, self.horizon), 1.0 / self.n_experts)


class _NotAnEWA(OnlineAggregator):
    """An OnlineAggregator that is deliberately not an EWA."""

    def weights(self):
        self._require_initialized()
        return np.full((self.n_experts, self.n_series, self.horizon), 1.0 / self.n_experts)


class TestConstructor:
    @pytest.mark.parametrize("gammas", [[], (), (0.0,), (-0.01, 0.05)])
    def test_invalid_gammas_raise(self, gammas):
        with pytest.raises(ValidationError, match="gammas"):
            AggregatedAdaptiveConformalInference(alpha=0.1, gammas=gammas)

    def test_a_string_is_not_a_gamma_grid(self):
        """A bare Sequence[float] annotation would happily validate "0.1"."""
        with pytest.raises(ValidationError, match="gammas"):
            AggregatedAdaptiveConformalInference(alpha=0.1, gammas="0.1")

    def test_gammas_are_coerced_to_a_tuple_of_floats(self):
        m = AggregatedAdaptiveConformalInference(alpha=0.1, gammas=[0.01, 0.05])
        assert m.gammas == (0.01, 0.05)
        assert m.n_experts == 2

    def test_n_experts_is_read_only(self):
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        with pytest.raises(AttributeError):
            m.n_experts = 3

    @pytest.mark.parametrize(
        "kwargs",
        [{"interval_clip_lower": -5.0}, {"interval_clip_upper": 5.0}],
    )
    def test_clip_both_or_neither(self, kwargs):
        with pytest.raises(ValidationError, match="both be provided or both be None"):
            AggregatedAdaptiveConformalInference(alpha=0.1, **kwargs)

    def test_clip_order(self):
        with pytest.raises(ValidationError, match="strictly less"):
            AggregatedAdaptiveConformalInference(
                alpha=0.1, interval_clip_lower=5.0, interval_clip_upper=-5.0
            )

    def test_equal_clips_are_rejected(self):
        with pytest.raises(ValidationError, match="strictly less"):
            AggregatedAdaptiveConformalInference(
                alpha=0.1, interval_clip_lower=1.0, interval_clip_upper=1.0
            )

    def test_defaults(self):
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        assert m.gammas == (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)
        assert m.n_experts == 6
        assert m.IS_ONLINE is True
        assert m.interval_clip_lower is None
        assert m.interval_clip_upper is None
        assert m.seed == 0


class TestSchema:
    def test_eta_is_visible_nested_under_aggregator(self):
        """The whole point of typing `aggregator` as the concrete EWA."""
        schema = AggregatedAdaptiveConformalInference.model_json_schema()
        props = schema["properties"]
        assert props["gammas"]["minItems"] == 1
        # pydantic < 2.9 wraps a $ref that carries a description in allOf;
        # 2.9+ inlines it. Accept either.
        aggregator = props["aggregator"]
        node = aggregator.get("allOf", [aggregator])[0]
        ref = node["$ref"].rsplit("/", 1)[-1]
        assert "eta" in schema["$defs"][ref]["properties"]

    def test_fitted_state_is_absent_from_the_schema(self):
        props = AggregatedAdaptiveConformalInference.model_json_schema()["properties"]
        assert not [name for name in props if name.endswith("_")]


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series, horizon):
        preds, truths = _make_data(n_series, 60, horizon)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds, truths)
        assert isinstance(f, AgACIForecaster)
        assert isinstance(f.calibration_, CalibrationResult)
        assert f.calibration_.n_calibration_samples.shape == (n_series,)
        assert f.alpha_t_per_expert_.shape == (m.n_experts, n_series, horizon)
        new_preds = _make_data(n_series, 7, horizon, seed=1)[0]
        iv = f.interval(new_preds)
        assert iv.shape == (n_series, 7, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_float32_preserved(self):
        preds, truths = _make_data(2, 60, 3)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds.astype(np.float32), truths.astype(np.float32))
        assert f.interval(preds.astype(np.float32)).dtype == np.float32


class TestDefaultClipResolution:
    def test_fit_resolves_the_clips_without_touching_the_config(self):
        """The common path: no clips passed, so fit derives them from the truths.

        The config attributes must stay None — the resolved values live only on
        the trailing-underscore fitted attributes.
        """
        preds, truths = _make_data(2, 60, 3)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds, truths)

        std = float(np.nanstd(truths))
        assert f.interval_clip_lower_ == pytest.approx(float(np.nanmin(truths)) - 5 * std)
        assert f.interval_clip_upper_ == pytest.approx(float(np.nanmax(truths)) + 5 * std)
        assert m.interval_clip_lower is None
        assert m.interval_clip_upper is None
        assert f.method_.interval_clip_lower is None

    def test_explicit_clips_are_copied_verbatim_to_the_fitted_attributes(self):
        preds, truths = _make_data(2, 60, 3)
        m = AggregatedAdaptiveConformalInference(
            alpha=0.1, interval_clip_lower=-7.5, interval_clip_upper=7.5
        )
        f = m.fit(preds, truths)
        assert (f.interval_clip_lower_, f.interval_clip_upper_) == (-7.5, 7.5)
        assert (m.interval_clip_lower, m.interval_clip_upper) == (-7.5, 7.5)


class TestOutOfRangePoints:
    def test_bounds_do_not_invert_for_ood_points(self):
        # Points far outside the auto clip range must still give lower <= upper.
        preds, truths = _make_data(2, 200, 3)
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        ood = np.array([-1e6, 1e6]).reshape(1, 2, 1) * np.ones((2, 2, 3))
        iv = f.interval(ood)
        assert (iv[..., 0] <= iv[..., 1]).all()


class TestCoverage:
    def test_empirical_coverage_matches_alpha(self):
        # tol 0.04 matches tests/unit/methods/test_agaci_forecaster.py.
        rng = np.random.default_rng(42)
        n_series, n_cal, n_test, horizon = 3, 500, 500, 4
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds_cal, truths_cal)
        preds_test = rng.normal(size=(n_series, n_test, horizon))
        truths_test = preds_test + rng.normal(size=(n_series, n_test, horizon))
        iv = f.interval(preds_test)
        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.04


class TestSingleExpertReducesToACI:
    def test_single_expert_matches_aci(self):
        preds, truths = _make_data(3, 80, 4)
        agaci = AggregatedAdaptiveConformalInference(
            alpha=0.1, gammas=(0.005,), interval_clip_lower=-1e10, interval_clip_upper=1e10
        )
        aci = AdaptiveConformalInference(alpha=0.1, gamma=0.005)
        fa = agaci.fit(preds, truths)
        fb = aci.fit(preds, truths)
        np.testing.assert_allclose(fa.alpha_t_per_expert_[0], fb.alpha_t_, rtol=1e-12)
        new = _make_data(3, 5, 4, seed=9)[0]
        np.testing.assert_allclose(fa.interval(new), fb.interval(new), rtol=1e-10, atol=1e-10)


class TestAggregatorWeights:
    def test_default_aggregator_is_ewa(self):
        preds, truths = _make_data(2, 60, 3)
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        assert isinstance(f.aggregator_lower_, EWA)

    def test_default_aggregator_is_ewa_with_eta_one(self):
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        assert isinstance(m.aggregator, EWA)
        assert m.aggregator.eta == 1.0

    def test_aggregator_prototype_is_cloned_per_bound(self):
        preds, truths = _make_data(2, 60, 3)
        m = AggregatedAdaptiveConformalInference(alpha=0.1, aggregator=EWA(eta=3.0))
        f = m.fit(preds, truths)

        assert f.aggregator_lower_ is not f.aggregator_upper_
        for held in (f.aggregator_lower_, f.aggregator_upper_):
            assert held is not m.aggregator
            assert held is not f.method_.aggregator
        assert f.aggregator_lower_.eta == 3.0
        assert f.aggregator_upper_.eta == 3.0
        assert m.aggregator.cumulative_losses_ is None
        assert f.method_.aggregator.cumulative_losses_ is None

    def test_lower_and_upper_aggregators_diverge(self):
        """Independent aggregators; sharing one would silently symmetrize."""
        preds, truths = _make_data(2, 200, 3)
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        assert f.aggregator_lower_ is not f.aggregator_upper_
        assert not np.array_equal(
            f.aggregator_lower_.cumulative_losses_,
            f.aggregator_upper_.cumulative_losses_,
        )

    def test_non_ewa_aggregator_is_rejected(self):
        """Deliberate narrowing: the field is typed as the concrete EWA."""
        with pytest.raises(ValidationError, match="aggregator"):
            AggregatedAdaptiveConformalInference(alpha=0.1, aggregator=_NotAnEWA())

    def test_ewa_subclass_is_accepted(self):
        m = AggregatedAdaptiveConformalInference(alpha=0.1, aggregator=_UniformAggregator())
        assert isinstance(m.aggregator, _UniformAggregator)

    def test_weights_drift_from_uniform(self):
        preds, truths = _make_data(2, 200, 3)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        f = m.fit(preds, truths)
        uniform = np.full_like(f.aggregator_lower_.weights(), 1.0 / m.n_experts)
        assert not np.allclose(f.aggregator_lower_.weights(), uniform, atol=1e-3)

    def test_uniform_prototype_equals_unweighted_mean(self):
        preds, truths = _make_data(2, 60, 3)

        m = AggregatedAdaptiveConformalInference(alpha=0.1, aggregator=_UniformAggregator())
        f = m.fit(preds, truths)
        # With uniform weights the interval is the plain mean of per-expert bounds.
        new = _make_data(2, 4, 3, seed=1)[0]
        iv = f.interval(new)
        # Per-expert bounds for one representative sample slice.
        lower_k, upper_k = f._per_expert_bounds(new[:, 0, :], f.scores_)
        np.testing.assert_allclose(iv[:, 0, :, 0], lower_k.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(iv[:, 0, :, 1], upper_k.mean(axis=0), rtol=1e-10)


class TestDeterminism:
    def test_same_config_same_arrays_same_output(self):
        preds, truths = _make_data(3, 80, 4)
        m1 = AggregatedAdaptiveConformalInference(alpha=0.1)
        m2 = AggregatedAdaptiveConformalInference(alpha=0.1)
        f1 = m1.fit(preds, truths)
        f2 = m2.fit(preds, truths)
        np.testing.assert_array_equal(f1.alpha_t_per_expert_, f2.alpha_t_per_expert_)
        new = _make_data(3, 5, 4, seed=9)[0]
        np.testing.assert_array_equal(f1.interval(new), f2.interval(new))

    def test_fitting_twice_gives_independent_objects(self):
        preds, truths = _make_data(3, 80, 4)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        f1, f2 = m.fit(preds, truths), m.fit(preds, truths)
        assert f1 is not f2
        assert f1.aggregator_lower_ is not f2.aggregator_lower_
        f1.update(np.zeros((3, 1, 4)), np.full((3, 1, 4), 5.0))
        assert f2.n_observations_ == 80


class TestOnlineUpdate:
    def test_update_grows_history(self):
        preds, truths = _make_data(2, 60, 3)
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        before_losses = f.aggregator_lower_.cumulative_losses_.copy()
        assert f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0)) is None
        assert f.scores_.shape[1] == 61
        assert f.n_observations_ == 61
        assert not np.array_equal(before_losses, f.aggregator_lower_.cumulative_losses_)

    def test_deep_copy_snapshot_is_independent_of_a_later_update(self):
        """``model_copy(deep=True)`` must not alias the EWA aggregators."""
        preds, truths = _make_data(2, 60, 3)
        f = AggregatedAdaptiveConformalInference(alpha=0.1).fit(preds, truths)
        snap = f.model_copy(deep=True)
        weights_lower = snap.aggregator_lower_.weights().copy()
        weights_upper = snap.aggregator_upper_.weights().copy()
        alpha_t = snap.alpha_t_per_expert_.copy()

        f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))

        assert snap.aggregator_lower_ is not f.aggregator_lower_
        assert snap.aggregator_upper_ is not f.aggregator_upper_
        np.testing.assert_array_equal(snap.aggregator_lower_.weights(), weights_lower)
        np.testing.assert_array_equal(snap.aggregator_upper_.weights(), weights_upper)
        assert not np.array_equal(f.aggregator_lower_.weights(), weights_lower)
        np.testing.assert_array_equal(snap.alpha_t_per_expert_, alpha_t)
        assert snap.n_observations_ == 60
        assert snap.scores_.shape[1] == 60

    def test_update_never_touches_the_method_or_its_prototype(self):
        preds, truths = _make_data(2, 60, 3)
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        before = m.model_dump()
        f = m.fit(preds, truths)
        f.update(np.zeros((2, 1, 3)), np.full((2, 1, 3), 5.0))
        assert m.model_dump() == before
        assert m.aggregator.cumulative_losses_ is None


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
        ragged = AggregatedAdaptiveConformalInference(
            alpha=0.1, interval_clip_lower=-1e10, interval_clip_upper=1e10
        )
        ragged = ragged.fit(preds, truths)
        for i, n in enumerate(lengths):
            dense = AggregatedAdaptiveConformalInference(
                alpha=0.1, interval_clip_lower=-1e10, interval_clip_upper=1e10
            ).fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_allclose(
                ragged.alpha_t_per_expert_[:, i : i + 1, :],
                dense.alpha_t_per_expert_,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                ragged.aggregator_lower_.cumulative_losses_[:, i : i + 1, :],
                dense.aggregator_lower_.cumulative_losses_,
                rtol=1e-10,
            )

    def test_insufficient_series_named(self):
        preds, truths = _make_data(2, 20, 2)
        preds[1, :15] = np.nan
        truths[1, :15] = np.nan
        m = AggregatedAdaptiveConformalInference(alpha=0.1)
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            m.fit(preds, truths)
        assert excinfo.value.series_indices == [1]

    def test_alpha_float_noise_does_not_raise_the_floor(self):
        # 1 - 0.8 == 0.19999999999999996; five windows meet ceil(1/0.2) = 5.
        preds, truths = _make_data(1, 5, 2)
        AggregatedAdaptiveConformalInference(alpha=1 - 0.8).fit(preds, truths)
