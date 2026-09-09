"""Tests for PyO3 binding input validation and edge cases."""

import numpy as np
import pytest

from synforecast._lib import (
    augmentation,
    distributions,
    multivariate,
    pattern_injection,
    statistical,
    stochastic,
)


class TestAugmentationBindings:
    """Validation and shape contracts for the native augmentation module."""

    @pytest.mark.parametrize("period", [None, 7, 12])
    def test_mbb_many_matches_independent_draws(self, period):
        values = np.random.default_rng(12).normal(size=101)
        seeds = [42, 7, 2**63 - 1]
        copies = augmentation.moving_block_bootstrap_many(values, 6, seeds, period)
        assert len(copies) == len(seeds)
        for copy, seed in zip(copies, seeds, strict=True):
            np.testing.assert_array_equal(
                copy, augmentation.moving_block_bootstrap(values, 6, seed, period)
            )

    def test_mbb_many_rejects_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            augmentation.moving_block_bootstrap_many(np.arange(8.0), 9, [42])

    def test_dtw_supports_unequal_lengths(self):
        first = np.array([0.0, 1.0, 2.0])
        second = np.array([0.0, 0.5, 1.0, 2.0])
        distance, path = augmentation.dtw_alignment(first, second, 1)
        assert np.isfinite(distance)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 3)

    def test_nearest_dtw_rejects_empty_or_nonfinite_series(self):
        with pytest.raises(ValueError, match="non-empty|finite"):
            augmentation.nearest_dtw_neighbors([np.ones(3), np.array([])], 0.5, 1)
        with pytest.raises(ValueError, match="non-empty|finite"):
            augmentation.nearest_dtw_neighbors(
                [np.ones(3), np.array([0.0, np.nan, 1.0])], 0.5, 1
            )

    def test_nearest_dtw_single_series_has_no_neighbors(self):
        assert augmentation.nearest_dtw_neighbors([np.arange(5.0)], 0.5, 1) == [[]]

    @pytest.mark.parametrize("band", [-1, -10])
    def test_dtw_rejects_negative_band(self, band):
        values = np.arange(4.0)
        with pytest.raises(ValueError, match="band must be non-negative"):
            augmentation.dtw_alignment(values, values, band)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (np.array([]), np.arange(3.0)),
            (np.array([0.0, np.nan, 2.0]), np.arange(3.0)),
        ],
    )
    def test_dtw_rejects_invalid_values(self, first, second):
        with pytest.raises(ValueError):
            augmentation.dtw_alignment(first, second)

    def test_dba_preserves_reference_length_for_unequal_inputs(self):
        reference = np.array([0.0, 1.0, 2.0, 1.0])
        neighbors = [np.array([0.0, 0.5, 1.0, 2.0, 1.0])]
        weights = np.array([0.75, 0.25])
        result = augmentation.dba_barycenter(
            reference, neighbors, weights, n_iterations=2, band=1
        )
        assert result.shape == reference.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        ("weights", "n_iterations"),
        [(np.array([1.0]), 2), (np.array([1.0, 1.0]), 0)],
    )
    def test_dba_rejects_invalid_configuration(self, weights, n_iterations):
        reference = np.arange(4.0)
        neighbors = [reference.copy()]
        with pytest.raises(ValueError):
            augmentation.dba_barycenter(
                reference, neighbors, weights, n_iterations=n_iterations
            )

    def test_decomposition_reconstructs_input(self):
        time = np.arange(48.0)
        values = 0.1 * time + np.sin(2 * np.pi * time / 12)
        trend, seasonal, remainder = augmentation.classical_decompose(values, 12)
        np.testing.assert_allclose(trend + seasonal + remainder, values)

    @pytest.mark.parametrize("period", [0, 1])
    def test_decomposition_rejects_short_period(self, period):
        with pytest.raises(ValueError, match="period must be >= 2"):
            augmentation.classical_decompose(np.arange(8.0), period)

    def test_feature_tuple_is_finite_for_constant_input(self):
        features = augmentation.compute_features(np.ones(24), 12)
        assert len(features) == 4
        assert np.all(np.isfinite(features))
        assert features[0] == 0.0
        assert features[3] == 0.0

    def test_moving_block_bootstrap_is_seeded(self):
        values = np.sin(2 * np.pi * np.arange(48.0) / 12)
        first = augmentation.moving_block_bootstrap(values, 6, 42, 12)
        second = augmentation.moving_block_bootstrap(values, 6, 42, 12)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("block_size", [1, 9])
    def test_moving_block_bootstrap_rejects_invalid_block_size(self, block_size):
        with pytest.raises(ValueError, match="block_size"):
            augmentation.moving_block_bootstrap(np.arange(8.0), block_size, 42)

    def test_readonly_binding_rejects_noncontiguous_input(self):
        values = np.arange(16.0).reshape(8, 2)[:, 0]
        assert not values.flags.c_contiguous
        with pytest.raises(TypeError):
            augmentation.compute_features(values)


class TestBindingNegativeLength:
    """Bindings should reject length <= 0 with ValueError."""

    def test_random_walk_negative_length(self):
        with pytest.raises(ValueError):
            statistical.random_walk(-1, 0.0, 1.0, 0.0, 42, 0, 0.0)

    def test_random_walk_zero_length(self):
        with pytest.raises(ValueError):
            statistical.random_walk(0, 0.0, 1.0, 0.0, 42, 0, 0.0)

    def test_seasonal_negative_length(self):
        with pytest.raises(ValueError):
            statistical.seasonal(-1, 12, 1.0, 0.0, 1.0, 0.0, 42)

    def test_garch_negative_length(self):
        alpha = np.array([0.1])
        beta = np.array([0.85])
        with pytest.raises(ValueError):
            stochastic.garch(-1, 1, 1, 0.01, alpha, beta, 0.0, 0.04, 42, 0, 0.0)

    def test_copula_negative_length(self):
        corr = np.array([1.0, 0.5, 0.5, 1.0])
        with pytest.raises(ValueError):
            multivariate.copula(-1, 2, 0, 5.0, corr, 0, 0.0, 1.0, 42)


class TestBindingInvalidParams:
    """Bindings should reject invalid parameters."""

    def test_seasonal_negative_period(self):
        with pytest.raises(ValueError):
            statistical.seasonal(100, -1, 1.0, 0.0, 1.0, 0.0, 42)

    def test_seasonal_zero_period(self):
        with pytest.raises(ValueError):
            statistical.seasonal(100, 0, 1.0, 0.0, 1.0, 0.0, 42)


class TestPatternInjectionBindings:
    """Edge cases for the pattern_injection PyO3 bindings."""

    def test_missingness_zero_block_size_raises(self):
        values = np.zeros(100)
        with pytest.raises(ValueError):
            pattern_injection.add_missingness(values, 42, "block", 0.2, 0, 7)

    def test_missingness_zero_seasonal_period_raises(self):
        values = np.zeros(100)
        with pytest.raises(ValueError):
            pattern_injection.add_missingness(values, 42, "seasonal", 0.2, 3, 0)

    def test_missingness_metadata_matches_nans(self):
        values = np.zeros(200)
        out, meta = pattern_injection.add_missingness(values, 42, "random", 0.2, 3, 7)
        nan_pos = set(np.flatnonzero(np.isnan(out)).tolist())
        assert set(np.asarray(meta["missing_indices"]).tolist()) == nan_pos

    def test_anomalies_zero_fraction_empty(self):
        values = np.ones(50)
        out, meta = pattern_injection.add_anomalies(
            values, 42, ["spike"], 0.0, 10.0, -10.0, 5.0, 10
        )
        assert len(meta["anomaly_indices"]) == 0
        np.testing.assert_array_equal(out, np.ones(50))

    def test_anomalies_modifies_in_place(self):
        values = np.zeros(100)
        out, _ = pattern_injection.add_anomalies(
            values, 42, ["spike"], 0.1, 10.0, -10.0, 5.0, 10
        )
        # binding returns the same buffer it mutated
        assert out is values or np.shares_memory(out, values)
        assert (values != 0.0).any()

    def test_changepoints_deterministic_locations(self):
        values = np.zeros(100)
        out, meta = pattern_injection.add_changepoints(
            values,
            42,
            1,
            np.array([0.5]),
            "level",
            np.array([10.0]),
            np.array([]),
            np.array([]),
        )
        np.testing.assert_array_equal(meta["changepoint_indices"], [50])
        assert np.all(out[:50] == 0.0)
        assert np.all(out[50:] == 10.0)

    def test_changepoints_empty_series(self):
        values = np.zeros(0)
        out, meta = pattern_injection.add_changepoints(
            values,
            42,
            1,
            np.array([0.5]),
            "level",
            np.array([10.0]),
            np.array([]),
            np.array([]),
        )
        assert out.shape == (0,)

    def test_anomalies_single_point_series(self):
        values = np.zeros(1)
        out, meta = pattern_injection.add_anomalies(
            values, 42, ["spike"], 1.0, 10.0, -10.0, 5.0, 10
        )
        assert out.shape == (1,)
        assert out[0] == 10.0


class TestPatternInjectionAliasing:
    """The bindings mutate a caller-provided array, so they must refuse any
    input they cannot claim exclusive access to rather than alias it."""

    def test_changepoints_rejects_aliased_argument(self):
        # Passing the same array as both the mutated output and a read-only
        # parameter would create an aliasing &mut/& pair in Rust.
        values = np.arange(10.0)
        with pytest.raises(TypeError, match="already borrowed"):
            pattern_injection.add_changepoints(
                values, 42, 1, values, "level", values, values, values
            )

    def test_changepoints_leaves_aliased_input_untouched(self):
        values = np.arange(10.0)
        with pytest.raises(TypeError):
            pattern_injection.add_changepoints(
                values, 42, 1, values, "level", values, values, values
            )
        np.testing.assert_array_equal(values, np.arange(10.0))

    @pytest.mark.parametrize("fn", ["add_anomalies", "add_missingness"])
    def test_rejects_readonly_array(self, fn):
        values = np.zeros(50)
        values.flags.writeable = False
        with pytest.raises(TypeError, match="not writeable"):
            if fn == "add_anomalies":
                pattern_injection.add_anomalies(
                    values, 42, ["spike"], 0.5, 10.0, -10.0, 5.0, 10
                )
            else:
                pattern_injection.add_missingness(values, 42, "random", 0.5, 3, 7)
        np.testing.assert_array_equal(values, np.zeros(50))

    @pytest.mark.parametrize(
        "fn", ["add_anomalies", "add_missingness", "add_changepoints"]
    )
    def test_rejects_noncontiguous_array(self, fn):
        base = np.zeros((50, 2))
        values = base[:, 0]
        assert not values.flags.c_contiguous
        with pytest.raises(TypeError, match="not contiguous"):
            if fn == "add_anomalies":
                pattern_injection.add_anomalies(
                    values, 42, ["spike"], 0.5, 10.0, -10.0, 5.0, 10
                )
            elif fn == "add_missingness":
                pattern_injection.add_missingness(values, 42, "random", 0.5, 3, 7)
            else:
                pattern_injection.add_changepoints(
                    values,
                    42,
                    1,
                    np.array([0.5]),
                    "level",
                    np.array([10.0]),
                    np.array([]),
                    np.array([]),
                )

    def test_missingness_mutates_in_place(self):
        values = np.zeros(200)
        out, _ = pattern_injection.add_missingness(values, 42, "random", 0.2, 3, 7)
        assert out is values
        assert np.isnan(values).any()

    def test_changepoints_mutates_in_place(self):
        values = np.zeros(100)
        out, _ = pattern_injection.add_changepoints(
            values,
            42,
            1,
            np.array([0.5]),
            "level",
            np.array([10.0]),
            np.array([]),
            np.array([]),
        )
        assert out is values
        assert np.all(values[50:] == 10.0)


class TestDistributionEdgeCases:
    """Distribution functions should handle edge cases."""

    def test_norm_cdf_empty_array(self):
        x = np.array([], dtype=np.float64)
        result = distributions.norm_cdf(x)
        assert result.shape == (0,)

    def test_norm_ppf_empty_array(self):
        x = np.array([], dtype=np.float64)
        result = distributions.norm_ppf(x)
        assert result.shape == (0,)

    def test_norm_cdf_single_value(self):
        x = np.array([0.0])
        result = distributions.norm_cdf(x)
        assert abs(result[0] - 0.5) < 1e-6
