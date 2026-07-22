"""Tests for PyO3 binding input validation and edge cases."""

import numpy as np
import pytest

from synforecast._lib import (
    distributions,
    multivariate,
    pattern_injection,
    statistical,
    stochastic,
)


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
