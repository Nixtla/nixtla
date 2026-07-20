"""Semantic parity tests between the Rust and Python pattern-injection paths.

Streams differ by design (Rust uses its own MT64 RNG seeded per call), so
these tests compare configured semantics: deterministic configurations must
match exactly, stochastic ones must match in counts/shapes/magnitudes.
"""

import numpy as np
import pytest

import synforecast._core as core_module
from synforecast._core import _add_anomalies, _add_changepoints, _add_missingness

pytestmark = pytest.mark.skipif(
    not core_module._HAS_RUST, reason="Rust backend not available"
)


def _run_both(monkeypatch, fn, values, *args):
    """Run fn on the Rust path and the Python fallback with fresh inputs."""
    out_rust, meta_rust = fn(values.copy(), np.random.default_rng(1), *args)
    monkeypatch.setattr(core_module, "_HAS_RUST", False)
    out_py, meta_py = fn(values.copy(), np.random.default_rng(2), *args)
    monkeypatch.setattr(core_module, "_HAS_RUST", True)
    return (out_rust, meta_rust), (out_py, meta_py)


def _base_series(length=500, seed=0):
    return np.random.default_rng(seed).standard_normal(length)


class TestChangepointParity:
    def test_level_deterministic_identical(self, monkeypatch):
        """Fully specified level changepoints are deterministic: both
        paths must produce bit-identical output and metadata."""
        args = (
            2,
            np.array([0.3, 0.7]),
            "level",
            np.array([5.0, -3.0]),
            np.array([]),
            np.array([]),
        )
        (out_r, meta_r), (out_p, meta_p) = _run_both(
            monkeypatch, _add_changepoints, _base_series(), *args
        )
        np.testing.assert_array_equal(out_r, out_p)
        np.testing.assert_array_equal(
            meta_r["changepoint_indices"], meta_p["changepoint_indices"]
        )

    def test_trend_deterministic_identical(self, monkeypatch):
        args = (
            1,
            np.array([0.5]),
            "trend",
            np.array([]),
            np.array([0.2]),
            np.array([]),
        )
        (out_r, _), (out_p, _) = _run_both(
            monkeypatch, _add_changepoints, _base_series(), *args
        )
        np.testing.assert_array_equal(out_r, out_p)

    @pytest.mark.parametrize("cp_type", ["level", "trend", "variance", "mixed"])
    def test_random_config_same_shape_and_count(self, cp_type, monkeypatch):
        args = (2, np.array([]), cp_type, np.array([]), np.array([]), np.array([]))
        (out_r, meta_r), (out_p, meta_p) = _run_both(
            monkeypatch, _add_changepoints, _base_series(), *args
        )
        assert out_r.shape == out_p.shape
        assert len(meta_r["changepoint_indices"]) == 2
        assert len(meta_p["changepoint_indices"]) == 2
        assert np.all(np.isfinite(out_p))

    def test_partial_locations_padded_both_paths(self, monkeypatch):
        """One location for two changepoints: both paths pad with a random
        location instead of failing."""
        args = (
            2,
            np.array([0.5]),
            "level",
            np.array([5.0, 5.0]),
            np.array([]),
            np.array([]),
        )
        (out_r, meta_r), (out_p, meta_p) = _run_both(
            monkeypatch, _add_changepoints, _base_series(), *args
        )
        assert len(meta_r["changepoint_indices"]) == 2
        assert len(meta_p["changepoint_indices"]) == 2
        assert meta_r["changepoint_indices"][0] == 250
        assert meta_p["changepoint_indices"][0] == 250

    def test_variance_bounded_both_paths(self, monkeypatch):
        """Variance changepoints stay finite and bounded on both paths
        (regression for the divergent Rust running-window implementation)."""
        args = (
            1,
            np.array([0.2]),
            "variance",
            np.array([]),
            np.array([]),
            np.array([2.0]),
        )
        (out_r, _), (out_p, _) = _run_both(
            monkeypatch, _add_changepoints, _base_series(2000), *args
        )
        assert np.max(np.abs(out_r)) < 100.0
        assert np.max(np.abs(out_p)) < 100.0

    def test_variance_deterministic_close(self, monkeypatch):
        """Fully specified variance changepoints apply the same rolling-mean
        rescaling on both paths (identical up to float summation order)."""
        args = (
            1,
            np.array([0.5]),
            "variance",
            np.array([]),
            np.array([]),
            np.array([2.0]),
        )
        (out_r, _), (out_p, _) = _run_both(
            monkeypatch, _add_changepoints, _base_series(1000), *args
        )
        np.testing.assert_allclose(out_r, out_p, rtol=1e-8, atol=1e-8)


class TestAnomalyParity:
    @pytest.mark.parametrize(
        "anomaly_types",
        [["spike"], ["dip"], ["level_shift"], ["spike", "dip", "level_shift"]],
    )
    def test_event_count_matches(self, anomaly_types, monkeypatch):
        args = (anomaly_types, 0.05, 10.0, -10.0, 5.0, 20)
        (out_r, meta_r), (out_p, meta_p) = _run_both(
            monkeypatch, _add_anomalies, np.zeros(500), *args
        )
        expected = int(500 * 0.05)
        assert len(meta_r["anomaly_indices"]) == expected
        assert len(meta_p["anomaly_indices"]) == expected
        assert (out_r != 0).any() and (out_p != 0).any()

    def test_spike_magnitudes_match(self, monkeypatch):
        args = (["spike"], 0.05, 12.5, -10.0, 5.0, 20)
        (out_r, _), (out_p, _) = _run_both(
            monkeypatch, _add_anomalies, np.zeros(1000), *args
        )
        # Both paths sample locations without replacement: exactly
        # floor(n*fraction) distinct points, each spiked exactly once.
        expected = int(1000 * 0.05)
        assert np.allclose(out_p[out_p != 0], 12.5)
        assert np.allclose(out_r[out_r != 0], 12.5)
        assert (out_r != 0).sum() == expected
        assert (out_p != 0).sum() == expected


class TestMissingnessParity:
    @pytest.mark.parametrize("pattern", ["random", "block", "seasonal"])
    def test_both_paths_inject_nans(self, pattern, monkeypatch):
        args = (pattern, 0.1, 10, 7)
        (out_r, meta_r), (out_p, meta_p) = _run_both(
            monkeypatch, _add_missingness, np.zeros(500), *args
        )
        assert out_r.shape == out_p.shape
        assert np.isnan(out_r).any() and np.isnan(out_p).any()
        # metadata matches actual NaN positions on both paths
        assert set(np.asarray(meta_p["missing_indices"]).tolist()) == set(
            np.flatnonzero(np.isnan(out_p)).tolist()
        )
        assert set(np.asarray(meta_r["missing_indices"]).tolist()) == set(
            np.flatnonzero(np.isnan(out_r)).tolist()
        )

    def test_random_counts_exact_both_paths(self, monkeypatch):
        """Both paths sample without replacement: exactly floor(n*rate)
        distinct NaNs."""
        n, rate = 2000, 0.2
        args = ("random", rate, 10, 7)
        (out_r, _), (out_p, _) = _run_both(
            monkeypatch, _add_missingness, np.zeros(n), *args
        )
        expected = int(n * rate)
        assert np.isnan(out_p).sum() == expected
        assert np.isnan(out_r).sum() == expected

    def test_block_validation_matches(self, monkeypatch):
        """Both paths raise ValueError for a non-positive block size."""
        with pytest.raises(ValueError):
            _add_missingness(
                np.zeros(100), np.random.default_rng(0), "block", 0.2, 0, 7
            )
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        with pytest.raises(ValueError):
            _add_missingness(
                np.zeros(100), np.random.default_rng(0), "block", 0.2, 0, 7
            )
