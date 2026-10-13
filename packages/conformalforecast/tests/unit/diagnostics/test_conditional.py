"""Tests for diagnostics.conditional."""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.diagnostics.conditional import (
    coverage_by_group,
    coverage_by_magnitude_bin,
)


class TestCoverageByMagnitudeBin:
    def test_high_magnitude_better_coverage_when_widths_scale(self) -> None:
        # Construct truth values across a wide magnitude range.
        rng = np.random.default_rng(0)
        n = 400
        truth = rng.normal(scale=5.0, size=(1, n, 1))

        # Width scales with |truth|: large-magnitude cells get correspondingly
        # wider intervals (covered), small-magnitude cells get narrow
        # intervals that miss often.
        half_width = np.where(np.abs(truth) > 3.0, 10.0, 0.05)
        interval = np.stack([truth - half_width, truth + half_width], axis=-1)
        # Force the small-width cells to miss by shifting the interval by a
        # value larger than the small width.
        small_mask = np.abs(truth) <= 3.0
        shifted_lower = interval[..., 0] + np.where(small_mask, 1.0, 0.0)
        shifted_upper = interval[..., 1] + np.where(small_mask, 1.0, 0.0)
        interval = np.stack([shifted_lower, shifted_upper], axis=-1)

        out = coverage_by_magnitude_bin(interval, truth, n_bins=5, by="truth")
        assert out["coverage_per_bin"].shape == (5,)
        assert out["mean_width_per_bin"].shape == (5,)
        assert out["bin_edges"].shape == (6,)
        assert out["n_per_bin"].shape == (5,)
        # High-magnitude bin (last) should have higher coverage than the
        # low-magnitude bin (first).
        assert out["coverage_per_bin"][-1] > out["coverage_per_bin"][0]

    def test_bin_edges_monotonic(self) -> None:
        rng = np.random.default_rng(1)
        truth = rng.normal(size=(2, 50, 3))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        out = coverage_by_magnitude_bin(interval, truth, n_bins=8)
        edges = out["bin_edges"]
        assert (np.diff(edges) >= 0).all()

    def test_n_per_bin_sums_to_total_cells(self) -> None:
        rng = np.random.default_rng(2)
        n_series, n_samples, horizon = 3, 50, 2
        truth = rng.normal(size=(n_series, n_samples, horizon))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        out = coverage_by_magnitude_bin(interval, truth, n_bins=10)
        assert int(out["n_per_bin"].sum()) == n_series * n_samples * horizon

    def test_by_midpoint(self) -> None:
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(1, 30, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        out = coverage_by_magnitude_bin(interval, truth, n_bins=4, by="midpoint")
        assert out["coverage_per_bin"].shape == (4,)

    def test_invalid_n_bins_raises(self) -> None:
        truth = np.zeros((1, 10, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        with pytest.raises(ValueError, match="n_bins"):
            coverage_by_magnitude_bin(interval, truth, n_bins=1)

    def test_invalid_by_raises(self) -> None:
        truth = np.zeros((1, 10, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        with pytest.raises(ValueError, match="by"):
            coverage_by_magnitude_bin(interval, truth, n_bins=4, by="bogus")  # type: ignore[arg-type]


class TestCoverageByGroup:
    def test_two_groups_differ(self) -> None:
        # Group 0 has narrow intervals (always miss).
        # Group 1 has wide intervals (always cover).
        rng = np.random.default_rng(0)
        n_samples = 20
        truth = rng.normal(size=(1, n_samples, 1))
        half = n_samples // 2
        narrow = np.stack([truth + 100.0, truth + 200.0], axis=-1)
        wide = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        interval = np.empty((1, n_samples, 1, 2))
        interval[:, :half, :, :] = narrow[:, :half, :, :]
        interval[:, half:, :, :] = wide[:, half:, :, :]

        groups = np.empty((1, n_samples, 1), dtype=np.int64)
        groups[:, :half, :] = 0
        groups[:, half:, :] = 1
        out = coverage_by_group(interval, truth, groups)

        assert out["groups"].shape == (2,)
        np.testing.assert_array_equal(out["groups"], np.array([0, 1]))
        np.testing.assert_allclose(out["coverage_per_group"], np.array([0.0, 1.0]))
        np.testing.assert_array_equal(out["n_per_group"], np.array([half, n_samples - half]))

    def test_string_groups(self) -> None:
        truth = np.zeros((1, 4, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        groups = np.array([[["a"], ["b"], ["a"], ["b"]]])
        out = coverage_by_group(interval, truth, groups)
        np.testing.assert_array_equal(out["groups"], np.array(["a", "b"]))

    def test_shape_mismatch_raises(self) -> None:
        truth = np.zeros((1, 4, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        groups = np.zeros((1, 5, 1), dtype=np.int64)  # wrong sample count
        with pytest.raises(ValueError, match="groups"):
            coverage_by_group(interval, truth, groups)
