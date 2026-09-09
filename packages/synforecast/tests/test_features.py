"""Tests for the minimal feature-targeting feature set."""

import numpy as np
import pytest

from synforecast._features import (
    acf1,
    classical_decompose,
    compute_features,
    seasonal_strength,
    spectral_entropy,
    trend_strength,
)
from tests.helpers import sample_acf


def _moving_average(values: np.ndarray, period: int | None) -> np.ndarray:
    """Compute a centered moving average with endpoint extension."""
    n = len(values)
    period = period if period is not None and period <= n // 2 else None
    weights: np.ndarray
    if period is None:
        window = max(3, (n // 10) | 1)
        if window > n:
            window = n if n % 2 else n - 1
        weights = np.full(window, 1.0 / window)
    elif period % 2:
        weights = np.full(period, 1.0 / period)
    else:
        weights = np.concatenate(
            ([0.5 / period], np.full(period - 1, 1.0 / period), [0.5 / period])
        )

    width = len(weights)
    if width > n:
        window = n if n % 2 else n - 1
        weights = np.full(window, 1.0 / window)
        width = window
    computed = np.convolve(values, weights, mode="valid")
    left = (width - 1) // 2
    right = n - len(computed) - left
    return np.pad(computed, (left, right), mode="edge")


class TestFeatureComputation:
    """Behavior and edge-case tests for private numeric features."""

    @pytest.mark.parametrize(
        ("period", "values", "trend", "phase", "remainder"),
        [
            (
                3,
                [-1, 3, 1, 2, 6, 4, 5, 9, 7],
                [1, 1, 2, 3, 4, 5, 6, 7, 7],
                [-4 / 3, 2, -2 / 3],
                [-2 / 3, 0, -1 / 3, 1 / 3, 0, -1 / 3, 1 / 3, 0, 2 / 3],
            ),
            (
                2,
                [1, 0, 3, 2, 5, 4, 7, 6],
                [1, 1, 2, 3, 4, 5, 6, 6],
                [0.75, -0.75],
                [-0.75, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 0.75],
            ),
        ],
    )
    def test_components_and_strengths_against_hand_calculation(
        self, period, values, trend, phase, remainder
    ) -> None:
        # Linear trend plus a repeating phase pattern. Endpoint extension
        # changes the phase means, so merely reconstructing y is insufficient.
        values, trend, remainder = map(np.asarray, (values, trend, remainder))
        seasonal = np.resize(phase, len(values))
        actual = classical_decompose(values, period)
        for got, expected in zip(actual, (trend, seasonal, remainder), strict=True):
            np.testing.assert_allclose(got, expected, atol=1e-12)
        features = compute_features(values, period)
        expected_trend = 1 - np.var(remainder) / np.var(trend + remainder)
        expected_seasonal = 1 - np.var(remainder) / np.var(seasonal + remainder)
        assert features["trend_strength"] == pytest.approx(expected_trend)
        assert features["seasonal_strength"] == pytest.approx(expected_seasonal)
        assert trend_strength(values, period) == pytest.approx(expected_trend)
        assert seasonal_strength(values, period) == pytest.approx(expected_seasonal)

    @pytest.mark.parametrize("period", [None, 2, 7, 12, 20_000])
    @pytest.mark.parametrize("offset", [0.0, 1e9])
    def test_rolling_trend_matches_direct_convolution(
        self, period: int | None, offset: float
    ) -> None:
        values = offset + np.random.default_rng(42).normal(size=10_001)
        expected = _moving_average(values, period)
        actual, _, _ = classical_decompose(values, period)
        np.testing.assert_allclose(
            actual, expected, rtol=0.0, atol=1e-12 if offset == 0 else 1e-6
        )

    def test_compute_features_contract(self) -> None:
        values = np.random.default_rng(0).normal(size=128)
        features = compute_features(values, 12)
        assert set(features) == {
            "spectral_entropy",
            "trend_strength",
            "seasonal_strength",
            "acf1",
        }
        assert all(isinstance(value, float) for value in features.values())
        assert all(np.isfinite(value) for value in features.values())

    @pytest.mark.parametrize("length", [63, 64, 121, 1000, 1001, 4093, 4095, 4096])
    def test_native_feature_tuple_matches_public_helpers(self, length: int) -> None:
        values = np.random.default_rng(length).normal(size=length)
        features = compute_features(values, 12)
        assert features["spectral_entropy"] == pytest.approx(
            spectral_entropy(values), abs=1e-12
        )
        assert features["trend_strength"] == pytest.approx(
            trend_strength(values, 12), abs=1e-12
        )
        assert features["seasonal_strength"] == pytest.approx(
            seasonal_strength(values, 12), abs=1e-12
        )
        assert features["acf1"] == pytest.approx(acf1(values), abs=1e-12)

    @pytest.mark.parametrize("period", [None, 7, 12])
    def test_decomposition_reconstructs_input(self, period: int | None) -> None:
        values = np.random.default_rng(1).normal(size=121)
        trend, seasonal, remainder = classical_decompose(values, period)
        np.testing.assert_allclose(
            trend + seasonal + remainder, values, rtol=0.0, atol=1e-15
        )

    def test_trend_strength_distinguishes_trend_and_noise(self) -> None:
        rng = np.random.default_rng(2)
        trend = np.linspace(0, 20, 300) + rng.normal(0, 0.1, 300)
        noise = rng.normal(size=300)
        assert trend_strength(trend, None) > 0.7
        assert trend_strength(noise, None) < 0.3

    def test_seasonal_strength_and_nonseasonal_mode(self) -> None:
        rng = np.random.default_rng(3)
        time = np.arange(480)
        values = np.sin(2 * np.pi * time / 24) + rng.normal(0, 0.05, len(time))
        assert seasonal_strength(values, 24) > 0.7
        assert seasonal_strength(values, None) == 0.0

    def test_spectral_entropy_distinguishes_sine_and_noise(self) -> None:
        time = np.arange(1024)
        sine = np.sin(2 * np.pi * time / 32)
        noise = np.random.default_rng(4).normal(size=len(time))
        assert spectral_entropy(sine) < 0.5
        assert spectral_entropy(noise) > 0.9

    def test_acf_matches_shared_helper(self) -> None:
        values = np.random.default_rng(5).normal(size=200)
        assert acf1(values) == pytest.approx(sample_acf(values, 1), abs=1e-6)

    def test_constant_and_length_three_edges(self) -> None:
        constant = np.ones(3)
        features = compute_features(constant, None)
        assert features["spectral_entropy"] == 0.0
        assert features["acf1"] == 0.0
        assert all(np.isfinite(value) for value in features.values())

    @pytest.mark.parametrize("values", [[1.0, np.nan, 2.0], [1.0, np.inf, 2.0]])
    def test_nonfinite_input_rejected(self, values: list[float]) -> None:
        with pytest.raises(ValueError, match="finite"):
            compute_features(np.asarray(values), None)

    @pytest.mark.parametrize("values", [np.ones((2, 3)), np.ones(2)])
    def test_invalid_shape_or_length_rejected(self, values: np.ndarray) -> None:
        with pytest.raises(ValueError, match="one-dimensional|at least 3"):
            compute_features(values, None)

    def test_invalid_decomposition_period_rejected(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 2"):
            classical_decompose(np.ones(12), 1)
