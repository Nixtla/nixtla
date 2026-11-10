"""Independent numerical oracles for the native coverage candidates."""

import numpy as np
import pytest

from synforecast._features import (
    FEATURE_NAMES,
    compute_feature_matrix,
    compute_feature_set,
    compute_targeting_features,
)
from tests.test_features import _moving_average


def oracle(values, period, window=None):
    """Direct calculations favor clarity over the Rust kernel's rolling updates."""
    x = (
        (values - values.mean()) / values.std()
        if values.std()
        else np.zeros_like(values)
    )

    def acf(a, lag):
        centered = a - a.mean()
        return (
            float(centered[:-lag] @ centered[lag:] / (centered @ centered))
            if np.var(a)
            else 0.0
        )

    width = window or period or 10
    trend = _moving_average(x, period)
    detrended = x - trend
    if period is None:
        remainder = detrended
    else:
        phases = np.array([detrended[i::period].mean() for i in range(period)])
        remainder = detrended - np.resize(phases - phases.mean(), len(x))
    windows = np.lib.stride_tricks.sliding_window_view(x, width)
    return {
        "x_acf10": sum(acf(x, lag) ** 2 for lag in range(1, 11)),
        "diff1_acf1": acf(np.diff(x), 1),
        "seas_acf1": acf(x, period) if period else 0.0,
        "spike": np.var([np.var(np.delete(remainder, i)) for i in range(len(x))]),
        "lumpiness": np.var(
            [np.var(x[i : i + width]) for i in range(0, len(x) - width + 1, width)]
        ),
        "max_level_shift": np.max(
            np.abs(windows.mean(axis=1)[width:] - windows.mean(axis=1)[:-width])
        ),
        "max_var_shift": np.max(
            np.abs(windows.var(axis=1)[width:] - windows.var(axis=1)[:-width])
        ),
        "crossing_points": np.count_nonzero(np.diff(x <= np.median(x))),
    }


@pytest.mark.parametrize("period", [2, 3, 4, 12])
@pytest.mark.parametrize("length", [30, 49, 103])
def test_against_direct_oracles(period, length):
    values = np.random.default_rng(length).normal(size=length)
    actual = compute_feature_set(values, period)
    assert tuple(actual) == FEATURE_NAMES
    for name, expected in oracle(values, period).items():
        assert actual[name] == pytest.approx(expected, abs=1e-12), name
    # The four MAR-targeting features are scale-free, so their raw-input
    # helper agrees with the normalized evaluation path up to rounding.
    for name, expected in compute_targeting_features(values, period).items():
        assert actual[name] == pytest.approx(expected, abs=1e-10), name


@pytest.mark.parametrize(
    "period,window,length",
    [(None, None, 64), (None, 5, 23), (12, 5, 40), (24, 7, 60), (4, 9, 50)],
)
def test_oracles_without_period_and_with_explicit_windows(period, window, length):
    values = np.random.default_rng(length).normal(size=length)
    actual = compute_feature_set(values, period, window)
    for name, expected in oracle(values, period, window).items():
        assert actual[name] == pytest.approx(expected, abs=1e-12), name


@pytest.mark.parametrize("kind", ["outlier", "intermittent"])
def test_rolling_window_updates_on_long_series(kind):
    rng = np.random.default_rng(3)
    if kind == "outlier":
        values = rng.normal(size=5_000)
        values[2_500] = 1e6
    else:
        values = np.where(rng.random(5_000) < 0.05, rng.gamma(2.0, size=5_000), 0.0)
    actual = compute_feature_set(values, 12)
    expected = oracle(values, 12)
    for name in ("lumpiness", "max_level_shift", "max_var_shift"):
        assert actual[name] == pytest.approx(expected[name], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1e-9, 0.01, 3.0, 1e160])
@pytest.mark.parametrize("offset", [0.0, 100.0])
def test_all_features_scale_invariant(scale, offset):
    values = np.random.default_rng(54).normal(size=120)
    expected = compute_feature_set(values, 12)
    actual = compute_feature_set(scale * values + offset * scale, 12)
    for name in FEATURE_NAMES:
        assert actual[name] == pytest.approx(expected[name], rel=1e-8, abs=1e-10), name


def test_analytic_values():
    n, period = 120, 12
    sine = np.sin(2 * np.pi * np.arange(n) / period)
    features = compute_feature_set(sine, period)
    assert features["seasonal_strength"] == pytest.approx(1.0, abs=1e-10)
    assert features["trend_strength"] == pytest.approx(0.0, abs=1e-10)
    assert features["seas_acf1"] == pytest.approx((n - period) / n, abs=1e-12)
    pure = np.sin(2 * np.pi * np.arange(1024) / 32)
    assert compute_feature_set(pure, None)["spectral_entropy"] < 1e-6
    alternating = np.tile([-1.0, 1.0], 20)
    assert compute_feature_set(alternating, None)["crossing_points"] == 39


def test_ar1_and_white_noise_statistics():
    rng = np.random.default_rng(1)
    phi, n = 0.6, 50_000
    noise = rng.normal(size=n)
    values = np.empty(n)
    values[0] = noise[0]
    for i in range(1, n):
        values[i] = phi * values[i - 1] + noise[i]
    features = compute_feature_set(values, None)
    assert features["acf1"] == pytest.approx(phi, abs=0.015)
    expected_x_acf10 = phi**2 * (1 - phi**20) / (1 - phi**2)
    assert features["x_acf10"] == pytest.approx(expected_x_acf10, abs=0.03)
    assert compute_feature_set(noise, None)["diff1_acf1"] == pytest.approx(
        -0.5, abs=0.015
    )


@pytest.mark.parametrize("n", range(3, 41))
@pytest.mark.parametrize("period", [None, 2, 3, 5, 12, 20])
@pytest.mark.parametrize("window", [None, 2, 7])
def test_undefined_value_rules(n, period, window):
    features = compute_feature_set(
        np.random.default_rng(n).normal(size=n), period, window
    )
    unusable = period is not None and period > n // 2
    for name in ("trend_strength", "seasonal_strength", "spike"):
        assert np.isnan(features[name]) == unusable, name
    assert np.isnan(features["x_acf10"]) == (n <= 10)
    assert np.isnan(features["seas_acf1"]) == (period is not None and period >= n)
    width = window or period or 10
    for name in ("lumpiness", "max_level_shift", "max_var_shift"):
        assert np.isnan(features[name]) == (width > n // 2), name
    for name in ("spectral_entropy", "acf1", "diff1_acf1", "crossing_points"):
        assert np.isfinite(features[name]), name


@pytest.mark.parametrize("period,window", [(None, None), (4, None), (12, 5)])
def test_batch_kernel_matches_per_series(period, window):
    rng = np.random.default_rng(9)
    series = [rng.normal(size=n) for n in (2, 30, 7, 55, 41, 30, 200)]
    offsets = np.r_[0, np.cumsum([len(x) for x in series])]
    for workers in (0, 1, 3):
        matrix = compute_feature_matrix(
            np.concatenate(series), offsets, period, window, workers
        )
        assert matrix.shape == (len(series), len(FEATURE_NAMES))
        for row, values in zip(matrix, series, strict=True):
            expected = list(compute_feature_set(values, period, window).values())
            np.testing.assert_array_equal(row, expected)


@pytest.mark.parametrize("offsets", [[1, 3], [0, 2, 2, 4], [0, 3, 2], [0, 5]])
def test_batch_kernel_rejects_invalid_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        compute_feature_matrix(np.arange(4.0), np.asarray(offsets), None)


def test_constant_and_short_contracts():
    actual = compute_feature_set(np.ones(100), 12)
    assert all(value == 0.0 for value in actual.values())
    assert all(np.isnan(x) for x in compute_feature_set(np.ones(2), None).values())
    short = compute_feature_set(np.arange(10.0), 12)
    assert np.isnan(short["seasonal_strength"])
    assert np.isnan(short["trend_strength"])
    assert np.isnan(short["spike"])
    assert np.isnan(short["x_acf10"])
    assert np.isnan(short["seas_acf1"])
    assert np.isnan(short["lumpiness"])
    assert np.isnan(short["max_level_shift"])
    # Evaluation marks unavailable cycles; MAR's existing fallback is unchanged.
    assert compute_targeting_features(np.arange(10.0), 12)["seasonal_strength"] == 0
    without_period = compute_feature_set(np.arange(20.0), None)
    assert without_period["seas_acf1"] == 0
    assert np.isfinite(without_period["lumpiness"])
    assert np.isnan(compute_feature_set(np.arange(19.0), None)["lumpiness"])


@pytest.mark.parametrize(
    "values", [np.array([]), np.array([1.0, np.inf]), np.ones((3, 3))]
)
def test_invalid_values(values):
    with pytest.raises(ValueError):
        compute_feature_set(values, None)


def test_step_and_variance_change():
    step = np.r_[np.zeros(20), np.ones(20)]
    result = compute_feature_set(step, None)
    assert result["max_level_shift"] == pytest.approx(2.0)
    assert result["crossing_points"] == 1
    variance_change = np.r_[np.tile([-1.0, 1.0], 10), np.tile([-3.0, 3.0], 10)]
    result = compute_feature_set(variance_change, None)
    assert result["max_var_shift"] == pytest.approx(1.6)
    assert result["lumpiness"] == pytest.approx(0.64)


def test_explicit_window_on_short_nonseasonal_series():
    values = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, 1.0, 2.0])
    default = compute_feature_set(values, None)
    actual = compute_feature_set(values, None, window_size=5)
    x = (values - values.mean()) / values.std()
    windows = np.lib.stride_tricks.sliding_window_view(x, 5)
    assert np.isnan(default["lumpiness"])
    assert actual["lumpiness"] == pytest.approx(
        np.var([np.var(x[:5]), np.var(x[5:10])])
    )
    assert actual["max_level_shift"] == pytest.approx(
        np.max(np.abs(windows.mean(axis=1)[5:] - windows.mean(axis=1)[:-5]))
    )
    assert actual["max_var_shift"] == pytest.approx(
        np.max(np.abs(windows.var(axis=1)[5:] - windows.var(axis=1)[:-5]))
    )
    for name in set(FEATURE_NAMES) - {"lumpiness", "max_level_shift", "max_var_shift"}:
        assert actual[name] == default[name]
