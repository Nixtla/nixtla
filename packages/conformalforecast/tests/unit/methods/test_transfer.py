import numpy as np
import pytest

from conformalforecast.methods.transfer import compute_series_scales, scales_from_truths


def test_std_scale_of_random_walk_matches_step_std():
    rng = np.random.default_rng(0)
    steps = rng.normal(scale=3.0, size=(1, 5000))
    series = np.cumsum(steps, axis=1)
    scale = compute_series_scales(series, "std")
    assert scale.shape == (1,)
    assert abs(scale[0] - 3.0) < 0.2


def test_mad_scale_is_robust_to_an_outlier_step():
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.normal(scale=1.0, size=(1, 2000)), axis=1)
    contaminated = series.copy()
    contaminated[0, 1000:] += 500.0  # one enormous jump
    mad = compute_series_scales(contaminated, "mad")[0]
    std = compute_series_scales(contaminated, "std")[0]
    assert mad < std / 5


def test_iqr_estimator_is_supported():
    rng = np.random.default_rng(2)
    series = np.cumsum(rng.normal(size=(2, 500)), axis=1)
    assert compute_series_scales(series, "iqr").shape == (2,)


def test_scales_are_per_series():
    rng = np.random.default_rng(3)
    quiet = np.cumsum(rng.normal(scale=1.0, size=(1, 2000)), axis=1)
    loud = np.cumsum(rng.normal(scale=10.0, size=(1, 2000)), axis=1)
    scales = compute_series_scales(np.vstack([quiet, loud]), "std")
    assert scales[1] > 5 * scales[0]


def test_leading_nan_padding_is_ignored():
    rng = np.random.default_rng(4)
    series = np.cumsum(rng.normal(size=(1, 1000)), axis=1)
    padded = np.concatenate([np.full((1, 200), np.nan), series], axis=1)
    np.testing.assert_allclose(
        compute_series_scales(padded, "std"),
        compute_series_scales(series, "std"),
    )


def test_scales_from_truths_handles_horizon_one():
    rng = np.random.default_rng(5)
    truths = rng.normal(size=(2, 100, 1))
    assert scales_from_truths(truths, "std").shape == (2,)


def test_scales_from_truths_ignores_nan_windows():
    rng = np.random.default_rng(6)
    truths = rng.normal(size=(2, 100, 3))
    truths[0, :10, :] = np.nan
    scales = scales_from_truths(truths, "std")
    assert np.all(np.isfinite(scales))


def test_scales_from_truths_raises_for_series_with_too_few_observations():
    rng = np.random.default_rng(8)
    truths = np.zeros((2, 3, 1))
    # Series 0: healthy random walk
    steps = rng.normal(size=3)
    truths[0, :, 0] = np.cumsum(steps)
    # Series 1: only 1 usable observation (rest are NaN)
    truths[1, :, 0] = np.nan
    truths[1, 0, 0] = 1.0
    with pytest.raises(ValueError, match=r"\[1\]"):
        scales_from_truths(truths, "std")


def test_unknown_estimator_raises():
    with pytest.raises(ValueError, match="estimator must be"):
        compute_series_scales(np.zeros((1, 10)), "variance")


def test_degenerate_constant_series_raises_naming_indices():
    # A noisy series, not arange: scales are computed on first differences, and
    # arange's differences are a constant 1.0 — as degenerate as ones's constant
    # 0.0, which would make both rows raise and defeat the point of this test.
    healthy = np.cumsum(np.random.default_rng(7).normal(size=100))
    series = np.vstack([healthy, np.ones(100)])
    with pytest.raises(ValueError, match=r"\[1\]"):
        compute_series_scales(series, "std")


def test_series_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        compute_series_scales(np.zeros((1, 1)), "std")
