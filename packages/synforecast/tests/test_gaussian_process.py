"""Tests for GaussianProcessGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import GaussianProcessGenerator
from tests.helpers import assert_distribution, assert_long_format, series_values

BASE = {"min_length": 30, "max_length": 60, "freq": "D", "seed": 42}

KERNELS = ["rbf", "matern_0.5", "matern_1.5", "matern_2.5", "periodic"]


def kernel_correlation(
    kernel: str, r: float, length_scale: float, period: float = 50.0
) -> float:
    """Theoretical correlation k(r)/k(0) for each kernel."""
    if kernel == "rbf":
        return float(np.exp(-0.5 * (r / length_scale) ** 2))
    if kernel == "matern_0.5":
        return float(np.exp(-r / length_scale))
    if kernel == "matern_1.5":
        s = np.sqrt(3.0) * r / length_scale
        return float((1.0 + s) * np.exp(-s))
    if kernel == "matern_2.5":
        s = np.sqrt(5.0) * r / length_scale
        return float((1.0 + s + s**2 / 3.0) * np.exp(-s))
    if kernel == "periodic":
        return float(np.exp(-2.0 * (np.sin(np.pi * r / period) / length_scale) ** 2))
    raise ValueError(kernel)


def correlation_known_mean(vals: list, mean: float, lag: int) -> float:
    """Pooled lag-k correlation using the known process mean (unbiased)."""
    num = den = 0.0
    for v in vals:
        d = v - mean
        num += d[:-lag] @ d[lag:]
        den += d @ d
    return num / den


class TestGPApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = GaussianProcessGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=30, max_length=60)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = GaussianProcessGenerator(**BASE, engine=engine).generate(n_series=3)
        df2 = GaussianProcessGenerator(**BASE, engine=engine).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_all_kernels_produce_finite_output(self, kernel: str) -> None:
        gen = GaussianProcessGenerator(**BASE, kernel=kernel)
        values = gen.generate_single_series(50)
        assert values.shape == (50,)
        assert np.all(np.isfinite(values))

    def test_invalid_kernel(self) -> None:
        with pytest.raises(ValueError):
            GaussianProcessGenerator(**BASE, kernel="polynomial")

    @pytest.mark.parametrize(
        "field,value",
        [("length_scale", 0.0), ("amplitude", -1.0), ("noise_variance", -0.1)],
    )
    def test_invalid_parameters(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            GaussianProcessGenerator(**BASE, **{field: value})

    def test_get_model_info(self) -> None:
        info = GaussianProcessGenerator(**BASE, kernel="periodic").get_model_info()
        assert info["kernel"] == "periodic"
        assert info["period"] == 50.0
        info_rbf = GaussianProcessGenerator(**BASE, kernel="rbf").get_model_info()
        assert info_rbf["period"] is None


@pytest.mark.stats
class TestGPStats:
    """Statistical property tests (fixed seeds)."""

    def test_marginal_distribution(self) -> None:
        """First point of each series is iid N(mean, amplitude^2 + noise)."""
        gen = GaussianProcessGenerator(
            min_length=256,
            max_length=256,
            freq="D",
            kernel="rbf",
            length_scale=10.0,
            amplitude=2.0,
            mean=1.0,
            seed=9,
        )
        firsts = np.array([gen.generate_single_series(256)[0] for _ in range(300)])
        sigma = np.sqrt(2.0**2 + gen.noise_variance)
        assert_distribution(firsts, stats.norm(1.0, sigma))

    @pytest.mark.parametrize(
        "kernel", ["rbf", "matern_0.5", "matern_1.5", "matern_2.5"]
    )
    def test_acf_matches_kernel(self, kernel: str) -> None:
        """Lag-k correlation matches k(r)/k(0) at small lags."""
        ls = 10.0
        gen = GaussianProcessGenerator(
            min_length=1024,
            max_length=1024,
            freq="D",
            kernel=kernel,
            length_scale=ls,
            amplitude=2.0,
            mean=1.0,
            seed=5,
        )
        vals = [gen.generate_single_series(1024) for _ in range(12)]
        for lag, tol in [(1, 0.03), (5, 0.05)]:
            got = correlation_known_mean(vals, 1.0, lag)
            expected = kernel_correlation(kernel, lag, ls)
            assert abs(got - expected) < tol, (
                f"{kernel} corr(lag={lag}) {got:.4f} != {expected:.4f}"
            )

    def test_periodic_kernel_periodicity(self) -> None:
        """Correlation at a full period is much higher than at a half period."""
        gen = GaussianProcessGenerator(
            min_length=500,
            max_length=500,
            freq="D",
            kernel="periodic",
            length_scale=1.0,
            period=50.0,
            amplitude=2.0,
            seed=5,
        )
        vals = [gen.generate_single_series(500) for _ in range(10)]
        corr_full = correlation_known_mean(vals, 0.0, 50)
        corr_half = correlation_known_mean(vals, 0.0, 25)
        assert corr_full > 0.5
        assert corr_full - corr_half > 0.3

    def test_smoothness_ordering(self) -> None:
        """Rougher kernels have larger first-difference variance: for a
        stationary GP, E[(y_{t+1}-y_t)^2] = 2 * k(0) * (1 - corr(1))."""
        diffs = {}
        for kernel in ["rbf", "matern_0.5", "matern_1.5", "matern_2.5"]:
            gen = GaussianProcessGenerator(
                min_length=512,
                max_length=512,
                freq="D",
                kernel=kernel,
                length_scale=10.0,
                seed=21,
            )
            vals = [gen.generate_single_series(512) for _ in range(8)]
            diffs[kernel] = float(np.mean([np.diff(v) ** 2 for v in vals]))
        assert diffs["matern_0.5"] > diffs["matern_1.5"] > diffs["rbf"]
        assert diffs["matern_0.5"] > diffs["matern_2.5"] > diffs["rbf"]

    def test_native_matches_theory(self) -> None:
        """The native implementation reproduces marginal scale and ACF."""
        ls = 10.0
        gen = GaussianProcessGenerator(
            min_length=384,
            max_length=384,
            freq="D",
            kernel="matern_0.5",
            length_scale=ls,
            amplitude=2.0,
            mean=0.0,
            seed=31,
        )
        vals = [gen.generate_single_series(384) for _ in range(12)]
        pooled = np.concatenate(vals)
        # marginal std = amplitude (pooled samples are correlated, so use a
        # manual 10% tolerance rather than an iid z-bound)
        assert abs(pooled.std() - 2.0) / 2.0 < 0.1
        got = correlation_known_mean(vals, 0.0, 1)
        assert abs(got - kernel_correlation("matern_0.5", 1, ls)) < 0.03
