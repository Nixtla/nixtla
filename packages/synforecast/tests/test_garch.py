"""Tests for GARCHGenerator."""

import numpy as np
import pytest
from scipy import stats

import synforecast.generators.garch as garch_module
from synforecast.generators import GARCHGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    sample_acf,
    series_values,
)


def _make(engine: str = "pandas", **overrides) -> GARCHGenerator:
    params = {
        "min_length": 50,
        "max_length": 80,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(overrides)
    return GARCHGenerator(**params)


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _make(engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine).generate(n_series=2)
        df2 = _make(engine).generate(n_series=2)
        vals1 = series_values(df1)
        vals2 = series_values(df2)
        assert vals1.keys() == vals2.keys()
        for uid in vals1:
            np.testing.assert_array_equal(vals1[uid], vals2[uid])

    def test_start_id(self, engine: str) -> None:
        df = _make(engine).generate(n_series=2, start_id=5)
        assert set(series_values(df).keys()) == {"5", "6"}

    def test_higher_order(self, engine: str) -> None:
        gen = _make(
            engine,
            p=2,
            q=2,
            alpha=[0.05, 0.05],
            beta=[0.4, 0.3],
            min_length=60,
            max_length=60,
        )
        df = gen.generate(n_series=1)
        assert_long_format(df, n_series=1, min_length=60, max_length=60)


class TestValidation:
    def test_stationarity_rejection(self) -> None:
        with pytest.raises(ValueError, match="stationarity"):
            _make(alpha=[0.5], beta=[0.6])

    def test_alpha_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="alpha must have q=1"):
            _make(q=1, alpha=[0.1, 0.1])

    def test_beta_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="beta must have p=2"):
            _make(p=2, beta=[0.5])

    def test_invalid_orders(self) -> None:
        with pytest.raises(ValueError):
            _make(p=0)
        with pytest.raises(ValueError):
            _make(q=0)


@pytest.mark.stats
class TestStatistics:
    def test_unconditional_variance(self) -> None:
        """Sample variance matches omega / (1 - sum(alpha) - sum(beta))."""
        gen = _make(
            min_length=2000,
            max_length=2000,
            omega=0.2,
            alpha=[0.1],
            beta=[0.7],
            mu=0.0,
            seed=7,
        )
        df = gen.generate(n_series=20)
        r = np.concatenate(list(series_values(df).values()))
        # kurtosis=6 covers both the GARCH kurtosis (~3.2 for these params)
        # and the positive autocorrelation of squared returns, which inflates
        # the variance of the variance estimator beyond the iid formula.
        assert_std(r, expected=1.0, kurtosis=6.0)
        assert_mean(r, expected=0.0, std=1.0)

    def test_volatility_clustering(self) -> None:
        """Squared returns are autocorrelated while returns are not."""
        gen = _make(
            min_length=5000,
            max_length=5000,
            omega=0.2,
            alpha=[0.2],
            beta=[0.6],
            mu=0.0,
            seed=11,
        )
        df = gen.generate(n_series=1)
        r = next(iter(series_values(df).values()))
        assert_acf(r, lag=1, expected=0.0)
        # Theoretical lag-1 ACF of r^2 is ~0.26 for alpha=0.2, beta=0.6
        acf_sq = sample_acf(r**2, lag=1)
        assert acf_sq > 5.0 / np.sqrt(r.size), f"acf(r^2) {acf_sq:.4f} not > 0"

    def test_heavy_tails(self) -> None:
        """Persistent GARCH produces excess kurtosis (theoretical ~5.6)."""
        gen = _make(
            min_length=2000,
            max_length=2000,
            omega=0.05,
            alpha=[0.15],
            beta=[0.8],
            mu=0.0,
            seed=3,
        )
        df = gen.generate(n_series=10)
        r = np.concatenate(list(series_values(df).values()))
        kurt = stats.kurtosis(r, fisher=False)
        assert kurt > 3.5, f"kurtosis {kurt:.2f} shows no heavy tails"

    def test_degenerate_case_is_gaussian(self) -> None:
        """With alpha=beta=0 returns are iid N(mu, omega)."""
        gen = _make(
            min_length=5000,
            max_length=5000,
            omega=1.0,
            alpha=[0.0],
            beta=[0.0],
            mu=0.0,
            seed=5,
        )
        df = gen.generate(n_series=1)
        r = next(iter(series_values(df).values()))
        assert_distribution(r, stats.norm(0.0, 1.0))

    def test_nonzero_mean_python_path(self, monkeypatch) -> None:
        """ARCH term uses innovations (r - mu), not raw returns.

        With mu=5, using raw returns in the ARCH term would inflate the
        variance of demeaned returns to (omega + alpha*mu^2)/(1-alpha-beta)
        = 13.5 instead of the correct omega/(1-alpha-beta) = 1.0.
        """
        monkeypatch.setattr(garch_module, "_HAS_RUST", False)
        gen = _make(
            min_length=6000,
            max_length=6000,
            omega=0.2,
            alpha=[0.1],
            beta=[0.7],
            mu=5.0,
            seed=13,
        )
        r = gen.generate_single_series(6000)
        assert_mean(r, expected=5.0, std=1.0)
        assert_std(r - 5.0, expected=1.0, kurtosis=6.0)
