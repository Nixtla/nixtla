"""Tests for IntermittentDemandGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import IntermittentDemandGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)


def make_gen(**kwargs) -> IntermittentDemandGenerator:
    params = {"min_length": 60, "max_length": 90, "freq": "D", "seed": 42}
    params.update(kwargs)
    return IntermittentDemandGenerator(**params)


def _run_lengths(values: np.ndarray) -> list[int]:
    """Lengths of maximal runs of non-zero values."""
    runs = []
    current = 0
    for v in values:
        if v > 0:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


class TestIntermittentDemandApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=60, max_length=90)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=7).generate(n_series=2)
        df2 = make_gen(engine=engine, seed=7).generate(n_series=2)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_values_nonnegative_with_zeros(self) -> None:
        values = make_gen(demand_probability=0.3).generate_single_series(500)
        assert np.all(values >= 0)
        assert np.any(values == 0), "intermittent series should contain zeros"
        assert np.any(values > 0), "intermittent series should contain demand"

    def test_min_demand_respected(self) -> None:
        gen = make_gen(min_demand=3, demand_probability=0.5)
        values = gen.generate_single_series(500)
        nonzero = values[values > 0]
        assert np.all(nonzero >= 3)

    def test_poisson_sizes_are_integers(self) -> None:
        values = make_gen(demand_probability=0.5).generate_single_series(500)
        np.testing.assert_array_equal(values, np.round(values))

    @pytest.mark.parametrize("pattern", ["random", "clustered", "seasonal"])
    def test_patterns_generate(self, pattern: str) -> None:
        values = make_gen(intermittent_pattern=pattern).generate_single_series(300)
        assert values.shape == (300,)
        assert np.all(np.isfinite(values))

    @pytest.mark.parametrize("pattern", ["random", "clustered", "seasonal"])
    def test_zero_probability_all_zero(self, engine: str, pattern: str) -> None:
        # demand_probability=0 must yield a finite all-zero series without
        # error for every occurrence pattern (regression: 'clustered' raised
        # from geometric(0) on the Python path and overflowed on Rust).
        # For 'seasonal' the peak probability must also be 0, otherwise
        # occurrences near the cycle start are expected.
        df = make_gen(
            engine=engine,
            min_length=60,
            max_length=60,
            demand_probability=0.0,
            intermittent_pattern=pattern,
            seasonal_peak_prob=0.0,
        ).generate(n_series=2)
        assert_long_format(df, n_series=2, min_length=60, max_length=60)
        for values in series_values(df).values():
            assert values.shape == (60,)
            assert np.all(np.isfinite(values))
            assert np.all(values == 0.0)

    @pytest.mark.parametrize("pattern", ["random", "clustered", "seasonal"])
    def test_zero_probability_python_fallback(self, monkeypatch, pattern: str) -> None:
        # Same contract on the pure-Python path (the original failure mode).
        import synforecast.generators.intermittent_demand as id_mod

        monkeypatch.setattr(id_mod, "_HAS_RUST", False)
        values = make_gen(
            demand_probability=0.0,
            intermittent_pattern=pattern,
            seasonal_peak_prob=0.0,
        ).generate_single_series(200)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values))
        assert np.all(values == 0.0)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(demand_probability=1.5)

    def test_invalid_mean_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(demand_mean=0.0)

    def test_invalid_distribution_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(demand_distribution="weibull")


@pytest.mark.stats
class TestIntermittentDemandStats:
    def test_random_zero_fraction(self) -> None:
        # In 'random' mode P(y_t > 0) = demand_probability exactly.
        p = 0.3
        gen = make_gen(demand_probability=p, seed=123)
        values = gen.generate_single_series(20000)
        assert_mean(values > 0, p, np.sqrt(p * (1 - p)))

    def test_poisson_size_mean(self) -> None:
        # Sizes are Poisson(5) clipped at 1: E[max(X,1)] = 5 + P(X=0).
        gen = make_gen(demand_probability=0.5, demand_mean=5.0, seed=456)
        values = gen.generate_single_series(20000)
        nonzero = values[values > 0]
        assert_mean(nonzero, 5.0 + np.exp(-5.0), np.sqrt(5.0))

    def test_lognormal_sizes_ks(self) -> None:
        # mu = ln(m^2/sqrt(v+m^2)), sigma^2 = ln(1+v/m^2); the min_demand
        # clip at 1 affects ~3e-5 of the mass and is negligible for KS.
        m, s = 5.0, 2.0
        gen = make_gen(
            demand_distribution="lognormal",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=s,
            seed=789,
        )
        values = gen.generate_single_series(12000)
        nonzero = values[values > 0]

        mu = np.log(m**2 / np.sqrt(s**2 + m**2))
        sigma = np.sqrt(np.log(1 + s**2 / m**2))
        assert_distribution(nonzero, stats.lognorm(s=sigma, scale=np.exp(mu)))
        assert_mean(nonzero, m, s)

    def test_gamma_sizes_ks(self) -> None:
        # shape = (m/s)^2, scale = s^2/m; P(X < 1) ~ 0 for these parameters.
        m, s = 8.0, 2.0
        gen = make_gen(
            demand_distribution="gamma",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=s,
            seed=321,
        )
        values = gen.generate_single_series(12000)
        nonzero = values[values > 0]

        assert_distribution(nonzero, stats.gamma(a=(m / s) ** 2, scale=s**2 / m))
        assert_mean(nonzero, m, s)

    def test_negative_binomial_size_moments(self) -> None:
        # m=6, var=12 gives n = m^2/(var-m) = 6 (integer on purpose: the
        # Rust backend rounds n to an integer, so a non-integer n would
        # test the rounding artifact rather than the parameterization).
        m, s = 6.0, np.sqrt(12.0)
        gen = make_gen(
            demand_distribution="negative_binomial",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=s,
            seed=654,
        )
        values = gen.generate_single_series(20000)
        nonzero = values[values > 0]

        # Clipping at 1 moves the P(X=0) mass to 1: E[max(X,1)] = m + pmf(0)
        var = s**2
        p = m / var
        n = m * p / (1 - p)
        pmf0 = stats.nbinom(n, p).pmf(0)
        mean_clipped = m + pmf0
        var_clipped = var + m**2 + pmf0 - mean_clipped**2
        assert_mean(nonzero, mean_clipped, np.sqrt(var_clipped))
        assert_std(nonzero, np.sqrt(var_clipped), kurtosis=6.0)
        assert nonzero.var() > 1.5 * nonzero.mean(), "NB sizes are overdispersed"

    def test_python_fallback_nb_real_valued_n(self, monkeypatch) -> None:
        # The Python path supports real-valued n = m^2/(var-m); the mean
        # must still match m (the Rust path rounds n and would be biased).
        import synforecast.generators.intermittent_demand as id_mod

        monkeypatch.setattr(id_mod, "_HAS_RUST", False)
        m, s = 5.0, 4.0  # n = 25/11, non-integer
        gen = make_gen(
            demand_distribution="negative_binomial",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=s,
            seed=654,
        )
        values = gen.generate_single_series(20000)
        nonzero = values[values > 0]

        var = s**2
        p = m / var
        n = m * p / (1 - p)
        pmf0 = stats.nbinom(n, p).pmf(0)
        mean_clipped = m + pmf0
        var_clipped = var + m**2 + pmf0 - mean_clipped**2
        assert_mean(nonzero, mean_clipped, np.sqrt(var_clipped))
        assert nonzero.var() > 1.5 * nonzero.mean(), "NB sizes are overdispersed"

    def test_nb_small_variance_falls_back_to_poisson(self) -> None:
        m = 5.0
        gen = make_gen(
            demand_distribution="negative_binomial",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=1.0,  # var < mean triggers Poisson fallback
            seed=987,
        )
        values = gen.generate_single_series(20000)
        nonzero = values[values > 0]
        assert_mean(nonzero, m + np.exp(-m), np.sqrt(m))
        assert_std(nonzero, np.sqrt(m), kurtosis=3.0 + 1.0 / m)

    def test_seasonal_phase_peaks_at_cycle_start(self) -> None:
        # Occurrence probability is seasonal_peak_prob at t % P == 0
        # (cos(0) = 1) and demand_probability at t % P == P/2 (cos(pi) = -1).
        period, base, peak = 8, 0.2, 0.6
        gen = make_gen(
            intermittent_pattern="seasonal",
            demand_probability=base,
            seasonal_peak_prob=peak,
            seasonal_period=period,
            seed=135,
        )
        values = gen.generate_single_series(24000)
        occurred = (values > 0).astype(float)
        position = np.arange(values.size) % period

        assert_mean(occurred[position == 0], peak, np.sqrt(peak * (1 - peak)))
        assert_mean(occurred[position == 4], base, np.sqrt(base * (1 - base)))
        assert occurred[position == 0].mean() > occurred[position == 4].mean()

    def test_clustered_run_lengths(self) -> None:
        cluster = 4
        gen = make_gen(
            intermittent_pattern="clustered",
            demand_probability=0.2,
            cluster_size=cluster,
            seed=246,
        )
        values = gen.generate_single_series(8000)
        runs = _run_lengths(values)

        assert len(runs) > 50, "expected many clusters"
        assert all(r <= cluster for r in runs)
        # All clusters are full-size except possibly the last (truncated)
        assert all(r == cluster for r in runs[:-1])

    def test_python_fallback_lognormal_ks(self, monkeypatch) -> None:
        # Regression: the Python path used to truncate continuous sizes to
        # int, which destroyed the lognormal shape and biased the mean down.
        import synforecast.generators.intermittent_demand as id_mod

        monkeypatch.setattr(id_mod, "_HAS_RUST", False)
        m, s = 5.0, 2.0
        gen = make_gen(
            demand_distribution="lognormal",
            demand_probability=0.5,
            demand_mean=m,
            demand_std=s,
            seed=468,
        )
        values = gen.generate_single_series(12000)
        nonzero = values[values > 0]

        mu = np.log(m**2 / np.sqrt(s**2 + m**2))
        sigma = np.sqrt(np.log(1 + s**2 / m**2))
        assert_distribution(nonzero, stats.lognorm(s=sigma, scale=np.exp(mu)))
        assert_mean(nonzero, m, s)

    def test_python_fallback_zero_fraction(self, monkeypatch) -> None:
        import synforecast.generators.intermittent_demand as id_mod

        monkeypatch.setattr(id_mod, "_HAS_RUST", False)
        p = 0.3
        gen = make_gen(demand_probability=p, seed=579)
        values = gen.generate_single_series(20000)
        assert_mean(values > 0, p, np.sqrt(p * (1 - p)))
