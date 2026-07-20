"""Tests for SeasonalGenerator."""

import numpy as np
import pytest
from scipy import stats

import synforecast.generators.seasonal as seasonal_mod
from synforecast.generators import SeasonalGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)


def make_gen(**kwargs) -> SeasonalGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "h", "seed": 42}
    params.update(kwargs)
    return SeasonalGenerator(**params)


def deterministic_part(gen: SeasonalGenerator, length: int) -> np.ndarray:
    """Closed-form noise-free component of the generated series."""
    t = np.arange(length)
    return (
        gen.base_level
        + gen.seasonality_amplitude * np.sin(2 * np.pi * t / gen.seasonality_period)
        + gen.trend * t
    )


@pytest.fixture(params=["rust", "python"])
def backend(request: pytest.FixtureRequest, monkeypatch) -> str:
    """Run generate_single_series through the Rust and pure-Python paths."""
    if request.param == "python":
        monkeypatch.setattr(seasonal_mod, "_HAS_RUST", False)
    return request.param


class TestSeasonalApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=7).generate(n_series=2)
        df2 = make_gen(engine=engine, seed=7).generate(n_series=2)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_n_jobs_independent(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=3).generate(n_series=4, n_jobs=1)
        df2 = make_gen(engine=engine, seed=3).generate(n_series=4, n_jobs=-1)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_start_id(self) -> None:
        df = make_gen().generate(n_series=3, start_id=4)
        assert set(series_values(df)) == {"4", "5", "6"}

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(seasonality_period=0)

    def test_negative_noise_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(noise_level=-1.0)

    def test_bad_length_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            make_gen(min_length=100, max_length=50)

    @pytest.mark.usefixtures("backend")
    def test_zero_noise_matches_closed_form(self) -> None:
        gen = make_gen(
            min_length=96,
            max_length=96,
            seasonality_period=24,
            seasonality_amplitude=8.0,
            trend=0.05,
            base_level=50.0,
            noise_level=0.0,
        )
        values = gen.generate_single_series(96)
        np.testing.assert_allclose(values, deterministic_part(gen, 96), atol=1e-9)


@pytest.mark.stats
class TestSeasonalStats:
    @pytest.mark.usefixtures("backend")
    def test_residual_moments(self) -> None:
        gen = make_gen(
            seasonality_period=24,
            seasonality_amplitude=8.0,
            trend=0.05,
            base_level=50.0,
            noise_level=1.5,
            seed=123,
        )
        n = 20000
        residuals = gen.generate_single_series(n) - deterministic_part(gen, n)
        assert_mean(residuals, 0.0, 1.5)
        assert_std(residuals, 1.5)
        assert_distribution(residuals, stats.norm(loc=0.0, scale=1.5))

    @pytest.mark.usefixtures("backend")
    def test_periodicity(self) -> None:
        # Differences at the seasonal lag remove the seasonal component:
        # y_{t+m} - y_t = trend * m + (eps_{t+m} - eps_t).
        m, trend, noise = 24, 0.1, 1.0
        gen = make_gen(seasonality_period=m, trend=trend, noise_level=noise, seed=456)
        values = gen.generate_single_series(20000)
        seasonal_diff = values[m:] - values[:-m]
        assert_mean(seasonal_diff, trend * m, np.sqrt(2) * noise)
        assert_std(seasonal_diff, np.sqrt(2) * noise)

    @pytest.mark.usefixtures("backend")
    def test_amplitude_via_regression(self) -> None:
        # Projecting onto sin/cos at the seasonal frequency recovers the
        # amplitude; the cos coefficient is 0 (sine phase).
        m, amp = 12, 6.0
        gen = make_gen(
            seasonality_period=m,
            seasonality_amplitude=amp,
            noise_level=1.0,
            base_level=0.0,
            seed=789,
        )
        n = 12000
        values = gen.generate_single_series(n)
        t = np.arange(n)
        sin_coef = 2 * np.mean(values * np.sin(2 * np.pi * t / m))
        cos_coef = 2 * np.mean(values * np.cos(2 * np.pi * t / m))
        # Regression coefficient SE ~ sigma * sqrt(2/n)
        se = 1.0 * np.sqrt(2.0 / n)
        assert abs(sin_coef - amp) < 5 * se
        assert abs(cos_coef) < 5 * se

    def test_innovation_distribution_laplace(self, monkeypatch) -> None:
        # Python path honors innovation_distribution for the noise term.
        monkeypatch.setattr(seasonal_mod, "_HAS_RUST", False)
        noise = 2.0
        gen = make_gen(
            seasonality_amplitude=0.0,
            base_level=0.0,
            noise_level=noise,
            innovation_distribution="laplace",
            seed=11,
        )
        values = gen.generate_single_series(20000)
        assert_distribution(values, stats.laplace(loc=0.0, scale=noise / np.sqrt(2)))

    @pytest.mark.usefixtures("backend")
    def test_t_innovations_heavier_tails(self) -> None:
        """t(3) innovations produce heavier tails than normal on both paths."""
        n = 20000
        common = {
            "seasonality_amplitude": 0.0,
            "base_level": 0.0,
            "noise_level": 1.0,
            "seed": 21,
        }
        normal_vals = make_gen(**common).generate_single_series(n)
        t_vals = make_gen(
            innovation_distribution="t", innovation_params={"df": 3}, **common
        ).generate_single_series(n)
        k_normal = stats.kurtosis(normal_vals)
        k_t = stats.kurtosis(t_vals)
        assert k_t > k_normal + 1.0, (
            f"t(3) should be heavier-tailed than normal: {k_t=} {k_normal=}"
        )

    @pytest.mark.usefixtures("backend")
    def test_t_innovations_ks_sanity(self) -> None:
        """Standardized t(5) noise passes a KS test against the scaled t
        distribution (variance-normalized to noise_level) on both paths."""
        df = 5.0
        noise = 2.0
        gen = make_gen(
            seasonality_amplitude=0.0,
            base_level=0.0,
            noise_level=noise,
            innovation_distribution="t",
            innovation_params={"df": df},
            seed=22,
        )
        values = gen.generate_single_series(20000)
        # Innovations are scaled so std == noise_level: t_df * noise * sqrt((df-2)/df)
        scale = noise * np.sqrt((df - 2) / df)
        assert_distribution(values, stats.t(df, loc=0.0, scale=scale))

    def test_t_innovations_batch_path_heavier_tails(self) -> None:
        """generate() (Rust rayon batch path) honors innovation_distribution."""

        def batch_values(**extra):
            gen = make_gen(
                min_length=5000,
                max_length=5000,
                seasonality_amplitude=0.0,
                base_level=0.0,
                noise_level=1.0,
                seed=31,
                **extra,
            )
            return np.concatenate(
                list(series_values(gen.generate(n_series=4)).values())
            )

        normal_vals = batch_values()
        t_vals = batch_values(innovation_distribution="t", innovation_params={"df": 3})
        assert stats.kurtosis(t_vals) > stats.kurtosis(normal_vals) + 1.0
