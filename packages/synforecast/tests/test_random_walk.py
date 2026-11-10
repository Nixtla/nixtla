"""Tests for RandomWalkGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import RandomWalkGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)


def make_gen(**kwargs) -> RandomWalkGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "D", "seed": 42}
    params.update(kwargs)
    return RandomWalkGenerator(**params)


def steps_of(values: np.ndarray, start_value: float = 0.0) -> np.ndarray:
    """Recover the random steps from a generated walk."""
    return np.diff(np.concatenate(([start_value], values)))


@pytest.fixture
def backend() -> None:
    """Mark tests that exercise native random-walk generation."""


class TestRandomWalkApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_integer_freq(self, engine: str) -> None:
        df = make_gen(engine=engine, freq=2).generate(n_series=2)
        assert_long_format(df, n_series=2, min_length=50, max_length=80)

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
        df = make_gen().generate(n_series=2, start_id=10)
        assert set(series_values(df)) == {"10", "11"}

    def test_exact_length(self) -> None:
        values = make_gen(min_length=64, max_length=64).generate_single_series(64)
        assert values.shape == (64,)

    def test_negative_volatility_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(volatility=-1.0)

    def test_bad_length_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            make_gen(min_length=100, max_length=50)

    def test_invalid_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency"):
            make_gen(freq="not-a-freq")

    @pytest.mark.usefixtures("backend")
    def test_zero_volatility_is_deterministic_drift(self) -> None:
        gen = make_gen(volatility=0.0, drift=0.5, start_value=2.0)
        values = gen.generate_single_series(5)
        np.testing.assert_allclose(values, 2.0 + 0.5 * np.arange(1, 6))


@pytest.mark.stats
class TestRandomWalkStats:
    @pytest.mark.usefixtures("backend")
    def test_increment_moments_and_independence(self) -> None:
        drift, vol, start = 0.3, 2.0, 5.0
        gen = make_gen(drift=drift, volatility=vol, start_value=start, seed=123)
        values = gen.generate_single_series(20000)
        steps = steps_of(values, start)

        assert_mean(steps, drift, vol)
        assert_std(steps, vol)
        # Increments of a random walk are independent
        for lag in (1, 2, 5):
            assert_acf(steps, lag, 0.0)

    @pytest.mark.usefixtures("backend")
    def test_steps_are_gaussian(self) -> None:
        gen = make_gen(drift=1.0, volatility=0.5, seed=99)
        steps = steps_of(gen.generate_single_series(20000))
        assert_distribution(steps, stats.norm(loc=1.0, scale=0.5))

    @pytest.mark.usefixtures("backend")
    def test_variance_grows_linearly(self) -> None:
        # Var(y_t) = t * volatility^2; check the ensemble at t = T.
        T, n_series, vol = 50, 2000, 1.5
        gen = make_gen(min_length=T, max_length=T, volatility=vol, seed=7)
        y_end = np.array([gen.generate_single_series(T)[-1] for _ in range(n_series)])
        assert_mean(y_end, 0.0, vol * np.sqrt(T))
        assert_std(y_end, vol * np.sqrt(T))

    def test_innovation_distribution_uniform(self) -> None:
        # Uniform innovations produce steps
        # on [drift - a, drift + a] with a = volatility * sqrt(3).
        drift, vol = 0.2, 1.0
        gen = make_gen(
            drift=drift, volatility=vol, innovation_distribution="uniform", seed=11
        )
        steps = steps_of(gen.generate_single_series(20000))
        a = vol * np.sqrt(3.0)
        assert_distribution(steps, stats.uniform(loc=drift - a, scale=2 * a))
        assert_std(steps, vol, kurtosis=1.8)

    @pytest.mark.usefixtures("backend")
    def test_innovation_distribution_t_scaled(self) -> None:
        # Student-t innovations are rescaled to the requested volatility.
        gen = make_gen(
            volatility=2.0,
            innovation_distribution="t",
            innovation_params={"df": 6.0},
            seed=17,
        )
        steps = steps_of(gen.generate_single_series(20000))
        assert_mean(steps, 0.0, 2.0)
        # t(6) has kurtosis 3 + 6/(df-4) = 6
        assert_std(steps, 2.0, kurtosis=6.0)
