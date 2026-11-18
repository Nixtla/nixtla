"""Tests for GeometricBrownianMotionGenerator.

The generator simulates the exact GBM solution: log-returns are iid with
mean (mu - sigma^2/2)*dt and standard deviation sigma*sqrt(dt).
"""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import GeometricBrownianMotionGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
    to_pandas,
)


def _gen(**overrides) -> GeometricBrownianMotionGenerator:
    params = {
        "min_length": 50,
        "max_length": 80,
        "freq": "D",
        "seed": 42,
        **overrides,
    }
    return GeometricBrownianMotionGenerator(**params)


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_exact_length(self, engine: str) -> None:
        df = _gen(engine=engine, min_length=64, max_length=64).generate(n_series=2)
        assert_long_format(df, n_series=2, min_length=64, max_length=64)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _gen(engine=engine).generate(n_series=3)
        df2 = _gen(engine=engine).generate(n_series=3)
        assert to_pandas(df1).equals(to_pandas(df2))

    def test_positive_values(self) -> None:
        df = _gen(mu=-0.1, sigma=0.5).generate(n_series=5)
        assert (to_pandas(df)["y"] > 0).all()

    def test_starts_at_initial_value(self) -> None:
        df = _gen(initial_value=42.5).generate(n_series=3)
        for values in series_values(df).values():
            assert values[0] == pytest.approx(42.5)

    def test_integer_freq(self) -> None:
        df = _gen(freq=2, min_length=10, max_length=10).generate(n_series=1)
        np.testing.assert_array_equal(to_pandas(df)["ds"].to_numpy(), np.arange(10) * 2)

    @pytest.mark.parametrize(
        "bad",
        [
            {"sigma": 0.0},
            {"sigma": -0.2},
            {"initial_value": 0.0},
            {"initial_value": -1.0},
            {"dt": 0.0},
            {"min_length": 100, "max_length": 50},
            {"freq": "not-a-freq"},
        ],
    )
    def test_validation_errors(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            _gen(**bad)

    def test_uniform_innovations_bounded(self) -> None:
        # Uniform innovations are bounded by sqrt(3), so every log-return
        # must lie within drift +/- sigma*sqrt(3*dt).
        gen = _gen(
            mu=0.0,
            sigma=0.2,
            dt=0.5,
            innovation_distribution="uniform",
            min_length=2000,
            max_length=2000,
        )
        log_returns = np.diff(np.log(gen.generate_single_series(2000)))
        drift = (gen.mu - 0.5 * gen.sigma**2) * gen.dt
        bound = gen.sigma * np.sqrt(3.0 * gen.dt)
        assert np.all(np.abs(log_returns - drift) <= bound * (1 + 1e-12))


@pytest.mark.stats
class TestStatistics:
    MU, SIGMA, DT = 0.05, 0.2, 1.0 / 252.0
    N = 100_000

    def _log_returns(self, seed: int = 7, **overrides) -> np.ndarray:
        gen = _gen(
            **{
                "seed": seed,
                "mu": self.MU,
                "sigma": self.SIGMA,
                "dt": self.DT,
                "min_length": self.N,
                "max_length": self.N,
                **overrides,
            }
        )
        return np.diff(np.log(gen.generate_single_series(self.N)))

    def test_log_return_moments(self) -> None:
        lr = self._log_returns()
        expected_mean = (self.MU - 0.5 * self.SIGMA**2) * self.DT
        expected_std = self.SIGMA * np.sqrt(self.DT)
        assert_mean(lr, expected_mean, std=expected_std)
        assert_std(lr, expected_std)

    def test_log_returns_normal(self) -> None:
        lr = self._log_returns(seed=11)
        dist = stats.norm(
            loc=(self.MU - 0.5 * self.SIGMA**2) * self.DT,
            scale=self.SIGMA * np.sqrt(self.DT),
        )
        assert_distribution(lr, dist)

    def test_dt_scaling(self) -> None:
        # Quadrupling dt doubles the log-return standard deviation.
        lr = self._log_returns(seed=13, dt=4.0 * self.DT)
        assert_std(lr, self.SIGMA * np.sqrt(4.0 * self.DT))
