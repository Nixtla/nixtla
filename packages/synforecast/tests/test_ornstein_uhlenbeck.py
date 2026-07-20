"""Tests for OrnsteinUhlenbeckGenerator.

The generator uses the Euler-Maruyama scheme, which is an AR(1) process:
    X_t = mu + phi*(X_{t-1} - mu) + sigma*sqrt(dt)*z_t,  phi = 1 - theta*dt
with stationary mean mu, stationary variance sigma^2*dt/(1-phi^2), and
lag-k autocorrelation phi^k. Assertions target this discretization (the
continuous-time limits are recovered as dt -> 0).
"""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import OrnsteinUhlenbeckGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    to_pandas,
)


def _gen(**overrides) -> OrnsteinUhlenbeckGenerator:
    params = {
        "min_length": 50,
        "max_length": 80,
        "freq": "D",
        "seed": 42,
        **overrides,
    }
    return OrnsteinUhlenbeckGenerator(**params)


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

    def test_starts_at_initial_value(self) -> None:
        gen = _gen(initial_value=-3.5, min_length=10, max_length=10)
        values = gen.generate_single_series(10)
        assert values[0] == pytest.approx(-3.5)

    @pytest.mark.parametrize(
        "bad",
        [
            {"theta": 0.0},
            {"theta": -0.5},
            {"sigma": 0.0},
            {"dt": 0.0},
            {"min_length": 100, "max_length": 50},
        ],
    )
    def test_validation_errors(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            _gen(**bad)

    def test_stability_constraint(self) -> None:
        # theta*dt >= 2 makes the Euler AR(1) coefficient |phi| >= 1.
        with pytest.raises(ValueError, match="theta \\* dt"):
            _gen(theta=2.5, dt=1.0)
        _gen(theta=3.0, dt=0.5)  # theta*dt = 1.5 is fine

    def test_uniform_innovations_bounded(self) -> None:
        # Uniform innovations are bounded by sqrt(3), so every increment
        # net of drift must lie within sigma*sqrt(3*dt).
        gen = _gen(
            theta=0.5,
            mu=0.0,
            sigma=1.0,
            dt=0.5,
            innovation_distribution="uniform",
            min_length=2000,
            max_length=2000,
        )
        x = gen.generate_single_series(2000)
        drift = gen.theta * (gen.mu - x[:-1]) * gen.dt
        residuals = np.diff(x) - drift
        bound = gen.sigma * np.sqrt(3.0 * gen.dt)
        assert np.all(np.abs(residuals) <= bound * (1 + 1e-12))


@pytest.mark.stats
class TestStatistics:
    THETA, MU, SIGMA, DT = 0.5, 5.0, 1.0, 1.0
    PHI = 1.0 - THETA * DT
    STAT_STD = SIGMA * np.sqrt(DT / (1.0 - PHI**2))
    N = 100_000

    def _series(self, seed: int = 7) -> np.ndarray:
        gen = _gen(
            seed=seed,
            theta=self.THETA,
            mu=self.MU,
            sigma=self.SIGMA,
            dt=self.DT,
            initial_value=self.MU,
            min_length=self.N,
            max_length=self.N,
        )
        # Burn in the transient from the deterministic start.
        return gen.generate_single_series(self.N)[200:]

    def test_stationary_mean_and_variance(self) -> None:
        # Thin to break autocorrelation (phi^10 ~ 1e-3) so the iid-based
        # standard errors in the helpers apply.
        thinned = self._series()[::10]
        assert_mean(thinned, self.MU, std=self.STAT_STD)
        assert_std(thinned, self.STAT_STD)

    def test_autocorrelation(self) -> None:
        x = self._series(seed=11)
        assert_acf(x, 1, self.PHI)
        assert_acf(x, 2, self.PHI**2)

    def test_stationary_distribution_normal(self) -> None:
        thinned = self._series(seed=13)[::10]
        assert_distribution(thinned, stats.norm(self.MU, self.STAT_STD))

    def test_mean_reversion_from_far_start(self) -> None:
        theta, mu, sigma, dt = 0.8, 5.0, 1.0, 1.0
        phi = 1.0 - theta * dt
        gen = _gen(
            seed=17,
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            initial_value=20.0,
            min_length=200,
            max_length=200,
        )
        tail = gen.generate_single_series(200)[-100:]
        stat_std = sigma * np.sqrt(dt / (1.0 - phi**2))
        # Variance of the mean of n autocorrelated samples inflated by
        # (1+phi)/(1-phi).
        se = stat_std * np.sqrt((1.0 + phi) / (1.0 - phi) / tail.size)
        assert abs(tail.mean() - mu) <= 5.0 * se
