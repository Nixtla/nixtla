"""Tests for JumpDiffusionGenerator (Merton jump diffusion).

Log-returns per step are X = (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z + sum(Y_k)
with N ~ Poisson(lambda_jump*dt) jumps of size Y ~ jump_mean + jump_std*eps.
With normal innovations:
    E[X]   = (mu - sigma^2/2)*dt + lambda*dt*jump_mean
    Var(X) = sigma^2*dt + lambda*dt*(jump_mean^2 + jump_std^2)
    kurt   = 3 + lambda*dt*E[Y^4] / Var(X)^2,
             E[Y^4] = m^4 + 6*m^2*s^2 + 3*s^4
"""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import JumpDiffusionGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
    to_pandas,
)


def _gen(**overrides) -> JumpDiffusionGenerator:
    params = {
        "min_length": 50,
        "max_length": 80,
        "freq": "D",
        "seed": 42,
        **overrides,
    }
    return JumpDiffusionGenerator(**params)


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
        df = _gen(mu=-0.1, lambda_jump=1.0, jump_mean=-0.2).generate(n_series=5)
        assert (to_pandas(df)["y"] > 0).all()

    def test_starts_at_initial_value(self) -> None:
        df = _gen(initial_value=12.5).generate(n_series=3)
        for values in series_values(df).values():
            assert values[0] == pytest.approx(12.5)

    @pytest.mark.parametrize(
        "bad",
        [
            {"sigma": 0.0},
            {"lambda_jump": -0.1},
            {"jump_std": -0.1},
            {"initial_value": 0.0},
            {"dt": 0.0},
            {"min_length": 100, "max_length": 50},
        ],
    )
    def test_validation_errors(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            _gen(**bad)


@pytest.mark.stats
class TestStatistics:
    def _log_returns(self, gen: JumpDiffusionGenerator, n: int) -> np.ndarray:
        return np.diff(np.log(gen.generate_single_series(n)))

    def test_log_return_moments(self) -> None:
        # mu chosen so the expected log-return is zero (no over/underflow
        # of the price level over a long path).
        sigma, lam, m, s, dt = 0.1, 0.3, 0.05, 0.4, 1.0
        mu = sigma**2 / 2 - lam * m
        n = 100_000
        gen = _gen(
            seed=3,
            mu=mu,
            sigma=sigma,
            lambda_jump=lam,
            jump_mean=m,
            jump_std=s,
            dt=dt,
            min_length=n,
            max_length=n,
        )
        lr = self._log_returns(gen, n)

        expected_mean = (mu - sigma**2 / 2) * dt + lam * dt * m
        expected_var = sigma**2 * dt + lam * dt * (m**2 + s**2)
        y4 = m**4 + 6 * m**2 * s**2 + 3 * s**4
        kurt = 3.0 + lam * dt * y4 / expected_var**2

        assert_mean(lr, expected_mean, std=np.sqrt(expected_var))
        assert_std(lr, np.sqrt(expected_var), kurtosis=kurt)

    def test_excess_kurtosis_positive(self) -> None:
        gen = _gen(
            seed=5,
            mu=0.005,
            sigma=0.1,
            lambda_jump=0.3,
            jump_mean=0.0,
            jump_std=0.4,
            dt=1.0,
            min_length=100_000,
            max_length=100_000,
        )
        lr = self._log_returns(gen, 100_000)
        # Theoretical excess kurtosis ~6.9; Gaussian diffusion alone has 0.
        assert stats.kurtosis(lr) > 1.0

    def test_jump_count_poisson(self) -> None:
        # Small diffusion, large jumps: a step contains a jump iff
        # |log-return| > 0.5. Steps with >=1 jump ~ Binomial(n, 1-exp(-lam*dt)).
        sigma, lam, dt = 0.02, 0.05, 1.0
        n = 50_000
        gen = _gen(
            seed=17,
            mu=sigma**2 / 2 - lam * 1.0,
            sigma=sigma,
            lambda_jump=lam,
            jump_mean=1.0,
            jump_std=0.1,
            dt=dt,
            min_length=n,
            max_length=n,
        )
        lr = self._log_returns(gen, n)
        detected = int(np.sum(np.abs(lr) > 0.5))

        p = 1.0 - np.exp(-lam * dt)
        expected = (n - 1) * p
        sd = np.sqrt((n - 1) * p * (1 - p))
        assert abs(detected - expected) <= 5.0 * sd, (
            f"jump count {detected} vs expected {expected:.0f} +/- {sd:.0f}"
        )

    def test_no_jumps_reduces_to_gbm(self) -> None:
        mu, sigma, dt = 0.05, 0.2, 1.0 / 252.0
        n = 50_000
        gen = _gen(
            seed=23,
            mu=mu,
            sigma=sigma,
            lambda_jump=0.0,
            dt=dt,
            min_length=n,
            max_length=n,
        )
        lr = self._log_returns(gen, n)
        dist = stats.norm(loc=(mu - sigma**2 / 2) * dt, scale=sigma * np.sqrt(dt))
        assert_distribution(lr, dist)
