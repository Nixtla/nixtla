"""Tests for INARGenerator."""

import numpy as np
import pytest

from synforecast.generators import INARGenerator
from tests.helpers import (
    assert_acf,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)


def make_gen(**kwargs) -> INARGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "D", "seed": 42}
    params.update(kwargs)
    return INARGenerator(**params)


class TestINARApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=7).generate(n_series=2)
        df2 = make_gen(engine=engine, seed=7).generate(n_series=2)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_integer_valued_and_nonnegative(self) -> None:
        values = make_gen().generate_single_series(500)
        assert np.all(values >= 0)
        np.testing.assert_array_equal(values, np.round(values))

    def test_start_id(self) -> None:
        df = make_gen().generate(n_series=2, start_id=5)
        assert set(series_values(df)) == {"5", "6"}

    def test_alpha_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length p=2"):
            make_gen(p=2, alpha=[0.5])

    def test_alpha_sum_ge_one_raises(self) -> None:
        with pytest.raises(ValueError, match="stationarity"):
            make_gen(p=2, alpha=[0.6, 0.5])

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            make_gen(p=1, alpha=[-0.1])

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(p=0)

    def test_invalid_innovation_mean_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(innovation_mean=0.0)

    def test_auto_alpha_is_stationary(self) -> None:
        info = make_gen(p=3, alpha=None).get_model_info()
        assert len(info["thinning_probabilities"]) == 3
        assert info["persistence"] < 1.0


@pytest.mark.stats
class TestINARStats:
    def test_poisson_inar1_moments_and_acf(self) -> None:
        # INAR(1) with Poisson innovations has a Poisson(mu/(1-a)) marginal
        # and geometric ACF a^k.
        a, mu = 0.5, 4.0
        gen = make_gen(p=1, alpha=[a], innovation_mean=mu, seed=123)
        values = gen.generate_single_series(20000)

        mean = mu / (1 - a)
        # SE of the sample mean is inflated by autocorrelation: (1+a)/(1-a)
        assert_mean(values, mean, np.sqrt(mean * (1 + a) / (1 - a)))
        assert_std(values, np.sqrt(mean), kurtosis=6.0)
        for lag in (1, 2, 3):
            assert_acf(values, lag, a**lag)

    def test_inar2_yule_walker_acf(self) -> None:
        # rho_1 = a1/(1-a2), rho_2 = a1*rho_1 + a2
        a1, a2 = 0.3, 0.2
        gen = make_gen(p=2, alpha=[a1, a2], innovation_mean=5.0, seed=456)
        values = gen.generate_single_series(20000)

        rho1 = a1 / (1 - a2)
        assert_acf(values, 1, rho1)
        assert_acf(values, 2, a1 * rho1 + a2)

    def test_negative_binomial_innovations_overdispersed(self) -> None:
        a, mu, r = 0.5, 5.0, 2.0
        gen = make_gen(
            p=1,
            alpha=[a],
            innovation_type="negative_binomial",
            innovation_mean=mu,
            innovation_dispersion=r,
            seed=789,
        )
        values = gen.generate_single_series(20000)

        mean = mu / (1 - a)
        # Var(X) = (a(1-a)E[X] + var_eps) / (1-a^2), var_eps = mu + mu^2/r
        var_eps = mu + mu**2 / r
        var = (a * (1 - a) * mean + var_eps) / (1 - a**2)
        assert_mean(values, mean, np.sqrt(var * (1 + a) / (1 - a)))
        assert_std(values, np.sqrt(var), kurtosis=8.0)
        assert values.var() > 1.5 * values.mean(), "NB-INAR should be overdispersed"

    def test_matches_theory(self) -> None:
        a, mu = 0.5, 4.0
        gen = make_gen(p=1, alpha=[a], innovation_mean=mu, seed=321)
        values = gen.generate_single_series(8000)

        mean = mu / (1 - a)
        assert np.all(values >= 0)
        np.testing.assert_array_equal(values, np.round(values))
        assert_mean(values, mean, np.sqrt(mean * (1 + a) / (1 - a)))
        assert_acf(values, 1, a)
