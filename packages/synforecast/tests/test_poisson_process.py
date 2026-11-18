"""Tests for PoissonProcessGenerator."""

import numpy as np
import pytest

from synforecast.generators import PoissonProcessGenerator
from tests.helpers import (
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)


def make_gen(**kwargs) -> PoissonProcessGenerator:
    params = {"min_length": 60, "max_length": 90, "freq": "D", "seed": 42}
    params.update(kwargs)
    return PoissonProcessGenerator(**params)


class TestPoissonProcessApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=60, max_length=90)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=7).generate(n_series=2)
        df2 = make_gen(engine=engine, seed=7).generate(n_series=2)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_n_jobs_invariance(self) -> None:
        df1 = make_gen(seed=11).generate(n_series=4, n_jobs=1)
        df2 = make_gen(seed=11).generate(n_series=4, n_jobs=4)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_counts_are_nonnegative_integers(self) -> None:
        values = make_gen().generate_single_series(500)
        assert np.all(values >= 0)
        np.testing.assert_array_equal(values, np.round(values))

    def test_cumulative_is_nondecreasing(self) -> None:
        values = make_gen(cumulative=True).generate_single_series(500)
        assert np.all(np.diff(values) >= 0)
        np.testing.assert_array_equal(values, np.round(values))

    def test_invalid_lambda_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gen(lambda_rate=0.0)
        with pytest.raises(ValueError):
            make_gen(lambda_rate=-1.0)


@pytest.mark.stats
class TestPoissonProcessStats:
    def test_count_moments(self) -> None:
        # y_t ~ Poisson(lam): mean == variance == lam
        lam = 4.0
        values = make_gen(lambda_rate=lam, seed=123).generate_single_series(50000)

        assert_mean(values, lam, np.sqrt(lam))
        assert_std(values, np.sqrt(lam), kurtosis=3.0 + 1.0 / lam)

    def test_zero_fraction_matches_pmf(self) -> None:
        # P(y_t = 0) = exp(-lam)
        lam = 2.0
        values = make_gen(lambda_rate=lam, seed=456).generate_single_series(50000)

        p0 = np.exp(-lam)
        assert_mean(values == 0, p0, np.sqrt(p0 * (1 - p0)))

    def test_dispersion_index_is_one(self) -> None:
        values = make_gen(lambda_rate=6.0, seed=789).generate_single_series(50000)
        assert abs(values.var() / values.mean() - 1.0) < 0.05

    def test_cumulative_increments_match_rate(self) -> None:
        lam = 3.0
        values = make_gen(
            lambda_rate=lam, cumulative=True, seed=321
        ).generate_single_series(20000)
        increments = np.diff(values)
        assert_mean(increments, lam, np.sqrt(lam))
        assert_std(increments, np.sqrt(lam), kurtosis=3.0 + 1.0 / lam)

    def test_matches_theory(self) -> None:
        lam = 4.0
        values = make_gen(lambda_rate=lam, seed=654).generate_single_series(50000)

        assert_mean(values, lam, np.sqrt(lam))
        assert_std(values, np.sqrt(lam), kurtosis=3.0 + 1.0 / lam)
