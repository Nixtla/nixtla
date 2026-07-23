"""Tests for HawkesProcessGenerator."""

import numpy as np
import pytest

from synforecast.generators import HawkesProcessGenerator
from tests.helpers import (
    assert_long_format,
    assert_mean,
    assert_std,
    sample_acf,
    series_values,
)


def make_gen(**kwargs) -> HawkesProcessGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "h", "seed": 42}
    params.update(kwargs)
    return HawkesProcessGenerator(**params)


class TestHawkesApi:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = make_gen(engine=engine, seed=7).generate(n_series=2)
        df2 = make_gen(engine=engine, seed=7).generate(n_series=2)
        for uid, values in series_values(df1).items():
            np.testing.assert_array_equal(values, series_values(df2)[uid])

    def test_counts_output_nonnegative_integers(self) -> None:
        values = make_gen(output_type="counts").generate_single_series(300)
        assert np.all(values >= 0)
        np.testing.assert_array_equal(values, np.round(values))

    def test_events_output_binary(self) -> None:
        values = make_gen(output_type="events").generate_single_series(300)
        assert set(np.unique(values)) <= {0.0, 1.0}

    def test_intensity_output_at_least_baseline(self) -> None:
        gen = make_gen(output_type="intensity", baseline_intensity=0.7)
        values = gen.generate_single_series(300)
        assert np.all(values >= 0.7 - 1e-9)

    def test_unstable_exponential_raises(self) -> None:
        with pytest.raises(ValueError, match="unstable"):
            make_gen(excitation_amplitude=1.5, decay_rate=1.0)

    def test_unstable_power_law_raises(self) -> None:
        # branching ratio alpha / (beta * (p-1)) = 1.0 / (1.0 * 0.5) = 2
        with pytest.raises(ValueError, match="unstable"):
            make_gen(
                kernel="power_law",
                excitation_amplitude=1.0,
                decay_rate=1.0,
                power_law_exponent=1.5,
            )

    def test_power_law_exponent_must_exceed_one(self) -> None:
        with pytest.raises(ValueError):
            make_gen(kernel="power_law", power_law_exponent=1.0)

    def test_model_info(self) -> None:
        info = make_gen(excitation_amplitude=0.5, decay_rate=1.0).get_model_info()
        assert info["branching_ratio"] == pytest.approx(0.5)
        assert info["expected_cluster_size"] == pytest.approx(2.0)
        assert info["is_stable"]


@pytest.mark.stats
class TestHawkesStats:
    def test_count_rate_matches_theory(self) -> None:
        # Long-run rate is mu / (1 - alpha/beta) events per bin.
        gen = make_gen(
            baseline_intensity=0.5,
            excitation_amplitude=0.5,
            decay_rate=1.0,
            max_events=20000,
            seed=123,
        )
        values = gen.generate_single_series(4000)
        # Var(N(T))/T ~ rate / (1-n)^2 = 4 for n = 0.5, so SE(mean) ~ 0.032
        assert abs(values.mean() - 1.0) < 0.2

    def test_clustering_overdispersion(self) -> None:
        # Self-excitation makes bin counts overdispersed relative to Poisson
        # (Fano factor ~3.2 for these parameters) and positively correlated.
        gen = make_gen(
            baseline_intensity=0.2,
            excitation_amplitude=0.8,
            decay_rate=1.0,
            max_events=30000,
            seed=456,
        )
        values = gen.generate_single_series(4000)

        assert values.var() > 1.8 * values.mean(), "counts should be overdispersed"
        assert sample_acf(values, 1) > 0.1, "counts should be positively correlated"

    def test_zero_excitation_reduces_to_poisson(self) -> None:
        mu = 3.0
        gen = make_gen(
            baseline_intensity=mu,
            excitation_amplitude=0.0,
            max_events=100000,
            seed=789,
        )
        values = gen.generate_single_series(20000)

        assert_mean(values, mu, np.sqrt(mu))
        assert_std(values, np.sqrt(mu), kurtosis=3.0 + 1.0 / mu)

    def test_intensity_output_mean(self) -> None:
        # E[lambda] = mu / (1 - branching) = 1.0
        gen = make_gen(
            output_type="intensity",
            baseline_intensity=0.5,
            excitation_amplitude=0.5,
            decay_rate=1.0,
            max_events=20000,
            seed=321,
        )
        values = gen.generate_single_series(4000)
        assert abs(values.mean() - 1.0) < 0.2

    def test_rate_and_clustering(self) -> None:
        gen = make_gen(
            baseline_intensity=0.5,
            excitation_amplitude=0.5,
            decay_rate=1.0,
            max_events=20000,
            seed=654,
        )
        values = gen.generate_single_series(1500)

        assert np.all(values >= 0)
        np.testing.assert_array_equal(values, np.round(values))
        assert abs(values.mean() - 1.0) < 0.3
        assert values.var() > 1.2 * values.mean(), "counts should be overdispersed"

    def test_power_law_runs(self) -> None:
        gen = make_gen(
            kernel="power_law",
            baseline_intensity=0.5,
            excitation_amplitude=0.3,
            decay_rate=1.0,
            power_law_exponent=2.0,
            max_events=10000,
            seed=987,
        )
        values = gen.generate_single_series(500)
        # branching = 0.3, rate = 0.5 / 0.7 ~ 0.714 per bin
        assert np.all(values >= 0)
        assert abs(values.mean() - 0.5 / 0.7) < 0.25
