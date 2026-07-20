"""Tests for ChaoticSystemGenerator."""

import numpy as np
import pytest
from scipy import stats

import synforecast.generators.chaotic_system as cs_mod
from synforecast.generators import ChaoticSystemGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)

BASE = {"min_length": 30, "max_length": 60, "freq": "D", "seed": 42}

SYSTEMS = ["lorenz", "logistic", "mackey_glass"]


class TestChaoticApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = ChaoticSystemGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=30, max_length=60)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = ChaoticSystemGenerator(**BASE, engine=engine).generate(n_series=3)
        df2 = ChaoticSystemGenerator(**BASE, engine=engine).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    @pytest.mark.parametrize("system", SYSTEMS)
    def test_all_systems_produce_finite_output(self, system: str) -> None:
        gen = ChaoticSystemGenerator(**BASE, system=system)
        values = gen.generate_single_series(50)
        assert values.shape == (50,)
        assert np.all(np.isfinite(values))

    def test_invalid_system(self) -> None:
        with pytest.raises(ValueError):
            ChaoticSystemGenerator(**BASE, system="henon")

    @pytest.mark.parametrize(
        "field,value",
        [("dt", 0.0), ("mg_tau", 0), ("observation_noise", -0.1)],
    )
    def test_invalid_parameters(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            ChaoticSystemGenerator(**BASE, **{field: value})

    def test_beta_param_accepted_by_field_name_and_alias(self) -> None:
        """Regression: beta_param used to be silently ignored (alias-only)."""
        gen_field = ChaoticSystemGenerator(**BASE, beta_param=2.5)
        assert gen_field.beta_param == 2.5
        gen_alias = ChaoticSystemGenerator(**BASE, lorenz_beta=2.5)
        assert gen_alias.beta_param == 2.5

    @pytest.mark.parametrize("system", SYSTEMS)
    def test_zero_perturbation_is_seed_independent(self, system: str) -> None:
        """With initial_perturbation=0 the system is fully deterministic."""
        v1 = ChaoticSystemGenerator(
            **{**BASE, "seed": 1}, system=system, initial_perturbation=0.0
        ).generate_single_series(50)
        v2 = ChaoticSystemGenerator(
            **{**BASE, "seed": 999}, system=system, initial_perturbation=0.0
        ).generate_single_series(50)
        np.testing.assert_array_equal(v1, v2)

    def test_get_model_info(self) -> None:
        info = ChaoticSystemGenerator(**BASE, system="lorenz").get_model_info()
        assert info["deterministic"] is True
        assert info["rho"] == 28.0
        info_log = ChaoticSystemGenerator(**BASE, system="logistic").get_model_info()
        assert info_log["r"] == 3.9


@pytest.mark.stats
class TestChaoticStats:
    """Statistical/dynamical property tests (fixed seeds)."""

    def test_lorenz_bounded_on_attractor(self) -> None:
        """The Lorenz x-component stays within the attractor's range."""
        gen = ChaoticSystemGenerator(
            min_length=500, max_length=500, freq="D", system="lorenz", seed=1
        )
        values = gen.generate_single_series(500)
        assert np.abs(values).max() < 30.0
        assert values.std() > 1.0  # non-degenerate, visits both wings

    def test_logistic_r4_invariant_density(self) -> None:
        """For r=4 the invariant density is Beta(1/2, 1/2) (arcsine law)."""
        gen = ChaoticSystemGenerator(
            min_length=2000,
            max_length=2000,
            freq="D",
            system="logistic",
            logistic_r=4.0,
            seed=2,
        )
        values = gen.generate_single_series(2000)
        assert values.min() >= 0.0 and values.max() <= 1.0
        assert_distribution(values, stats.beta(0.5, 0.5))

    def test_logistic_period_two_orbit(self) -> None:
        """For r=3.2 (below chaos) the map settles on a period-2 orbit."""
        gen = ChaoticSystemGenerator(
            min_length=200,
            max_length=200,
            freq="D",
            system="logistic",
            logistic_r=3.2,
            seed=3,
        )
        values = gen.generate_single_series(200)
        assert len(np.unique(np.round(values, 8))) <= 2

    def test_mackey_glass_bounded_oscillation(self) -> None:
        gen = ChaoticSystemGenerator(
            min_length=500, max_length=500, freq="D", system="mackey_glass", seed=4
        )
        values = gen.generate_single_series(500)
        assert values.min() > 0.0
        assert values.max() < 2.0
        assert values.std() > 0.05  # oscillates, not a fixed point

    @pytest.mark.parametrize("system", ["lorenz", "logistic"])
    def test_sensitivity_to_initial_conditions(self, system: str) -> None:
        """Tiny IC perturbations (different seeds) decorrelate trajectories,
        while identical seeds reproduce them exactly."""
        kwargs = {
            "min_length": 300,
            "max_length": 300,
            "freq": "D",
            "system": system,
            "initial_perturbation": 1e-6,
        }
        a = ChaoticSystemGenerator(**kwargs, seed=6).generate_single_series(300)
        a2 = ChaoticSystemGenerator(**kwargs, seed=6).generate_single_series(300)
        b = ChaoticSystemGenerator(**kwargs, seed=7).generate_single_series(300)
        np.testing.assert_array_equal(a, a2)
        assert not np.allclose(a, b)
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.5

    def test_observation_noise_is_additive_gaussian(self) -> None:
        """Same seed => same trajectory; the difference is the N(0, s^2) noise."""
        kwargs = {
            "min_length": 1000,
            "max_length": 1000,
            "freq": "D",
            "system": "logistic",
        }
        clean = ChaoticSystemGenerator(
            **kwargs, observation_noise=0.0, seed=8
        ).generate_single_series(1000)
        noisy = ChaoticSystemGenerator(
            **kwargs, observation_noise=0.5, seed=8
        ).generate_single_series(1000)
        diff = noisy - clean
        assert_mean(diff, expected=0.0, std=0.5)
        assert_std(diff, expected=0.5)

    def test_python_fallback_logistic_matches_theory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cs_mod, "_HAS_RUST", False)
        gen = ChaoticSystemGenerator(
            min_length=2000,
            max_length=2000,
            freq="D",
            system="logistic",
            logistic_r=4.0,
            seed=9,
        )
        values = gen.generate_single_series(2000)
        assert_distribution(values, stats.beta(0.5, 0.5))

    def test_python_fallback_lorenz_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cs_mod, "_HAS_RUST", False)
        gen = ChaoticSystemGenerator(
            min_length=100, max_length=100, freq="D", system="lorenz", seed=10
        )
        values = gen.generate_single_series(100)
        assert np.abs(values).max() < 30.0
