"""Tests for StateSpaceGenerator."""

import numpy as np
import pytest
from scipy import stats

import synforecast.generators.state_space as state_space_module
from synforecast.generators import StateSpaceGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_std,
    series_values,
)


def make_gen(engine: str = "pandas", **kwargs) -> StateSpaceGenerator:
    params = {
        "min_length": 100,
        "max_length": 100,
        "freq": "h",
        "engine": engine,
        "seed": 42,
    }
    params.update(kwargs)
    return StateSpaceGenerator(**params)


def ar1_params(phi: float = 0.8, q: float = 0.25, r: float = 0.0) -> dict:
    """AR(1)-as-SSM: x[t] = phi x[t-1] + w, y[t] = x[t] + v.

    The initial state covariance is set to the stationary variance
    q / (1 - phi^2) so the series starts in its stationary distribution.
    """
    return {
        "transition_matrix": [[phi]],
        "state_covariance": [[q]],
        "obs_covariance": [[r]],
        "initial_state": [0.0],
        "initial_state_covariance": [[q / (1.0 - phi**2)]],
    }


class TestStateSpaceAPI:
    """Structural / API contract tests."""

    def test_long_format(self, engine: str) -> None:
        gen = make_gen(engine, min_length=40, max_length=60)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=40, max_length=60)

    def test_start_id(self, engine: str) -> None:
        gen = make_gen(engine)
        values = series_values(gen.generate(n_series=2, start_id=7))
        assert set(values) == {"7", "8"}

    def test_seed_determinism(self, engine: str) -> None:
        df_a = make_gen(engine, seed=7).generate(n_series=3)
        df_b = make_gen(engine, seed=7).generate(n_series=3)
        values_a, values_b = series_values(df_a), series_values(df_b)
        assert set(values_a) == set(values_b)
        for uid in values_a:
            np.testing.assert_array_equal(values_a[uid], values_b[uid])
        df_c = make_gen(engine, seed=8).generate(n_series=3)
        assert not np.array_equal(series_values(df_c)["0"], values_a["0"]), (
            "different seeds should give different values"
        )

    def test_n_jobs_independence(self) -> None:
        """Results are identical regardless of the number of workers."""
        values_1 = series_values(make_gen(seed=3).generate(n_series=4, n_jobs=1))
        values_4 = series_values(make_gen(seed=3).generate(n_series=4, n_jobs=4))
        for uid in values_1:
            np.testing.assert_array_equal(values_1[uid], values_4[uid])

    def test_multidimensional_state_and_obs(self, engine: str) -> None:
        gen = make_gen(
            engine,
            state_dim=3,
            obs_dim=2,
            observation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        )
        df = gen.generate(n_series=2)
        assert_long_format(df, n_series=2)

    def test_singular_state_covariance_accepted(self) -> None:
        """PSD (singular) Q is valid, e.g. deterministic state components."""
        gen = make_gen(
            state_dim=2,
            transition_matrix=[[0.9, 0.0], [0.0, 0.5]],
            state_covariance=[[0.1, 0.0], [0.0, 0.0]],
        )
        values = gen.generate_single_series(50)
        assert np.isfinite(values).all()

    def test_custom_transition_and_observation_fn(self) -> None:
        """Custom functions replace the linear equations entirely."""

        def transition(x, t, rng):  # noqa: ARG001
            return 0.5 * x

        def observation(x, t, rng):  # noqa: ARG001
            return np.full(1, 5.0)

        gen = make_gen(transition_fn=transition, observation_fn=observation)
        values = gen.generate_single_series(20)
        np.testing.assert_allclose(values, 5.0)

    def test_generate_with_states(self, engine: str) -> None:
        gen = make_gen(engine, state_dim=2, min_length=30, max_length=30)
        obs_df, states_df = gen.generate_with_states(n_series=2, start_id=3)

        assert_long_format(obs_df, n_series=2, min_length=30, max_length=30)

        import narwhals.stable.v2 as nw

        states_nw = nw.from_native(states_df)
        assert set(states_nw.columns) == {"unique_id", "ds", "state_0", "state_1"}
        assert states_nw.schema["unique_id"] == nw.Categorical()
        states_pd = states_nw.to_pandas()
        assert set(states_pd["unique_id"].astype(str)) == {"3", "4"}
        # One state row per observation row
        obs_pd = nw.from_native(obs_df).to_pandas()
        assert len(states_pd) == len(obs_pd)


class TestStateSpaceValidation:
    """Constructor validation of state space parameters."""

    def test_bad_transition_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="transition_matrix"):
            make_gen(state_dim=2, transition_matrix=[[0.5]])

    def test_bad_observation_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="observation_matrix"):
            make_gen(state_dim=2, observation_matrix=[[1.0]])

    def test_bad_state_covariance_shape(self) -> None:
        with pytest.raises(ValueError, match="state_covariance"):
            make_gen(state_dim=1, state_covariance=[[1.0, 0.0], [0.0, 1.0]])

    def test_bad_obs_covariance_shape(self) -> None:
        with pytest.raises(ValueError, match="obs_covariance"):
            make_gen(obs_dim=1, obs_covariance=[[1.0, 0.0], [0.0, 1.0]])

    def test_bad_initial_state_shape(self) -> None:
        with pytest.raises(ValueError, match="initial_state"):
            make_gen(state_dim=2, initial_state=[0.0])

    def test_bad_initial_state_covariance_shape(self) -> None:
        with pytest.raises(ValueError, match="initial_state_covariance"):
            make_gen(state_dim=2, initial_state_covariance=[[1.0]])

    def test_negative_definite_state_covariance(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            make_gen(state_covariance=[[-1.0]])

    def test_negative_definite_obs_covariance(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            make_gen(obs_covariance=[[-1.0]])


@pytest.mark.stats
class TestStateSpaceStats:
    """Statistical property tests (fixed seeds)."""

    def test_pure_observation_noise(self) -> None:
        """With F = 0 and Q = 0 the output is exactly the observation
        noise: y ~ N(0, R)."""
        gen = make_gen(
            min_length=4000,
            max_length=4000,
            seed=11,
            transition_matrix=[[0.0]],
            state_covariance=[[0.0]],
            obs_covariance=[[2.25]],
            initial_state=[0.0],
            initial_state_covariance=[[0.0]],
        )
        values = series_values(gen.generate(n_series=1))["0"]
        assert_distribution(values, stats.norm(loc=0.0, scale=1.5))

    def test_ar1_dynamics(self) -> None:
        """F = [[phi]], no obs noise: y is AR(1) with acf(1) = phi and
        stationary std sqrt(q / (1 - phi^2))."""
        phi, q = 0.8, 0.25
        gen = make_gen(min_length=4000, max_length=4000, seed=22, **ar1_params(phi, q))
        values = series_values(gen.generate(n_series=1))["0"]
        assert_acf(values, lag=1, expected=phi)
        # kurtosis widened: AR dependence inflates the variance of s^2
        assert_std(values, expected=np.sqrt(q / (1 - phi**2)), kurtosis=12.0)

    def test_observation_noise_separation(self) -> None:
        """Measurement noise adds to the variance but dilutes the ACF:
        var(y) = var(x) + r, acf_y(1) = phi * var(x) / (var(x) + r)."""
        phi, q, r = 0.8, 0.25, 0.5
        var_x = q / (1 - phi**2)
        gen = make_gen(
            min_length=4000, max_length=4000, seed=33, **ar1_params(phi, q, r)
        )
        values = series_values(gen.generate(n_series=1))["0"]
        assert_std(values, expected=np.sqrt(var_x + r), kurtosis=12.0)
        assert_acf(values, lag=1, expected=phi * var_x / (var_x + r))

    def test_python_fallback_ar1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pure-Python simulation loop has the same AR(1) properties."""
        monkeypatch.setattr(state_space_module, "_HAS_RUST", False)
        phi, q = 0.8, 0.25
        gen = make_gen(min_length=4000, max_length=4000, seed=44, **ar1_params(phi, q))
        values = gen.generate_single_series(4000)
        assert_acf(values, lag=1, expected=phi)
        assert_std(values, expected=np.sqrt(q / (1 - phi**2)), kurtosis=12.0)
