"""Tests for ETSGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import ETSGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)

NO_TREND_NO_SEASONAL = {"trend_type": None, "seasonal_type": None}


def make_gen(**kwargs) -> ETSGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "D", "seed": 42}
    params.update(kwargs)
    return ETSGenerator(**params)


@pytest.fixture
def backend() -> None:
    """Mark tests that exercise native ETS generation."""


class TestETSApi:
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
        df = make_gen().generate(n_series=2, start_id=8)
        assert set(series_values(df)) == {"8", "9"}

    def test_seasonal_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="seasonal has"):
            make_gen(seasonal_period=4, seasonal=[1.0, 2.0])

    def test_multiplicative_requires_positive_level(self) -> None:
        with pytest.raises(ValueError, match="level > 0"):
            make_gen(error_type="mul", level=-1.0)
        with pytest.raises(ValueError, match="level > 0"):
            make_gen(seasonal_type="mul", level=0.0)

    def test_smoothing_parameter_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_gen(alpha=1.5)
        with pytest.raises(ValueError):
            make_gen(beta=-0.1)
        with pytest.raises(ValueError):
            make_gen(noise_std=0.0)
        with pytest.raises(ValueError):
            make_gen(seasonal_period=0)

    def test_mul_trend_nonpositive_initial_reset(self) -> None:
        gen = make_gen(trend_type="mul", trend=0.0)
        assert gen.trend == 1.0

    def test_model_notation(self) -> None:
        assert make_gen()._get_model_notation() == "ETS(A,A,A)"
        assert (
            make_gen(
                error_type="mul", trend_type="add", damped=True, seasonal_type="mul"
            )._get_model_notation()
            == "ETS(M,Ad,M)"
        )
        assert make_gen(**NO_TREND_NO_SEASONAL)._get_model_notation() == "ETS(A,N,N)"

    def test_get_model_info(self) -> None:
        info = make_gen(
            seasonal_period=4, seasonal=[1.0, -1.0, 2.0, -2.0]
        ).get_model_info()
        assert info["model"] == "ETS(A,A,A)"
        assert info["seasonal"] == [1.0, -1.0, 2.0, -2.0]
        assert info["alpha"] == 0.3

    def test_generate_with_states(self, engine: str) -> None:
        gen = make_gen(engine=engine, seasonal_period=4)
        obs_df, states_df = gen.generate_with_states(n_series=2)
        obs = series_values(obs_df)
        assert len(obs) == 2
        import narwhals.stable.v2 as nw

        states_cols = nw.from_native(states_df).columns
        assert {"level", "trend"} <= set(states_cols)
        assert {f"seasonal_{j}" for j in range(4)} <= set(states_cols)
        n_obs = sum(len(v) for v in obs.values())
        assert len(nw.from_native(states_df)) == n_obs

    def test_generate_with_states_no_seasonal(self) -> None:
        gen = make_gen(**NO_TREND_NO_SEASONAL)
        _, states_df = gen.generate_with_states(n_series=1)
        import narwhals.stable.v2 as nw

        assert not any(
            c.startswith("seasonal_") for c in nw.from_native(states_df).columns
        )

    @pytest.mark.usefixtures("backend")
    def test_box_cox_inverse_positive(self) -> None:
        gen = make_gen(
            **NO_TREND_NO_SEASONAL, level=1.0, noise_std=0.5, box_cox_lambda=0.0
        )
        values = gen.generate_single_series(200)
        assert np.all(values > 0)


@pytest.mark.stats
class TestETSStats:
    @pytest.mark.usefixtures("backend")
    def test_ses_differences_are_ma1(self) -> None:
        # ETS(A,N,N): y_t - y_{t-1} = eps_t + (alpha - 1) eps_{t-1}, an MA(1)
        # with theta = alpha - 1.
        alpha, sigma = 0.3, 1.5
        gen = make_gen(
            **NO_TREND_NO_SEASONAL, level=10.0, alpha=alpha, noise_std=sigma, seed=123
        )
        diffs = np.diff(gen.generate_single_series(20000))
        theta = alpha - 1.0
        assert_std(diffs, sigma * np.sqrt(1 + theta**2))
        assert_acf(diffs, 1, theta / (1 + theta**2))
        assert_acf(diffs, 2, 0.0)

    @pytest.mark.usefixtures("backend")
    def test_holt_ensemble_moments(self) -> None:
        # ETS(A,A,N): y_T = l0 + T b0 + eps_T + sum_k (alpha + k beta) eps_.
        T, R = 40, 2000
        l0, b0, alpha, beta, sigma = 50.0, 0.5, 0.3, 0.1, 1.0
        gen = make_gen(
            min_length=T,
            max_length=T,
            trend_type="add",
            seasonal_type=None,
            level=l0,
            trend=b0,
            alpha=alpha,
            beta=beta,
            noise_std=sigma,
            seed=7,
        )
        y_end = np.array([gen.generate_single_series(T)[-1] for _ in range(R)])
        k = np.arange(1, T)
        theo_std = sigma * np.sqrt(1 + np.sum((alpha + k * beta) ** 2))
        assert_mean(y_end, l0 + T * b0, theo_std)
        assert_std(y_end, theo_std)

    @pytest.mark.usefixtures("backend")
    def test_mnn_mean_preserved(self) -> None:
        # ETS(M,N,N) is a martingale: E[y_t] = l0 for all t.
        T, R, l0 = 30, 2000, 100.0
        gen = make_gen(
            min_length=T,
            max_length=T,
            error_type="mul",
            trend_type=None,
            seasonal_type=None,
            level=l0,
            alpha=0.2,
            noise_std=0.05,
            seed=11,
        )
        y_end = np.array([gen.generate_single_series(T)[-1] for _ in range(R)])
        # Var(y_T) = l0^2 [(1 + alpha^2 s^2)^(T-1) (1 + s^2) - 1]
        var = l0**2 * ((1 + 0.2**2 * 0.05**2) ** (T - 1) * (1 + 0.05**2) - 1)
        assert_mean(y_end, l0, np.sqrt(var))

    @pytest.mark.usefixtures("backend")
    def test_damped_trend_converges(self) -> None:
        # With alpha=beta=0 and damping, the forecast path converges to
        # l0 + b0 * phi / (1 - phi).
        l0, b0, phi = 10.0, 5.0, 0.8
        gen = make_gen(
            min_length=300,
            max_length=300,
            trend_type="add",
            seasonal_type=None,
            damped=True,
            phi=phi,
            level=l0,
            trend=b0,
            alpha=0.0,
            beta=0.0,
            noise_std=1e-9,
        )
        values = gen.generate_single_series(300)
        np.testing.assert_allclose(values[-1], l0 + b0 * phi / (1 - phi), rtol=1e-6)

    @pytest.mark.usefixtures("backend")
    def test_additive_seasonal_pattern(self) -> None:
        # With alpha=gamma=0 the seasonal pattern repeats exactly.
        s0 = [4.0, -2.0, 1.0, -3.0]
        gen = make_gen(
            min_length=40,
            max_length=40,
            trend_type=None,
            seasonal_type="add",
            seasonal_period=4,
            seasonal=s0,
            level=20.0,
            alpha=0.0,
            gamma=0.0,
            noise_std=1e-9,
        )
        values = gen.generate_single_series(40)
        expected = 20.0 + np.tile(s0, 10)
        np.testing.assert_allclose(values, expected, atol=1e-6)

    @pytest.mark.usefixtures("backend")
    def test_multiplicative_seasonal_pattern(self) -> None:
        s0 = [1.3, 0.7, 1.1, 0.9]
        gen = make_gen(
            min_length=40,
            max_length=40,
            error_type="mul",
            trend_type=None,
            seasonal_type="mul",
            seasonal_period=4,
            seasonal=s0,
            level=50.0,
            alpha=0.0,
            gamma=0.0,
            noise_std=1e-9,
        )
        values = gen.generate_single_series(40)
        expected = 50.0 * np.tile(s0, 10)
        np.testing.assert_allclose(values, expected, rtol=1e-6)

    def test_innovation_distribution_uniform(self) -> None:
        # ETS(A,N,N) with alpha=1 reduces to a random walk whose steps are
        # the raw innovations; the native path must honor the distribution.
        sigma = 1.0
        gen = make_gen(
            **NO_TREND_NO_SEASONAL,
            level=0.0,
            alpha=1.0,
            noise_std=sigma,
            innovation_distribution="uniform",
            seed=13,
        )
        values = gen.generate_single_series(20000)
        steps = np.diff(values)
        a = sigma * np.sqrt(3.0)
        assert_distribution(steps, stats.uniform(loc=-a, scale=2 * a))
