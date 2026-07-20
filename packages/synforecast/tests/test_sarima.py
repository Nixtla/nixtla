"""Tests for SARIMAGenerator."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.arima_process import arma_acf, arma_acovf

import synforecast.generators.sarima as sarima_mod
from synforecast.generators import SARIMAGenerator
from tests.helpers import (
    assert_acf,
    assert_distribution,
    assert_long_format,
    assert_mean,
    sample_acf,
    series_values,
)

# Orders default to (1,0,1)(1,0,1); tests use explicit orders/coefficients.
WHITE_NOISE = {"p": 0, "q": 0, "P": 0, "Q": 0}


def make_gen(**kwargs) -> SARIMAGenerator:
    params = {"min_length": 50, "max_length": 80, "freq": "D", "seed": 42}
    params.update(kwargs)
    return SARIMAGenerator(**params)


@pytest.fixture(params=["rust", "python"])
def backend(request: pytest.FixtureRequest, monkeypatch) -> str:
    """Run generate_single_series through the Rust and pure-Python paths."""
    if request.param == "python":
        monkeypatch.setattr(sarima_mod, "_HAS_RUST", False)
    return request.param


class TestSARIMAApi:
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
        df = make_gen().generate(n_series=2, start_id=3)
        assert set(series_values(df)) == {"3", "4"}

    def test_param_length_mismatches_raise(self) -> None:
        with pytest.raises(ValueError, match="ar_params length"):
            make_gen(p=2, ar_params=[0.5])
        with pytest.raises(ValueError, match="ma_params length"):
            make_gen(q=2, ma_params=[0.5])
        with pytest.raises(ValueError, match="seasonal_ar_params length"):
            make_gen(P=2, seasonal_ar_params=[0.5])
        with pytest.raises(ValueError, match="seasonal_ma_params length"):
            make_gen(Q=2, seasonal_ma_params=[0.5])

    def test_nonstationary_ar_raises(self) -> None:
        with pytest.raises(ValueError, match="non-stationary"):
            make_gen(p=1, ar_params=[1.05])

    def test_nonstationary_combined_polynomial_raises(self) -> None:
        # phi(B)Phi(B^s) with both factors near the unit circle
        with pytest.raises(ValueError, match="non-stationary"):
            make_gen(p=1, P=1, ar_params=[1.0], seasonal_ar_params=[0.5])

    def test_validate_stationarity_false_allows(self) -> None:
        gen = make_gen(p=1, ar_params=[1.05], validate_stationarity=False, burn_in=10)
        assert gen.generate_single_series(20).shape == (20,)

    def test_order_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_gen(d=3)
        with pytest.raises(ValueError):
            make_gen(noise_std=0.0)
        with pytest.raises(ValueError):
            make_gen(seasonal_period=0)

    def test_polynomial_expansion(self) -> None:
        # (1 - phi B)(1 - Phi B^4) = 1 - phi B - Phi B^4 + phi Phi B^5
        gen = make_gen(
            p=1,
            q=1,
            P=1,
            Q=1,
            seasonal_period=4,
            ar_params=[0.5],
            ma_params=[0.3],
            seasonal_ar_params=[0.4],
            seasonal_ma_params=[0.2],
        )
        np.testing.assert_allclose(gen._full_ar_poly, [0.5, 0, 0, 0.4, -0.2])
        np.testing.assert_allclose(gen._full_ma_poly, [0.3, 0, 0, 0.2, 0.06])

    def test_ar_filter_matches_recurrence(self) -> None:
        gen = make_gen(
            p=2,
            q=0,
            P=0,
            Q=0,
            ar_params=[0.5, -0.25],
        )

        result = gen._apply_ar_filter(np.array([1.0, 2.0, 3.0, 4.0]))

        np.testing.assert_allclose(result, [1.0, 2.5, 4.0, 5.375])

    def test_get_model_info(self) -> None:
        gen = make_gen(p=1, ar_params=[0.5], q=0, P=0, Q=0, d=1, drift=0.3)
        info = gen.get_model_info()
        assert info["model"] == "SARIMA(1,1,0)(0,0,0)[12]"
        assert info["ar_params"] == [0.5]
        assert info["drift"] == 0.3
        assert info["mean"] is None

    def test_exog_effects(self, monkeypatch) -> None:
        # Same seed with/without exog: outputs differ by exog @ coefficients.
        # Python path for both runs (exog is Python-only) so streams match.
        monkeypatch.setattr(sarima_mod, "_HAS_RUST", False)
        kwargs = dict(**WHITE_NOISE, exog_coefficients=[2.0, -1.0], seed=5, burn_in=10)
        exog = np.column_stack([np.ones(30), np.arange(30.0)])
        with_exog = make_gen(**kwargs).generate_single_series(30, exog=exog)
        without = make_gen(**kwargs).generate_single_series(30, exog=None)
        np.testing.assert_allclose(with_exog - without, exog @ [2.0, -1.0])

    def test_exog_shape_mismatch_raises(self) -> None:
        gen = make_gen(**WHITE_NOISE, exog_coefficients=[1.0], burn_in=10)
        with pytest.raises(ValueError, match="rows"):
            gen.generate_single_series(30, exog=np.ones((10, 1)))
        with pytest.raises(ValueError, match="columns"):
            gen.generate_single_series(30, exog=np.ones((30, 2)))


@pytest.mark.stats
class TestSARIMAStats:
    @pytest.mark.usefixtures("backend")
    def test_white_noise(self) -> None:
        gen = make_gen(**WHITE_NOISE, mean=5.0, noise_std=2.0, seed=123)
        values = gen.generate_single_series(20000)
        assert_mean(values, 5.0, 2.0)
        assert_distribution(values, stats.norm(loc=5.0, scale=2.0))
        for lag in (1, 2, 12):
            assert_acf(values, lag, 0.0)

    @pytest.mark.usefixtures("backend")
    def test_arma11_acf_matches_statsmodels(self) -> None:
        ar, ma, sigma = 0.6, 0.4, 1.5
        gen = make_gen(
            p=1,
            q=1,
            P=0,
            Q=0,
            ar_params=[ar],
            ma_params=[ma],
            mean=3.0,
            noise_std=sigma,
            seed=123,
        )
        n = 50000
        values = gen.generate_single_series(n)

        theo_acf = arma_acf([1, -ar], [1, ma], lags=6)
        # Bartlett-style SE for the sample ACF of a correlated series is
        # larger than 1/sqrt(n); use a widened bound.
        se = 3.0 / np.sqrt(n)
        for lag in (1, 2, 3, 5):
            got = sample_acf(values, lag)
            assert abs(got - theo_acf[lag]) < 5 * se, (
                f"acf({lag})={got:.4f} vs {theo_acf[lag]:.4f}"
            )

        # Marginal variance gamma_0 and mean; the sample mean SE is inflated
        # by the ACF sum, the sample variance SE by correlation.
        gamma0 = arma_acovf([1, -ar], [1, ma], nobs=200, sigma2=sigma**2)[0]
        inflation = np.sqrt(np.sum(arma_acf([1, -ar], [1, ma], lags=100) ** 2))
        assert_mean(values, 3.0, np.sqrt(gamma0) * 2 * inflation)
        var_se = gamma0 * np.sqrt(2.0 / n) * 2 * inflation
        assert abs(values.var() - gamma0) < 5 * var_se

    @pytest.mark.usefixtures("backend")
    def test_seasonal_ma_acf_structure(self) -> None:
        # SARIMA(0,0,1)(0,0,1)_12 has MA dependence at lags 1, 11, 12, 13
        # from the expanded polynomial (1 + theta B)(1 + Theta B^12).
        theta, Theta, s = 0.5, 0.6, 12
        gen = make_gen(
            p=0,
            q=1,
            P=0,
            Q=1,
            seasonal_period=s,
            ma_params=[theta],
            seasonal_ma_params=[Theta],
            seed=456,
        )
        n = 50000
        values = gen.generate_single_series(n)

        full_ma = np.concatenate(([1.0], gen._full_ma_poly))
        theo_acf = arma_acf([1.0], full_ma, lags=15)
        se = 2.0 / np.sqrt(n)
        for lag in (1, 2, 6, 11, 12, 13, 14):
            got = sample_acf(values, lag)
            assert abs(got - theo_acf[lag]) < 5 * se, (
                f"acf({lag})={got:.4f} vs {theo_acf[lag]:.4f}"
            )

    def test_integrated_drift(self, monkeypatch) -> None:
        # d=1: the differenced series is ARMA with mean `drift`, so the
        # series has slope `drift` per step. (Python path; the Rust kernel
        # handles drift differently.)
        monkeypatch.setattr(sarima_mod, "_HAS_RUST", False)
        ar, drift = 0.5, 0.4
        gen = make_gen(p=1, q=0, P=0, Q=0, d=1, ar_params=[ar], drift=drift, seed=99)
        n = 10000
        diffs = np.diff(gen.generate_single_series(n))
        # AR(1) diffs: SE of the mean inflated by sqrt((1+ar)/(1-ar))
        sigma_arma = 1.0 / np.sqrt(1 - ar**2)
        assert_mean(diffs, drift, sigma_arma * np.sqrt((1 + ar) / (1 - ar)))
        assert_acf(diffs, 1, ar, z=10.0)

    @pytest.mark.usefixtures("backend")
    def test_seasonal_differencing_recovers_arma(self) -> None:
        # D=1: seasonally differencing the output recovers the stationary
        # ARMA process (white noise here).
        s = 12
        gen = make_gen(**WHITE_NOISE, D=1, seasonal_period=s, seed=7)
        values = gen.generate_single_series(20000)
        sdiff = values[s:] - values[:-s]
        assert_mean(sdiff, 0.0, 1.0)
        assert_acf(sdiff, 1, 0.0)

    def test_burn_in_removes_startup_transient(self) -> None:
        # With persistent AR, early values must already be stationary:
        # ensemble of y_0 has the stationary variance, not variance ~0.
        ar = 0.95
        gen = make_gen(
            p=1, q=0, P=0, Q=0, ar_params=[ar], min_length=10, max_length=10, seed=13
        )
        first = np.array([gen.generate_single_series(10)[0] for _ in range(500)])
        sigma_stat = 1.0 / np.sqrt(1 - ar**2)
        assert_mean(first, 0.0, sigma_stat)
        assert first.std() > 0.7 * sigma_stat

    def test_innovation_distribution_t(self, monkeypatch) -> None:
        # Python path honors innovation_distribution: white-noise output is
        # the (variance-rescaled) t distribution.
        monkeypatch.setattr(sarima_mod, "_HAS_RUST", False)
        df, sigma = 6.0, 2.0
        gen = make_gen(
            **WHITE_NOISE,
            noise_std=sigma,
            innovation_distribution="t",
            innovation_params={"df": df},
            seed=17,
        )
        values = gen.generate_single_series(20000)
        t_scale = sigma * np.sqrt((df - 2) / df)
        assert_distribution(values, stats.t(df=df, loc=0.0, scale=t_scale))
