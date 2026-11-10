"""Tests for StochasticVolatilityGenerator."""

import numpy as np
import pytest

from synforecast.generators import StochasticVolatilityGenerator
from tests.helpers import (
    assert_acf,
    assert_long_format,
    assert_mean,
    series_values,
)


def _make(engine: str = "pandas", **overrides) -> StochasticVolatilityGenerator:
    params = {
        "min_length": 60,
        "max_length": 100,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(overrides)
    return StochasticVolatilityGenerator(**params)


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _make(engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=60, max_length=100)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine).generate(n_series=2)
        df2 = _make(engine).generate(n_series=2)
        vals1 = series_values(df1)
        vals2 = series_values(df2)
        assert vals1.keys() == vals2.keys()
        for uid in vals1:
            np.testing.assert_array_equal(vals1[uid], vals2[uid])

    @pytest.mark.parametrize("model", ["heston", "sabr"])
    @pytest.mark.parametrize("output_type", ["price", "returns", "volatility"])
    def test_output_types(self, engine: str, model: str, output_type: str) -> None:
        gen = _make(engine, model=model, output_type=output_type)
        df = gen.generate(n_series=2)
        assert_long_format(df, n_series=2, min_length=60, max_length=100)
        values = np.concatenate(list(series_values(df).values()))
        if output_type in ("price", "volatility"):
            assert np.all(values >= 0.0)

    def test_get_model_info(self) -> None:
        info = _make().get_model_info()
        assert info["model"] == "heston"
        assert info["leverage_effect"] is True
        assert info["feller_condition_satisfied"] is True  # 2*2*0.04 > 0.3^2
        sabr_info = _make(model="sabr").get_model_info()
        assert sabr_info["beta_interpretation"] == "CIR-like"


class TestValidation:
    def test_invalid_correlation(self) -> None:
        with pytest.raises(ValueError):
            _make(correlation=1.5)
        with pytest.raises(ValueError):
            _make(correlation=-1.5)

    def test_invalid_positive_params(self) -> None:
        for field in ("initial_price", "initial_vol", "vol_of_vol", "dt"):
            with pytest.raises(ValueError):
                _make(**{field: 0.0})

    def test_invalid_beta(self) -> None:
        with pytest.raises(ValueError):
            _make(beta=1.5)


@pytest.mark.stats
class TestStatistics:
    def test_heston_variance_mean_reverts_to_theta(self) -> None:
        """Long-run mean of the variance path equals theta."""
        gen = _make(
            min_length=1000,
            max_length=1000,
            model="heston",
            output_type="volatility",
            initial_vol=0.04,
            mean_vol=0.04,
            vol_mean_reversion=4.0,
            vol_of_vol=0.3,
            dt=1 / 12,
            seed=7,
        )
        df = gen.generate(n_series=10)
        variance = np.concatenate(list(series_values(df).values())) ** 2
        # Stationary std of V is sigma_v*sqrt(theta/(2*kappa)) ~= 0.0212;
        # inflate ~3x for the AR(1) autocorrelation (phi = 1 - kappa*dt = 2/3)
        assert_mean(variance, expected=0.04, std=0.0212 * 3.0)

    def test_heston_volatility_persistence(self) -> None:
        """Euler-discretized variance is AR(1)-like with phi = 1 - kappa*dt."""
        kappa, dt = 4.0, 1 / 12
        gen = _make(
            min_length=3000,
            max_length=3000,
            model="heston",
            output_type="volatility",
            initial_vol=0.04,
            mean_vol=0.04,
            vol_mean_reversion=kappa,
            vol_of_vol=0.3,
            dt=dt,
            seed=9,
        )
        df = gen.generate(n_series=1)
        variance = next(iter(series_values(df).values())) ** 2
        assert_acf(variance, lag=1, expected=1.0 - kappa * dt)

    def test_heston_leverage_effect(self) -> None:
        """Negative rho: returns and variance changes correlate negatively."""
        gen = _make(
            min_length=4000,
            max_length=4000,
            model="heston",
            correlation=-0.8,
            seed=21,
        )
        prices, vols, _ = gen.generate_with_volatility(n_series=1)
        returns = np.diff(np.log(prices))
        dvar = np.diff(vols**2)
        corr = np.corrcoef(returns, dvar)[0, 1]
        assert corr < -0.5, f"leverage correlation {corr:.3f} not strongly negative"

    def test_heston_returns_scale(self) -> None:
        """Log returns have std ~ sqrt(theta*dt) when V stays near theta."""
        theta, dt = 0.04, 1 / 252
        gen = _make(
            min_length=2000,
            max_length=2000,
            model="heston",
            output_type="returns",
            drift=0.0,
            initial_vol=theta,
            mean_vol=theta,
            vol_mean_reversion=5.0,
            vol_of_vol=0.1,
            dt=dt,
            seed=17,
        )
        df = gen.generate(n_series=5)
        r = np.concatenate([v[1:] for v in series_values(df).values()])
        expected_std = np.sqrt(theta * dt)
        assert 0.8 * expected_std < r.std() < 1.2 * expected_std

    def test_sabr_paths_positive(self) -> None:
        gen = _make(
            min_length=1000,
            max_length=1000,
            model="sabr",
            beta=0.5,
            seed=3,
        )
        prices, vols, _ = gen.generate_with_volatility(n_series=3)
        assert np.all(prices > 0.0)
        assert np.all(vols > 0.0)
