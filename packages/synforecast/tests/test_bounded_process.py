"""Tests for BoundedProcessGenerator."""

import numpy as np
import pytest

from synforecast.generators import BoundedProcessGenerator
from tests.helpers import (
    assert_long_format,
    assert_mean,
    sample_acf,
    series_values,
)


def _make(engine: str = "pandas", **overrides) -> BoundedProcessGenerator:
    params = {
        "min_length": 50,
        "max_length": 100,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(overrides)
    return BoundedProcessGenerator(**params)


class TestAPI:
    @pytest.mark.parametrize("model", ["beta_ar", "logit_normal"])
    def test_long_format(self, engine: str, model: str) -> None:
        df = _make(engine, model=model).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=100)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine).generate(n_series=2)
        df2 = _make(engine).generate(n_series=2)
        vals1 = series_values(df1)
        vals2 = series_values(df2)
        assert vals1.keys() == vals2.keys()
        for uid in vals1:
            np.testing.assert_array_equal(vals1[uid], vals2[uid])

    def test_custom_bounds_determinism(self, engine: str) -> None:
        # Custom bounds take the Python (non-batch) path; still deterministic
        df1 = _make(engine, lower=10.0, upper=20.0).generate(n_series=2)
        df2 = _make(engine, lower=10.0, upper=20.0).generate(n_series=2)
        for uid, vals in series_values(df1).items():
            np.testing.assert_array_equal(vals, series_values(df2)[uid])

    def test_get_model_info(self) -> None:
        info = _make(omega=0.1, phi=0.8).get_model_info()
        assert info["model"] == "beta_ar"
        assert info["bounds"] == (0.0, 1.0)
        assert info["stationary_mean"] == pytest.approx(0.5)

        scaled = _make(omega=0.1, phi=0.8, lower=10.0, upper=20.0).get_model_info()
        assert scaled["stationary_mean"] == pytest.approx(15.0)

        ln_info = _make(model="logit_normal").get_model_info()
        assert "sigma" in ln_info


class TestValidation:
    def test_bounds_ordering(self) -> None:
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            _make(lower=1.0, upper=0.0)
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            _make(lower=0.5, upper=0.5)

    def test_beta_ar_mean_range(self) -> None:
        with pytest.raises(ValueError, match="omega \\+ max\\(phi, 0\\)"):
            _make(model="beta_ar", omega=0.3, phi=0.8)
        with pytest.raises(ValueError, match="omega \\+ min\\(phi, 0\\)"):
            _make(model="beta_ar", omega=0.1, phi=-0.2)

    def test_invalid_positive_params(self) -> None:
        with pytest.raises(ValueError):
            _make(kappa=0.0)
        with pytest.raises(ValueError):
            _make(sigma=0.0)
        with pytest.raises(ValueError):
            _make(initial_value=0.0)


@pytest.mark.stats
class TestStatistics:
    @pytest.mark.parametrize("model", ["beta_ar", "logit_normal"])
    def test_within_default_bounds(self, model: str) -> None:
        gen = _make(min_length=1000, max_length=1000, model=model, seed=7)
        values = np.concatenate(list(series_values(gen.generate(n_series=5)).values()))
        assert np.all(values >= 0.0) and np.all(values <= 1.0)

    @pytest.mark.parametrize("model", ["beta_ar", "logit_normal"])
    def test_within_custom_bounds(self, model: str) -> None:
        gen = _make(
            min_length=1000,
            max_length=1000,
            model=model,
            lower=10.0,
            upper=20.0,
            seed=7,
        )
        values = np.concatenate(list(series_values(gen.generate(n_series=5)).values()))
        assert np.all(values >= 10.0) and np.all(values <= 20.0)

    def test_beta_ar_stationary_mean(self) -> None:
        """Beta-AR mean converges to omega / (1 - phi) = 0.5."""
        gen = _make(
            min_length=2000,
            max_length=2000,
            model="beta_ar",
            omega=0.1,
            phi=0.8,
            kappa=20.0,
            seed=11,
        )
        values = np.concatenate(list(series_values(gen.generate(n_series=5)).values()))
        # Stationary std ~= 0.18; inflate ~3x for the AR(1) autocorrelation
        assert_mean(values, expected=0.5, std=0.18 * 3.0)

    def test_beta_ar_mean_reversion_from_extreme_start(self) -> None:
        """Series started near the boundary reverts to the stationary mean."""
        gen = _make(
            min_length=2000,
            max_length=2000,
            model="beta_ar",
            omega=0.1,
            phi=0.8,
            initial_value=0.99,
            seed=13,
        )
        values = next(iter(series_values(gen.generate(n_series=1)).values()))
        tail = values[500:]
        assert abs(tail.mean() - 0.5) < 0.1

    def test_logit_normal_symmetric_around_half(self) -> None:
        """With zero-mean logit AR(1), values are symmetric around 0.5."""
        gen = _make(
            min_length=2000,
            max_length=2000,
            model="logit_normal",
            phi=0.9,
            sigma=0.3,
            seed=17,
        )
        values = np.concatenate(list(series_values(gen.generate(n_series=5)).values()))
        # x is bounded so std < 0.5; inflate ~4.4x for phi=0.9 autocorrelation
        assert_mean(values, expected=0.5, std=0.25 * 4.4)

    def test_phi_controls_persistence(self) -> None:
        """Higher phi gives higher lag-1 autocorrelation (ACF(1) = phi)."""
        base = {"min_length": 3000, "max_length": 3000, "model": "beta_ar", "seed": 19}
        low = _make(**base, phi=0.2, omega=0.4)
        high = _make(**base, phi=0.8, omega=0.1)
        low_vals = next(iter(series_values(low.generate(n_series=1)).values()))
        high_vals = next(iter(series_values(high.generate(n_series=1)).values()))
        low_acf = sample_acf(low_vals, 1)
        high_acf = sample_acf(high_vals, 1)
        assert abs(low_acf - 0.2) < 0.1
        assert abs(high_acf - 0.8) < 0.1
        assert high_acf > low_acf
