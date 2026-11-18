"""Tests for RegimeSwitchingGenerator."""

import numpy as np
import pytest

from synforecast.generators import RegimeSwitchingGenerator
from tests.helpers import (
    assert_acf,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)

# P = [[0.9, 0.1], [0.3, 0.7]] has stationary distribution pi = (0.75, 0.25)
TWO_REGIME = {
    "n_regimes": 2,
    "regime_means": [-8.0, 8.0],
    "regime_variances": [1.0, 1.0],
    "transition_matrix": [[0.9, 0.1], [0.3, 0.7]],
}


def _make(engine: str = "pandas", **overrides) -> RegimeSwitchingGenerator:
    params = {
        "min_length": 50,
        "max_length": 100,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(overrides)
    return RegimeSwitchingGenerator(**params)


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _make(engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=100)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine, **TWO_REGIME).generate(n_series=2)
        df2 = _make(engine, **TWO_REGIME).generate(n_series=2)
        vals1 = series_values(df1)
        vals2 = series_values(df2)
        assert vals1.keys() == vals2.keys()
        for uid in vals1:
            np.testing.assert_array_equal(vals1[uid], vals2[uid])

    def test_stationary_distribution(self) -> None:
        gen = _make(**TWO_REGIME)
        pi = gen.get_model_info()["stationary_distribution"]
        np.testing.assert_allclose(pi, [0.75, 0.25], atol=1e-10)

    def test_generate_with_regimes(self) -> None:
        gen = _make(min_length=100, max_length=100, **TWO_REGIME)
        values, regimes, ids = gen.generate_with_regimes(n_series=2)
        assert values.shape == regimes.shape == ids.shape == (200,)
        assert set(np.unique(regimes)) <= {0, 1}


class TestValidation:
    def test_rows_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="rows must sum to 1"):
            _make(transition_matrix=[[0.8, 0.1], [0.3, 0.7]])

    def test_wrong_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            _make(n_regimes=3, transition_matrix=[[0.9, 0.1], [0.3, 0.7]])

    def test_wrong_parameter_lengths(self) -> None:
        with pytest.raises(ValueError, match="regime_means must have 2"):
            _make(regime_means=[0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="regime_variances must have 2"):
            _make(regime_variances=[1.0])
        with pytest.raises(ValueError, match="regime_ar_coeffs must have 2"):
            _make(regime_ar_coeffs=[0.5])

    def test_nonpositive_variance(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _make(regime_variances=[1.0, -1.0])

    def test_unstable_ar(self) -> None:
        with pytest.raises(ValueError, match="stability"):
            _make(regime_ar_coeffs=[0.5, 1.0])

    def test_initial_regime_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="initial_regime"):
            _make(initial_regime=2)


@pytest.mark.stats
class TestStatistics:
    def test_regime_occupancy_matches_stationary_distribution(self) -> None:
        """Fraction of time in each regime converges to pi = (0.75, 0.25)."""
        gen = _make(min_length=1000, max_length=1000, seed=7, **TWO_REGIME)
        df = gen.generate(n_series=10)
        values = np.concatenate(list(series_values(df).values()))
        # Means are +-8 with unit variances, so sign classifies the regime
        frac_low = np.mean(values < 0)
        # SE ~ sqrt(0.75*0.25/n) * 2 (chain autocorrelation) ~= 0.009
        assert abs(frac_low - 0.75) < 0.05, f"occupancy {frac_low:.3f} != 0.75"

    def test_batch_initial_regime_drawn_per_series(self) -> None:
        """The Rust batch path draws s_0 per series: across a large batch,
        first-step regimes vary and match the stationary distribution
        pi = (0.75, 0.25) rather than sharing one initial regime."""
        gen = _make(
            min_length=2,
            max_length=2,
            seed=13,
            n_regimes=2,
            regime_means=[-8.0, 8.0],
            regime_variances=[1e-6, 1e-6],
            transition_matrix=[[0.9, 0.1], [0.3, 0.7]],
        )
        n_series = 2000
        df = gen.generate(n_series=n_series)
        first_values = np.array([vals[0] for vals in series_values(df).values()])
        first_regimes = (first_values > 0).astype(int)
        assert len(np.unique(first_regimes)) == 2, (
            "all series in the batch share the same initial regime"
        )
        frac_high = first_regimes.mean()
        # Binomial SE = sqrt(0.75*0.25/2000) ~= 0.0097; allow ~5 SE
        assert abs(frac_high - 0.25) < 0.05, (
            f"t=0 occupancy {frac_high:.3f} != stationary 0.25"
        )

    def test_per_regime_means_and_vols(self) -> None:
        """Values within each regime have that regime's mean and variance."""
        gen = _make(
            min_length=1000,
            max_length=1000,
            seed=11,
            n_regimes=2,
            regime_means=[-8.0, 8.0],
            regime_variances=[1.0, 4.0],
            transition_matrix=[[0.9, 0.1], [0.3, 0.7]],
        )
        df = gen.generate(n_series=10)
        values = np.concatenate(list(series_values(df).values()))
        low, high = values[values < 0], values[values > 0]
        assert_mean(low, expected=-8.0, std=1.0)
        assert_std(low, expected=1.0)
        assert_mean(high, expected=8.0, std=2.0)
        assert_std(high, expected=2.0)

    def test_ar_dynamics_within_regime(self) -> None:
        """An absorbing regime with phi=0.7 behaves as a plain AR(1)."""
        phi = 0.7
        gen = _make(
            min_length=4000,
            max_length=4000,
            seed=13,
            n_regimes=2,
            regime_means=[0.0, 100.0],
            regime_variances=[1.0, 1.0],
            regime_ar_coeffs=[phi, 0.0],
            transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
            initial_regime=0,
        )
        df = gen.generate(n_series=1)
        values = next(iter(series_values(df).values()))
        assert_acf(values, lag=1, expected=phi)
        # AR(1) stationary std = sigma / sqrt(1 - phi^2); kurtosis widened to
        # cover the autocorrelation-inflated variance of the estimator
        assert_std(values, expected=1.0 / np.sqrt(1.0 - phi**2), kurtosis=9.0)
        assert_mean(values, expected=0.0, std=1.0 / np.sqrt(1.0 - phi**2) * 2.4)
