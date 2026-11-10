"""Tests for CyclicGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import CyclicGenerator
from tests.helpers import (
    assert_distribution,
    assert_long_format,
    assert_mean,
    assert_std,
    series_values,
)

BASE = {"min_length": 30, "max_length": 60, "freq": "D", "seed": 42}


def dominant_period(values: np.ndarray) -> float:
    """Period of the largest non-DC FFT peak."""
    demeaned = values - values.mean()
    power = np.abs(np.fft.rfft(demeaned))
    freqs = np.fft.rfftfreq(len(values))
    peak = np.argmax(power[1:]) + 1
    return 1.0 / freqs[peak]


class TestCyclicApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = CyclicGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=30, max_length=60)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = CyclicGenerator(**BASE, engine=engine).generate(n_series=3)
        df2 = CyclicGenerator(**BASE, engine=engine).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_different_seeds_differ(self) -> None:
        v1 = CyclicGenerator(**{**BASE, "seed": 1}).generate_single_series(50)
        v2 = CyclicGenerator(**{**BASE, "seed": 2}).generate_single_series(50)
        assert not np.array_equal(v1, v2)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("cycle_period_mean", 0.0),
            ("cycle_period_std", -1.0),
            ("cycle_amplitude_std", -1.0),
            ("num_cycles", 0),
            ("noise_std", -0.5),
        ],
    )
    def test_invalid_parameters(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            CyclicGenerator(**BASE, **{field: value})


@pytest.mark.stats
class TestCyclicStats:
    """Statistical property tests (fixed seeds)."""

    def test_dominant_frequency_near_expected_period(self) -> None:
        """FFT peak of a single-cycle series sits near cycle_period_mean.

        The instantaneous period is modulated +-20% around the drawn period,
        so the peak lands within roughly that band.
        """
        for seed in range(5):
            gen = CyclicGenerator(
                min_length=600,
                max_length=600,
                freq="D",
                cycle_period_mean=50.0,
                cycle_period_std=2.0,
                num_cycles=1,
                noise_std=0.5,
                trend=0.0,
                seed=seed,
            )
            period = dominant_period(gen.generate_single_series(600))
            assert 50.0 * 0.75 < period < 50.0 * 1.3, f"seed {seed}: {period}"

    def test_level_and_trend(self) -> None:
        """Detrended series averages to base_level; cycles and noise are
        zero-mean."""
        gen = CyclicGenerator(
            min_length=1000,
            max_length=1000,
            freq="D",
            base_level=100.0,
            trend=0.5,
            seed=0,
        )
        values = gen.generate_single_series(1000)
        t = np.arange(1000)
        residual = values - (100.0 + 0.5 * t)
        assert abs(residual.mean()) < 5.0
        slope = np.polyfit(t, values, 1)[0]
        assert abs(slope - 0.5) < 0.1

    def test_zero_amplitude_reduces_to_gaussian_noise(self) -> None:
        """With zero cycle amplitude the series is base_level + N(0, noise^2)."""
        gen = CyclicGenerator(
            min_length=2000,
            max_length=2000,
            freq="D",
            base_level=10.0,
            trend=0.0,
            cycle_amplitude_mean=0.0,
            cycle_amplitude_std=0.0,
            noise_std=2.0,
            seed=1,
        )
        values = gen.generate_single_series(2000)
        assert_mean(values, expected=10.0, std=2.0)
        assert_std(values, expected=2.0)
        assert_distribution(values, stats.norm(10.0, 2.0))

    def test_dominant_frequency(self) -> None:
        gen = CyclicGenerator(
            min_length=600,
            max_length=600,
            freq="D",
            cycle_period_mean=50.0,
            cycle_period_std=2.0,
            num_cycles=1,
            noise_std=0.5,
            seed=3,
        )
        period = dominant_period(gen.generate_single_series(600))
        assert 50.0 * 0.75 < period < 50.0 * 1.3
