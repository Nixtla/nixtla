"""Tests for EnergyLoadGenerator."""

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast.generators import EnergyLoadGenerator
from tests.helpers import (
    assert_long_format,
    assert_mean,
    assert_std,
    sample_acf,
    series_values,
)


def make_gen(**kwargs):
    params = {
        "min_length": 96,
        "max_length": 120,
        "freq": "h",
        "seed": 42,
    }
    params.update(kwargs)
    return EnergyLoadGenerator(**params)


def flat_values(df) -> np.ndarray:
    return np.concatenate(list(series_values(df).values()))


class TestEnergyLoadAPI:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=96, max_length=120)

    def test_custom_column_names(self, engine: str) -> None:
        df = make_gen(
            engine=engine, id_col="series", time_col="time", target_col="load"
        ).generate(n_series=2)
        assert_long_format(
            df, n_series=2, id_col="series", time_col="time", target_col="load"
        )

    def test_seed_determinism(self) -> None:
        df1 = make_gen(seed=7).generate(n_series=3)
        df2 = make_gen(seed=7).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_load_types_produce_nonnegative_output(self) -> None:
        for load_type in ["residential", "commercial", "industrial"]:
            values = make_gen(load_type=load_type).generate_single_series(200)
            assert values.shape == (200,)
            assert np.isfinite(values).all()
            assert (values >= 0).all()

    def test_batch_params_available_for_fractional_freqs(self) -> None:
        """Sub-hourly and non-fixed freqs no longer opt out of batching."""
        for freq in ["h", "15min", "30min", "D", "MS", 1]:
            assert make_gen(freq=freq)._get_batch_params() is not None

    def test_integer_freq(self) -> None:
        df = make_gen(freq=1).generate(n_series=2)
        assert_long_format(df, n_series=2)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            make_gen(base_load=0.0)
        with pytest.raises(ValidationError):
            make_gen(morning_peak_hour=24)
        with pytest.raises(ValidationError):
            make_gen(holiday_effect=1.5)
        with pytest.raises(ValidationError):
            make_gen(load_type="nuclear")
        with pytest.raises(ValueError, match="max_length must be >= min_length"):
            EnergyLoadGenerator(min_length=100, max_length=50, freq="h")


@pytest.mark.stats
class TestEnergyLoadStats:
    def test_noise_std_and_base_load(self) -> None:
        """With all patterns off, load = base_load + N(0, noise_std)."""
        gen = make_gen(
            min_length=800,
            max_length=800,
            base_load=100.0,
            daily_pattern=False,
            weekly_pattern=False,
            yearly_pattern=False,
            temperature_sensitive=False,
            noise_std=5.0,
            seed=123,
        )
        values = flat_values(gen.generate(n_series=5))
        assert_mean(values, 100.0, 5.0)
        assert_std(values, 5.0)

    def test_daily_cycle_acf_hourly(self) -> None:
        """Hourly residential load has a strong daily cycle at lag 24."""
        gen = make_gen(
            min_length=480,
            max_length=480,
            weekly_pattern=False,
            yearly_pattern=False,
            temperature_sensitive=False,
            noise_std=2.0,
            seed=0,
        )
        values = gen.generate_single_series(480)
        assert sample_acf(values, 24) > 0.5

    def test_daily_cycle_acf_sub_hourly(self) -> None:
        """At 15min frequency the daily cycle sits at lag 96 (Rust path:
        the kernel takes fractional step-hours, so sub-hourly freqs no
        longer fall back to Python)."""
        gen = make_gen(
            min_length=768,
            max_length=768,
            freq="15min",
            weekly_pattern=False,
            yearly_pattern=False,
            temperature_sensitive=False,
            noise_std=2.0,
            seed=0,
        )
        values = gen.generate_single_series(768)
        assert sample_acf(values, 96) > 0.5

    def test_daily_cycle_hourly_vs_15min_rust_batch_path(self) -> None:
        """generate() (Rust batch path) shows the daily cycle at the right
        lag for both hourly (24) and 15min (96) frequencies."""
        common = {
            "weekly_pattern": False,
            "yearly_pattern": False,
            "temperature_sensitive": False,
            "noise_std": 2.0,
            "seed": 0,
        }
        hourly = make_gen(min_length=480, max_length=480, freq="h", **common)
        hourly_values = flat_values(hourly.generate(n_series=1))
        assert sample_acf(hourly_values, 24) > 0.5

        sub = make_gen(min_length=1920, max_length=1920, freq="15min", **common)
        sub_values = flat_values(sub.generate(n_series=1))
        assert sample_acf(sub_values, 96) > 0.5
        # The cycle sits at one day (96 steps), not at 24 steps (= 6 hours)
        assert sample_acf(sub_values, 96) > sample_acf(sub_values, 24) + 0.2

    def test_weekend_effect_sign_hourly(self) -> None:
        """Residential weekend load drops by weekly_amplitude."""
        gen = make_gen(
            min_length=24 * 28,
            max_length=24 * 28,
            daily_pattern=False,
            weekly_pattern=True,
            weekly_amplitude=30.0,
            yearly_pattern=False,
            temperature_sensitive=False,
            noise_std=1.0,
            seed=1,
        )
        values = gen.generate_single_series(24 * 28)
        day_of_week = (np.arange(len(values)) // 24) % 7
        weekend = values[day_of_week >= 5]
        weekday = values[day_of_week < 5]
        diff = weekday.mean() - weekend.mean()
        assert 28 < diff < 32

    def test_weekend_effect_daily_freq(self) -> None:
        """With freq='D', the weekly cycle repeats every 7 steps."""
        gen = make_gen(
            min_length=350,
            max_length=350,
            freq="D",
            daily_pattern=False,
            weekly_pattern=True,
            weekly_amplitude=30.0,
            yearly_pattern=False,
            temperature_sensitive=False,
            noise_std=1.0,
            seed=1,
        )
        values = gen.generate_single_series(350)
        day_of_week = np.arange(len(values)) % 7
        diff = values[day_of_week < 5].mean() - values[day_of_week >= 5].mean()
        assert 28 < diff < 32

    def test_holiday_effect(self) -> None:
        """Holidays reduce the load by holiday_effect."""
        holidays = [10, 50, 100, 150, 200]
        gen = make_gen(
            min_length=365,
            max_length=365,
            freq="D",
            base_load=100.0,
            daily_pattern=False,
            weekly_pattern=False,
            yearly_pattern=False,
            temperature_sensitive=False,
            holiday_days=holidays,
            holiday_effect=0.5,
            noise_std=0.5,
            seed=2,
        )
        values = gen.generate_single_series(365)
        mask = np.zeros(365, dtype=bool)
        mask[holidays] = True
        assert abs(values[mask].mean() - 50.0) < 3.0
        assert abs(values[~mask].mean() - 100.0) < 3.0

    def test_temperature_effect_increases_load(self) -> None:
        """Temperature sensitivity adds load proportional to |deviation|."""
        common = {
            "min_length": 600,
            "max_length": 600,
            "daily_pattern": False,
            "weekly_pattern": False,
            "yearly_pattern": False,
            "noise_std": 1.0,
            "seed": 3,
        }
        with_temp = flat_values(
            make_gen(temperature_sensitive=True, **common).generate(n_series=3)
        )
        without_temp = flat_values(
            make_gen(temperature_sensitive=False, **common).generate(n_series=3)
        )
        assert with_temp.mean() > without_temp.mean() + 5

    def test_extreme_weather_raises_mean(self) -> None:
        common = {
            "min_length": 600,
            "max_length": 600,
            "daily_pattern": False,
            "weekly_pattern": False,
            "yearly_pattern": False,
            "temperature_sensitive": False,
            "noise_std": 1.0,
            "extreme_weather_impact": 2.0,
            "seed": 4,
        }
        with_ew = flat_values(
            make_gen(extreme_weather_prob=0.2, **common).generate(n_series=3)
        )
        without_ew = flat_values(
            make_gen(extreme_weather_prob=0.0, **common).generate(n_series=3)
        )
        # ~20% of steps doubled -> mean up by ~20%
        assert with_ew.mean() > without_ew.mean() * 1.1
