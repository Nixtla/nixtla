"""Tests for DailyActiveUsersGenerator."""

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast.generators import DailyActiveUsersGenerator
from tests.helpers import assert_long_format, to_pandas


def make_gen(**kwargs):
    params = {
        "min_length": 100,
        "max_length": 150,
        "freq": "D",
        "seed": 42,
    }
    params.update(kwargs)
    return DailyActiveUsersGenerator(**params)


class TestDailyActiveUsersAPI:
    def test_long_format_with_event_column(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(
            df, n_series=3, min_length=100, max_length=150, extra_cols={"event"}
        )

    def test_event_column_is_binary(self, engine: str) -> None:
        df = make_gen(engine=engine, event_probability=0.2).generate(n_series=2)
        events = to_pandas(df)["event"].to_numpy()
        assert set(np.unique(events)) <= {0, 1}

    def test_custom_event_col_name(self, engine: str) -> None:
        df = make_gen(engine=engine, event_col="launch").generate(n_series=2)
        assert_long_format(df, n_series=2, extra_cols={"launch"})

    def test_seed_determinism(self) -> None:
        pdf1 = to_pandas(make_gen(seed=7).generate(n_series=3))
        pdf2 = to_pandas(make_gen(seed=7).generate(n_series=3))
        np.testing.assert_array_equal(pdf1["y"].to_numpy(), pdf2["y"].to_numpy())
        np.testing.assert_array_equal(
            pdf1["event"].to_numpy(), pdf2["event"].to_numpy()
        )

    def test_start_id(self) -> None:
        df = make_gen().generate(n_series=2, start_id=10)
        ids = sorted(int(i) for i in to_pandas(df)["unique_id"].unique())
        assert ids == [10, 11]

    def test_default_weekend_factor_by_app_type(self) -> None:
        assert make_gen(app_type="gaming").weekend_factor == 1.2
        assert make_gen(app_type="consumer").weekend_factor == 0.8
        assert make_gen(app_type="business").weekend_factor == 0.8
        assert make_gen(weekend_factor=1.5).weekend_factor == 1.5

    def test_values_non_negative(self) -> None:
        values = make_gen(noise_std=0.5).generate_single_series(300)
        assert (values >= 0).all()

    def test_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            make_gen(base_users=0.0)
        with pytest.raises(ValidationError):
            make_gen(event_probability=1.5)
        with pytest.raises(ValidationError):
            make_gen(growth_rate_std=-0.1)
        with pytest.raises(ValueError, match="event_impact_max"):
            make_gen(event_impact_min=1.5, event_impact_max=1.2)


@pytest.mark.stats
class TestDailyActiveUsersStats:
    def test_event_frequency_matches_probability(self) -> None:
        """Observed event fraction ~ Binomial(n, event_probability)."""
        p = 0.05
        gen = make_gen(min_length=1000, max_length=1000, event_probability=p, seed=11)
        events = to_pandas(gen.generate(n_series=4))["event"].to_numpy()
        n = events.size
        se = np.sqrt(p * (1 - p) / n)
        assert abs(events.mean() - p) < 5 * se

    def test_no_events_when_probability_zero(self) -> None:
        df = make_gen(event_probability=0.0).generate(n_series=3)
        assert (to_pandas(df)["event"].to_numpy() == 0).all()

    def test_weekend_factor_direction(self) -> None:
        """Weekend (day 5, 6 of each week) DAU scales by weekend_factor."""
        gen = make_gen(
            min_length=700,
            max_length=700,
            base_users=1000.0,
            growth_rate=0.0,
            event_probability=0.0,
            weekend_factor=0.5,
            noise_std=0.01,
            seed=3,
        )
        values = gen.generate_single_series(700)
        day_of_week = np.arange(700) % 7
        ratio = values[day_of_week >= 5].mean() / values[day_of_week < 5].mean()
        assert abs(ratio - 0.5) < 0.02

    def test_gaming_weekend_boost(self) -> None:
        gen = make_gen(
            min_length=700,
            max_length=700,
            app_type="gaming",
            growth_rate=0.0,
            event_probability=0.0,
            noise_std=0.01,
            seed=3,
        )
        values = gen.generate_single_series(700)
        day_of_week = np.arange(700) % 7
        assert values[day_of_week >= 5].mean() > values[day_of_week < 5].mean()

    def test_growth_rate_applies_per_day_hourly_freq(self) -> None:
        """With freq='h', growth compounds per day (24 steps), not per step."""
        gen = make_gen(
            min_length=72,
            max_length=72,
            freq="h",
            base_users=1000.0,
            growth_rate=0.1,
            weekly_pattern=False,
            event_probability=0.0,
            noise_std=0.0,
            seed=5,
        )
        values = gen.generate_single_series(72)
        np.testing.assert_allclose(values[:24], 1000.0)
        np.testing.assert_allclose(values[24:48], 1100.0)
        np.testing.assert_allclose(values[48:72], 1210.0)

    def test_growth_rate_applies_per_day_15min_freq(self) -> None:
        """With freq='15min', growth compounds per day (96 steps), not per
        step; the Rust kernel handles fractional step-hours directly."""
        gen = make_gen(
            min_length=288,
            max_length=288,
            freq="15min",
            base_users=1000.0,
            growth_rate=0.1,
            weekly_pattern=False,
            event_probability=0.0,
            noise_std=0.0,
            seed=5,
        )
        values = gen.generate_single_series(288)
        np.testing.assert_allclose(values[:96], 1000.0)
        np.testing.assert_allclose(values[96:192], 1100.0)
        np.testing.assert_allclose(values[192:288], 1210.0)

    def test_growth_rate_daily_freq(self) -> None:
        gen = make_gen(
            min_length=10,
            max_length=10,
            freq="D",
            base_users=1000.0,
            growth_rate=0.1,
            weekly_pattern=False,
            event_probability=0.0,
            noise_std=0.0,
            seed=5,
        )
        values = gen.generate_single_series(10)
        np.testing.assert_allclose(values, 1000.0 * 1.1 ** np.arange(10))

    def test_events_boost_dau(self) -> None:
        """Values right after an event exceed the event-free baseline."""
        common = {
            "min_length": 500,
            "max_length": 500,
            "base_users": 1000.0,
            "growth_rate": 0.0,
            "weekly_pattern": False,
            "noise_std": 0.01,
            "event_impact_min": 1.5,
            "event_impact_max": 2.0,
            "seed": 8,
        }
        gen = make_gen(event_probability=0.05, **common)
        df = to_pandas(gen.generate(n_series=2))
        event_steps = df["event"].to_numpy() == 1
        assert event_steps.any()
        assert df.loc[event_steps, "y"].mean() > df.loc[~event_steps, "y"].mean()

    def test_noise_scale(self) -> None:
        """Noise std is proportional to the DAU level."""
        gen = make_gen(
            min_length=2000,
            max_length=2000,
            base_users=1000.0,
            growth_rate=0.0,
            weekly_pattern=False,
            event_probability=0.0,
            noise_std=0.05,
            seed=9,
        )
        values = gen.generate_single_series(2000)
        # values = level * (1 + N(0, 0.05)) with level=1000
        rel = values / 1000.0 - 1.0
        assert 0.04 < rel.std() < 0.06

    def test_lengths_vary_between_series(self) -> None:
        df = to_pandas(make_gen(min_length=50, max_length=200).generate(n_series=8))
        lengths = df.groupby("unique_id", observed=True).size()
        assert lengths.nunique() > 1
