"""Tests for IoTSensorGenerator."""

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast.generators import IoTSensorGenerator
from tests.helpers import (
    assert_long_format,
    assert_mean,
    assert_std,
    sample_acf,
    series_values,
)


def make_gen(**kwargs):
    params = {
        "min_length": 80,
        "max_length": 120,
        "freq": "min",
        "seed": 42,
    }
    params.update(kwargs)
    return IoTSensorGenerator(**params)


class TestIoTSensorAPI:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=80, max_length=120)

    def test_multivariate_networks(self, engine: str) -> None:
        """Each network of n_sensors becomes n_sensors output series."""
        df = make_gen(engine=engine, n_sensors=3).generate(n_series=2)
        assert_long_format(df, n_series=6, min_length=80, max_length=120)

    def test_seed_determinism(self) -> None:
        v1 = series_values(make_gen(seed=7).generate(n_series=3))
        v2 = series_values(make_gen(seed=7).generate(n_series=3))
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_default_base_value_per_sensor_type(self) -> None:
        assert make_gen(sensor_type="temperature").base_value == 20.0
        assert make_gen(sensor_type="pressure").base_value == 1013.25
        assert make_gen(sensor_type="humidity").base_value == 50.0
        assert make_gen(sensor_type="light", base_value=3.0).base_value == 3.0

    def test_failures_produce_nan(self, engine: str) -> None:
        df = make_gen(
            engine=engine,
            failure_type="intermittent",
            failure_probability=0.1,
            failure_duration=5,
        ).generate(n_series=2)
        assert_long_format(df, n_series=2, allow_nan=True)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            make_gen(n_sensors=0)
        with pytest.raises(ValidationError):
            make_gen(failure_probability=1.5)
        with pytest.raises(ValidationError):
            make_gen(failure_duration=0)
        with pytest.raises(ValidationError):
            make_gen(sensor_type="sonar")
        with pytest.raises(ValidationError):
            make_gen(spatial_correlation=2.0)


@pytest.mark.stats
class TestIoTSensorStats:
    def test_measurement_noise_scale(self) -> None:
        """With no trend/drift/failures, readings = base + N(0, noise)."""
        gen = make_gen(
            min_length=2000,
            max_length=2000,
            base_value=20.0,
            measurement_noise=0.5,
            drift_rate=0.0,
            drift_noise=0.0,
            seed=1,
        )
        values = gen.generate_single_series(2000)
        assert_mean(values, 20.0, 0.5)
        assert_std(values, 0.5)

    def test_linear_trend(self) -> None:
        gen = make_gen(
            min_length=500,
            max_length=500,
            base_value=0.0,
            trend=0.1,
            measurement_noise=0.01,
            drift_rate=0.0,
            drift_noise=0.0,
            seed=2,
        )
        values = gen.generate_single_series(500)
        slope = np.polyfit(np.arange(500), values, 1)[0]
        assert abs(slope - 0.1) < 0.005

    def test_seasonality_acf(self) -> None:
        """Seasonal cycle detectable at the configured period."""
        gen = make_gen(
            min_length=960,
            max_length=960,
            seasonal_period=48,
            seasonal_amplitude=5.0,
            measurement_noise=0.5,
            drift_rate=0.0,
            drift_noise=0.0,
            seed=3,
        )
        values = gen.generate_single_series(960)
        assert sample_acf(values, 48) > 0.7

    def test_intermittent_failure_rate(self) -> None:
        """NaN fraction tracks failure_probability x failure_duration."""
        gen = make_gen(
            min_length=1000,
            max_length=1000,
            failure_type="intermittent",
            failure_probability=0.05,
            failure_duration=5,
            seed=4,
        )
        values = np.concatenate([gen.generate_single_series(1000) for _ in range(3)])
        nan_frac = np.isnan(values).mean()
        # Expected fraction ~ p*d / (1 + p*(d-1)) ~ 0.21
        assert 0.10 < nan_frac < 0.35

    def test_no_failures_when_probability_zero(self) -> None:
        values = make_gen(failure_probability=0.0).generate_single_series(500)
        assert np.isfinite(values).all()

    def test_complete_failure_is_permanent(self) -> None:
        """After a complete failure, all subsequent readings are NaN."""
        gen = make_gen(
            min_length=300,
            max_length=300,
            failure_type="complete",
            failure_probability=1.0,
            seed=5,
        )
        values = gen.generate_single_series(300)
        nan_mask = np.isnan(values)
        assert nan_mask.any()
        first_nan = int(np.argmax(nan_mask))
        assert nan_mask[first_nan:].all()

    def test_stuck_failure_freezes_readings(self) -> None:
        """Stuck episodes produce runs of identical consecutive values."""
        gen = make_gen(
            min_length=1000,
            max_length=1000,
            failure_type="stuck",
            failure_probability=0.05,
            failure_duration=5,
            measurement_noise=0.5,
            seed=6,
        )
        values = gen.generate_single_series(1000)
        assert np.isfinite(values).all()
        # Exact repeats only occur in stuck runs (noise is continuous)
        assert (np.diff(values) == 0).sum() > 10

    def test_battery_degradation_increases_noise(self) -> None:
        gen = make_gen(
            min_length=600,
            max_length=600,
            base_value=100.0,
            measurement_noise=0.5,
            battery_life=200,
            battery_degradation_rate=0.01,
            drift_rate=0.0,
            drift_noise=0.0,
            seed=7,
        )
        values = gen.generate_single_series(600)
        head = values[:150]
        tail = values[450:]
        assert tail.std() > 2 * head.std()

    def test_drift_random_walk(self) -> None:
        """drift_rate accumulates: the series ends drift_rate*length higher."""
        gen = make_gen(
            min_length=1000,
            max_length=1000,
            base_value=0.0,
            measurement_noise=0.01,
            drift_rate=0.05,
            drift_noise=0.001,
            seed=8,
        )
        values = gen.generate_single_series(1000)
        assert 30 < values[-1] < 70  # ~ 0.05 * 1000 = 50

    def test_spatial_correlation_between_sensors(self) -> None:
        """Adjacent sensors in a network are positively correlated."""
        gen = make_gen(
            min_length=800,
            max_length=800,
            n_sensors=2,
            spatial_correlation=0.9,
            measurement_noise=1.0,
            drift_rate=0.0,
            drift_noise=0.0,
            seed=9,
        )
        samples = gen._generate_multivariate(800, 2)
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert corr > 0.5
