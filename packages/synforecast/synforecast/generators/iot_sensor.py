"""IoT Sensor data generator with drift, noise, and failure patterns."""

from typing import Literal

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

try:
    from synforecast._lib import domain as _rs_dom

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Typical readings per sensor type, used when base_value is not given
_DEFAULT_BASE_VALUES = {
    "temperature": 20.0,  # Celsius
    "humidity": 50.0,  # percent
    "pressure": 1013.25,  # hPa (standard atmosphere)
    "light": 500.0,  # lux
    "motion": 0.0,
    "generic": 0.0,
}


class IoTSensorGenerator(BaseGenerator):
    """Generate IoT sensor readings with realistic degradation patterns.

    The signal is ``base_value + trend * t`` plus optional sinusoidal
    seasonality, a calibration offset, a cumulative drift random walk
    (``drift_rate`` per step with ``drift_noise`` variation), and Gaussian
    measurement noise. After ``battery_life`` steps, noise grows and the
    signal is attenuated at ``battery_degradation_rate``. Failures can be
    injected as NaN gaps ('intermittent'), a permanent NaN tail ('complete'),
    or frozen readings ('stuck').

    With ``n_sensors > 1``, each generated "series" is a network of sensors
    whose measurement noise is spatially correlated
    (``corr[i, j] = spatial_correlation ** |i - j|``); each sensor becomes a
    separate output series.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data, e.g. 's', 'min', 'h'.
        n_sensors (int): Sensors per network (default: 1).
        sensor_type (str): 'temperature', 'humidity', 'pressure', 'light',
            'motion' or 'generic' (default: 'temperature').
        base_value (float | None): Base sensor reading (default: typical value
            for sensor_type).
        trend (float): Linear trend per time step (default: 0.0).
        seasonal_period (int): Seasonal cycle length in steps, 0 disables
            (default: 0).
        seasonal_amplitude (float): Amplitude of seasonal variation
            (default: 0.0).
        measurement_noise (float): Std of measurement noise (default: 0.1).
        drift_rate (float): Deterministic sensor drift per step (default: 0.0).
        drift_noise (float): Std of the random drift component (default: 0.01).
        calibration_error (float): Constant calibration offset (default: 0.0).
        battery_life (int | None): Steps until battery degradation starts,
            None disables (default: None).
        battery_degradation_rate (float): Rate of quality loss per step after
            battery_life (default: 0.001).
        failure_probability (float): Probability of sensor failure
            (default: 0.0). For 'complete': probability the series fails at
            all; for 'intermittent'/'stuck': per-step probability of starting
            a failure episode.
        failure_type (str): 'intermittent', 'complete' or 'stuck'
            (default: 'intermittent').
        failure_duration (int): Length of intermittent/stuck episodes
            (default: 10).
        stuck_value (float | None): Reading during 'stuck' failures
            (default: the reading at episode start).
        spatial_correlation (float): Noise correlation between adjacent
            sensors in a network (default: 0.5).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    n_sensors: int = Field(default=1, ge=1, description="Number of sensors to generate")
    sensor_type: Literal[
        "temperature", "humidity", "pressure", "light", "motion", "generic"
    ] = Field(default="temperature", description="Type of sensor")

    base_value: float | None = Field(default=None, description="Base sensor reading")
    trend: float = Field(default=0.0, description="Linear drift rate per time step")
    seasonal_period: int = Field(default=0, ge=0, description="Seasonal cycle length")
    seasonal_amplitude: float = Field(
        default=0.0, description="Amplitude of seasonal variation"
    )

    measurement_noise: float = Field(
        default=0.1, description="Std of measurement noise"
    )
    drift_rate: float = Field(default=0.0, description="Rate of sensor drift")
    drift_noise: float = Field(default=0.01, description="Random variation in drift")

    calibration_error: float = Field(
        default=0.0, description="Initial calibration offset"
    )
    battery_life: int | None = Field(
        default=None, description="Steps until battery degradation starts"
    )
    battery_degradation_rate: float = Field(
        default=0.001, description="Rate of quality loss"
    )

    failure_probability: float = Field(
        default=0.0, ge=0, le=1, description="Probability of sensor failure"
    )
    failure_type: Literal["intermittent", "complete", "stuck"] = Field(
        default="intermittent", description="Type of failure"
    )
    failure_duration: int = Field(
        default=10, ge=1, description="Duration of intermittent failures"
    )
    stuck_value: float | None = Field(default=None, description="Value when stuck")

    spatial_correlation: float = Field(
        default=0.5, ge=0, le=1, description="Correlation between nearby sensors"
    )

    @model_validator(mode="after")
    def set_default_base_value(self) -> "IoTSensorGenerator":
        """Set default base value based on sensor type if not provided."""
        if self.base_value is None:
            object.__setattr__(
                self, "base_value", _DEFAULT_BASE_VALUES[self.sensor_type]
            )
        return self

    def _generate_base_signal(self, length: int) -> np.ndarray:
        """Base value plus linear trend and optional sinusoidal seasonality."""
        t = np.arange(length)
        signal = np.full(length, self.base_value) + self.trend * t
        if self.seasonal_period > 0 and self.seasonal_amplitude > 0:
            signal += self.seasonal_amplitude * np.sin(
                2 * np.pi * t / self.seasonal_period
            )
        return signal

    def _add_sensor_drift(self, signal: np.ndarray) -> np.ndarray:
        """Add cumulative drift: random walk with per-step mean drift_rate."""
        steps = self.rng.normal(self.drift_rate, self.drift_noise, len(signal))
        steps[0] = 0.0
        return signal + np.cumsum(steps)

    def _add_battery_degradation(self, signal: np.ndarray) -> np.ndarray:
        """Increase noise and attenuate the signal after battery_life steps."""
        if self.battery_life is None:
            return signal

        degraded = signal.copy()
        for i in range(self.battery_life, len(signal)):
            steps_since = i - self.battery_life
            degradation_factor = 1 + steps_since * self.battery_degradation_rate
            degraded[i] += self.rng.normal(
                0, self.measurement_noise * degradation_factor
            )
            # Attenuate towards zero; never flip sign for very long series
            degraded[i] *= max(
                1 - steps_since * self.battery_degradation_rate * 0.1, 0.0
            )
        return degraded

    def _add_failures(self, signal: np.ndarray) -> np.ndarray:
        """Inject the configured failure mode into the signal."""
        if self.failure_probability == 0:
            return signal

        length = len(signal)
        failed_signal = signal.copy()

        if self.failure_type == "complete":
            if self.rng.random() < self.failure_probability:
                failure_start = self.rng.integers(0, length)
                failed_signal[failure_start:] = np.nan

        elif self.failure_type == "intermittent":
            i = 0
            while i < length:
                if self.rng.random() < self.failure_probability:
                    duration = min(self.failure_duration, length - i)
                    failed_signal[i : i + duration] = np.nan
                    i += duration
                else:
                    i += 1

        elif self.failure_type == "stuck":
            i = 0
            while i < length:
                if self.rng.random() < self.failure_probability:
                    stuck_val = (
                        self.stuck_value
                        if self.stuck_value is not None
                        else failed_signal[i]
                    )
                    duration = min(self.failure_duration, length - i)
                    failed_signal[i : i + duration] = stuck_val
                    i += duration
                else:
                    i += 1

        return failed_signal

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single sensor.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of sensor readings (NaN during failures).
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            failure_t = {"intermittent": 2, "complete": 1, "stuck": 3}[
                self.failure_type
            ]
            if self.failure_probability == 0.0:
                failure_t = 0
            return _rs_dom.iot_sensor(
                length,
                self.base_value,
                self.trend,
                self.seasonal_amplitude,
                float(self.seasonal_period),
                self.measurement_noise,
                self.drift_rate,
                self.battery_life is not None,
                float(self.battery_life) if self.battery_life is not None else 0.0,
                self.calibration_error,
                failure_t,
                self.failure_probability,
                self.failure_duration,
                self.spatial_correlation,
                seed,
            )

        signal = self._generate_base_signal(length)
        signal += self.calibration_error
        signal = self._add_sensor_drift(signal)
        signal += self.rng.normal(0, self.measurement_noise, length)
        signal = self._add_battery_degradation(signal)
        return self._add_failures(signal)

    def _generate_multivariate(self, length: int, n_sensors: int) -> np.ndarray:
        """Generate one sensor network with spatially correlated noise.

        Returns:
            np.ndarray: Array of shape (length, n_sensors).
        """
        # AR(1)-style correlation: decays exponentially with sensor distance
        corr_matrix = self.spatial_correlation ** np.abs(
            np.arange(n_sensors)[:, None] - np.arange(n_sensors)[None, :]
        )

        base_signals = np.zeros((length, n_sensors))
        for i in range(n_sensors):
            base_signals[:, i] = self._generate_base_signal(length)

        base_signals += self.rng.multivariate_normal(
            np.zeros(n_sensors), corr_matrix * (self.measurement_noise**2), size=length
        )

        for i in range(n_sensors):
            # Each sensor has slightly different calibration
            base_signals[:, i] += self.calibration_error + self.rng.normal(0, 0.1)
            base_signals[:, i] = self._add_sensor_drift(base_signals[:, i])
            base_signals[:, i] = self._add_battery_degradation(base_signals[:, i])
            base_signals[:, i] = self._add_failures(base_signals[:, i])

        return base_signals

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,  # noqa: ARG002
    ) -> IntoDataFrameT:
        """Generate IoT sensor data.

        With ``n_sensors == 1``, generates ``n_series`` independent sensors.
        With ``n_sensors > 1``, generates ``n_series`` sensor networks, each
        contributing ``n_sensors`` correlated series.

        Args:
            n_series (int): Number of series/networks to generate.
            start_id (int): Starting ID for the series numbering (default: 0).
            n_jobs (int): Ignored; generation is sequential.

        Returns:
            DataFrame in long format with columns [id_col, time_col,
            target_col].
        """
        all_metadata: list[SeriesMetadata] = []
        series_counter = start_id

        for _ in range(n_series):
            length = int(self.rng.integers(self.min_length, self.max_length + 1))
            timestamps = self._timestamps(length)

            if self.n_sensors == 1:
                samples = self.generate_single_series(length)[:, None]
            else:
                samples = self._generate_multivariate(length, self.n_sensors)

            for sensor_idx in range(samples.shape[1]):
                values = np.ascontiguousarray(samples[:, sensor_idx])
                values, cp_indices, anom_indices, miss_indices = (
                    self._apply_pattern_injection(values)
                )

                all_metadata.append(
                    SeriesMetadata(
                        values=values,
                        timestamps=timestamps,
                        series_id=series_counter,
                        length=length,
                        anomaly_indices=anom_indices,
                        changepoint_indices=cp_indices,
                        missing_indices=miss_indices,
                    )
                )

                series_counter += 1

        return self._build_dataframe(all_metadata)
