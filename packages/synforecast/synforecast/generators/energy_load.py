"""Energy Load generator with realistic consumption patterns."""

from typing import Literal

import numpy as np
from pydantic import Field

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import domain as _rs_dom

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

_LOAD_TYPE_IDS = {"residential": 0, "commercial": 1, "industrial": 2}


class EnergyLoadGenerator(BaseGenerator):
    """Generate electricity demand with nested daily/weekly/yearly cycles.

    The load is a base level plus:

    - A daily profile depending on ``load_type``: residential has Gaussian
      morning/evening peaks, commercial a broad midday peak, industrial a
      near-constant profile with a night dip.
    - A weekly cycle: weekend reduction for residential/commercial, a
      sinusoidal pattern for industrial.
    - A yearly cosine cycle peaking around the series start (winter).
    - Temperature-driven load: both heating (cold) and cooling (hot) increase
      demand proportionally to ``|temperature - base_temperature|``.
    - Holiday reductions, random extreme-weather multipliers, and Gaussian
      noise. The result is clipped at zero.

    Hour of day, day of week and day of year are derived from the step
    position relative to the series start using the step size implied by
    ``freq`` (an integer ``freq`` is treated as hourly).

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data, e.g. 'h', '15min', 'D'.
        base_load (float): Base load in kW or MW (default: 100.0).
        load_type (str): 'residential', 'commercial' or 'industrial'
            (default: 'residential').
        daily_pattern (bool): Enable the daily cycle (default: True).
        daily_amplitude (float): Amplitude of the daily variation
            (default: 30.0).
        weekly_pattern (bool): Enable the weekly cycle (default: True).
        weekly_amplitude (float): Amplitude of the weekly variation
            (default: 15.0).
        yearly_pattern (bool): Enable the yearly cycle (default: True).
        yearly_amplitude (float): Amplitude of the yearly variation
            (default: 20.0).
        temperature_sensitive (bool): Enable temperature effects
            (default: True).
        temperature_sensitivity (float): Load change per degree of deviation
            from base_temperature (default: 2.0).
        base_temperature (float): Reference temperature in Celsius
            (default: 20.0).
        morning_peak_hour (int): Hour of the residential morning peak
            (default: 8).
        evening_peak_hour (int): Hour of the residential evening peak
            (default: 19).
        peak_amplitude (float): Additional load at the residential peaks
            (default: 40.0).
        holiday_effect (float): Fractional load reduction on holidays
            (default: 0.3).
        holiday_days (list[int]): Day-of-year indices (0-364) that are
            holidays (default: []).
        extreme_weather_prob (float): Per-step probability of extreme weather
            (default: 0.0).
        extreme_weather_impact (float): Load multiplier during extreme weather
            (default: 1.5).
        noise_std (float): Standard deviation of additive noise (default: 5.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    base_load: float = Field(default=100.0, gt=0, description="Base load in kW or MW")
    load_type: Literal["residential", "commercial", "industrial"] = Field(
        default="residential", description="Type of load"
    )

    daily_pattern: bool = Field(default=True, description="Enable daily cycle")
    daily_amplitude: float = Field(
        default=30.0, description="Amplitude of daily variation"
    )
    weekly_pattern: bool = Field(default=True, description="Enable weekly cycle")
    weekly_amplitude: float = Field(
        default=15.0, description="Amplitude of weekly variation"
    )
    yearly_pattern: bool = Field(default=True, description="Enable yearly cycle")
    yearly_amplitude: float = Field(
        default=20.0, description="Amplitude of yearly variation"
    )

    temperature_sensitive: bool = Field(
        default=True, description="Enable temperature effects"
    )
    temperature_sensitivity: float = Field(
        default=2.0, description="Load change per degree"
    )
    base_temperature: float = Field(
        default=20.0, description="Base temperature in Celsius"
    )

    morning_peak_hour: int = Field(
        default=8, ge=0, le=23, description="Hour of morning peak"
    )
    evening_peak_hour: int = Field(
        default=19, ge=0, le=23, description="Hour of evening peak"
    )
    peak_amplitude: float = Field(
        default=40.0, description="Additional load during peaks"
    )

    holiday_effect: float = Field(
        default=0.3, ge=0, le=1, description="Load reduction on holidays"
    )
    holiday_days: list[int] = Field(default=[], description="Day indices for holidays")
    extreme_weather_prob: float = Field(
        default=0.0, ge=0, le=1, description="Probability of extreme weather"
    )
    extreme_weather_impact: float = Field(
        default=1.5, description="Load multiplier during extreme weather"
    )

    noise_std: float = Field(
        default=5.0, description="Standard deviation of random noise"
    )

    def _get_hour_of_day(self, t: int) -> int:
        """Hour of day (0-23) at step ``t``, relative to the series start."""
        return int(t * self._step_hours(default=1.0)) % 24

    def _get_day_of_week(self, t: int) -> int:
        """Day of week (0-6) at step ``t``, relative to the series start."""
        return (int(t * self._step_hours(default=1.0)) // 24) % 7

    def _get_day_of_year(self, t: int) -> int:
        """Day of year (0-364) at step ``t``, relative to the series start."""
        return (int(t * self._step_hours(default=1.0)) // 24) % 365

    def _generate_temperature(self, length: int) -> np.ndarray:
        """Temperature with daily and yearly cycles plus Gaussian noise."""
        temperature = np.zeros(length)
        for t in range(length):
            hour = self._get_hour_of_day(t)
            daily_temp = -5 * np.cos(2 * np.pi * hour / 24)
            day_of_year = self._get_day_of_year(t)
            yearly_temp = -10 * np.cos(2 * np.pi * day_of_year / 365)
            noise = self.rng.normal(0, 2)
            temperature[t] = self.base_temperature + daily_temp + yearly_temp + noise
        return temperature

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        holidays_f64 = np.array(self.holiday_days, dtype=np.float64)
        return (
            np.array(
                [
                    self.base_load,
                    float(_LOAD_TYPE_IDS[self.load_type]),
                    1.0 if self.daily_pattern else 0.0,
                    self.daily_amplitude,
                    1.0 if self.weekly_pattern else 0.0,
                    self.weekly_amplitude,
                    1.0 if self.yearly_pattern else 0.0,
                    self.yearly_amplitude,
                    1.0 if self.temperature_sensitive else 0.0,
                    self.temperature_sensitivity,
                    self.base_temperature,
                    float(self.morning_peak_hour),
                    float(self.evening_peak_hour),
                    self.peak_amplitude,
                    self.holiday_effect,
                    self.extreme_weather_prob,
                    self.extreme_weather_impact,
                    self.noise_std,
                    self._step_hours(default=1.0),
                ]
            ),
            [holidays_f64],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single energy load series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of energy load values.
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            holidays = np.array(self.holiday_days, dtype=np.int32)
            return _rs_dom.energy_load(
                length,
                self.base_load,
                _LOAD_TYPE_IDS[self.load_type],
                self.daily_pattern,
                self.daily_amplitude,
                self.weekly_pattern,
                self.weekly_amplitude,
                self.yearly_pattern,
                self.yearly_amplitude,
                self.temperature_sensitive,
                self.temperature_sensitivity,
                self.base_temperature,
                self.morning_peak_hour,
                self.evening_peak_hour,
                self.peak_amplitude,
                self.holiday_effect,
                holidays,
                self.extreme_weather_prob,
                self.extreme_weather_impact,
                self.noise_std,
                self._step_hours(default=1.0),
                seed,
            )

        load = np.full(length, self.base_load)

        if self.temperature_sensitive:
            temperature = self._generate_temperature(length)

        for t in range(length):
            hour = self._get_hour_of_day(t)
            day_of_week = self._get_day_of_week(t)
            day_of_year = self._get_day_of_year(t)

            if self.daily_pattern:
                if self.load_type == "residential":
                    morning_peak = self.peak_amplitude * np.exp(
                        -((hour - self.morning_peak_hour) ** 2) / 8
                    )
                    evening_peak = self.peak_amplitude * np.exp(
                        -((hour - self.evening_peak_hour) ** 2) / 8
                    )
                    daily_load = morning_peak + evening_peak
                elif self.load_type == "commercial":
                    daily_load = self.daily_amplitude * np.exp(-((hour - 14) ** 2) / 50)
                else:  # industrial: slight dip at night
                    daily_load = -self.daily_amplitude * np.exp(-((hour - 3) ** 2) / 20)
                load[t] += daily_load

            if self.weekly_pattern:
                if self.load_type in ["residential", "commercial"]:
                    if day_of_week >= 5:  # weekend
                        load[t] -= self.weekly_amplitude
                else:  # industrial
                    load[t] += self.weekly_amplitude * np.sin(
                        2 * np.pi * day_of_week / 7
                    )

            if self.yearly_pattern:
                load[t] += self.yearly_amplitude * (
                    1 + np.cos(2 * np.pi * day_of_year / 365)
                )

            if self.temperature_sensitive:
                # Heating (cold) and cooling (hot) both increase load
                temp_deviation = temperature[t] - self.base_temperature
                load[t] += self.temperature_sensitivity * abs(temp_deviation)

            if day_of_year in self.holiday_days:
                load[t] *= 1 - self.holiday_effect

            if self.rng.random() < self.extreme_weather_prob:
                load[t] *= self.extreme_weather_impact

        load += self.rng.normal(0, self.noise_std, length)
        return np.maximum(load, 0)
