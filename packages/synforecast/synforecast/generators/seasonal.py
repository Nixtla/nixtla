"""Seasonal time series generator."""

import numpy as np
from pydantic import Field

from synforecast._lib import statistical as _rs_stat
from synforecast.base import BaseGenerator


class SeasonalGenerator(BaseGenerator):
    """Generate time series with seasonal patterns.

    y_t = base_level + amplitude · sin(2π t / period) + trend · t + ε_t,
    where ε_t has standard deviation `noise_level`.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency, a pandas offset alias (e.g. 'D', 'h',
            '5min', 'MS') or an integer time step.
        seasonality_period (int): Period of seasonality in time steps
            (default: 24).
        seasonality_amplitude (float): Amplitude of seasonal component
            (default: 10.0).
        trend (float): Linear trend coefficient per time step (default: 0.0).
        noise_level (float): Standard deviation of noise (default: 1.0).
        base_level (float): Base level of the series (default: 50.0).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp of every series
            (default: '2000-01-01').
    """

    seasonality_period: int = Field(
        default=24, ge=1, description="Period of seasonality"
    )
    seasonality_amplitude: float = Field(
        default=10.0, description="Amplitude of seasonal component"
    )
    trend: float = Field(default=0.0, description="Linear trend coefficient")
    noise_level: float = Field(
        default=1.0, ge=0, description="Standard deviation of noise"
    )
    base_level: float = Field(default=50.0, description="Base level of the series")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    float(self.seasonality_period),
                    self.seasonality_amplitude,
                    self.trend,
                    self.noise_level,
                    self.base_level,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single seasonal time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stat.seasonal(
            length,
            self.seasonality_period,
            self.seasonality_amplitude,
            self.trend,
            self.noise_level,
            self.base_level,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
