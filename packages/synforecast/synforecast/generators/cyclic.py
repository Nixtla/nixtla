"""Cyclic generator for time series with irregular business cycles."""

import numpy as np
from pydantic import Field

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class CyclicGenerator(BaseGenerator):
    """Generate time series with irregular cyclic patterns.

    Models business cycles and economic indicators: a linear trend plus
    ``num_cycles`` superposed sinusoids whose periods and amplitudes are
    drawn once per series (period ~ |N(period_mean, period_std)|,
    amplitude ~ N(amplitude_mean, amplitude_std)), plus additive noise drawn
    from the configured ``innovation_distribution``.
    Each sinusoid's instantaneous frequency is slowly modulated (+-20%
    around 2*pi/period, integrated as a cumulative phase), so cycle
    lengths vary within a series, unlike regular seasonal patterns.

    Args:
        base_level (float): Base level of the series (default: 100.0).
        trend (float): Linear trend coefficient per step (default: 0.0).
        cycle_period_mean (float): Mean cycle period in steps (default: 50.0).
        cycle_period_std (float): Std of the per-series period draw
            (default: 10.0).
        cycle_amplitude_mean (float): Mean cycle amplitude (default: 20.0).
        cycle_amplitude_std (float): Std of the per-series amplitude draw
            (default: 5.0).
        num_cycles (int): Number of superposed cycle components (default: 3).
        noise_std (float): Standard deviation of additive noise (default: 1.0).
    """

    base_level: float = Field(default=100.0, description="Base level of series")
    trend: float = Field(default=0.0, description="Linear trend coefficient")
    cycle_period_mean: float = Field(
        default=50.0, gt=0, description="Mean cycle period"
    )
    cycle_period_std: float = Field(
        default=10.0, ge=0, description="Std of cycle period variation"
    )
    cycle_amplitude_mean: float = Field(
        default=20.0, description="Mean cycle amplitude"
    )
    cycle_amplitude_std: float = Field(
        default=5.0, ge=0, description="Std of cycle amplitude variation"
    )
    num_cycles: int = Field(default=3, gt=0, description="Number of cycles to generate")
    noise_std: float = Field(
        default=1.0, ge=0, description="Standard deviation of noise"
    )

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.base_level,
                    self.trend,
                    self.cycle_period_mean,
                    self.cycle_period_std,
                    self.cycle_amplitude_mean,
                    self.cycle_amplitude_std,
                    float(self.num_cycles),
                    self.noise_std,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single time series with irregular cycles.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.cyclic(
            length,
            self.base_level,
            self.trend,
            self.cycle_period_mean,
            self.cycle_period_std,
            self.cycle_amplitude_mean,
            self.cycle_amplitude_std,
            self.num_cycles,
            self.noise_std,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
