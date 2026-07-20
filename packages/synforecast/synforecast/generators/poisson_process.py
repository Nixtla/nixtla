"""Poisson Process generator."""

import numpy as np
from pydantic import Field

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import stochastic as _rs_stoch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class PoissonProcessGenerator(BaseGenerator):
    """Generate time series based on a homogeneous Poisson process.

    Each observation is the event count in one time step:
    y_t ~ Poisson(lambda_rate), i.i.d., so mean and variance both equal
    lambda_rate. With cumulative=True the running total N(t) = sum y_s is
    returned instead (the counting process itself). lambda_rate is
    expressed per time step of `freq`.

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data (e.g. 'D', 'h', '5min') or int
        lambda_rate (float): Expected events per time step (default: 5.0)
        cumulative (bool): Return cumulative counts (default: False)
        seed (int | None): Random seed for reproducibility (default: None)
    """

    lambda_rate: float = Field(default=5.0, gt=0, description="Rate parameter")
    cumulative: bool = Field(default=False, description="Return cumulative counts")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array([self.lambda_rate, 1.0 if self.cumulative else 0.0]),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single Poisson Process time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values (counts per time period)
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            return _rs_stoch.poisson_process(
                length, self.lambda_rate, self.cumulative, seed
            )
        counts = self.rng.poisson(self.lambda_rate, length)
        if self.cumulative:
            return np.cumsum(counts).astype(float)
        return counts.astype(float)
