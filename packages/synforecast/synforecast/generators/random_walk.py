"""Random Walk time series generator."""

import numpy as np
from pydantic import Field

from synforecast._lib import statistical as _rs_stat
from synforecast.base import BaseGenerator


class RandomWalkGenerator(BaseGenerator):
    """Generate random walk time series.

    y_t = y_{t-1} + drift + ε_t, where ε_t has standard deviation
    `volatility` and is drawn from `innovation_distribution`. The first
    output value already includes one step: y_1 = start_value + drift + ε_1.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency, a pandas offset alias (e.g. 'D', 'h',
            '5min', 'MS') or an integer time step.
        drift (float): Mean of the random steps (default: 0.0).
        volatility (float): Standard deviation of random steps (default: 1.0).
        start_value (float): Initial value for all series (default: 0.0).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp of every series
            (default: '2000-01-01').
    """

    drift: float = Field(default=0.0, description="Mean of the random steps")
    volatility: float = Field(
        default=1.0, ge=0, description="Standard deviation of random steps"
    )
    start_value: float = Field(default=0.0, description="Initial value for all series")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.drift,
                    self.volatility,
                    self.start_value,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single random walk time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stat.random_walk(
            length,
            self.drift,
            self.volatility,
            self.start_value,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
