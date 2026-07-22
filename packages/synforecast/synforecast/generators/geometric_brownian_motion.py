"""Geometric Brownian Motion generator."""

import numpy as np
from pydantic import Field

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class GeometricBrownianMotionGenerator(BaseGenerator):
    """Generate time series from Geometric Brownian Motion.

    GBM models strictly positive processes such as asset prices:

        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Simulated via the exact solution of the SDE,
    ``S_t = S_{t-1} * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * z_t)``,
    where ``z_t`` are unit-variance draws from ``innovation_distribution``
    (exact for normal innovations). ``dt`` is the model time per observation
    and is independent of ``freq``: with annualized ``mu``/``sigma``, daily
    observations correspond to ``dt=1/252``. Note that the default
    ``dt=1.0`` treats ``mu`` and ``sigma`` as per-step rates; long series
    with a large ``mu * dt`` grow explosively.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min', 'MS')
            or an integer for an integer time index.
        mu (float): Drift per unit of model time (default: 0.05).
        sigma (float): Volatility per sqrt unit of model time (default: 0.2).
        initial_value (float): Initial value S_0, must be > 0 (default: 100.0).
        dt (float): Model time step per observation (default: 1.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    mu: float = Field(default=0.05, description="Drift coefficient")
    sigma: float = Field(default=0.2, gt=0, description="Volatility")
    initial_value: float = Field(default=100.0, gt=0, description="Initial value")
    dt: float = Field(default=1.0, gt=0, description="Time step")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.mu,
                    self.sigma,
                    self.initial_value,
                    self.dt,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single Geometric Brownian Motion time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values.
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.geometric_brownian_motion(
            length,
            self.mu,
            self.sigma,
            self.initial_value,
            self.dt,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
