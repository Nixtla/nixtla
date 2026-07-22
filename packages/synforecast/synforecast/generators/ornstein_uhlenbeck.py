"""Ornstein-Uhlenbeck process generator."""

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class OrnsteinUhlenbeckGenerator(BaseGenerator):
    """Generate time series from an Ornstein-Uhlenbeck (mean-reverting) process.

    The OU process is commonly used to model interest rates, volatility, and
    other mean-reverting phenomena:

        dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Simulated with the Euler-Maruyama scheme
    ``X_t = X_{t-1} + theta * (mu - X_{t-1}) * dt + sigma * sqrt(dt) * z_t``,
    where ``z_t`` are unit-variance draws from ``innovation_distribution``.
    This is an AR(1) process with coefficient ``phi = 1 - theta * dt``,
    stationary mean ``mu``, stationary variance
    ``sigma^2 * dt / (1 - phi^2)`` (which approaches the continuous-time
    ``sigma^2 / (2 * theta)`` as dt -> 0), and lag-1 autocorrelation ``phi``.
    Stability requires ``theta * dt < 2``. ``dt`` is the model time per
    observation and is independent of ``freq``.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min', 'MS')
            or an integer for an integer time index.
        theta (float): Speed of mean reversion, must satisfy
            ``theta * dt < 2`` (default: 0.5).
        mu (float): Long-term mean (default: 0.0).
        sigma (float): Volatility (default: 1.0).
        initial_value (float): Initial value X_0 (default: 0.0).
        dt (float): Model time step per observation (default: 1.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    theta: float = Field(default=0.5, gt=0, description="Speed of mean reversion")
    mu: float = Field(default=0.0, description="Long-term mean")
    sigma: float = Field(default=1.0, gt=0, description="Volatility")
    initial_value: float = Field(default=0.0, description="Initial value")
    dt: float = Field(default=1.0, gt=0, description="Time step")

    @model_validator(mode="after")
    def validate_stability(self) -> "OrnsteinUhlenbeckGenerator":
        """The Euler scheme is an AR(1) with phi = 1 - theta*dt; it diverges
        (|phi| >= 1) instead of mean-reverting when theta*dt >= 2."""
        if self.theta * self.dt >= 2.0:
            raise ValueError(
                "Euler discretization requires theta * dt < 2 for stability; "
                f"got theta * dt = {self.theta * self.dt}. "
                "Reduce theta or use a smaller dt."
            )
        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.theta,
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
        """Generate values for a single Ornstein-Uhlenbeck time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values.
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.ornstein_uhlenbeck(
            length,
            self.theta,
            self.mu,
            self.sigma,
            self.initial_value,
            self.dt,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
