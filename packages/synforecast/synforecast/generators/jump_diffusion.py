"""Jump Diffusion process generator."""

import numpy as np
from pydantic import Field

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class JumpDiffusionGenerator(BaseGenerator):
    """Generate time series from a jump diffusion process (Merton model).

    Combines Geometric Brownian Motion with discontinuous jumps from a
    compound Poisson process, commonly used for asset prices with rare
    events:

        dS_t = mu * S_t * dt + sigma * S_t * dW_t + S_{t-} * dJ_t

    Each step multiplies the price by
    ``exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * z_t + sum_k Y_k)`` with
    ``N_t ~ Poisson(lambda_jump * dt)`` jumps of log-size
    ``Y_k = jump_mean + jump_std * eps_k``. Both ``z_t`` and ``eps_k`` are
    unit-variance draws from ``innovation_distribution`` (normal by default,
    giving Merton's log-normal jumps). The drift is not compensated for
    jumps, so the expected log-return per step is
    ``(mu - sigma^2/2) * dt + lambda_jump * dt * jump_mean``. ``dt`` is the
    model time per observation and is independent of ``freq``.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min', 'MS')
            or an integer for an integer time index.
        mu (float): Drift per unit of model time (default: 0.05).
        sigma (float): Diffusion volatility (default: 0.2).
        lambda_jump (float): Jump intensity, expected jumps per unit of
            model time (default: 0.1).
        jump_mean (float): Mean jump size in log-price (default: 0.0).
        jump_std (float): Std of jump size in log-price (default: 0.1).
        initial_value (float): Initial value S_0, must be > 0 (default: 100.0).
        dt (float): Model time step per observation (default: 1.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    mu: float = Field(default=0.05, description="Drift coefficient")
    sigma: float = Field(default=0.2, gt=0, description="Diffusion volatility")
    lambda_jump: float = Field(
        default=0.1, ge=0, description="Jump intensity (jumps per unit time)"
    )
    jump_mean: float = Field(default=0.0, description="Mean of jump size (log scale)")
    jump_std: float = Field(
        default=0.1, ge=0, description="Std of jump size (log scale)"
    )
    initial_value: float = Field(default=100.0, gt=0, description="Initial value")
    dt: float = Field(default=1.0, gt=0, description="Time step")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.mu,
                    self.sigma,
                    self.lambda_jump,
                    self.jump_mean,
                    self.jump_std,
                    self.initial_value,
                    self.dt,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single jump diffusion time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values.
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.jump_diffusion(
            length,
            self.mu,
            self.sigma,
            self.lambda_jump,
            self.jump_mean,
            self.jump_std,
            self.initial_value,
            self.dt,
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )
