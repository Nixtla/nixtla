"""Levy (alpha-stable) process generator."""

from typing import Any

import numpy as np
from pydantic import Field

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class LevyProcessGenerator(BaseGenerator):
    """Generate time series with alpha-stable (Levy) increments.

    Each observation step adds an independent increment
    ``scale * X + location`` where ``X ~ S(alpha, beta_skew; 1)`` is a
    standard alpha-stable random variable in the S1 parameterization
    (matching ``scipy.stats.levy_stable``), sampled with the
    Chambers-Mallows-Stuck algorithm. For ``alpha < 2`` the increments have
    infinite variance, producing extreme jumps far beyond Gaussian or
    t-distributed innovations. There is no separate ``dt``: ``scale`` is the
    per-step scale (a step of duration ``dt`` in model time corresponds to
    ``scale ~ dt**(1/alpha)`` by self-similarity).

    Special cases:
        - ``alpha=2``: Gaussian with standard deviation ``scale * sqrt(2)``
        - ``alpha=1, beta_skew=0``: Cauchy
        - ``alpha=0.5, beta_skew=1``: Levy distribution

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min', 'MS')
            or an integer for an integer time index.
        alpha (float): Stability index in (0, 2] (default: 1.5).
        beta_skew (float): Skewness parameter in [-1, 1] (default: 0.0).
        scale (float): Scale of each increment (default: 1.0).
        location (float): Location shift of each increment (default: 0.0).
        cumulative (bool): Return the cumulative sum (Levy flight) instead
            of raw increments (default: True).
        initial_value (float): Starting value for cumulative mode
            (default: 0.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    alpha: float = Field(
        default=1.5, gt=0.0, le=2.0, description="Stability parameter (0, 2]"
    )
    beta_skew: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Skewness parameter [-1, 1]"
    )
    scale: float = Field(default=1.0, gt=0, description="Scale parameter")
    location: float = Field(default=0.0, description="Location parameter")
    cumulative: bool = Field(
        default=True, description="Return cumulative sum (Levy flight)"
    )
    initial_value: float = Field(
        default=0.0, description="Starting value for cumulative mode"
    )

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    self.alpha,
                    self.beta_skew,
                    self.scale,
                    self.location,
                    1.0 if self.cumulative else 0.0,
                    self.initial_value,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single Levy process time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of time series values.
        """
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.levy_process(
            length,
            self.alpha,
            self.beta_skew,
            self.scale,
            self.location,
            self.cumulative,
            self.initial_value,
            seed,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the Levy process configuration."""
        if abs(self.alpha - 2.0) < 1e-10:
            dist_type = "Gaussian"
        elif abs(self.alpha - 1.0) < 1e-10 and abs(self.beta_skew) < 1e-10:
            dist_type = "Cauchy"
        elif abs(self.alpha - 0.5) < 1e-10 and abs(self.beta_skew - 1.0) < 1e-10:
            dist_type = "Levy"
        else:
            dist_type = f"alpha-stable(alpha={self.alpha}, beta={self.beta_skew})"

        return {
            "alpha": self.alpha,
            "beta_skew": self.beta_skew,
            "scale": self.scale,
            "distribution": dist_type,
            "infinite_variance": self.alpha < 2.0,
            "infinite_mean": self.alpha <= 1.0,
        }
