"""Levy (alpha-stable) process generator."""

from typing import Any

import numpy as np
from pydantic import Field

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import stochastic as _rs_stoch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


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

    def _sample_stable(self, n: int) -> np.ndarray:
        """Sample n standard alpha-stable variates (S1 parameterization).

        Chambers, Mallows, Stuck (1976). "A Method for Simulating Stable
        Random Variables". JASA, 71(354), 340-344; formulas as in
        Weron (1996).
        """
        alpha = self.alpha
        beta = self.beta_skew

        if abs(alpha - 2.0) < 1e-10:
            # Gaussian case: S(2, 0; 1) = N(0, 2)
            return self.rng.normal(0.0, np.sqrt(2.0), n)

        V = self.rng.uniform(-np.pi / 2, np.pi / 2, n)
        W = self.rng.exponential(1.0, n)

        if abs(alpha - 1.0) < 1e-10:
            b_term = (np.pi / 2 + beta * V) * np.tan(V)
            x = (1.0 / (np.pi / 2)) * (
                b_term
                - beta * np.log((np.pi / 2) * W * np.cos(V) / (np.pi / 2 + beta * V))
            )
        else:
            b_alpha = np.arctan(beta * np.tan(np.pi * alpha / 2.0)) / alpha
            s_alpha = (1.0 + beta**2 * np.tan(np.pi * alpha / 2.0) ** 2) ** (
                1.0 / (2.0 * alpha)
            )

            B = alpha * (V + b_alpha)
            numerator = np.sin(B)
            denom = np.cos(V) ** (1.0 / alpha)
            factor = (np.cos(V - B) / W) ** ((1.0 - alpha) / alpha)

            x = s_alpha * numerator / denom * factor

        return x

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
            np.ndarray: Array of time series values.
        """
        if _HAS_RUST:
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

        increments = self.scale * self._sample_stable(length) + self.location
        if self.cumulative:
            return self.initial_value + np.cumsum(increments)
        return increments

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
