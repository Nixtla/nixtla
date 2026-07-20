"""GARCH (Generalized Autoregressive Conditional Heteroskedasticity) generator."""

from typing import Any

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import stochastic as _rs_stoch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class GARCHGenerator(BaseGenerator):
    """Generate return series from a GARCH(p, q) model.

    The model is ``r_t = mu + eps_t`` with ``eps_t = sigma_t * z_t`` and
    conditional variance

        sigma2_t = omega + sum_i alpha_i * eps_{t-i}^2
                         + sum_j beta_j * sigma2_{t-j}

    Stationarity requires ``sum(alpha) + sum(beta) < 1``, giving an
    unconditional variance of ``omega / (1 - sum(alpha) - sum(beta))``.
    Squared returns are positively autocorrelated (volatility clustering)
    while the returns themselves are serially uncorrelated.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min', 'MS')
            or an integer time step.
        p (int): GARCH order (number of variance lags, default: 1).
        q (int): ARCH order (number of squared-innovation lags, default: 1).
        omega (float): Constant term in the variance equation (default: 0.1).
        alpha (list[float] | None): ARCH coefficients; auto-generated when None.
        beta (list[float] | None): GARCH coefficients; auto-generated when None.
        mu (float): Mean of returns (default: 0.0).
        initial_variance (float): Variance used to start the recursion
            (default: 1.0). A 100-step burn-in removes its influence.
        seed (int | None): Random seed for reproducibility (default: None).
    """

    p: int = Field(default=1, ge=1, description="GARCH order")
    q: int = Field(default=1, ge=1, description="ARCH order")
    omega: float = Field(default=0.1, description="Constant term in variance equation")
    alpha: list[float] | None = Field(default=None, description="ARCH parameters")
    beta: list[float] | None = Field(default=None, description="GARCH parameters")
    mu: float = Field(default=0.0, description="Mean of returns")
    initial_variance: float = Field(default=1.0, description="Initial variance")

    _alpha_array: Any = None
    _beta_array: Any = None

    @model_validator(mode="after")
    def compute_garch_params(self) -> "GARCHGenerator":
        """Compute GARCH parameters if not provided and validate stationarity."""
        if self.alpha is None:
            alpha_array = self.rng.uniform(0.05, 0.15, self.q)
        else:
            if len(self.alpha) != self.q:
                raise ValueError(f"alpha must have q={self.q} elements")
            alpha_array = np.array(self.alpha)
        object.__setattr__(self, "_alpha_array", alpha_array)

        if self.beta is None:
            remaining = 0.95 - np.sum(alpha_array)
            beta_array = self.rng.uniform(0.1, remaining / self.p, self.p)
        else:
            if len(self.beta) != self.p:
                raise ValueError(f"beta must have p={self.p} elements")
            beta_array = np.array(self.beta)
        object.__setattr__(self, "_beta_array", beta_array)

        if np.sum(self._alpha_array) + np.sum(self._beta_array) >= 1:
            raise ValueError(
                "Sum of alpha and beta must be less than 1 for stationarity"
            )

        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    float(self.p),
                    float(self.q),
                    self.omega,
                    self.mu,
                    self.initial_variance,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [
                np.asarray(self._alpha_array, dtype=np.float64),
                np.asarray(self._beta_array, dtype=np.float64),
            ],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single GARCH time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            return _rs_stoch.garch(
                length,
                self.p,
                self.q,
                self.omega,
                self._alpha_array,
                self._beta_array,
                self.mu,
                self.initial_variance,
                seed,
                self._rs_innov_dist,
                self._rs_innov_param,
            )

        burn_in = 100
        total_length = length + burn_in

        returns = np.zeros(total_length)
        # eps_t = sigma_t * z_t: the ARCH term must use lagged squared
        # innovations, not lagged squared returns (they differ when mu != 0).
        eps = np.zeros(total_length)
        variances = np.ones(total_length) * self.initial_variance
        errors = self._sample_innovations(total_length)

        for t in range(max(self.p, self.q), total_length):
            variance = self.omega
            for i in range(self.q):
                variance += self._alpha_array[i] * eps[t - i - 1] ** 2
            for j in range(self.p):
                variance += self._beta_array[j] * variances[t - j - 1]

            variances[t] = variance
            eps[t] = np.sqrt(variance) * errors[t]
            returns[t] = self.mu + eps[t]

        return returns[burn_in:]
