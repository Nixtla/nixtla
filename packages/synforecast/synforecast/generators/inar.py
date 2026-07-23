"""Integer-valued AutoRegressive (INAR) generator."""

from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import statistical as _rs_stat
from synforecast.base import BaseGenerator


class INARGenerator(BaseGenerator):
    """Generate integer-valued time series with autoregressive structure.

    INAR(p) models use binomial thinning to maintain integer values while
    preserving autoregressive dynamics:

        X_t = alpha_1 o X_{t-1} + ... + alpha_p o X_{t-p} + epsilon_t

    where 'o' is binomial thinning, alpha o X = sum_{i=1}^{X} Bernoulli(alpha),
    and epsilon_t are i.i.d. count innovations (Poisson or negative binomial).

    Stationarity requires sum(alpha) < 1, giving unconditional mean
    E[X] = E[epsilon] / (1 - sum(alpha)). The autocorrelation function
    follows the same Yule-Walker recursions as a Gaussian AR(p); for
    INAR(1), acf(k) = alpha^k. With Poisson innovations the INAR(1)
    stationary marginal is Poisson(innovation_mean / (1 - alpha)).

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data (e.g. 'D', 'h', '5min') or int
        p (int): Autoregressive order (default: 1)
        alpha (list[float] | None): Thinning probabilities, each in [0, 1]
            with sum < 1 (default: random with sum < 0.8)
        innovation_type (str): 'poisson' or 'negative_binomial'
            (default: 'poisson')
        innovation_mean (float): Mean of innovations (default: 5.0)
        innovation_dispersion (float): Dispersion r for the negative binomial;
            innovation variance is mean + mean^2 / r (default: 2.0)
        seed (int | None): Random seed for reproducibility (default: None)
    """

    p: int = Field(default=1, ge=1, description="INAR order")
    alpha: list[float] | None = Field(
        default=None, description="Thinning probabilities (each in [0,1])"
    )
    innovation_type: Literal["poisson", "negative_binomial"] = Field(
        default="poisson", description="Innovation distribution type"
    )
    innovation_mean: float = Field(
        default=5.0, gt=0, description="Mean of innovation distribution"
    )
    innovation_dispersion: float = Field(
        default=2.0, gt=0, description="Dispersion for negative binomial"
    )

    _alpha_array: Any = None

    @model_validator(mode="after")
    def compute_inar_params(self) -> "INARGenerator":
        """Compute thinning probabilities if not provided."""
        if self.alpha is None:
            # Random but stationary: per-element cap keeps sum(alpha) < 0.8
            max_per = 0.8 / self.p
            alpha_array = self.rng.uniform(0.1, max_per, self.p)
        else:
            if len(self.alpha) != self.p:
                raise ValueError(
                    f"alpha must have length p={self.p}, got {len(self.alpha)}"
                )
            alpha_array = np.array(self.alpha)

        if np.any(alpha_array < 0) or np.any(alpha_array > 1):
            raise ValueError("All thinning probabilities must be in [0, 1]")
        if np.sum(alpha_array) >= 1:
            raise ValueError(
                "Sum of thinning probabilities must be < 1 for stationarity"
            )
        object.__setattr__(self, "_alpha_array", alpha_array)
        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        innov_type_id = 0 if self.innovation_type == "poisson" else 1
        return (
            np.array(
                [
                    float(self.p),
                    float(innov_type_id),
                    self.innovation_mean,
                    self.innovation_dispersion,
                ]
            ),
            [np.asarray(self._alpha_array, dtype=np.float64)],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single INAR time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of non-negative integer time series values
        """
        seed = int(self.rng.integers(0, 2**63))
        innov_type_id = 0 if self.innovation_type == "poisson" else 1
        return _rs_stat.inar(
            length,
            self.p,
            self._alpha_array,
            innov_type_id,
            self.innovation_mean,
            self.innovation_dispersion,
            seed,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the INAR configuration."""
        alpha_sum = float(np.sum(self._alpha_array))
        stationary_mean = self.innovation_mean / (1.0 - alpha_sum)
        return {
            "order": self.p,
            "thinning_probabilities": self._alpha_array.tolist(),
            "innovation_type": self.innovation_type,
            "innovation_mean": self.innovation_mean,
            "stationary_mean": stationary_mean,
            "persistence": alpha_sum,
        }
