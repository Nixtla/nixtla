"""Bounded process generator for [0, 1] or [a, b] valued time series."""

from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


def _logit(x: np.ndarray) -> np.ndarray:
    """Logit transform: log(x / (1-x))."""
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid transform: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class BoundedProcessGenerator(BaseGenerator):
    """Generate time series constrained to a bounded interval.

    Values are simulated on the unit interval and affinely mapped to
    [lower, upper] (default [0, 1]). Useful for proportions, market shares,
    probabilities, and other bounded quantities.

    Models:
        - beta_ar: Beta AR(1) via conditional mean parameterization.
          mu_t = omega + phi * x_{t-1}, x_t ~ Beta(mu_t * kappa,
          (1 - mu_t) * kappa), so E[x_t | x_{t-1}] = mu_t and the stationary
          mean is omega / (1 - phi).
        - logit_normal: AR(1) on the logit scale, sigmoid-transformed back:
          z_t = phi * z_{t-1} + sigma * eps_t, x_t = sigmoid(z_t).

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min') or an
            integer time step.
        model (str): 'beta_ar' or 'logit_normal' (default: 'beta_ar').
        phi (float): AR coefficient in [-1, 1] (default: 0.8).
        omega (float): Intercept of the beta_ar conditional mean
            (default: 0.1). Must satisfy 0 < omega + phi * x < 1 for
            x in (0, 1).
        kappa (float): Beta precision; larger = less noise (default: 20.0).
        sigma (float): Logit-scale innovation std (logit_normal only,
            default: 0.3).
        initial_value (float): Starting value on the unit scale, in (0, 1)
            (default: 0.5).
        lower (float): Lower bound of the output interval (default: 0.0).
        upper (float): Upper bound of the output interval (default: 1.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    model: Literal["beta_ar", "logit_normal"] = Field(
        default="beta_ar", description="Bounded process model type"
    )
    phi: float = Field(default=0.8, ge=-1.0, le=1.0, description="AR coefficient")
    omega: float = Field(
        default=0.1, description="Intercept for beta_ar conditional mean"
    )
    kappa: float = Field(default=20.0, gt=0, description="Beta precision parameter")
    sigma: float = Field(default=0.3, gt=0, description="Logit-normal innovation std")
    initial_value: float = Field(
        default=0.5, gt=0, lt=1, description="Starting value in (0, 1)"
    )
    lower: float = Field(default=0.0, description="Lower bound")
    upper: float = Field(default=1.0, description="Upper bound")

    _model_id: int = 0

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundedProcessGenerator":
        """Validate that bounds are properly ordered."""
        if self.lower >= self.upper:
            raise ValueError(
                f"lower ({self.lower}) must be less than upper ({self.upper})"
            )
        if self.model == "beta_ar":
            # mu_t = omega + phi * x_{t-1} must stay in (0, 1) for all x in (0, 1)
            if self.omega + max(self.phi, 0.0) >= 1.0:
                raise ValueError(
                    "omega + max(phi, 0) must be < 1 for beta_ar "
                    "to keep conditional mean in (0, 1)"
                )
            if self.omega + min(self.phi, 0.0) <= 0.0:
                raise ValueError(
                    "omega + min(phi, 0) must be > 0 for beta_ar "
                    "to keep conditional mean in (0, 1)"
                )
        model_map = {"beta_ar": 0, "logit_normal": 1}
        object.__setattr__(self, "_model_id", model_map[self.model])
        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        return (
            np.array(
                [
                    float(self._model_id),
                    self.phi,
                    self.omega,
                    self.kappa,
                    self.sigma,
                    self.initial_value,
                    self.lower,
                    self.upper,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single bounded time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values in [lower, upper]
        """
        # The Rust kernel applies the [lower, upper] affine map itself.
        seed = int(self.rng.integers(0, 2**63))
        return _rs_stoch.bounded_process(
            length,
            self._model_id,
            self.phi,
            self.omega,
            self.kappa,
            self.sigma,
            self.initial_value,
            self.lower,
            self.upper,
            seed,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the bounded process configuration."""
        info: dict[str, Any] = {
            "model": self.model,
            "phi": self.phi,
            "bounds": (self.lower, self.upper),
        }
        if self.model == "beta_ar":
            # Unit-scale stationary mean omega / (1 - phi), mapped to bounds
            unit_mean = self.omega / (1.0 - self.phi)
            stationary_mean = self.lower + unit_mean * (self.upper - self.lower)
            info.update(
                omega=self.omega,
                kappa=self.kappa,
                stationary_mean=stationary_mean,
            )
        else:
            info["sigma"] = self.sigma
        return info
