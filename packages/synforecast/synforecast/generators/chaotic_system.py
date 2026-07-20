"""Chaotic system generator."""

from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import stochastic as _rs_stoch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class ChaoticSystemGenerator(BaseGenerator):
    """Generate time series from deterministic chaotic dynamical systems.

    Produces series that look stochastic but are fully deterministic given
    the initial condition; randomness enters only through a seeded
    perturbation of the initial condition and optional observation noise.

    Systems:
        - lorenz: Lorenz attractor ``x' = sigma(y-x), y' = x(rho-z) - y,
          z' = xy - beta*z``, integrated with RK4 and sampled every
          1/dt steps (one time unit per observation); the x-component is
          returned.
        - logistic: Logistic map ``x_{n+1} = r * x_n * (1 - x_n)``
          (chaotic for r ~ 3.57..4; for r=4 the invariant density is
          Beta(1/2, 1/2)).
        - mackey_glass: Mackey-Glass delay differential equation
          ``x' = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x``,
          Euler-integrated with unit step (chaotic for tau >= 17 at the
          default parameters).

    Args:
        system (str): 'lorenz', 'logistic' or 'mackey_glass' (default: 'lorenz').
        sigma (float): Lorenz sigma (default: 10.0).
        rho (float): Lorenz rho (default: 28.0).
        beta_param (float): Lorenz beta, alias 'lorenz_beta' (default: 2.6667).
        dt (float): Lorenz RK4 integration step (default: 0.01).
        logistic_r (float): Logistic map parameter r (default: 3.9).
        mg_beta (float): Mackey-Glass beta (default: 0.2).
        mg_gamma (float): Mackey-Glass gamma (default: 0.1).
        mg_n (float): Mackey-Glass exponent n (default: 10.0).
        mg_tau (int): Mackey-Glass delay tau (default: 17).
        observation_noise (float): Std of additive Gaussian observation
            noise (default: 0.0).
        initial_perturbation (float): Scale of the random initial-condition
            perturbation; 0 makes the output seed-independent (default: 0.01).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    system: Literal["lorenz", "logistic", "mackey_glass"] = Field(
        default="lorenz", description="Chaotic system type"
    )

    # Lorenz parameters
    sigma: float = Field(default=10.0, description="Lorenz sigma")
    rho: float = Field(default=28.0, description="Lorenz rho")
    beta_param: float = Field(
        default=2.6667, description="Lorenz beta", alias="lorenz_beta"
    )
    dt: float = Field(default=0.01, gt=0, description="Integration step size")

    # Logistic map parameters
    logistic_r: float = Field(default=3.9, description="Logistic map parameter r")

    # Mackey-Glass parameters
    mg_beta: float = Field(default=0.2, description="Mackey-Glass beta")
    mg_gamma: float = Field(default=0.1, description="Mackey-Glass gamma")
    mg_n: float = Field(default=10.0, description="Mackey-Glass exponent n")
    mg_tau: int = Field(default=17, ge=1, description="Mackey-Glass delay tau")

    # Shared parameters
    observation_noise: float = Field(
        default=0.0, ge=0, description="Observation noise std"
    )
    initial_perturbation: float = Field(
        default=0.01, ge=0, description="Initial condition perturbation scale"
    )

    _system_id: int = 0

    @model_validator(mode="after")
    def setup_system_id(self) -> "ChaoticSystemGenerator":
        """Map system name to integer ID for Rust backend."""
        system_map = {"lorenz": 0, "logistic": 1, "mackey_glass": 2}
        object.__setattr__(self, "_system_id", system_map[self.system])
        return self

    def _generate_lorenz(self, length: int) -> np.ndarray:
        """Generate Lorenz attractor trajectory (x-component) via RK4."""
        burn_in = 1000
        total = length + burn_in
        steps_per_obs = max(1, int(1.0 / self.dt))
        total_steps = total * steps_per_obs

        perturb = self.rng.standard_normal(3) * self.initial_perturbation
        x, y, z = 1.0 + perturb[0], 1.0 + perturb[1], 1.0 + perturb[2]

        out = np.empty(total)
        idx = 0
        for i in range(total_steps):
            dx1 = self.sigma * (y - x)
            dy1 = x * (self.rho - z) - y
            dz1 = x * y - self.beta_param * z

            x2 = x + 0.5 * self.dt * dx1
            y2 = y + 0.5 * self.dt * dy1
            z2 = z + 0.5 * self.dt * dz1

            dx2 = self.sigma * (y2 - x2)
            dy2 = x2 * (self.rho - z2) - y2
            dz2 = x2 * y2 - self.beta_param * z2

            x3 = x + 0.5 * self.dt * dx2
            y3 = y + 0.5 * self.dt * dy2
            z3 = z + 0.5 * self.dt * dz2

            dx3 = self.sigma * (y3 - x3)
            dy3 = x3 * (self.rho - z3) - y3
            dz3 = x3 * y3 - self.beta_param * z3

            x4 = x + self.dt * dx3
            y4 = y + self.dt * dy3
            z4 = z + self.dt * dz3

            dx4 = self.sigma * (y4 - x4)
            dy4 = x4 * (self.rho - z4) - y4
            dz4 = x4 * y4 - self.beta_param * z4

            x += self.dt / 6.0 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
            y += self.dt / 6.0 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            z += self.dt / 6.0 * (dz1 + 2 * dz2 + 2 * dz3 + dz4)

            if (i + 1) % steps_per_obs == 0 and idx < total:
                out[idx] = x
                idx += 1

        return out[burn_in:]

    def _generate_logistic(self, length: int) -> np.ndarray:
        """Generate logistic map trajectory."""
        burn_in = 500
        total = length + burn_in
        r = self.logistic_r

        x = 0.5 + self.rng.standard_normal() * self.initial_perturbation
        x = np.clip(x, 0.01, 0.99)

        out = np.empty(total)
        for i in range(total):
            x = r * x * (1.0 - x)
            out[i] = x

        return out[burn_in:]

    def _generate_mackey_glass(self, length: int) -> np.ndarray:
        """Generate Mackey-Glass trajectory (Euler, unit step, ring-buffer delay)."""
        burn_in = 500
        total = length + burn_in
        tau = self.mg_tau
        dt_mg = 1.0

        history_len = tau + 1
        history = np.full(
            history_len,
            1.2 + self.rng.standard_normal() * self.initial_perturbation,
        )

        out = np.empty(total)
        x = history[-1]

        for i in range(total):
            x_tau = history[i % history_len]
            dx = self.mg_beta * x_tau / (1.0 + x_tau**self.mg_n) - self.mg_gamma * x
            x = x + dx * dt_mg
            out[i] = x
            history[(i + tau + 1) % history_len] = x

        return out[burn_in:]

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    float(self._system_id),
                    self.sigma,
                    self.rho,
                    self.beta_param,
                    self.dt,
                    self.logistic_r,
                    self.mg_beta,
                    self.mg_gamma,
                    self.mg_n,
                    float(self.mg_tau),
                    self.observation_noise,
                    self.initial_perturbation,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single chaotic time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values.
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            return _rs_stoch.chaotic_system(
                length,
                self._system_id,
                self.sigma,
                self.rho,
                self.beta_param,
                self.dt,
                self.logistic_r,
                self.mg_beta,
                self.mg_gamma,
                self.mg_n,
                self.mg_tau,
                self.observation_noise,
                self.initial_perturbation,
                seed,
            )

        if self.system == "lorenz":
            values = self._generate_lorenz(length)
        elif self.system == "logistic":
            values = self._generate_logistic(length)
        elif self.system == "mackey_glass":
            values = self._generate_mackey_glass(length)
        else:
            raise ValueError(f"Unknown system: {self.system}")

        if self.observation_noise > 0:
            values = values + self.rng.normal(0, self.observation_noise, length)

        return values

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the chaotic system configuration."""
        info: dict[str, Any] = {"system": self.system, "deterministic": True}
        if self.system == "lorenz":
            info.update(
                sigma=self.sigma,
                rho=self.rho,
                beta=self.beta_param,
                dt=self.dt,
            )
        elif self.system == "logistic":
            info["r"] = self.logistic_r
        elif self.system == "mackey_glass":
            info.update(
                beta=self.mg_beta,
                gamma=self.mg_gamma,
                n=self.mg_n,
                tau=self.mg_tau,
            )
        return info
