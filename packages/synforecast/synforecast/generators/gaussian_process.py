"""Gaussian Process generator."""

from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import multivariate as _rs_mv

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class GaussianProcessGenerator(BaseGenerator):
    """Generate time series by sampling from a Gaussian Process.

    Samples ``f ~ GP(mean, k)`` on the integer grid t = 0..length-1, so the
    marginal distribution is ``N(mean, amplitude^2 + noise_variance)`` and
    the correlation at lag r is ``k(r) / k(0)``.

    Kernels (r = |t - t'|, l = length_scale, a = amplitude):
        - rbf: ``a^2 exp(-r^2 / (2 l^2))`` — infinitely differentiable,
          very smooth paths
        - matern_0.5: ``a^2 exp(-r/l)`` — rough, Ornstein-Uhlenbeck-like
        - matern_1.5: ``a^2 (1+s) exp(-s)``, s = sqrt(3) r / l —
          once-differentiable
        - matern_2.5: ``a^2 (1+s+s^2/3) exp(-s)``, s = sqrt(5) r / l —
          twice-differentiable
        - periodic: ``a^2 exp(-2 sin^2(pi r / period) / l^2)`` — exact
          periodicity

    Args:
        kernel (str): Kernel type (default: 'rbf').
        length_scale (float): Kernel length scale (default: 20.0).
        amplitude (float): Signal amplitude / output scale (default: 1.0).
        period (float): Period for the periodic kernel (default: 50.0).
        mean (float): Mean function value (default: 0.0).
        noise_variance (float): Observation noise variance, also acts as
            jitter for the Cholesky factorization (default: 1e-6).
    """

    kernel: Literal["rbf", "matern_0.5", "matern_1.5", "matern_2.5", "periodic"] = (
        Field(default="rbf", description="Kernel function type")
    )
    length_scale: float = Field(default=20.0, gt=0, description="Kernel length scale")
    amplitude: float = Field(default=1.0, gt=0, description="Signal amplitude")
    period: float = Field(default=50.0, gt=0, description="Period for periodic kernel")
    mean: float = Field(default=0.0, description="Mean function value")
    noise_variance: float = Field(
        default=1e-6, ge=0, description="Observation noise variance"
    )

    _kernel_id: int = 0

    @model_validator(mode="after")
    def setup_kernel_id(self) -> "GaussianProcessGenerator":
        """Map kernel name to integer ID for Rust backend."""
        kernel_map = {
            "rbf": 0,
            "matern_0.5": 1,
            "matern_1.5": 2,
            "matern_2.5": 3,
            "periodic": 4,
        }
        object.__setattr__(self, "_kernel_id", kernel_map[self.kernel])
        return self

    def _kernel_fn(self, r: np.ndarray) -> np.ndarray:
        """Evaluate kernel function for distance array r."""
        ls = self.length_scale
        a2 = self.amplitude**2

        if self.kernel == "rbf":
            return a2 * np.exp(-0.5 * (r / ls) ** 2)
        elif self.kernel == "matern_0.5":
            return a2 * np.exp(-r / ls)
        elif self.kernel == "matern_1.5":
            s = np.sqrt(3.0) * r / ls
            return a2 * (1.0 + s) * np.exp(-s)
        elif self.kernel == "matern_2.5":
            s = np.sqrt(5.0) * r / ls
            return a2 * (1.0 + s + s**2 / 3.0) * np.exp(-s)
        elif self.kernel == "periodic":
            return a2 * np.exp(-2.0 * (np.sin(np.pi * r / self.period) / ls) ** 2)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    float(self._kernel_id),
                    self.length_scale,
                    self.amplitude,
                    self.period,
                    self.mean,
                    self.noise_variance,
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single GP sample path.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            return _rs_mv.gaussian_process(
                length,
                self._kernel_id,
                self.length_scale,
                self.amplitude,
                self.period,
                self.mean,
                self.noise_variance,
                seed,
            )

        t = np.arange(length, dtype=np.float64)
        dists = np.abs(t[:, None] - t[None, :])
        K = self._kernel_fn(dists)
        K += self.noise_variance * np.eye(length)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition with clamped eigenvalues
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals = np.maximum(eigvals, 0.0)
            L = eigvecs * np.sqrt(eigvals)[None, :]

        z = self.rng.standard_normal(length)
        return self.mean + L @ z

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the GP configuration."""
        smoothness = {
            "rbf": "infinitely smooth (C-infinity)",
            "matern_0.5": "rough / non-differentiable (C-0)",
            "matern_1.5": "once-differentiable (C-1)",
            "matern_2.5": "twice-differentiable (C-2)",
            "periodic": "smooth periodic (C-infinity)",
        }
        return {
            "kernel": self.kernel,
            "length_scale": self.length_scale,
            "amplitude": self.amplitude,
            "smoothness": smoothness[self.kernel],
            "period": self.period if self.kernel == "periodic" else None,
        }
