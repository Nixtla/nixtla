"""KernelSynth generator: sample from randomly composed GP kernels."""

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

# Divergence guards, mirroring the other pretraining generators (TSI/TCM).
_MAX_ABS = 1e8
_MIN_STD = 1e-8
_MAX_RETRIES = 8


class KernelSynthGenerator(BaseGenerator):
    """Generate series by sampling from randomly composed Gaussian-process kernels.

    This adapts the KernelSynth recipe introduced for pretraining the Chronos
    forecasting models (Ansari et al. 2024,
    "Chronos: Learning the Language of Time Series",
    https://arxiv.org/abs/2403.07815) and its Apache-2.0-licensed reference
    implementation (https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py).
    For each series the
    generator draws ``1..max_kernels`` base kernels (with replacement) from a
    fixed bank, folds them together with randomly chosen binary operators
    (``+`` or ``*``), and samples one path from the resulting GP prior on the
    normalized grid ``x = linspace(0, 1, length)``. Kernel addition mixes
    behaviors (e.g. trend + seasonality); kernel multiplication modulates them
    (e.g. locally periodic, amplitude-varying seasonality). SynForecast makes
    the bank configurable, expresses seasonal periods in time steps on a
    normalized grid, and adds bounded retries, divergence guards, and optional
    standardization.

    Base kernels (r = |x_i - x_j|, all on the normalized grid):
        - rbf: ``exp(-r^2 / (2 l^2))`` — smooth, length-scale ``l``
        - rational_quadratic: ``(1 + r^2 / (2 a))^(-a)`` — scale mixture of
          RBFs, shape ``a``
        - periodic (ExpSineSquared): ``exp(-2 sin^2(pi r / p_norm))`` with
          ``p_norm = period / length`` so ``period`` is expressed in time
          steps
        - linear (DotProduct): ``s^2 + x_i x_j`` — trend / drift
        - white: ``w`` on the diagonal — independent noise
        - constant: a constant offset

    Because a composed kernel can be near-degenerate or produce an exploding
    scale, non-finite, near-constant, or ``|y| >= 1e8`` draws are redrawn a
    bounded number of times before falling back to Gaussian noise.

    Args:
        max_kernels (int): Maximum number of base kernels composed per series;
            the count is drawn uniformly from ``1..max_kernels`` (default: 5).
        seasonal_periods (list[float]): Periodic-kernel periods, in time steps,
            forming the periodic entries of the bank (default: a broad set from
            4 up to 730 covering common hourly/daily/weekly/quarterly/yearly
            seasonalities).
        rbf_length_scales (list[float]): RBF length scales on the normalized
            grid (default: [0.1, 1.0, 10.0]).
        rational_quadratic_alphas (list[float]): Rational-quadratic shape
            parameters (default: [0.1, 1.0, 10.0]).
        linear_sigmas (list[float]): ``sigma_0`` offsets for the linear
            (DotProduct) kernel (default: [0.0, 1.0, 10.0]).
        white_noise_levels (list[float]): Diagonal noise levels for the white
            kernel (default: [0.1, 1.0]).
        include_constant (bool): Include a constant kernel in the bank
            (default: True).
        jitter (float): Diagonal jitter added before factorization for
            numerical stability (default: 1e-6).
        standardize (bool): Standardize each sampled series to zero mean and
            unit variance. Kernel compositions span extreme scales, so
            standardization keeps the pool comparable for pretraining
            (default: True).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    max_kernels: int = Field(
        default=5, ge=1, description="Max base kernels composed per series"
    )
    seasonal_periods: list[float] = Field(
        default=[
            4.0,
            6.0,
            7.0,
            10.0,
            12.0,
            14.0,
            24.0,
            26.0,
            30.0,
            48.0,
            52.0,
            96.0,
            168.0,
            336.0,
            365.0,
            730.0,
        ],
        description="Periodic-kernel periods in time steps",
    )
    rbf_length_scales: list[float] = Field(
        default=[0.1, 1.0, 10.0], description="RBF length scales (normalized grid)"
    )
    rational_quadratic_alphas: list[float] = Field(
        default=[0.1, 1.0, 10.0], description="Rational-quadratic shape parameters"
    )
    linear_sigmas: list[float] = Field(
        default=[0.0, 1.0, 10.0], description="Linear (DotProduct) sigma_0 offsets"
    )
    white_noise_levels: list[float] = Field(
        default=[0.1, 1.0], description="White-kernel diagonal noise levels"
    )
    include_constant: bool = Field(
        default=True, description="Include a constant kernel in the bank"
    )
    jitter: float = Field(
        default=1e-6, ge=0, description="Diagonal jitter for numerical stability"
    )
    standardize: bool = Field(
        default=True, description="Standardize each series to zero mean, unit variance"
    )

    # Bank of (kind, param) specs; built once so rng.integers can index it.
    _bank: list[tuple[str, float]] = []

    @model_validator(mode="after")
    def build_kernel_bank(self) -> "KernelSynthGenerator":
        """Validate pools and assemble the kernel bank in a stable order."""
        if any(p <= 1 for p in self.seasonal_periods):
            raise ValueError("seasonal_periods must all be > 1")
        for name in ("rbf_length_scales", "rational_quadratic_alphas"):
            values = getattr(self, name)
            if any(v <= 0 for v in values):
                raise ValueError(f"{name} values must be positive")
        if any(w < 0 for w in self.white_noise_levels):
            raise ValueError("white_noise_levels must be non-negative")
        if any(s < 0 for s in self.linear_sigmas):
            raise ValueError("linear_sigmas must be non-negative")

        bank: list[tuple[str, float]] = []
        bank += [("rbf", float(v)) for v in self.rbf_length_scales]
        bank += [
            ("rational_quadratic", float(v)) for v in self.rational_quadratic_alphas
        ]
        bank += [("periodic", float(v)) for v in self.seasonal_periods]
        bank += [("linear", float(v)) for v in self.linear_sigmas]
        bank += [("white", float(v)) for v in self.white_noise_levels]
        if self.include_constant:
            bank.append(("constant", 1.0))
        if not bank:
            raise ValueError("kernel bank is empty; enable at least one kernel family")

        object.__setattr__(self, "_bank", bank)
        return self

    def _gram(
        self,
        spec: tuple[str, float],
        length: int,
        r: np.ndarray,
        r2: np.ndarray,
        outer: np.ndarray,
    ) -> np.ndarray:
        """Gram matrix for one base kernel on the normalized grid."""
        kind, param = spec
        if kind == "rbf":
            return np.exp(-0.5 * r2 / param**2)
        if kind == "rational_quadratic":
            return (1.0 + r2 / (2.0 * param)) ** (-param)
        if kind == "periodic":
            p_norm = param / length
            return np.exp(-2.0 * np.sin(np.pi * r / p_norm) ** 2)
        if kind == "linear":
            return param**2 + outer
        if kind == "white":
            return param * np.eye(length)
        # constant
        return param * np.ones((length, length))

    def _sample_prior(self, K: np.ndarray, length: int) -> np.ndarray:
        """Draw one sample from N(0, K), robust to ill-conditioned K."""
        K = K + self.jitter * np.eye(length)
        try:
            factor = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals = np.maximum(eigvals, 0.0)
            factor = eigvecs * np.sqrt(eigvals)[None, :]
        return factor @ self.rng.standard_normal(length)

    def _compose_and_sample(
        self,
        length: int,
        r: np.ndarray,
        r2: np.ndarray,
        outer: np.ndarray,
    ) -> np.ndarray:
        """Draw a kernel composition and sample one path from its GP prior."""
        n_bank = len(self._bank)
        n_kernels = int(self.rng.integers(1, self.max_kernels + 1))
        indices = self.rng.integers(0, n_bank, size=n_kernels)

        K = self._gram(self._bank[indices[0]], length, r, r2, outer)
        for j in range(1, n_kernels):
            comp = self._gram(self._bank[indices[j]], length, r, r2, outer)
            # 0 -> add (mix behaviors), 1 -> multiply (modulate behaviors)
            K = K + comp if self.rng.integers(0, 2) == 0 else K * comp

        return self._sample_prior(K, length)

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate one KernelSynth series.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        x = np.linspace(0.0, 1.0, length) if length > 1 else np.zeros(1)
        diff = x[:, None] - x[None, :]
        r = np.abs(diff)
        r2 = diff * diff
        outer = x[:, None] * x[None, :]

        for _ in range(_MAX_RETRIES):
            values = self._compose_and_sample(length, r, r2, outer)
            if not np.all(np.isfinite(values)):
                continue
            if np.abs(values).max() >= _MAX_ABS:
                continue
            std = float(values.std()) if length > 1 else 1.0
            if length > 1 and std <= _MIN_STD:
                continue
            if self.standardize and length > 1:
                values = (values - values.mean()) / std
            return values

        # Retry budget exhausted (pathological bank): fall back to noise.
        return self.rng.normal(0.0, 1.0, length)
