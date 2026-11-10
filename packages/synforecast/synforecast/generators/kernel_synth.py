"""KernelSynth generator: sample from randomly composed GP kernels."""

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import multivariate as _rs_mv
from synforecast.base import BaseGenerator

# Shared with rust/src/generators/kernel_synth.rs.
_KIND_IDS = {
    "rbf": 0,
    "rational_quadratic": 1,
    "periodic": 2,
    "linear": 3,
    "white": 4,
    "constant": 5,
}
# Mirrors MAX_KERNELS in rust/src/generators/kernel_synth.rs.
_MAX_KERNELS = 64


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
    standardization. It also replaces the reference's dense Cholesky sampler
    with the exact O(n^2) polynomial decomposition described below, which
    adds ``jitter`` to each term and escalates it on a numerical breakdown.

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

    Each composition is a polynomial ``K = sum_d (x x')^d T_d(x - x')`` with
    stationary PSD ``T_d`` (only the linear kernel is non-stationary), sampled
    exactly as ``sum_d x^d * f_d`` with independent O(n^2) Durbin-Levinson
    draws ``f_d ~ GP(0, T_d)``. Both bulk ``generate()`` and
    ``generate_single_series`` run this sampler in Rust.

    Because a composed kernel can be near-degenerate or produce an exploding
    scale, non-finite, near-constant, or ``|y| >= 1e8`` draws are redrawn a
    bounded number of times before falling back to Gaussian noise.

    Args:
        max_kernels (int): Maximum number of base kernels composed per series;
            the count is drawn uniformly from ``1..max_kernels`` (default: 5,
            at most 64).
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
        jitter (float): Diagonal jitter added to each term ``T_d``. On a
            numerical breakdown it is escalated tenfold up to three times, to
            at least ``1e-6`` times the term's variance even when ``jitter`` is
            0 (default: 1e-6).
        standardize (bool): Standardize each sampled series to zero mean and
            unit variance. Kernel compositions span extreme scales, so
            standardization keeps the pool comparable for pretraining
            (default: True).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    max_kernels: int = Field(
        default=5,
        ge=1,
        le=_MAX_KERNELS,
        description="Max base kernels composed per series",
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

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encode the bank for ``GEN_KERNEL_SYNTH`` in ``rust/src/batch.rs``.

        scalars: [max_kernels, jitter, standardize]
        arrays:  [kernel kind ids (``_KIND_IDS``), kernel parameters]
        """
        kinds = np.array([_KIND_IDS[kind] for kind, _ in self._bank], dtype=float)
        params = np.array([param for _, param in self._bank], dtype=float)
        scalars = np.array(
            [float(self.max_kernels), self.jitter, float(self.standardize)]
        )
        return scalars, [kinds, params]

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate one KernelSynth series.

        Runs the same Rust sampler as the batch path, seeded from ``self.rng``.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        scalars, (kinds, params) = self._get_batch_params()
        seed = int(self.rng.integers(0, 2**63))
        return _rs_mv.kernel_synth(length, scalars, kinds, params, seed)
