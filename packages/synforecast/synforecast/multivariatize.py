"""Multivariatizer: turn any univariate generator into correlated channels.

Implements the Chronos-2-style multivariate synthetic-data recipe: draw
independent univariate series from a wrapped generator, then couple them
cotemporaneously (random well-conditioned mixing) and/or sequentially
(lead-lag transforms) so the channels carry cross-series dependence.
"""

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

_COUPLINGS = ("mixing", "leadlag")

# Divergence guard thresholds (mirror the generators' synthetic-pool guard)
_MAX_ABS = 1e8
_MIN_STD = 1e-8
# Base-series redraws allowed before falling back to unit Gaussian noise
_MAX_REDRAWS = 5
# Probability that a non-root channel becomes a lead-lag transform (at least
# one channel always does when 'leadlag' is active and n_series >= 2)
_LEADLAG_PROB = 0.5


class Multivariatizer(BaseModel):
    """Wrap a univariate :class:`BaseGenerator` to produce correlated channels.

    ``generate(n_series)`` draws ``n_series`` independent series of one
    shared length from the wrapped generator, standardizes them, applies the
    configured couplings, then restores each channel's original level and
    scale. The output is the same long-format frame the wrapped generator
    produces (its ``id_col``/``time_col``/``target_col`` and ``engine``).

    Couplings (both may compose; ``mixing`` is applied first):

    - ``"mixing"`` (cotemporaneous): channels become instantaneous linear
      combinations ``Z @ L.T`` of the standardized bases, where ``L`` is the
      Cholesky factor of a random well-conditioned correlation target
      ``C = (1 - s) I + s Q``. ``Q`` is the correlation matrix of a random
      Gaussian Gram matrix and ``s`` (the mixing strength, drawn from
      ``mixing_strength_range``) directly sets the magnitude of the induced
      cross-correlations; ``s < 1`` keeps ``C`` positive definite, so ``L``
      is well conditioned.
    - ``"leadlag"`` (sequential): each non-root channel becomes, with
      probability 0.5 (at least one always does), a lagged, sign-flipped,
      noise-perturbed copy of an earlier channel:
      ``z_j = sign * roll(z_src, lag) + sigma * eps`` with ``lag`` from
      ``lag_range`` (clamped below the series length, circular wrap so all
      channels share the same timestamps) and ``sigma`` from
      ``noise_scale_range``. Per-channel scaling comes from the level/scale
      restore.

    Guards: every drawn base series must be finite with ``|x| < 1e8`` and
    ``std > 1e-8``; violating draws are redrawn up to 5 times, then replaced
    by unit Gaussian noise.

    Seeding: the multivariatizer's own ``seed`` fully determines the output.
    The wrapped generator is copied and reseeded from the multivariatizer's
    rng on every ``generate`` call, so the base generator's own seed and rng
    state never influence the result and the original object is not mutated.

    The last sampled coupling recipe (mixing strength/matrix, lead-lag
    pairs with their lags) is exposed as ``last_recipe`` for introspection.

    Args:
        base (BaseGenerator): Wrapped univariate generator; supplies the
            length range, frequency, column names, and dataframe engine.
        couplings (list[str]): Couplings to apply, subset of
            ['mixing', 'leadlag'] (default: both).
        mixing_strength_range (tuple[float, float]): Range for the mixing
            strength s in [0, 1); s scales the induced cotemporaneous
            cross-correlations (default: (0.2, 0.9)).
        lag_range (tuple[int, int]): Inclusive range for lead-lag offsets in
            time steps (default: (1, 24)).
        noise_scale_range (tuple[float, float]): Range for the lead-lag
            perturbation noise std, relative to the unit-variance
            standardized channels (default: (0.02, 0.2)).
        seed (int | None): Random seed for reproducibility (default: None).

    Example:
        >>> from synforecast.generators import TSIGenerator
        >>> base = TSIGenerator(min_length=256, max_length=512, freq="h")
        >>> mv = Multivariatizer(base=base, seed=42)
        >>> df = mv.generate(n_series=4)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base: BaseGenerator = Field(..., description="Wrapped univariate generator")
    couplings: list[str] = Field(
        default=list(_COUPLINGS), description="Couplings to apply"
    )
    mixing_strength_range: tuple[float, float] = Field(
        default=(0.2, 0.9), description="Range for the mixing strength in [0, 1)"
    )
    lag_range: tuple[int, int] = Field(
        default=(1, 24), description="Inclusive range for lead-lag offsets"
    )
    noise_scale_range: tuple[float, float] = Field(
        default=(0.02, 0.2), description="Range for lead-lag noise std"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    _rng: np.random.Generator = PrivateAttr()
    _last_recipe: dict | None = PrivateAttr(default=None)

    @property
    def rng(self) -> np.random.Generator:
        """Multivariatizer-local random number stream."""
        return self._rng

    @model_validator(mode="after")
    def validate_multivariatizer(self) -> "Multivariatizer":
        """Validate coupling pool and sampling ranges; seed the rng."""
        if not self.couplings:
            raise ValueError("couplings must not be empty")
        for coupling in self.couplings:
            if coupling not in _COUPLINGS:
                raise ValueError(f"couplings must be a subset of {_COUPLINGS}")

        s_low, s_high = self.mixing_strength_range
        if not (0.0 <= s_low <= s_high < 1.0):
            raise ValueError("mixing_strength_range must satisfy 0 <= low <= high < 1")

        l_low, l_high = self.lag_range
        if l_low < 1 or l_high < l_low:
            raise ValueError("lag_range must satisfy 1 <= low <= high")

        n_low, n_high = self.noise_scale_range
        if n_low <= 0 or n_high < n_low:
            raise ValueError("noise_scale_range must satisfy 0 < low <= high")

        self._rng = np.random.default_rng(self.seed)
        return self

    @property
    def last_recipe(self) -> dict | None:
        """The coupling recipe sampled by the most recent generate() call."""
        return self._last_recipe

    @staticmethod
    def _series_ok(values: np.ndarray) -> bool:
        """Finiteness/scale guard: finite, |x| < 1e8, std > 1e-8."""
        return bool(
            np.isfinite(values).all()
            and np.abs(values).max() < _MAX_ABS
            and values.std() > _MIN_STD
        )

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        """Standardize to zero mean, unit std (identity scale if constant)."""
        std = float(values.std())
        if std < _MIN_STD:
            std = 1.0
        return (values - values.mean()) / std

    def _draw_base(self, gen: BaseGenerator, length: int) -> np.ndarray:
        """Draw one guarded base series; fall back to noise after redraws."""
        for _ in range(_MAX_REDRAWS + 1):
            values = np.asarray(gen.generate_single_series(length), dtype=np.float64)
            if self._series_ok(values):
                return values
        return self.rng.normal(0.0, 1.0, length)

    def _mixing_matrix(self, n: int) -> tuple[np.ndarray, float]:
        """Sample a well-conditioned mixing matrix L with L @ L.T = C.

        C = (1 - s) I + s Q is a strictly positive definite correlation
        target (Q the correlation matrix of a random Gaussian Gram matrix),
        so mixing independent unit-variance channels by L yields channel
        correlations equal to C's off-diagonals, of magnitude ~ s.
        """
        s = float(self.rng.uniform(*self.mixing_strength_range))
        a = self.rng.normal(0.0, 1.0, (n, n))
        gram = a @ a.T
        d = np.sqrt(np.diag(gram))
        q = gram / np.outer(d, d)
        c = (1.0 - s) * np.eye(n) + s * q
        return np.linalg.cholesky(c), s

    def _apply_mixing(self, z: np.ndarray, recipe: dict) -> np.ndarray:
        """Cotemporaneous coupling: instantaneous linear combinations."""
        n = z.shape[1]
        mix, strength = self._mixing_matrix(n)
        recipe["mixing"] = {"strength": strength, "matrix": mix}
        mixed = z @ mix.T
        for j in range(n):
            mixed[:, j] = self._standardize(mixed[:, j])
        return mixed

    def _apply_leadlag(self, z: np.ndarray, recipe: dict) -> np.ndarray:
        """Sequential coupling: lagged noise-perturbed channel transforms."""
        length, n = z.shape
        coupled = self.rng.random(n - 1) < _LEADLAG_PROB
        if not coupled.any():
            coupled[-1] = True

        pairs = []
        for j in range(1, n):
            if not coupled[j - 1]:
                continue
            src = int(self.rng.integers(0, j))
            lag = int(self.rng.integers(self.lag_range[0], self.lag_range[1] + 1))
            lag = min(lag, length - 1)
            if lag < 1:
                continue
            sign = float(self.rng.choice([-1.0, 1.0]))
            sigma = float(self.rng.uniform(*self.noise_scale_range))
            z[:, j] = sign * np.roll(z[:, src], lag) + sigma * self.rng.normal(
                0.0, 1.0, length
            )
            z[:, j] = self._standardize(z[:, j])
            pairs.append(
                {"src": src, "dst": j, "lag": lag, "sign": sign, "noise": sigma}
            )
        recipe["leadlag"] = pairs
        return z

    def generate(self, n_series: int, start_id: int = 0) -> IntoDataFrameT:
        """Generate n_series cross-dependent channels from the wrapped base.

        All channels share one length drawn from the base generator's
        [min_length, max_length]; each channel is one ``unique_id`` in the
        long-format output.

        Args:
            n_series (int): Number of coupled channels to generate.
            start_id (int): Starting ID for series numbering (default: 0).

        Returns:
            DataFrame in long format with the wrapped generator's [id_col, time_col, target_col] columns and dataframe engine.
        """
        if n_series < 1:
            raise ValueError("n_series must be >= 1")

        # Copy + reseed so the base's own rng/seed never affects the output
        gen = self.base.model_copy()
        gen._rng = np.random.default_rng(int(self.rng.integers(0, 2**63)))

        length = int(self.rng.integers(gen.min_length, gen.max_length + 1))

        z = np.empty((length, n_series))
        levels = np.empty(n_series)
        scales = np.empty(n_series)
        for i in range(n_series):
            values = self._draw_base(gen, length)
            levels[i] = values.mean()
            scales[i] = max(float(values.std()), _MIN_STD)
            z[:, i] = self._standardize(values)

        recipe: dict = {"couplings": [], "length": length}
        if n_series >= 2:
            if "mixing" in self.couplings:
                z = self._apply_mixing(z, recipe)
                recipe["couplings"].append("mixing")
            if "leadlag" in self.couplings:
                z = self._apply_leadlag(z, recipe)
                recipe["couplings"].append("leadlag")
        self._last_recipe = recipe

        timestamps = gen._timestamps(length)
        all_metadata = [
            SeriesMetadata(
                values=z[:, i] * scales[i] + levels[i],
                timestamps=timestamps,
                series_id=start_id + i,
                length=length,
            )
            for i in range(n_series)
        ]
        return gen._build_dataframe(all_metadata)
