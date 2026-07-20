"""TSI (Trend / Seasonality / Irregularity) composition generator."""

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

# Ordered tuples: default pool order must be stable across processes
# (rng.integers indexes into them), so no sets here.
_TREND_TYPES = (
    "none",
    "linear",
    "exponential",
    "logistic",
    "piecewise_linear",
    "damped",
)
_IRREGULAR_TYPES = ("gaussian", "ar1", "garch_like", "student_t", "laplace")

_MAX_ABS = 1e8
_MIN_STD = 1e-8
_MAX_RETRIES = 8


class TSIGenerator(BaseGenerator):
    """Generate series by composing randomized Trend, Seasonality and
    Irregularity components.

    The component-based construction is based on Bahrpeyma et al. (2021),
    "A Methodology for Validating Diversity in Synthetic Time Series
    Generation," https://doi.org/10.1016/j.mex.2021.101459. SynForecast's
    component families, sampling distributions, and stability guards are its
    own extensions rather than a reproduction of that paper's generator.

    Every series draws a fresh random configuration: a trend type from
    `trend_types`, 0-3 seasonal harmonics with periods from
    `seasonal_periods` (integer and non-integer, so multiple harmonics are
    incommensurate), and an irregular (noise) process from
    `irregular_types`. The components are combined additively, or
    multiplicatively with probability `multiplicative_prob` when the trend
    base can be kept positive:

        additive:        y_t = T_t + S_t + e_t
        multiplicative:  y_t = T_t · (1 + S_t / c) + e_t,  min_t T_t > 0

    where c caps the relative seasonal swing so the factor stays positive.
    Trend shapes are normalized so their total movement over the series is
    drawn from `trend_slope_range` regardless of length; harmonic
    amplitudes are log-uniform; the noise scale is a log-uniform fraction
    of the structural signal's standard deviation, so the pool spans
    signal-dominated through noise-dominated series. A per-series level
    and log-uniform scale spread series across magnitudes. Degenerate or
    exploding draws (non-finite, |y| >= 1e8, or constant) are redrawn a
    bounded number of times.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min') or an
            integer time step.
        trend_types (list[str]): Trend shapes sampled per series. Options:
            'none', 'linear', 'exponential', 'logistic', 'piecewise_linear',
            'damped' (default: all six).
        trend_slope_range (tuple[float, float]): Range of the signed total
            trend movement over the whole series (default: (-8.0, 8.0)).
        trend_growth_range (tuple[float, float]): Range of the exponential
            trend's total log-curvature (default: (1.0, 4.0)).
        n_breakpoints_range (tuple[int, int]): Breakpoint count for
            piecewise-linear trends (default: (1, 3)).
        level_range (tuple[float, float]): Per-series base level draw
            (default: (-10.0, 10.0)).
        n_seasonal_range (tuple[int, int]): Number of seasonal harmonics per
            series (default: (0, 3)).
        seasonal_periods (list[float]): Period pool, in time steps; mixes
            integer and non-integer/co-prime periods (default includes 7,
            12, 24, ..., 365.25 and 5.5, 11.3, 19.7, 29.53).
        seasonal_amplitude_range (tuple[float, float]): Log-uniform harmonic
            amplitude range (default: (0.2, 3.0)).
        amplitude_modulation_prob (float): Probability a harmonic gets a
            slowly varying amplitude envelope (default: 0.4).
        harmonics_prob (float): Probability a harmonic gets phase-locked
            2f/3f overtones at decaying amplitude (default: 0.4).
        irregular_types (list[str]): Noise processes sampled per series.
            Options: 'gaussian', 'ar1', 'garch_like', 'student_t', 'laplace'
            (default: all five).
        noise_scale_range (tuple[float, float]): Log-uniform noise std as a
            fraction of the structural signal's std (default: (0.5, 12.0)).
        ar1_phi_range (tuple[float, float]): AR(1) coefficient range for
            'ar1' noise, |phi| < 1 (default: (0.3, 0.95)).
        tail_df_range (tuple[float, float]): Student-t degrees of freedom
            range for 'student_t' noise, > 2 (default: (2.5, 12.0)).
        multiplicative_prob (float): Probability of multiplicative
            trend-season composition (default: 0.3).
        scale_range (tuple[float, float]): Log-uniform overall output scale
            (default: (0.1, 100.0)).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    trend_types: list[str] = Field(
        default=list(_TREND_TYPES),
        description="Trend shapes sampled per series",
    )
    trend_slope_range: tuple[float, float] = Field(
        default=(-8.0, 8.0),
        description="Signed total trend movement over the series",
    )
    trend_growth_range: tuple[float, float] = Field(
        default=(1.0, 4.0),
        description="Total log-curvature of exponential trends",
    )
    n_breakpoints_range: tuple[int, int] = Field(
        default=(1, 3),
        description="Breakpoint count range for piecewise-linear trends",
    )
    level_range: tuple[float, float] = Field(
        default=(-10.0, 10.0), description="Per-series base level range"
    )
    n_seasonal_range: tuple[int, int] = Field(
        default=(0, 3), description="Number of seasonal harmonics per series"
    )
    seasonal_periods: list[float] = Field(
        default=[
            5.5,
            7.0,
            11.3,
            12.0,
            19.7,
            24.0,
            29.53,
            48.0,
            96.0,
            168.0,
            336.0,
            365.25,
        ],
        description="Pool of seasonal periods in time steps",
    )
    seasonal_amplitude_range: tuple[float, float] = Field(
        default=(0.2, 3.0),
        description="Log-uniform seasonal harmonic amplitude range",
    )
    amplitude_modulation_prob: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Probability of a slowly varying amplitude envelope",
    )
    harmonics_prob: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Probability of 2f/3f overtones on a harmonic",
    )
    irregular_types: list[str] = Field(
        default=list(_IRREGULAR_TYPES),
        description="Irregular (noise) processes sampled per series",
    )
    noise_scale_range: tuple[float, float] = Field(
        default=(0.5, 12.0),
        description="Log-uniform noise std as a fraction of structural std",
    )
    ar1_phi_range: tuple[float, float] = Field(
        default=(0.3, 0.95), description="AR(1) coefficient range for 'ar1' noise"
    )
    tail_df_range: tuple[float, float] = Field(
        default=(2.5, 12.0),
        description="Student-t degrees of freedom range for 'student_t' noise",
    )
    multiplicative_prob: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Probability of multiplicative trend-season composition",
    )
    scale_range: tuple[float, float] = Field(
        default=(0.1, 100.0), description="Log-uniform overall output scale range"
    )

    @model_validator(mode="after")
    def validate_tsi_parameters(self) -> "TSIGenerator":
        """Validate component pools and sampling ranges."""
        if not self.trend_types:
            raise ValueError("trend_types must not be empty")
        unknown = set(self.trend_types) - set(_TREND_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown trend_types {sorted(unknown)}; valid: {sorted(_TREND_TYPES)}"
            )

        if not self.irregular_types:
            raise ValueError("irregular_types must not be empty")
        unknown = set(self.irregular_types) - set(_IRREGULAR_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown irregular_types {sorted(unknown)}; "
                f"valid: {sorted(_IRREGULAR_TYPES)}"
            )

        if not self.seasonal_periods:
            raise ValueError("seasonal_periods must not be empty")
        if any(p <= 1 for p in self.seasonal_periods):
            raise ValueError("seasonal_periods must all be > 1")

        for name in (
            "trend_slope_range",
            "trend_growth_range",
            "n_breakpoints_range",
            "level_range",
            "n_seasonal_range",
            "seasonal_amplitude_range",
            "noise_scale_range",
            "ar1_phi_range",
            "tail_df_range",
            "scale_range",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must satisfy low <= high, got ({lo}, {hi})")

        for name in (
            "trend_growth_range",
            "seasonal_amplitude_range",
            "noise_scale_range",
            "scale_range",
        ):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} values must be positive")

        if self.n_seasonal_range[0] < 0:
            raise ValueError("n_seasonal_range values must be >= 0")
        if self.n_breakpoints_range[0] < 1:
            raise ValueError("n_breakpoints_range values must be >= 1")
        if self.ar1_phi_range[0] <= -1 or self.ar1_phi_range[1] >= 1:
            raise ValueError("ar1_phi_range must lie within (-1, 1) for stationarity")
        if self.tail_df_range[0] <= 2:
            raise ValueError("tail_df_range values must be > 2 for finite variance")

        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encode field-level ranges/pools for the Rust batch kernel.

        The Rust kernel draws the whole per-series configuration itself
        (trend type, harmonics, irregular process, composition) from these
        ranges, so only range endpoints, probabilities and pools are passed.
        Scalar ordering and pool id encodings must match the ``GEN_TSI``
        dispatch arm in ``rust/src/batch.rs`` / ``rust/src/generators/tsi.rs``:

        scalars: [trend_slope_lo, trend_slope_hi, trend_growth_lo,
                  trend_growth_hi, n_breakpoints_lo, n_breakpoints_hi,
                  level_lo, level_hi, n_seasonal_lo, n_seasonal_hi,
                  seasonal_amp_lo, seasonal_amp_hi,
                  amplitude_modulation_prob, harmonics_prob,
                  noise_scale_lo, noise_scale_hi, ar1_phi_lo, ar1_phi_hi,
                  tail_df_lo, tail_df_hi, multiplicative_prob,
                  scale_lo, scale_hi]
        arrays:  [trend type ids (index into _TREND_TYPES),
                  seasonal period pool,
                  irregular type ids (index into _IRREGULAR_TYPES)]
        """
        return (
            np.array(
                [
                    self.trend_slope_range[0],
                    self.trend_slope_range[1],
                    self.trend_growth_range[0],
                    self.trend_growth_range[1],
                    float(self.n_breakpoints_range[0]),
                    float(self.n_breakpoints_range[1]),
                    self.level_range[0],
                    self.level_range[1],
                    float(self.n_seasonal_range[0]),
                    float(self.n_seasonal_range[1]),
                    self.seasonal_amplitude_range[0],
                    self.seasonal_amplitude_range[1],
                    self.amplitude_modulation_prob,
                    self.harmonics_prob,
                    self.noise_scale_range[0],
                    self.noise_scale_range[1],
                    self.ar1_phi_range[0],
                    self.ar1_phi_range[1],
                    self.tail_df_range[0],
                    self.tail_df_range[1],
                    self.multiplicative_prob,
                    self.scale_range[0],
                    self.scale_range[1],
                ],
                dtype=np.float64,
            ),
            [
                np.array(
                    [_TREND_TYPES.index(t) for t in self.trend_types],
                    dtype=np.float64,
                ),
                np.asarray(self.seasonal_periods, dtype=np.float64),
                np.array(
                    [_IRREGULAR_TYPES.index(t) for t in self.irregular_types],
                    dtype=np.float64,
                ),
            ],
        )

    def _log_uniform(self, low: float, high: float) -> float:
        return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))

    def _sample_trend(self, length: int) -> np.ndarray:
        """Sample a trend component: level + normalized trend shape."""
        kind = self.trend_types[int(self.rng.integers(len(self.trend_types)))]
        level = self.rng.uniform(*self.level_range)
        u = np.arange(length) / max(length - 1, 1)
        # Total movement over the series, independent of length
        movement = self.rng.uniform(*self.trend_slope_range)

        if kind == "none":
            shape = np.zeros(length)
        elif kind == "linear":
            shape = movement * u
        elif kind == "exponential":
            g = self.rng.uniform(*self.trend_growth_range)
            g *= self.rng.choice([-1.0, 1.0])
            shape = movement * (np.exp(g * u) - 1.0) / (np.exp(g) - 1.0)
        elif kind == "logistic":
            steepness = self.rng.uniform(5.0, 15.0)
            center = self.rng.uniform(0.25, 0.75)
            raw = 1.0 / (1.0 + np.exp(-steepness * (u - center)))
            shape = movement * (raw - raw[0]) / (raw[-1] - raw[0] + 1e-12)
        elif kind == "piecewise_linear":
            lo, hi = self.n_breakpoints_range
            n_breaks = int(self.rng.integers(lo, hi + 1))
            knots_x = np.concatenate(
                [[0.0], np.sort(self.rng.uniform(0.1, 0.9, n_breaks)), [1.0]]
            )
            knots_y = self.rng.uniform(-abs(movement), abs(movement), n_breaks + 2)
            shape = np.interp(u, knots_x, knots_y)
        else:  # damped: rises toward `movement` then flattens
            tau = self.rng.uniform(0.15, 0.5)
            shape = movement * (1.0 - np.exp(-u / tau))

        return level + shape

    def _sample_seasonality(self, length: int) -> np.ndarray:
        """Sample 0+ harmonics with random period/amplitude/phase and
        optional overtones and amplitude modulation."""
        lo, hi = self.n_seasonal_range
        n_harmonics = int(self.rng.integers(lo, hi + 1))
        t = np.arange(length)
        season = np.zeros(length)
        a_lo, a_hi = self.seasonal_amplitude_range

        for _ in range(n_harmonics):
            period = float(self.rng.choice(np.asarray(self.seasonal_periods)))
            amplitude = self._log_uniform(a_lo, a_hi)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            angle = 2.0 * np.pi * t / period + phase
            wave = amplitude * np.sin(angle)
            if self.rng.uniform() < self.harmonics_prob:
                # Phase-locked overtones -> non-sinusoidal seasonal shape
                wave += amplitude * (
                    0.4 * np.sin(2.0 * angle) + 0.2 * np.sin(3.0 * angle)
                )
            if self.rng.uniform() < self.amplitude_modulation_prob:
                env_period = length * self.rng.uniform(0.3, 1.5)
                env_depth = self.rng.uniform(0.3, 0.8)
                env_phase = self.rng.uniform(0.0, 2.0 * np.pi)
                wave *= 1.0 + env_depth * np.sin(
                    2.0 * np.pi * t / env_period + env_phase
                )
            season += wave

        return season

    def _sample_irregular(self, length: int, sigma: float) -> np.ndarray:
        """Sample an irregular component with marginal std ~= sigma."""
        kind = self.irregular_types[int(self.rng.integers(len(self.irregular_types)))]

        if kind == "gaussian":
            return self.rng.normal(0.0, sigma, length)
        if kind == "laplace":
            # Laplace with scale b has variance 2 b^2
            return self.rng.laplace(0.0, sigma / np.sqrt(2.0), length)
        if kind == "student_t":
            df = self.rng.uniform(*self.tail_df_range)
            # standard_t has variance df/(df-2); rescale to sigma
            return self.rng.standard_t(df, length) * sigma * np.sqrt((df - 2.0) / df)
        if kind == "ar1":
            phi = self.rng.uniform(*self.ar1_phi_range)
            # Innovation variance sigma^2 (1 - phi^2) gives marginal std sigma
            innovations = self.rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), length)
            values = np.empty(length)
            values[0] = self.rng.normal(0.0, sigma)
            for i in range(1, length):
                values[i] = phi * values[i - 1] + innovations[i]
            return values

        # garch_like: GARCH(1,1) recursion, kept stationary (alpha+beta < 1)
        alpha = self.rng.uniform(0.05, 0.25)
        beta = self.rng.uniform(0.5, 0.9)
        persistence = alpha + beta
        if persistence > 0.98:
            alpha *= 0.98 / persistence
            beta *= 0.98 / persistence
        omega = sigma**2 * (1.0 - alpha - beta)
        z = self.rng.normal(0.0, 1.0, length)
        values = np.empty(length)
        h = sigma**2
        for i in range(length):
            values[i] = np.sqrt(h) * z[i]
            h = omega + alpha * values[i] ** 2 + beta * h
        return values

    def _compose(self, length: int) -> np.ndarray:
        """Draw one random T/S/I configuration and compose a series."""
        trend = self._sample_trend(length)
        season = self._sample_seasonality(length)

        multiplicative = season.any() and (
            self.rng.uniform() < self.multiplicative_prob
        )
        if multiplicative:
            base = trend
            low = base.min()
            if low <= 0:  # shift so the base level stays positive
                base = base - low + 0.1 * (np.abs(base).max() + 1.0)
            # Cap the relative seasonal swing so the factor stays positive
            relative = season / max(1.0, 1.25 * np.abs(season).max())
            structural = base * (1.0 + relative)
        else:
            structural = trend + season

        signal_std = float(structural.std()) if length > 1 else 0.0
        if signal_std < 1e-9:
            signal_std = 1.0  # structureless draw: noise gets an absolute scale
        noise_fraction = self._log_uniform(*self.noise_scale_range)
        irregular = self._sample_irregular(length, noise_fraction * signal_std)

        scale = self._log_uniform(*self.scale_range)
        return scale * (structural + irregular)

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single TSI-composed time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        for _ in range(_MAX_RETRIES):
            values = self._compose(length)
            if not np.all(np.isfinite(values)):
                continue
            if np.abs(values).max() >= _MAX_ABS:
                continue
            if length > 1 and values.std() <= _MIN_STD:
                continue
            return values
        # Retry budget exhausted (pathological ranges): fall back to noise
        return self.rng.normal(0.0, 1.0, length)
