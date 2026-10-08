"""Pre-configured generator pools for balanced time series generation."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd
from pandas.tseries import offsets as _offsets

from synforecast.base import BaseGenerator
from synforecast.generators.bounded_process import BoundedProcessGenerator
from synforecast.generators.chaotic_system import ChaoticSystemGenerator
from synforecast.generators.cyclic import CyclicGenerator
from synforecast.generators.energy_load import EnergyLoadGenerator
from synforecast.generators.ets import ETSGenerator
from synforecast.generators.fractional_brownian_motion import (
    FractionalBrownianMotionGenerator,
)
from synforecast.generators.garch import GARCHGenerator
from synforecast.generators.gaussian_process import GaussianProcessGenerator
from synforecast.generators.inar import INARGenerator
from synforecast.generators.intermittent_demand import IntermittentDemandGenerator
from synforecast.generators.iot_sensor import IoTSensorGenerator
from synforecast.generators.kernel_synth import KernelSynthGenerator
from synforecast.generators.levy_process import LevyProcessGenerator
from synforecast.generators.mar import MARGenerator
from synforecast.generators.regime_switching import RegimeSwitchingGenerator
from synforecast.generators.sarima import SARIMAGenerator
from synforecast.generators.seasonal import SeasonalGenerator
from synforecast.generators.tcm import TCMGenerator
from synforecast.generators.tsi import TSIGenerator
from synforecast.generators.vital_signs import VitalSignsGenerator

# Steps per unit of the next-larger calendar cycle for each pandas offset
# family. Matching on offset classes rather than alias strings keeps the lookup
# stable across pandas versions, whose rule codes differ ("H"/"h", "T"/"min",
# "A-DEC"/"YE-DEC").
_CYCLE_STEPS: list[tuple[tuple[type[_offsets.BaseOffset], ...], int]] = [
    ((_offsets.Second,), 60),  # seconds per minute
    ((_offsets.Minute,), 60),  # minutes per hour
    ((_offsets.Hour,), 24),  # hours per day
    ((_offsets.Day,), 7),  # days per week
    ((_offsets.BusinessDay, _offsets.CustomBusinessDay), 5),  # per week
    ((_offsets.Week,), 52),  # weeks per year
    ((_offsets.SemiMonthEnd, _offsets.SemiMonthBegin), 24),  # per year
    (
        (
            _offsets.MonthEnd,
            _offsets.MonthBegin,
            _offsets.BusinessMonthEnd,
            _offsets.BusinessMonthBegin,
            _offsets.CustomBusinessMonthEnd,
            _offsets.CustomBusinessMonthBegin,
        ),
        12,  # months per year
    ),
    (
        (
            _offsets.QuarterEnd,
            _offsets.QuarterBegin,
            _offsets.BQuarterEnd,
            _offsets.BQuarterBegin,
        ),
        4,  # quarters per year
    ),
    (
        (_offsets.YearEnd, _offsets.YearBegin, _offsets.BYearEnd, _offsets.BYearBegin),
        1,  # no sub-annual cycle
    ),
]

_DEFAULT_SEASONAL_PERIOD = 12


def _seasonal_period_from_freq(
    freq: str | int, default: int = _DEFAULT_SEASONAL_PERIOD
) -> int:
    """Conventional seasonal period, in time steps, for a frequency.

    The period is the number of steps in the next-larger calendar cycle:
    seconds per minute, minutes per hour, hours per day, days per week, and
    weeks, months or quarters per year. Multiples divide the unit period, so
    ``"15min"`` gives 4, ``"2h"`` gives 12 and ``"3MS"`` gives 4. Yearly data
    has no sub-annual cycle and gives 1.

    Args:
        freq: Pandas offset alias or integer step.
        default: Period for integer frequencies and offsets without a calendar
            convention (e.g. milliseconds, business hours).

    Returns:
        Seasonal period, at least 1.
    """
    if isinstance(freq, int):
        return default
    offset = pd.tseries.frequencies.to_offset(freq)
    for classes, unit_period in _CYCLE_STEPS:
        if isinstance(offset, classes):
            return max(1, round(unit_period / offset.n))
    return default


def interpretable_pool(
    min_length: int = 200,
    max_length: int = 200,
    freq: str | int = "D",
    seed: int | None = 42,
    seasonal_period: int | None = None,
    **base_kwargs: Any,
) -> list[BaseGenerator]:
    """Create a pool of interpretable single-mechanism generators.

    Returns 42 pre-configured generator instances across 15 behavioral niches,
    with allocation proportional to each generator's behavioral range, so no
    domain dominates. Every instance is one named data-generating process;
    the meta-generators that randomize their own composition per series are
    left to :func:`pretraining_pool`. Until version 0.2 this function was
    called ``balanced_pool``; that name remains available as a deprecated alias.

    The list is ordered round-robin across the niches (one variant of every
    niche, then second variants, and so on), so any prefix spans as many
    distinct behaviors as possible: the first 15 entries cover all 15 niches.
    Consumers that use only the first k generators — such as
    ``generate_series`` with ``n_series < 42`` — therefore still get a
    behaviorally diverse panel.

    Behavioral niches covered:
        - ARMA + seasonality (SARIMA, 5 variants)
        - Exponential smoothing (ETS, 4 variants)
        - Long-range memory (Fractional Brownian Motion, 3 Hurst regimes)
        - Structural breaks (Regime Switching, 2 variants)
        - Volatility clustering (GARCH, 2 persistence levels)
        - Irregular cycles (Cyclic, 2 regularity levels)
        - Sparse/intermittent (Intermittent Demand, 3 patterns)
        - Multi-seasonal (Energy Load, 2 load types)
        - Sensor artifacts (IoT Sensor, 3 health states)
        - Physiological (Vital Signs, 3 patient types)
        - Smooth/rough functions (Gaussian Process, 4 kernels)
        - Deterministic chaos (Chaotic System, 3 systems)
        - Count time series (INAR, 2 innovation types)
        - Bounded/proportion data (Bounded Process, 2 models)
        - Heavy-tailed processes (Levy Process, 2 stability levels)

    The seasonal variants (seasonal SARIMA, Holt-Winters ETS, seasonal
    intermittent demand and the periodic Gaussian process) share one seasonal
    period. Unless ``seasonal_period`` is given it is derived from ``freq``
    as the number of steps in the next-larger calendar cycle: hourly data
    gets 24, daily 7, business-daily 5, weekly 52, monthly 12, quarterly 4,
    and multiples divide accordingly (``"15min"`` gives 4, ``"2h"`` gives 12).
    Integer frequencies and offsets without a calendar convention fall back
    to 12. Yearly data has no sub-annual cycle, so its period is 1 and the
    seasonal variants reduce to their non-seasonal counterparts.

    Args:
        min_length: Minimum time series length for all generators.
        max_length: Maximum time series length for all generators.
        freq: Frequency for all generators, as a pandas offset alias or integer.
        seed: Base random seed. Each generator gets seed + i for reproducibility.
            Set to None for random seeds.
        seasonal_period: Seasonal period, in time steps, for the seasonal
            variants. Defaults to None, which derives it from ``freq``.
        **base_kwargs: Additional keyword arguments passed to all generators
            (e.g., engine, id_col, time_col, target_col).

    Returns:
        List of 42 BaseGenerator instances ready for use with SynSet.

    Examples:
        >>> from synforecast import SynSet, interpretable_pool
        >>> dataset = SynSet(
        ...     interpretable_pool(min_length=100, max_length=100, freq="D")
        ... )
        >>> df = dataset.generate(n_series_per_generator=1)
    """
    base: dict[str, Any] = {
        "min_length": min_length,
        "max_length": max_length,
        "freq": freq,
    }
    base.update(base_kwargs)

    if seasonal_period is None:
        season = _seasonal_period_from_freq(freq)
    elif seasonal_period < 1:
        raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
    else:
        season = seasonal_period

    def _seed(i: int) -> int | None:
        return seed + i if seed is not None else None

    # Generators are declared grouped by behavioral niche and returned
    # interleaved round-robin across niches, so any prefix of the pool spans
    # as many distinct behaviors as possible. generate_series draws from the
    # front when n_series < len(pool), so the interleaving is what makes a
    # small panel diverse. Seed offsets are fixed per variant at declaration.
    niches: list[list[BaseGenerator]] = [
        # --- SARIMA: 5 variants covering stationary, integrated, seasonal ---
        [
            # Stationary AR(1)
            SARIMAGenerator(
                **base,
                seed=_seed(0),
                p=1,
                d=0,
                q=0,
                P=0,
                D=0,
                Q=0,
                seasonal_period=1,
            ),
            # Integrated MA — random walk with smoothing
            SARIMAGenerator(
                **base,
                seed=_seed(1),
                p=0,
                d=1,
                q=1,
                P=0,
                D=0,
                Q=0,
                seasonal_period=1,
            ),
            # Seasonal AR
            SARIMAGenerator(
                **base,
                seed=_seed(2),
                p=1,
                d=0,
                q=0,
                P=1,
                D=0,
                Q=0,
                seasonal_period=season,
            ),
            # Airline model — classic seasonal integrated
            SARIMAGenerator(
                **base,
                seed=_seed(3),
                p=0,
                d=1,
                q=1,
                P=0,
                D=1,
                Q=1,
                seasonal_period=season,
            ),
            # Stationary ARMA(2,2) — complex short-range dynamics
            SARIMAGenerator(
                **base,
                seed=_seed(4),
                p=2,
                d=0,
                q=2,
                P=0,
                D=0,
                Q=0,
                seasonal_period=1,
            ),
        ],
        # --- ETS: 4 combos spanning the error/trend/season taxonomy ---
        [
            # Simple exponential smoothing (no trend, no season)
            ETSGenerator(
                **base,
                seed=_seed(5),
                error_type="add",
                trend_type=None,
                seasonal_type=None,
            ),
            # Damped trend, no season
            ETSGenerator(
                **base,
                seed=_seed(6),
                error_type="add",
                trend_type="add",
                seasonal_type=None,
                damped=True,
            ),
            # Additive Holt-Winters
            ETSGenerator(
                **base,
                seed=_seed(7),
                error_type="add",
                trend_type="add",
                seasonal_type="add",
                seasonal_period=season,
            ),
            # Multiplicative Holt-Winters
            ETSGenerator(
                **base,
                seed=_seed(8),
                error_type="mul",
                trend_type="mul",
                seasonal_type="mul",
                seasonal_period=season,
                level=100.0,
            ),
        ],
        # --- Fractional Brownian Motion: 3 Hurst regimes ---
        [
            # Mean-reverting (anti-persistent)
            FractionalBrownianMotionGenerator(**base, seed=_seed(9), hurst=0.2),
            # Standard Brownian motion
            FractionalBrownianMotionGenerator(**base, seed=_seed(10), hurst=0.5),
            # Trending (persistent, long memory)
            FractionalBrownianMotionGenerator(**base, seed=_seed(11), hurst=0.85),
        ],
        # --- Regime Switching: 2 variants ---
        [
            # 2-regime (e.g., expansion/contraction)
            RegimeSwitchingGenerator(**base, seed=_seed(12), n_regimes=2),
            # 3-regime with distinct dynamics
            RegimeSwitchingGenerator(**base, seed=_seed(13), n_regimes=3),
        ],
        # --- GARCH: 2 persistence levels ---
        [
            # Low persistence — mild volatility clustering
            GARCHGenerator(
                **base,
                seed=_seed(14),
                omega=0.1,
                alpha=[0.1],
                beta=[0.3],
            ),
            # High persistence — strong volatility clustering
            GARCHGenerator(
                **base,
                seed=_seed(15),
                omega=0.01,
                alpha=[0.15],
                beta=[0.8],
            ),
        ],
        # --- Cyclic: 2 regularity levels ---
        [
            # Nearly regular single cycle
            CyclicGenerator(
                **base,
                seed=_seed(16),
                num_cycles=1,
                cycle_period_std=2.0,
            ),
            # Highly irregular multi-cycle overlay
            CyclicGenerator(
                **base,
                seed=_seed(17),
                num_cycles=4,
                cycle_period_std=10.0,
            ),
        ],
        # --- Intermittent Demand: 3 sparsity patterns ---
        [
            IntermittentDemandGenerator(
                **base, seed=_seed(18), intermittent_pattern="random"
            ),
            IntermittentDemandGenerator(
                **base, seed=_seed(19), intermittent_pattern="clustered"
            ),
            IntermittentDemandGenerator(
                **base,
                seed=_seed(20),
                intermittent_pattern="seasonal",
                seasonal_period=season,
            ),
        ],
        # --- Energy Load: 2 consumption profiles ---
        [
            EnergyLoadGenerator(**base, seed=_seed(21), load_type="residential"),
            EnergyLoadGenerator(**base, seed=_seed(22), load_type="industrial"),
        ],
        # --- IoT Sensor: 3 health states ---
        [
            # Healthy sensor
            IoTSensorGenerator(
                **base,
                seed=_seed(23),
                failure_probability=0.0,
                drift_rate=0.0,
            ),
            # Degrading sensor
            IoTSensorGenerator(
                **base,
                seed=_seed(24),
                drift_rate=0.01,
                battery_degradation_rate=0.001,
            ),
            # Failing sensor
            IoTSensorGenerator(
                **base,
                seed=_seed(25),
                failure_probability=0.05,
                failure_type="intermittent",
            ),
        ],
        # --- Vital Signs: 3 patient types ---
        [
            VitalSignsGenerator(**base, seed=_seed(26), patient_type="healthy"),
            VitalSignsGenerator(**base, seed=_seed(27), patient_type="cardiac"),
            VitalSignsGenerator(**base, seed=_seed(28), patient_type="sepsis"),
        ],
        # --- Gaussian Process: 4 kernel variants ---
        [
            # RBF — infinitely smooth
            GaussianProcessGenerator(**base, seed=_seed(29), kernel="rbf"),
            # Matern 0.5 — rough (exponential correlation)
            GaussianProcessGenerator(**base, seed=_seed(30), kernel="matern_0.5"),
            # Matern 2.5 — moderately smooth
            GaussianProcessGenerator(**base, seed=_seed(31), kernel="matern_2.5"),
            # Periodic — smooth exact periodicity
            GaussianProcessGenerator(
                **base, seed=_seed(32), kernel="periodic", period=float(season)
            ),
        ],
        # --- Chaotic System: 3 deterministic systems ---
        [
            ChaoticSystemGenerator(**base, seed=_seed(33), system="lorenz"),
            ChaoticSystemGenerator(**base, seed=_seed(34), system="logistic"),
            ChaoticSystemGenerator(**base, seed=_seed(35), system="mackey_glass"),
        ],
        # --- INAR: 2 innovation types ---
        [
            # Poisson innovations
            INARGenerator(**base, seed=_seed(36), innovation_type="poisson"),
            # Negative binomial innovations (overdispersed)
            INARGenerator(**base, seed=_seed(37), innovation_type="negative_binomial"),
        ],
        # --- Bounded Process: 2 models ---
        [
            # Beta-AR — bounded with mean-reversion
            BoundedProcessGenerator(**base, seed=_seed(38), model="beta_ar"),
            # Logit-normal — bounded random walk
            BoundedProcessGenerator(**base, seed=_seed(39), model="logit_normal"),
        ],
        # --- Levy Process: 2 stability levels ---
        [
            # Moderate heavy tails (alpha=1.5)
            LevyProcessGenerator(**base, seed=_seed(40), alpha=1.5),
            # Extreme heavy tails (alpha=1.0, Cauchy-like)
            LevyProcessGenerator(**base, seed=_seed(41), alpha=1.0),
        ],
    ]

    # Round-robin across niches: first variant of every niche, then second
    # variants, and so on.
    generators: list[BaseGenerator] = [
        group[variant]
        for variant in range(max(len(group) for group in niches))
        for group in niches
        if variant < len(group)
    ]

    return generators


def balanced_pool(*args: Any, **kwargs: Any) -> list[BaseGenerator]:
    """Deprecated alias of :func:`interpretable_pool`.

    The pool is balanced across behavioral niches, but so is
    :func:`pretraining_pool`; the distinguishing property is that every
    instance is an interpretable single-mechanism process. Use
    :func:`interpretable_pool`. This alias warns and will be removed in a
    future release.
    """
    warnings.warn(
        "balanced_pool is deprecated and will be removed in a future release; "
        "use interpretable_pool, which returns the same generators.",
        DeprecationWarning,
        stacklevel=2,
    )
    return interpretable_pool(*args, **kwargs)


def pretraining_pool(
    min_length: int = 256,
    max_length: int = 1024,
    freq: str | int = "D",
    seed: int | None = 42,
    include_interpretable: bool = True,
    n_meta_variants: int = 3,
    seasonal_period: int | None = None,
    include_seasonal: bool = False,
    **base_kwargs: Any,
) -> list[BaseGenerator]:
    """Create a breadth-maximizing pool for foundation-model pretraining.

    This is the pretraining-oriented counterpart to :func:`interpretable_pool`.
    It adds the diversity-targeted *meta-generators* that ``interpretable_pool``
    deliberately excludes — ``TSIGenerator`` (randomized trend/seasonal/
    irregular composition), ``TCMGenerator`` (random temporal causal graphs),
    ``KernelSynthGenerator`` (samples from randomly composed GP kernels), and
    ``MARGenerator`` (GRATIS-style mixtures of autoregressive components).
    Each resamples a fresh configuration per series, so a handful of instances
    spans a very wide distribution. By default it also includes the full
    ``interpretable_pool`` so the corpus carries interpretable single-mechanism
    behaviors alongside the meta-generators.

    Unlike ``interpretable_pool``, the default length range is wide
    (256-1024 steps), matching the longer contexts typical of pretraining.

    Args:
        min_length: Minimum series length for all generators.
        max_length: Maximum series length for all generators.
        freq: Frequency for all generators, as a pandas offset alias or integer.
        seed: Base random seed. Each generator gets a distinct offset. Set to
            None for random seeds.
        include_interpretable: When True (default), prepend the full
            :func:`interpretable_pool`; when False, return only the
            meta-generators (a purely procedural pretraining corpus), plus the
            seasonal instances if ``include_seasonal`` is also set. The former
            keyword ``include_balanced`` is still accepted with a
            ``DeprecationWarning``.
        n_meta_variants: Number of independently-seeded instances of each
            meta-generator (default 3). More instances give the meta-generators
            a larger share when series are spread evenly across the pool, as in
            :func:`generate_series`.
        seasonal_period: Seasonal period for the seasonal variants of the
            included :func:`interpretable_pool` and :func:`seasonal_pool`.
            Defaults to None, which derives it from ``freq``.
        include_seasonal: When True, append the eight :func:`seasonal_pool`
            instances (strong seasonality on a moving level) at the same
            period. Skipped with a warning when the period is below 2, as for
            yearly data. Recommended for regularly seasonal quarterly and
            monthly targets, where it raised feature-space coverage on every
            panel tested that had a seasonal gap to fill; it does not help
            intermittent monthly data, where displacing pool breadth cost
            about three points. Leave it off for weekly, daily, hourly, and
            intermittent targets. Default False.
        **base_kwargs: Additional keyword arguments passed to all generators
            (e.g., engine, id_col, time_col, target_col).

    Returns:
        List of BaseGenerator instances ready for use with SynSet.

    Examples:
        >>> from synforecast import SynSet, pretraining_pool
        >>> pool = pretraining_pool(min_length=512, max_length=512, freq="h")
        >>> df = SynSet(pool).generate(n_series_per_generator=1)

        >>> # Purely procedural corpus (meta-generators only)
        >>> meta = pretraining_pool(include_interpretable=False)

        >>> # Add the strongly seasonal instances for monthly targets
        >>> seasonal = pretraining_pool(freq="MS", include_seasonal=True)
    """
    if "include_balanced" in base_kwargs:
        warnings.warn(
            "include_balanced is deprecated and will be removed in a future "
            "release; use include_interpretable.",
            DeprecationWarning,
            stacklevel=2,
        )
        include_interpretable = base_kwargs.pop("include_balanced")
    if n_meta_variants < 1:
        raise ValueError("n_meta_variants must be >= 1")
    if seasonal_period is not None and seasonal_period < 1:
        raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")

    base: dict[str, Any] = {
        "min_length": min_length,
        "max_length": max_length,
        "freq": freq,
    }
    base.update(base_kwargs)

    def _seed(i: int) -> int | None:
        return seed + i if seed is not None else None

    # Meta-generator seeds are offset well past interpretable_pool's 0..41 range so
    # the two sets never collide when combined.
    meta_classes = (TSIGenerator, TCMGenerator, KernelSynthGenerator, MARGenerator)
    meta: list[BaseGenerator] = [
        cls(**base, seed=_seed(1000 + 100 * family + variant))
        for family, cls in enumerate(meta_classes)
        for variant in range(n_meta_variants)
    ]

    seasonal: list[BaseGenerator] = []
    if include_seasonal:
        period = (
            _seasonal_period_from_freq(freq)
            if seasonal_period is None
            else seasonal_period
        )
        if period < 2:
            source = (
                f"freq={freq!r} yields a seasonal period of {period}"
                if seasonal_period is None
                else f"seasonal_period={seasonal_period} has no seasonal cycle"
            )
            hint = (
                " Pass seasonal_period explicitly to add it anyway."
                if seasonal_period is None
                else ""
            )
            warnings.warn(
                f"include_seasonal=True was ignored: {source}, and "
                f"seasonal_pool needs a period of at least 2.{hint}",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Seed offsets 2000..2007 sit past both the 0..41 and 1000.. ranges.
            seasonal = seasonal_pool(
                min_length=min_length,
                max_length=max_length,
                freq=freq,
                seed=_seed(2000),
                seasonal_period=period,
                **base_kwargs,
            )

    if not include_interpretable:
        return meta + seasonal

    balanced = interpretable_pool(
        min_length=min_length,
        max_length=max_length,
        freq=freq,
        seed=seed,
        seasonal_period=seasonal_period,
        **base_kwargs,
    )
    return balanced + meta + seasonal


def _seasonal_factors(period: int, sharpness: float) -> list[float]:
    """Unit-mean multiplicative seasonal factors exp(sharpness * cos)."""
    factors = [
        math.exp(sharpness * math.cos(2 * math.pi * k / period)) for k in range(period)
    ]
    mean = sum(factors) / period
    return [f / mean for f in factors]


def seasonal_pool(
    min_length: int = 200,
    max_length: int = 200,
    *,
    freq: str | int,
    seed: int | None = 42,
    seasonal_period: int | None = None,
    **base_kwargs: Any,
) -> list[BaseGenerator]:
    """Create an opt-in pool of strongly seasonal series on a moving level.

    Feature-space coverage benchmarks found that real quarterly and monthly
    panels with a pronounced, regular seasonal cycle whose swing grows with
    the level (tourism demand, hospital patient counts) have few near
    neighbours in :func:`interpretable_pool`. This pool adds eight configured
    instances of existing generators that fill that region: ``TSIGenerator``
    with one harmonic at the seasonal period under moderate to heavy noise,
    including persistent AR(1) and heavy-tailed variants and a weakly seasonal
    variant; noisy multiplicative and additive Holt-Winters ``ETSGenerator``
    instances with fast level adaptation; and ``SeasonalGenerator`` sines with
    level or slope breaks.

    It is meant to be **added on top of** ``interpretable_pool`` or
    ``pretraining_pool``, never to replace them or any share of them. On
    tourism, hospital, and M1 monthly panels, adding it raised the fraction of
    real series with a close synthetic neighbour by 8 to 32 points; on M4
    panels, which have no such gap, it was neutral; and on intermittent
    car-parts sales it slightly lowered coverage when it displaced the
    pretraining pool's share. As a standalone corpus it covers far fewer real
    series than the pools. It was evaluated only at quarterly and monthly
    periods; other periods are supported but untested, and its effect on
    forecasting accuracy has not been measured.

    Args:
        min_length: Minimum series length for all generators.
        max_length: Maximum series length for all generators.
        freq: Frequency for all generators, as a pandas offset alias or
            integer. Required, because the pool is meant to be combined with
            another and its period derives from ``freq``: pass the same value
            to :func:`interpretable_pool` or :func:`pretraining_pool`, whose
            default is ``"D"``. Evaluated at ``"QS"`` and ``"MS"``.
        seed: Base random seed; generator ``i`` receives ``seed + i``. Set to
            None for random seeds.
        seasonal_period: Seasonal period in time steps. Defaults to None,
            which derives it from ``freq`` as in :func:`interpretable_pool`; it
            must be at least 2, so yearly frequencies need an explicit value.
        **base_kwargs: Additional keyword arguments passed to all generators
            (e.g., engine, id_col, time_col, target_col). Keys the pool sets
            per generator, such as ``alias`` or ``level``, raise ValueError.

    Returns:
        List of 8 BaseGenerator instances ready for use with SynSet.

    Examples:
        >>> from synforecast import SynSet, interpretable_pool, seasonal_pool
        >>> pool = interpretable_pool(freq="MS") + seasonal_pool(freq="MS")
        >>> df = SynSet(pool).generate(n_series_per_generator=1)
    """
    base: dict[str, Any] = {
        "min_length": min_length,
        "max_length": max_length,
        "freq": freq,
    }
    base.update(base_kwargs)
    period = (
        _seasonal_period_from_freq(freq) if seasonal_period is None else seasonal_period
    )
    if period < 2:
        raise ValueError(
            "seasonal_pool needs a seasonal period of at least 2; pass "
            "seasonal_period explicitly for frequencies without a sub-annual cycle"
        )

    def _seed(i: int) -> int | None:
        return seed + i if seed is not None else None

    def _make(
        cls: type[BaseGenerator], **settings: Any
    ) -> tuple[type[BaseGenerator], dict[str, Any]]:
        return cls, settings

    tsi: dict[str, Any] = {
        "seasonal_periods": [float(period)],
        "n_seasonal_range": (1, 1),
        "seasonal_amplitude_range": (0.5, 3.0),
        "harmonics_prob": 0.5,
        "amplitude_modulation_prob": 0.5,
        "trend_types": [
            "linear",
            "piecewise_linear",
            "damped",
            "logistic",
            "exponential",
        ],
        "trend_slope_range": (-3.0, 6.0),
        "multiplicative_prob": 1.0,
        "noise_scale_range": (0.3, 3.0),
        "irregular_types": ["gaussian", "ar1", "student_t"],
        "ar1_phi_range": (0.5, 0.95),
        "tail_df_range": (3.0, 8.0),
        "level_range": (5.0, 10.0),
        "scale_range": (1.0, 1.0),
    }
    ets: dict[str, Any] = {
        "seasonal_period": period,
        "level": 100.0,
        "alpha": 0.3,
        "beta": 0.02,
        "phi": 0.9,
    }
    specs = [
        _make(TSIGenerator, seed=_seed(0), alias="tsi_moderate_multiplicative", **tsi),
        _make(
            TSIGenerator,
            seed=_seed(1),
            alias="tsi_moderate_additive",
            **{**tsi, "multiplicative_prob": 0.0, "trend_slope_range": (-3.0, 3.0)},
        ),
        _make(
            TSIGenerator,
            seed=_seed(2),
            alias="tsi_persistent",
            **{
                **tsi,
                "multiplicative_prob": 0.0,
                "trend_slope_range": (-3.0, 3.0),
                "noise_scale_range": (1.0, 4.0),
                "irregular_types": ["ar1"],
                "ar1_phi_range": (0.8, 0.97),
            },
        ),
        _make(
            TSIGenerator,
            seed=_seed(3),
            alias="tsi_weak_seasonal",
            **{
                **tsi,
                "multiplicative_prob": 0.3,
                "seasonal_amplitude_range": (0.2, 1.0),
                "noise_scale_range": (1.0, 6.0),
            },
        ),
        _make(
            ETSGenerator,
            seed=_seed(4),
            alias="ets_MAdM_noisy",
            error_type="mul",
            trend_type="add",
            seasonal_type="mul",
            trend=0.3,
            damped=True,
            gamma=0.1,
            noise_std=0.15,
            seasonal=_seasonal_factors(period, 0.5),
            **ets,
        ),
        _make(
            ETSGenerator,
            seed=_seed(5),
            alias="ets_AAdA_noisy",
            error_type="add",
            trend_type="add",
            seasonal_type="add",
            trend=0.2,
            damped=True,
            gamma=0.1,
            noise_std=10.0,
            seasonal=[20 * math.cos(2 * math.pi * k / period) for k in range(period)],
            **ets,
        ),
        _make(
            SeasonalGenerator,
            seed=_seed(6),
            alias="seasonal_noisy_level_breaks",
            seasonality_period=period,
            seasonality_amplitude=1.0,
            trend=0.008,
            noise_level=0.6,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="level",
            changepoint_level_changes=[0.8, -0.7],
        ),
        _make(
            SeasonalGenerator,
            seed=_seed(7),
            alias="seasonal_noisy_trend_breaks",
            seasonality_period=period,
            seasonality_amplitude=1.0,
            noise_level=0.6,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="trend",
            changepoint_trend_changes=[0.01, -0.015],
        ),
    ]
    clash = sorted(base_kwargs.keys() & set().union(*(spec[1] for spec in specs)))
    if clash:
        raise ValueError(
            f"seasonal_pool sets {clash} per generator; do not pass them as "
            "keyword arguments"
        )
    return [cls(**base, **settings) for cls, settings in specs]
