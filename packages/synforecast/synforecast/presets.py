"""Pre-configured generator pools for balanced time series generation."""

from __future__ import annotations

from typing import Any

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
from synforecast.generators.levy_process import LevyProcessGenerator
from synforecast.generators.regime_switching import RegimeSwitchingGenerator
from synforecast.generators.sarima import SARIMAGenerator
from synforecast.generators.vital_signs import VitalSignsGenerator


def balanced_pool(
    min_length: int = 200,
    max_length: int = 200,
    freq: str | int = "D",
    seed: int | None = 42,
    **base_kwargs: Any,
) -> list[BaseGenerator]:
    """Create a balanced pool of generators covering diverse temporal behaviors.

    Returns 42 pre-configured generator instances across 15 behavioral niches,
    with allocation proportional to each generator's behavioral range. This
    avoids the implicit bias toward financial processes that occurs when using
    all generators equally.

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

    Args:
        min_length: Minimum time series length for all generators.
        max_length: Maximum time series length for all generators.
        freq: Frequency for all generators, as a pandas offset alias or integer.
        seed: Base random seed. Each generator gets seed + i for reproducibility.
            Set to None for random seeds.
        **base_kwargs: Additional keyword arguments passed to all generators
            (e.g., engine, id_col, time_col, target_col).

    Returns:
        List of 42 BaseGenerator instances ready for use with SynSet.

    Examples:
        >>> from synforecast import SynSet, balanced_pool
        >>> dataset = SynSet(balanced_pool(min_length=100, max_length=100, freq="D"))
        >>> df = dataset.generate(n_series_per_generator=1)
    """
    base = {"min_length": min_length, "max_length": max_length, "freq": freq}
    base.update(base_kwargs)

    def _seed(i: int) -> int | None:
        return seed + i if seed is not None else None

    generators: list[BaseGenerator] = [
        # --- SARIMA: 5 variants covering stationary, integrated, seasonal ---
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
            seasonal_period=12,
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
            seasonal_period=12,
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
        # --- ETS: 4 combos spanning the error/trend/season taxonomy ---
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
        ),
        # Multiplicative Holt-Winters
        ETSGenerator(
            **base,
            seed=_seed(8),
            error_type="mul",
            trend_type="mul",
            seasonal_type="mul",
            level=100.0,
        ),
        # --- Fractional Brownian Motion: 3 Hurst regimes ---
        # Mean-reverting (anti-persistent)
        FractionalBrownianMotionGenerator(**base, seed=_seed(9), hurst=0.2),
        # Standard Brownian motion
        FractionalBrownianMotionGenerator(**base, seed=_seed(10), hurst=0.5),
        # Trending (persistent, long memory)
        FractionalBrownianMotionGenerator(**base, seed=_seed(11), hurst=0.85),
        # --- Regime Switching: 2 variants ---
        # 2-regime (e.g., expansion/contraction)
        RegimeSwitchingGenerator(**base, seed=_seed(12), n_regimes=2),
        # 3-regime with distinct dynamics
        RegimeSwitchingGenerator(**base, seed=_seed(13), n_regimes=3),
        # --- GARCH: 2 persistence levels ---
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
        # --- Cyclic: 2 regularity levels ---
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
        # --- Intermittent Demand: 3 sparsity patterns ---
        IntermittentDemandGenerator(
            **base, seed=_seed(18), intermittent_pattern="random"
        ),
        IntermittentDemandGenerator(
            **base, seed=_seed(19), intermittent_pattern="clustered"
        ),
        IntermittentDemandGenerator(
            **base, seed=_seed(20), intermittent_pattern="seasonal"
        ),
        # --- Energy Load: 2 consumption profiles ---
        EnergyLoadGenerator(**base, seed=_seed(21), load_type="residential"),
        EnergyLoadGenerator(**base, seed=_seed(22), load_type="industrial"),
        # --- IoT Sensor: 3 health states ---
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
        # --- Vital Signs: 3 patient types ---
        VitalSignsGenerator(**base, seed=_seed(26), patient_type="healthy"),
        VitalSignsGenerator(**base, seed=_seed(27), patient_type="cardiac"),
        VitalSignsGenerator(**base, seed=_seed(28), patient_type="sepsis"),
        # --- Gaussian Process: 4 kernel variants ---
        # RBF — infinitely smooth
        GaussianProcessGenerator(**base, seed=_seed(29), kernel="rbf"),
        # Matern 0.5 — rough (exponential correlation)
        GaussianProcessGenerator(**base, seed=_seed(30), kernel="matern_0.5"),
        # Matern 2.5 — moderately smooth
        GaussianProcessGenerator(**base, seed=_seed(31), kernel="matern_2.5"),
        # Periodic — smooth exact periodicity
        GaussianProcessGenerator(
            **base, seed=_seed(32), kernel="periodic", period=50.0
        ),
        # --- Chaotic System: 3 deterministic systems ---
        ChaoticSystemGenerator(**base, seed=_seed(33), system="lorenz"),
        ChaoticSystemGenerator(**base, seed=_seed(34), system="logistic"),
        ChaoticSystemGenerator(**base, seed=_seed(35), system="mackey_glass"),
        # --- INAR: 2 innovation types ---
        # Poisson innovations
        INARGenerator(**base, seed=_seed(36), innovation_type="poisson"),
        # Negative binomial innovations (overdispersed)
        INARGenerator(**base, seed=_seed(37), innovation_type="negative_binomial"),
        # --- Bounded Process: 2 models ---
        # Beta-AR — bounded with mean-reversion
        BoundedProcessGenerator(**base, seed=_seed(38), model="beta_ar"),
        # Logit-normal — bounded random walk
        BoundedProcessGenerator(**base, seed=_seed(39), model="logit_normal"),
        # --- Levy Process: 2 stability levels ---
        # Moderate heavy tails (alpha=1.5)
        LevyProcessGenerator(**base, seed=_seed(40), alpha=1.5),
        # Extreme heavy tails (alpha=1.0, Cauchy-like)
        LevyProcessGenerator(**base, seed=_seed(41), alpha=1.0),
    ]

    return generators
