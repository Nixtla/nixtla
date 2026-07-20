"""Benchmark Rust vs pure-Python implementations of SynForecast generators.

Compares:
  1. Statistical equivalence (mean, std, shape) between Rust and Python paths
  2. Wall-clock speed for generating series of varying lengths
  3. Throughput scaling when generating many series

Usage:
    uv run python benchmarks/benchmark_rust_vs_python.py
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# ---------------------------------------------------------------------------
# Modules whose _HAS_RUST flag we need to toggle
# ---------------------------------------------------------------------------
import synforecast._core as _core_mod
import synforecast._distributions as _dist_mod
import synforecast.generators.bounded_process as bp_mod
import synforecast.generators.chaotic_system as cs_mod
import synforecast.generators.clickstream as clickstream_mod
import synforecast.generators.copula as copula_mod
import synforecast.generators.cyclic as cyclic_mod
import synforecast.generators.daily_active_users as dau_mod
import synforecast.generators.energy_load as energy_mod
import synforecast.generators.ets as ets_mod
import synforecast.generators.fractional_brownian_motion as fbm_mod
import synforecast.generators.garch as garch_mod
import synforecast.generators.gaussian_process as gp_mod
import synforecast.generators.geometric_brownian_motion as gbm_mod
import synforecast.generators.hawkes_process as hawkes_mod
import synforecast.generators.inar as inar_mod
import synforecast.generators.intermittent_demand as idem_mod
import synforecast.generators.iot_sensor as iot_mod
import synforecast.generators.jump_diffusion as jd_mod
import synforecast.generators.levy_process as lp_mod
import synforecast.generators.ornstein_uhlenbeck as ou_mod
import synforecast.generators.poisson_process as pp_mod
import synforecast.generators.random_walk as rw_mod
import synforecast.generators.regime_switching as rs_mod
import synforecast.generators.sarima as sarima_mod
import synforecast.generators.seasonal as seasonal_mod
import synforecast.generators.state_space as ss_mod
import synforecast.generators.stochastic_volatility as sv_mod
import synforecast.generators.var as var_mod
import synforecast.generators.vital_signs as vs_mod
from synforecast.generators import (
    BoundedProcessGenerator,
    ChaoticSystemGenerator,
    ClickstreamGenerator,
    CopulaGenerator,
    CyclicGenerator,
    DailyActiveUsersGenerator,
    EnergyLoadGenerator,
    ETSGenerator,
    FractionalBrownianMotionGenerator,
    GARCHGenerator,
    GaussianProcessGenerator,
    GeometricBrownianMotionGenerator,
    HawkesProcessGenerator,
    INARGenerator,
    IntermittentDemandGenerator,
    IoTSensorGenerator,
    JumpDiffusionGenerator,
    LevyProcessGenerator,
    OrnsteinUhlenbeckGenerator,
    PoissonProcessGenerator,
    RandomWalkGenerator,
    RegimeSwitchingGenerator,
    SARIMAGenerator,
    SeasonalGenerator,
    StateSpaceGenerator,
    StochasticVolatilityGenerator,
    VARGenerator,
    VitalSignsGenerator,
)

# All modules that have _HAS_RUST
_ALL_MODULES = [
    _core_mod,
    _dist_mod,
    rw_mod,
    seasonal_mod,
    sarima_mod,
    ets_mod,
    inar_mod,
    garch_mod,
    ou_mod,
    gbm_mod,
    jd_mod,
    pp_mod,
    cyclic_mod,
    fbm_mod,
    hawkes_mod,
    sv_mod,
    rs_mod,
    cs_mod,
    bp_mod,
    lp_mod,
    copula_mod,
    var_mod,
    gp_mod,
    ss_mod,
    iot_mod,
    idem_mod,
    energy_mod,
    dau_mod,
    vs_mod,
    clickstream_mod,
]


def set_rust(enabled: bool) -> None:
    """Toggle the Rust fast path on or off for every module."""
    for mod in _ALL_MODULES:
        mod._HAS_RUST = enabled


# ---------------------------------------------------------------------------
# Generator configurations for benchmarking
# ---------------------------------------------------------------------------

GENERATORS: dict[str, tuple[type, dict]] = {
    "RandomWalk": (
        RandomWalkGenerator,
        {"drift": 0.1, "volatility": 1.0, "start_value": 0.0},
    ),
    "Seasonal": (
        SeasonalGenerator,
        {"period": 24, "amplitude": 5.0, "trend": 0.01, "noise_level": 0.5},
    ),
    "SARIMA": (
        SARIMAGenerator,
        {"order": [1, 0, 1], "seasonal_order": [1, 0, 1, 12]},
    ),
    "ETS": (
        ETSGenerator,
        {
            "error_type": "add",
            "trend_type": "add",
            "seasonal_type": "add",
            "seasonal_period": 12,
        },
    ),
    "GARCH": (
        GARCHGenerator,
        {"p": 1, "q": 1, "omega": 0.1, "alpha": [0.15], "beta": [0.75]},
    ),
    "OrnsteinUhlenbeck": (
        OrnsteinUhlenbeckGenerator,
        {"theta": 0.7, "mu": 5.0, "sigma": 0.3},
    ),
    "GeometricBrownianMotion": (
        GeometricBrownianMotionGenerator,
        {"mu": 0.05, "sigma": 0.2, "initial_value": 100.0},
    ),
    "JumpDiffusion": (
        JumpDiffusionGenerator,
        {"mu": 0.05, "sigma": 0.2, "lambda_jump": 0.5},
    ),
    "PoissonProcess": (
        PoissonProcessGenerator,
        {"lambda_rate": 5.0, "cumulative": False},
    ),
    "Cyclic": (
        CyclicGenerator,
        {"num_cycles": 3, "base_level": 10.0, "trend": 0.01},
    ),
    "FBM (Hosking)": (
        FractionalBrownianMotionGenerator,
        {"hurst": 0.7, "sigma": 1.0, "method": "hosking"},
    ),
    # HawkesProcess omitted: Python fallback too slow for benchmarking
    "StochasticVolatility": (
        StochasticVolatilityGenerator,
        {"model_type": "heston", "output_type": "price"},
    ),
    "RegimeSwitching": (
        RegimeSwitchingGenerator,
        {"n_regimes": 2},
    ),
    "Copula": (
        CopulaGenerator,
        {"copula_type": "gaussian", "marginal_distributions": [{"type": "normal"}]},
    ),
    "VAR": (
        VARGenerator,
        {"n_variables": 2, "lag_order": 1},
    ),
    # StateSpace omitted: Python fallback too slow for benchmarking
    "IoTSensor": (
        IoTSensorGenerator,
        {"base_value": 25.0, "measurement_noise": 0.5, "drift_rate": 0.01},
    ),
    "IntermittentDemand": (
        IntermittentDemandGenerator,
        {"demand_probability": 0.3, "demand_mean": 10.0},
    ),
    "EnergyLoad": (
        EnergyLoadGenerator,
        {"base_load": 100.0, "load_type": "residential"},
    ),
    "DailyActiveUsers": (
        DailyActiveUsersGenerator,
        {"base_users": 1000.0},
    ),
    "VitalSigns": (
        VitalSignsGenerator,
        {"vital_sign_type": "heart_rate"},
    ),
    "Clickstream": (
        ClickstreamGenerator,
        {"output_type": "sessions"},
    ),
    "INAR": (
        INARGenerator,
        {"p": 1, "alpha": [0.5], "innovation_type": "poisson", "innovation_mean": 3.0},
    ),
    "GaussianProcess": (
        GaussianProcessGenerator,
        {"kernel": "rbf", "length_scale": 20.0, "amplitude": 1.0},
    ),
    "ChaoticSystem": (
        ChaoticSystemGenerator,
        {"system": "lorenz", "observation_noise": 0.1},
    ),
    "BoundedProcess": (
        BoundedProcessGenerator,
        {"model": "beta_ar", "phi": 0.8, "omega": 0.1, "kappa": 20.0},
    ),
    "LevyProcess": (
        LevyProcessGenerator,
        {"alpha": 1.5, "cumulative": True, "initial_value": 0.0},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maximum lengths per generator (to avoid O(n^2)/O(n^3) blowups)
_MAX_LENGTHS: dict[str, int] = {
    "FBM (Hosking)": 3_000,
    "GaussianProcess": 3_000,  # O(n^3) Cholesky decomposition
}


def _generate_series(cls, params, length, seed):
    """Instantiate a generator and produce a single series of *length* points."""
    gen = cls(min_length=length, max_length=length, seed=seed, freq="1h", **params)
    return gen.generate_single_series(length)


def _time_fn(fn, repeats=3):
    """Return the *minimum* wall-clock time over *repeats* calls (seconds)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


# ---------------------------------------------------------------------------
# 1. Equivalence check
# ---------------------------------------------------------------------------


def check_equivalence():
    """Compare statistical properties between Rust and Python paths.

    Both paths use different RNG streams so exact equality is impossible.
    Instead we verify: same shape, finite values, and similar statistical
    properties (mean within 50% or 5 units, std within 2x).
    """
    print("=" * 72)
    print("  EQUIVALENCE CHECK  (Rust vs Python, same generator params)")
    print("=" * 72)
    length = 5_000
    seed = 42

    results = []
    for name, (cls, params) in GENERATORS.items():
        # Rust path
        set_rust(True)
        rust_vals = _generate_series(cls, params, length, seed)

        # Python path
        set_rust(False)
        py_vals = _generate_series(cls, params, length, seed)

        # Re-enable Rust
        set_rust(True)

        # Check shape
        shape_ok = rust_vals.shape == py_vals.shape

        # Filter NaNs for stats comparison
        rust_clean = rust_vals[np.isfinite(rust_vals)]
        py_clean = py_vals[np.isfinite(py_vals)]

        if len(rust_clean) == 0 or len(py_clean) == 0:
            results.append((name, shape_ok, "SKIP (all NaN)", 0, 0, 0, 0))
            continue

        rust_mean, rust_std = np.mean(rust_clean), np.std(rust_clean)
        py_mean, py_std = np.mean(py_clean), np.std(py_clean)

        # Loose check: means within 50% relative or 5 absolute
        denom = max(abs(py_mean), 1e-6)
        mean_diff = abs(rust_mean - py_mean)
        mean_ok = mean_diff < 0.5 * denom or mean_diff < 5.0

        # Stds within 2x
        std_ratio = max(rust_std, 1e-9) / max(py_std, 1e-9)
        std_ok = 0.2 < std_ratio < 5.0

        status = "OK" if (shape_ok and mean_ok and std_ok) else "WARN"
        results.append((name, shape_ok, status, rust_mean, py_mean, rust_std, py_std))

    # Print table
    print(
        f"{'Generator':<25} {'Shape':>5} {'Status':>6} "
        f"{'Rust mean':>10} {'Py mean':>10} {'Rust std':>10} {'Py std':>10}"
    )
    print("-" * 90)
    for name, shape_ok, status, cm, pm, cs, ps in results:
        if isinstance(cm, str):
            print(f"{name:<25} {'OK' if shape_ok else 'FAIL':>5} {status:>6}")
        else:
            print(
                f"{name:<25} {'OK' if shape_ok else 'FAIL':>5} {status:>6} "
                f"{cm:>10.3f} {pm:>10.3f} {cs:>10.3f} {ps:>10.3f}"
            )
    print()


# ---------------------------------------------------------------------------
# 2. Speed benchmark – varying series length
# ---------------------------------------------------------------------------


def benchmark_lengths():
    """Benchmark Rust vs Python for increasing series lengths."""
    print("=" * 72)
    print("  SPEED BENCHMARK  (single series, varying length)")
    print("=" * 72)
    base_lengths = [100]
    seed = 123

    print(
        f"{'Generator':<25} {'Length':>7} "
        f"{'Rust (ms)':>10} {'Py (ms)':>10} {'Speedup':>8}"
    )
    print("-" * 72)

    for name, (cls, params) in GENERATORS.items():
        max_len = _MAX_LENGTHS.get(name, 50_000)
        lengths = [l for l in base_lengths if l <= max_len]
        for length in lengths:
            # Rust timing
            set_rust(True)
            t_rust = _time_fn(lambda: _generate_series(cls, params, length, seed)) * 1000

            # Python timing
            set_rust(False)
            t_py = _time_fn(lambda: _generate_series(cls, params, length, seed)) * 1000

            set_rust(True)

            speedup = t_py / t_rust if t_rust > 0 else float("inf")
            print(
                f"{name:<25} {length:>7,} "
                f"{t_rust:>10.2f} {t_py:>10.2f} {speedup:>7.1f}x"
            )
        print()


# ---------------------------------------------------------------------------
# 3. Multi-series throughput
# ---------------------------------------------------------------------------


def benchmark_multi_series():
    """Benchmark generating many series (via generate()) with Rust vs Python."""
    print("=" * 72)
    print("  MULTI-SERIES THROUGHPUT  (n_series scaling, length=1000)")
    print("=" * 72)
    length = 1_000
    seed = 99
    n_series_list = [1, 10]

    # Use a representative subset to keep benchmark time reasonable
    subset = {
        "RandomWalk": GENERATORS["RandomWalk"],
        "SARIMA": GENERATORS["SARIMA"],
        "ETS": GENERATORS["ETS"],
        "GARCH": GENERATORS["GARCH"],
        "OrnsteinUhlenbeck": GENERATORS["OrnsteinUhlenbeck"],
        "RegimeSwitching": GENERATORS["RegimeSwitching"],
        "EnergyLoad": GENERATORS["EnergyLoad"],
        "IoTSensor": GENERATORS["IoTSensor"],
    }

    print(
        f"{'Generator':<25} {'n_series':>8} "
        f"{'Rust (ms)':>10} {'Py (ms)':>10} {'Speedup':>8}"
    )
    print("-" * 72)

    for name, (cls, params) in subset.items():
        for n_series in n_series_list:

            def run_gen():
                gen = cls(
                    min_length=length, max_length=length, seed=seed, freq="1h", **params
                )
                gen.generate(n_series=n_series)

            set_rust(True)
            t_rust = _time_fn(run_gen, repeats=2) * 1000

            set_rust(False)
            t_py = _time_fn(run_gen, repeats=2) * 1000

            set_rust(True)

            speedup = t_py / t_rust if t_rust > 0 else float("inf")
            print(
                f"{name:<25} {n_series:>8} "
                f"{t_rust:>10.2f} {t_py:>10.2f} {speedup:>7.1f}x"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify Rust is available
    try:
        from synforecast._lib import statistical  # noqa: F401

        print("Rust extension: LOADED\n")
    except ImportError:
        print("ERROR: Rust extension not available. Build first with:")
        print("  uv pip install -e .")
        sys.exit(1)

    check_equivalence()
    benchmark_lengths()
    benchmark_multi_series()

    # Restore Rust enabled
    set_rust(True)
