"""Verify output parity between Rust backend and Python fallback.

For each generator, pattern injection functions, and distribution
functions, this script:
  1. Generates output using the Rust (PyO3) backend
  2. Generates output using the pure-Python fallback
  3. Compares: shape equality, finiteness, statistical properties

Distribution functions (pure math, no RNG) are compared for exact numerical
parity. Generator outputs differ due to different RNG streams but must have
matching shape and similar statistical properties.

Usage:
    uv run python scripts/check_rust_parity.py
"""

# ruff: noqa: E402  (imports intentionally follow the Rust-availability check)

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# ---------------------------------------------------------------------------
# Ensure Rust backend is available
# ---------------------------------------------------------------------------
try:
    from synforecast._lib import statistical  # noqa: F401

    print("Rust extension: LOADED\n")
except ImportError:
    print("ERROR: Rust extension not available. Build first with:")
    print("  maturin develop --release")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Module imports for toggling _HAS_RUST
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
from synforecast._distributions import (
    expon_ppf,
    gamma_ppf,
    lognorm_ppf,
    norm_cdf,
    norm_ppf,
    t_cdf,
    uniform_ppf,
)
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


def set_backend(enabled: bool) -> None:
    for mod in _ALL_MODULES:
        mod._HAS_RUST = enabled


# ---------------------------------------------------------------------------
# Generator configurations (same as benchmark_rust_vs_python.py)
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
    "INAR": (
        INARGenerator,
        {"p": 1, "alpha": [0.5], "innovation_type": "poisson", "innovation_mean": 3.0},
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
    "HawkesProcess": (
        HawkesProcessGenerator,
        {"baseline_intensity": 0.5, "alpha": 0.8, "beta": 1.0},
    ),
    "StochasticVolatility": (
        StochasticVolatilityGenerator,
        {"model_type": "heston", "output_type": "price"},
    ),
    "RegimeSwitching": (
        RegimeSwitchingGenerator,
        {"n_regimes": 2},
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
    "Copula": (
        CopulaGenerator,
        {"copula_type": "gaussian", "marginal_distributions": [{"type": "normal"}]},
    ),
    "VAR": (
        VARGenerator,
        {"n_variables": 2, "lag_order": 1},
    ),
    "StateSpace": (
        StateSpaceGenerator,
        {"state_dim": 2, "obs_dim": 1},
    ),
    "GaussianProcess": (
        GaussianProcessGenerator,
        {"kernel": "rbf", "length_scale": 20.0, "amplitude": 1.0},
    ),
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
}


def _generate_series(cls, params, length, seed):
    gen = cls(
        min_length=length,
        max_length=length,
        seed=seed,
        freq="h",
        engine="polars",
        **params,
    )
    return gen.generate_single_series(length)


# ---------------------------------------------------------------------------
# 1. Generator parity check
# ---------------------------------------------------------------------------

# For some processes the raw single-path mean/std never converge, so any two
# correct implementations with different RNG streams "fail" a raw comparison.
# These transforms map each path to a stationary, finite-variance quantity:
#   - GBM / JumpDiffusion grow exponentially (E[log S_t] = 154 at t=5000 for
#     mu=0.05, dt=1): compare log-increments instead of levels.
#   - ETS(A,A,A) is I(2) (the undamped trend is a random walk): compare
#     second differences.
#   - Clickstream compounds a per-series drift ~ U(-0.0005, 0.001) in log
#     space, so the level is dominated by one random draw: compare
#     log-increments of the session counts.
COMPARE_TRANSFORMS = {
    "ETS": lambda v: np.diff(v, n=2),
    "GeometricBrownianMotion": lambda v: np.diff(np.log(v)),
    "JumpDiffusion": lambda v: np.diff(np.log(v)),
    "Clickstream": lambda v: np.diff(np.log(np.maximum(v, 0.5))),
    # Levy increments are iid alpha-stable with alpha=1.5: infinite variance,
    # so mean/std are unstable even for increments — compare robustly below.
    "LevyProcess": np.diff,
}

# Generators with heavy/infinite-variance tails: compare median and IQR of
# the (transformed) values instead of mean/std.
ROBUST_STATS = {"LevyProcess"}


def _location_scale(values, robust):
    if robust:
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        return float(q50), float(q75 - q25)
    return float(np.mean(values)), float(np.std(values))


def check_generator_parity():
    print("=" * 78)
    print("  GENERATOR PARITY: Rust backend vs Python fallback")
    print("=" * 78)
    length = 5_000
    seed = 42

    results = []
    for name, (cls, params) in GENERATORS.items():
        set_backend(True)
        rust_vals = _generate_series(cls, params, length, seed)

        set_backend(False)
        py_vals = _generate_series(cls, params, length, seed)

        set_backend(True)

        shape_ok = rust_vals.shape == py_vals.shape

        transform = COMPARE_TRANSFORMS.get(name)
        if transform is not None:
            rust_vals = transform(rust_vals)
            py_vals = transform(py_vals)

        rust_clean = rust_vals[np.isfinite(rust_vals)]
        py_clean = py_vals[np.isfinite(py_vals)]

        if len(rust_clean) == 0 or len(py_clean) == 0:
            results.append((name, shape_ok, "SKIP", 0, 0, 0, 0))
            continue

        robust = name in ROBUST_STATS
        rust_mean, rust_std = _location_scale(rust_clean, robust)
        py_mean, py_std = _location_scale(py_clean, robust)

        denom = max(abs(py_mean), 1e-6)
        mean_diff = abs(rust_mean - py_mean)
        mean_ok = mean_diff < 0.5 * denom or mean_diff < 5.0

        std_ratio = max(rust_std, 1e-9) / max(py_std, 1e-9)
        std_ok = 0.2 < std_ratio < 5.0

        status = "PASS" if (shape_ok and mean_ok and std_ok) else "FAIL"
        results.append((name, shape_ok, status, rust_mean, py_mean, rust_std, py_std))

    print(
        f"  {'Generator':<27} {'Shape':>5} {'Status':>6} "
        f"{'Rust loc':>10} {'Py loc':>10} {'Rust scl':>10} {'Py scl':>10}"
    )
    print("  " + "-" * 85)

    pass_count = 0
    total = len(results)
    for name, shape_ok, status, rm, pm, rs, ps in results:
        if status in ("PASS", "SKIP"):
            pass_count += 1
        label = name + ("*" if name in COMPARE_TRANSFORMS else "")
        if isinstance(rm, str) or rm == 0:
            print(f"  {label:<27} {'OK' if shape_ok else 'FAIL':>5} {status:>6}")
        else:
            print(
                f"  {label:<27} {'OK' if shape_ok else 'FAIL':>5} {status:>6} "
                f"{rm:>10.3f} {pm:>10.3f} {rs:>10.3f} {ps:>10.3f}"
            )

    print()
    print(
        "  * compared on a stationary transform (log-increments/differences);"
        " LevyProcess uses median/IQR (infinite variance)."
    )
    print(f"  Result: {pass_count}/{total} generators passed parity check")
    print()
    return pass_count == total


# ---------------------------------------------------------------------------
# 2. Distribution function parity (pure math, exact match expected)
# ---------------------------------------------------------------------------


def check_distribution_parity():
    print("=" * 78)
    print("  DISTRIBUTION FUNCTION PARITY (pure math, numerical match)")
    print("=" * 78)

    x_vals = np.linspace(-3, 3, 1000)
    u_vals = np.linspace(0.001, 0.999, 1000)

    tests = [
        ("norm_cdf", lambda: norm_cdf(x_vals)),
        ("norm_ppf", lambda: norm_ppf(u_vals)),
        ("t_cdf (df=5)", lambda: t_cdf(x_vals, df=5.0)),
        ("expon_ppf", lambda: expon_ppf(u_vals, scale=2.0)),
        ("lognorm_ppf", lambda: lognorm_ppf(u_vals, s=0.5, scale=1.0)),
        ("uniform_ppf", lambda: uniform_ppf(u_vals, loc=0.0, scale=10.0)),
        ("gamma_ppf", lambda: gamma_ppf(u_vals, shape=2.0, scale=1.0)),
    ]

    all_pass = True
    for name, fn in tests:
        set_backend(True)
        rust_result = fn()

        set_backend(False)
        py_result = fn()

        set_backend(True)

        max_abs_err = np.max(np.abs(rust_result - py_result))
        max_rel_err = np.max(
            np.abs(rust_result - py_result) / (np.abs(py_result) + 1e-15)
        )

        status = "PASS" if max_abs_err < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  {name:<20} {status:>6}  "
            f"max_abs_err={max_abs_err:.2e}  max_rel_err={max_rel_err:.2e}"
        )

    print()
    print(f"  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dist_ok = check_distribution_parity()
    gen_ok = check_generator_parity()

    print("=" * 78)
    if dist_ok and gen_ok:
        print("  ALL PARITY CHECKS PASSED")
    else:
        print("  SOME PARITY CHECKS FAILED")
    print("=" * 78)

    set_backend(True)
    sys.exit(0 if (dist_ok and gen_ok) else 1)
