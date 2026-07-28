"""Benchmark backend performance with optional save/compare support.

Measures wall-clock time per generator and balanced_pool throughput grid.
Can save results to JSON for cross-branch or cross-machine comparison.

Usage:
    # Run and display results
    uv run python benchmarks/benchmark_performance.py

    # Save results to JSON (for later comparison)
    uv run python benchmarks/benchmark_performance.py --save results.json

    # Compare against saved reference
    uv run python benchmarks/benchmark_performance.py --compare reference.json
"""

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure compiled backend is available
# ---------------------------------------------------------------------------
try:
    from synforecast._lib import statistical  # noqa: F401

    print("Compiled extension: LOADED\n")
except ImportError:
    print("ERROR: Compiled extension not available. Build first.")
    sys.exit(1)

from _env import environment_metadata  # noqa: E402

from synforecast import balanced_pool  # noqa: E402
from synforecast.generators import (  # noqa: E402
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

# ---------------------------------------------------------------------------
# Generator configurations
# ---------------------------------------------------------------------------

GENERATORS: dict[str, tuple[type, dict]] = {
    "RandomWalk": (
        RandomWalkGenerator,
        {"drift": 0.1, "volatility": 1.0, "start_value": 0.0},
    ),
    "Seasonal": (
        SeasonalGenerator,
        {
            "seasonality_period": 24,
            "seasonality_amplitude": 5.0,
            "trend": 0.01,
            "noise_level": 0.5,
        },
    ),
    "SARIMA": (
        SARIMAGenerator,
        {"p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1, "seasonal_period": 12},
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
        {"baseline_intensity": 0.5, "excitation_amplitude": 0.8, "decay_rate": 1.0},
    ),
    "StochasticVolatility": (
        StochasticVolatilityGenerator,
        {"model": "heston", "output_type": "price"},
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
        {"lag_order": 1},
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
        {"vital_sign": "heart_rate"},
    ),
    "Clickstream": (
        ClickstreamGenerator,
        {"output_type": "sessions"},
    ),
}

# Maximum lengths to avoid O(n^2)/O(n^3) blowups
_MAX_LENGTHS: dict[str, int] = {
    "FBM (Hosking)": 3_000,
    "GaussianProcess": 3_000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERIES_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SERIES_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
REPEATS = 1


def _generate_series(cls, params, length, seed):
    gen = cls(min_length=length, max_length=length, seed=seed, freq="1h", **params)
    return gen.generate_single_series(length)


def _time_fn(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def generate_n_series(generators: list, n_total: int, length: int) -> int:
    n_gens = len(generators)
    total_points = 0
    for i in range(n_total):
        gen = generators[i % n_gens]
        arr = gen.generate_single_series(length)
        total_points += len(arr)
    return total_points


def benchmark_one(generators: list, n_total: int, length: int, repeats: int) -> float:
    generate_n_series(generators, min(n_total, 2), length)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        generate_n_series(generators, n_total, length)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


# ---------------------------------------------------------------------------
# 1. Per-generator benchmark
# ---------------------------------------------------------------------------


def benchmark_generators():
    print("=" * 72)
    print("  PER-GENERATOR BENCHMARK  (single series, length=1000)")
    print("=" * 72)
    length = 1_000
    seed = 123
    repeats = 5

    results = {}
    print(f"  {'Generator':<25} {'Time (ms)':>10} {'Throughput':>15}")
    print("  " + "-" * 55)

    for name, (cls, params) in GENERATORS.items():
        t_ms = (
            _time_fn(lambda: _generate_series(cls, params, length, seed), repeats)
            * 1000
        )
        throughput = length / (t_ms / 1000)
        results[name] = t_ms
        print(f"  {name:<25} {t_ms:>10.3f} {throughput:>12,.0f} pts/s")

    print()
    times = list(results.values())
    print(f"  Median: {np.median(times):.3f} ms  |  Mean: {np.mean(times):.3f} ms")
    print()
    return results


# ---------------------------------------------------------------------------
# 2. Balanced pool grid benchmark
# ---------------------------------------------------------------------------


def benchmark_grid():
    print("=" * 72)
    print("  BALANCED POOL GRID BENCHMARK")
    print("=" * 72)
    print(f"  Repeats    : {REPEATS} (reporting minimum)")
    print(f"  Series     : {SERIES_COUNTS}")
    print(f"  Lengths    : {SERIES_LENGTHS}")
    print()

    pools: dict[int, list] = {}
    for length in SERIES_LENGTHS:
        pools[length] = balanced_pool(
            min_length=length,
            max_length=length,
            freq="1d",
            seed=42,
        )

    n_gens = len(pools[SERIES_LENGTHS[0]])
    print(f"  Pool size  : {n_gens} generators")
    print()

    results: dict[str, dict[str, float]] = {}
    total_cells = len(SERIES_COUNTS) * len(SERIES_LENGTHS)
    cell = 0

    for n in SERIES_COUNTS:
        results[str(n)] = {}
        for length in SERIES_LENGTHS:
            cell += 1
            print(
                f"\r  Benchmarking [{cell}/{total_cells}] N={n:>5}, L={length:>5} ...",
                end="",
                flush=True,
            )
            t = benchmark_one(pools[length], n, length, REPEATS)
            results[str(n)][str(length)] = t
    print("\r" + " " * 60 + "\r", end="")

    # Print table
    col_w = 10
    print("  Time (s)  — rows: N series, columns: series length")
    print("  " + "=" * (col_w + col_w * len(SERIES_LENGTHS)))
    label = "N \\ L"
    header = f"{label:>{col_w}}"
    for length in SERIES_LENGTHS:
        header += f"{length:>{col_w}}"
    print(f"  {header}")
    print("  " + "-" * (col_w + col_w * len(SERIES_LENGTHS)))
    for n in SERIES_COUNTS:
        row = f"{n:>{col_w}}"
        for length in SERIES_LENGTHS:
            t = results[str(n)][str(length)]
            row += f"{t:>{col_w}.4f}"
        print(f"  {row}")
    print()

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(gen_results, ref_gen, grid_results, ref_grid, ref_label):
    # Per-generator comparison
    print("=" * 72)
    print(f"  PER-GENERATOR COMPARISON (current vs {ref_label})")
    print("=" * 72)
    print(f"  {'Generator':<25} {'Current':>10} {ref_label:>10} {'Speedup':>8}")
    print("  " + "-" * 58)

    speedups = []
    for name in gen_results:
        t_cur = gen_results[name]
        t_ref = ref_gen.get(name)
        if t_ref is not None and t_ref > 0:
            speedup = t_ref / t_cur
            speedups.append(speedup)
            print(f"  {name:<25} {t_cur:>9.3f}ms {t_ref:>9.3f}ms {speedup:>7.2f}x")
        else:
            print(f"  {name:<25} {t_cur:>9.3f}ms {'N/A':>10} {'N/A':>8}")

    if speedups:
        print()
        print(
            f"  Geometric mean speedup: {np.exp(np.mean(np.log(speedups))):.2f}x  "
            f"| Median: {np.median(speedups):.2f}x"
        )
    print()

    # Grid comparison
    if ref_grid:
        print("=" * 72)
        print(f"  GRID SPEEDUP (current vs {ref_label}) — >1.0 means current is faster")
        print("=" * 72)
        col_w = 10
        label = "N \\ L"
        header = f"{label:>{col_w}}"
        for length in SERIES_LENGTHS:
            header += f"{length:>{col_w}}"
        print(f"  {header}")
        print("  " + "-" * (col_w + col_w * len(SERIES_LENGTHS)))
        for n in SERIES_COUNTS:
            row = f"{n:>{col_w}}"
            for length in SERIES_LENGTHS:
                t_cur = grid_results.get(str(n), {}).get(str(length))
                t_ref = ref_grid.get(str(n), {}).get(str(length))
                if t_cur and t_ref and t_cur > 0:
                    speedup = t_ref / t_cur
                    row += f"{speedup:>{col_w - 1}.2f}x"
                else:
                    row += f"{'N/A':>{col_w}}"
            print(f"  {row}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark backend performance")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare against saved reference JSON file",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="reference",
        help="Label for the reference in comparison output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny grid for smoke runs; results are not comparable",
    )
    args = parser.parse_args()

    if args.quick:
        global SERIES_COUNTS, SERIES_LENGTHS
        SERIES_COUNTS = [16, 32]
        SERIES_LENGTHS = [128, 256]

    gen_results = benchmark_generators()
    grid_results = benchmark_grid()

    if args.save:
        data = {
            "environment": environment_metadata(),
            "generators": gen_results,
            "grid": grid_results,
        }
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Results saved to {args.save}")
        print()

    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        ref_gen = ref.get("generators", {})
        ref_grid = ref.get("grid", {})
        print_comparison(
            gen_results, ref_gen, grid_results, ref_grid, args.compare_label
        )


if __name__ == "__main__":
    main()
