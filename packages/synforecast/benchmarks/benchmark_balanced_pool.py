"""Benchmark balanced_pool throughput scaling (Rust backend only).

Measures wall-clock time for generating N series of length L using the
balanced_pool preset with Rust acceleration.

Two modes are compared:
  Sequential : generate_single_series() called N times in a Python loop
  Parallel   : ThreadPoolExecutor dispatching generate_single_series() concurrently
               (Rust backend releases the GIL, so threads run truly in parallel)

Grid:
    N (series) : 16, 32, 64, 128, 256, 512, 1024, 2048
    L (length) : 128, 256, 512, 1024

Usage:
    uv run python benchmarks/benchmark_balanced_pool.py
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure Rust backend is active
# ---------------------------------------------------------------------------
try:
    from synforecast._lib import statistical  # noqa: F401
except ImportError:
    print("ERROR: Rust extension not available. Build first with:")
    print("  uv pip install -e .")
    sys.exit(1)

import synforecast._core as _core_mod
import synforecast._distributions as _dist_mod
import synforecast.generators.bounded_process as bp_mod
import synforecast.generators.chaotic_system as cs_mod
import synforecast.generators.cyclic as cyclic_mod
import synforecast.generators.energy_load as energy_mod
import synforecast.generators.ets as ets_mod
import synforecast.generators.fractional_brownian_motion as fbm_mod
import synforecast.generators.garch as garch_mod
import synforecast.generators.gaussian_process as gp_mod
import synforecast.generators.inar as inar_mod
import synforecast.generators.intermittent_demand as idem_mod
import synforecast.generators.iot_sensor as iot_mod
import synforecast.generators.levy_process as lp_mod
import synforecast.generators.regime_switching as rs_mod
import synforecast.generators.sarima as sarima_mod
import synforecast.generators.vital_signs as vs_mod

_RUST_MODULES = [
    _core_mod,
    _dist_mod,
    sarima_mod,
    ets_mod,
    inar_mod,
    garch_mod,
    cyclic_mod,
    fbm_mod,
    rs_mod,
    cs_mod,
    bp_mod,
    lp_mod,
    gp_mod,
    iot_mod,
    idem_mod,
    energy_mod,
    vs_mod,
]

for mod in _RUST_MODULES:
    if not getattr(mod, "_HAS_RUST", False):
        print(f"WARNING: {mod.__name__} does not have Rust enabled")


def force_rust(enabled: bool) -> None:
    for mod in _RUST_MODULES:
        mod._HAS_RUST = enabled


force_rust(True)

from synforecast import balanced_pool  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERIES_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SERIES_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
REPEATS = 1  # single run to keep total benchmark time short


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def generate_n_series(generators: list, n_total: int, length: int) -> int:
    """Generate exactly n_total series by cycling through generators.

    Returns the total number of data points generated.
    """
    n_gens = len(generators)
    total_points = 0
    for i in range(n_total):
        gen = generators[i % n_gens]
        arr = gen.generate_single_series(length)
        total_points += len(arr)
    return total_points


def benchmark_one(generators: list, n_total: int, length: int, repeats: int) -> float:
    """Return minimum wall-clock time (seconds) over `repeats` runs."""
    # Warm-up run (not timed) — just 2 series to JIT/cache-prime
    generate_n_series(generators, min(n_total, 2), length)

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        generate_n_series(generators, n_total, length)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def generate_n_series_parallel(
    generators: list, n_total: int, length: int, n_threads: int
) -> None:
    """Generate exactly n_total series by cycling through generators,
    dispatching each call to a thread-pool worker."""
    n_gens = len(generators)
    tasks = [(generators[i % n_gens], length) for i in range(n_total)]
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(lambda t: t[0].generate_single_series(t[1]), tasks))


def benchmark_one_parallel(
    generators: list, n_total: int, length: int, n_threads: int, repeats: int
) -> float:
    """Return minimum wall-clock time for generating n_total series in parallel
    via ThreadPoolExecutor (each worker calls into the Rust backend)."""
    # Warm-up
    generate_n_series_parallel(generators, min(n_total, 2), length, n_threads)

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        generate_n_series_parallel(generators, n_total, length, n_threads)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    n_threads = os.cpu_count() or 1

    print("=" * 78)
    print("  BALANCED POOL BENCHMARK  (Rust backend)")
    print("=" * 78)
    print(f"  Repeats    : {REPEATS} (reporting minimum)")
    print(f"  Series     : {SERIES_COUNTS}")
    print(f"  Lengths    : {SERIES_LENGTHS}")
    print(f"  Threads    : {n_threads} (rayon default — os.cpu_count())")
    print()

    # Build one pool per length (length is baked into the generator config)
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

    total_cells = len(SERIES_COUNTS) * len(SERIES_LENGTHS)

    # ---- Sequential grid --------------------------------------------------
    seq_results: dict[int, dict[int, float]] = {}

    cell = 0
    for n in SERIES_COUNTS:
        seq_results[n] = {}
        for length in SERIES_LENGTHS:
            cell += 1
            print(
                f"\r  Sequential [{cell}/{total_cells}] "
                f"N={n:>5}, L={length:>5} ...",
                end="",
                flush=True,
            )
            t = benchmark_one(pools[length], n, length, REPEATS)
            seq_results[n][length] = t
    print("\r" + " " * 60 + "\r", end="")

    # ---- Parallel grid ----------------------------------------------------
    # Same N series, same generators, but dispatched via ThreadPoolExecutor
    # so each generate_single_series() call runs in its own thread (the Rust
    # backend releases the GIL, so threads run truly concurrently).
    par_results: dict[int, dict[int, float]] = {}

    cell = 0
    for n in SERIES_COUNTS:
        par_results[n] = {}
        for length in SERIES_LENGTHS:
            cell += 1
            print(
                f"\r  Parallel   [{cell}/{total_cells}] "
                f"N={n:>5}, L={length:>5} ...",
                end="",
                flush=True,
            )
            t = benchmark_one_parallel(pools[length], n, length, n_threads, REPEATS)
            par_results[n][length] = t
    print("\r" + " " * 60 + "\r", end="")

    # ---- Print tables -----------------------------------------------------
    col_w = 10

    def print_table(title: str, data: dict[int, dict[int, float]]) -> None:
        print("=" * 78)
        print(f"  {title}")
        print("=" * 78)
        label = "N \\ L"
        header = f"{label:>{col_w}}"
        for length in SERIES_LENGTHS:
            header += f"{length:>{col_w}}"
        print(header)
        print("-" * (col_w + col_w * len(SERIES_LENGTHS)))
        for n in SERIES_COUNTS:
            row = f"{n:>{col_w}}"
            for length in SERIES_LENGTHS:
                row += f"{data[n][length]:>{col_w}.4f}"
            print(row)
        print()

    print_table(
        "Sequential time (s)  — generate_single_series loop",
        seq_results,
    )
    print_table(
        f"Parallel time (s)    — ThreadPoolExecutor, {n_threads} threads",
        par_results,
    )

    # ---- Speedup table ----------------------------------------------------
    print("=" * 78)
    print("  Speedup  (sequential / parallel)  — >1.0x means parallel is faster")
    print("=" * 78)
    label = "N \\ L"
    header = f"{label:>{col_w}}"
    for length in SERIES_LENGTHS:
        header += f"{length:>{col_w}}"
    print(header)
    print("-" * (col_w + col_w * len(SERIES_LENGTHS)))
    for n in SERIES_COUNTS:
        row = f"{n:>{col_w}}"
        for length in SERIES_LENGTHS:
            t_seq = seq_results[n][length]
            t_par = par_results[n][length]
            speedup = t_seq / t_par if t_par > 0 else float("inf")
            marker = " **" if speedup > 1.5 else ""
            row += f"{speedup:>{col_w - len(marker)}.2f}x{marker}"
        print(row)
    print()


if __name__ == "__main__":
    main()
