"""Benchmark: Rust rayon batch parallelization vs Python ThreadPoolExecutor.

Compares two parallelization strategies:
  1. **Threaded (old)**: Python ThreadPoolExecutor calling Rust per-series
  2. **Rayon batch (new)**: Single Python→Rust call, rayon parallelizes all series

Tests both levels:
  - Per-generator: BaseGenerator.generate(n_series=N) for a single generator
  - Cross-generator: SynSet.generate(n_series_per_generator=N) with multiple generators

Usage:
    uv run python benchmarks/benchmark_batch_parallel.py
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure Rust backend is active
# ---------------------------------------------------------------------------
try:
    from synforecast._lib import batch as _rs_batch  # noqa: F401
    from synforecast._lib import statistical  # noqa: F401
except ImportError:
    print("ERROR: Rust extension not available (batch module required).")
    print("  Build with: maturin develop --release")
    sys.exit(1)

import synforecast.base as base_mod
import synforecast.dataset as dataset_mod
from synforecast import SynSet, balanced_pool
from synforecast.generators import (
    ETSGenerator,
    GARCHGenerator,
    OrnsteinUhlenbeckGenerator,
    RandomWalkGenerator,
    SARIMAGenerator,
    SeasonalGenerator,
)

# Keep reference to the real batch module
_REAL_BATCH = base_mod._rs_batch


def set_batch_mode(enabled: bool) -> None:
    """Toggle between rayon batch (True) and Python threaded (False) paths."""
    val = _REAL_BATCH if enabled else None
    base_mod._rs_batch = val
    dataset_mod._rs_batch = val


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERIES_COUNTS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
SERIES_LENGTHS = [128, 512, 2048, 8192]
REPEATS = 3
WARMUP = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def time_fn(fn, repeats=REPEATS, warmup=WARMUP):
    """Return minimum wall-clock time (seconds) over `repeats` runs."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def print_table(title, row_label, col_values, row_values, data, fmt=".4f"):
    """Print a 2D table with row/column headers."""
    col_w = 10
    print(f"  {title}")
    print("  " + "=" * (col_w + col_w * len(col_values)))
    header = f"{row_label:>{col_w}}"
    for c in col_values:
        header += f"{c:>{col_w}}"
    print(f"  {header}")
    print("  " + "-" * (col_w + col_w * len(col_values)))
    for r in row_values:
        row = f"{r:>{col_w}}"
        for c in col_values:
            val = data.get(r, {}).get(c)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(val, str):
                row += f"{val:>{col_w}}"
            else:
                row += f"{val:>{col_w}{fmt}}"
        print(f"  {row}")
    print()


# ---------------------------------------------------------------------------
# 1. Single-generator benchmark
# ---------------------------------------------------------------------------


def benchmark_single_generator():
    print("=" * 78)
    print("  SINGLE GENERATOR: Rayon Batch vs Python Threaded")
    print("  (RandomWalkGenerator, n_series scaling)")
    print("=" * 78)
    print()

    results_threaded = {}
    results_batch = {}
    results_speedup = {}

    for length in SERIES_LENGTHS:
        results_threaded[length] = {}
        results_batch[length] = {}
        results_speedup[length] = {}

        for n in SERIES_COUNTS:
            # Threaded (old path)
            set_batch_mode(False)
            gen = RandomWalkGenerator(
                min_length=length,
                max_length=length,
                seed=42,
                freq="1h",
                drift=0.1,
                volatility=1.0,
            )
            t_threaded = time_fn(lambda: gen.generate(n_series=n))

            # Batch (new path)
            set_batch_mode(True)
            gen = RandomWalkGenerator(
                min_length=length,
                max_length=length,
                seed=42,
                freq="1h",
                drift=0.1,
                volatility=1.0,
            )
            t_batch = time_fn(lambda: gen.generate(n_series=n))

            results_threaded[length][n] = t_threaded
            results_batch[length][n] = t_batch
            speedup = t_threaded / t_batch if t_batch > 0 else float("inf")
            results_speedup[length][n] = speedup

    # Print tables per length
    for length in SERIES_LENGTHS:
        print(f"  Series length = {length}")
        print(
            f"  {'N':>8} {'Threaded (s)':>14} {'Batch (s)':>14} {'Speedup':>10}"
        )
        print("  " + "-" * 50)
        for n in SERIES_COUNTS:
            t_t = results_threaded[length][n]
            t_b = results_batch[length][n]
            sp = results_speedup[length][n]
            marker = " **" if sp > 1.5 else ""
            print(
                f"  {n:>8} {t_t:>14.5f} {t_b:>14.5f} {sp:>9.2f}x{marker}"
            )
        print()

    # Compact speedup table
    speedup_display = {}
    for n in SERIES_COUNTS:
        speedup_display[n] = {}
        for length in SERIES_LENGTHS:
            speedup_display[n][length] = f"{results_speedup[length][n]:.2f}x"

    print_table(
        "Speedup (Threaded / Batch) — >1.0x means batch is faster",
        "N \\ L",
        SERIES_LENGTHS,
        SERIES_COUNTS,
        speedup_display,
        "s",
    )

    return results_threaded, results_batch


# ---------------------------------------------------------------------------
# 2. Multi-generator (SynSet) benchmark
# ---------------------------------------------------------------------------


def benchmark_synset():
    print("=" * 78)
    print("  SYNSET (MULTI-GENERATOR): Rayon Multi-Batch vs Python Threaded")
    print("  (6 generators, n_series_per_generator scaling)")
    print("=" * 78)
    print()

    synset_n_counts = [4, 8, 16, 32, 64, 128, 256]
    synset_lengths = [256, 1024, 4096]

    results_threaded = {}
    results_batch = {}
    results_speedup = {}

    for length in synset_lengths:
        results_threaded[length] = {}
        results_batch[length] = {}
        results_speedup[length] = {}

        for n in synset_n_counts:
            # Create SynSet with several generators
            def make_synset():
                return SynSet(
                    generators=[
                        RandomWalkGenerator(
                            min_length=length, max_length=length,
                            seed=42, freq="1h",
                        ),
                        SeasonalGenerator(
                            min_length=length, max_length=length,
                            seed=43, freq="1h", period=24,
                        ),
                        SARIMAGenerator(
                            min_length=length, max_length=length,
                            seed=44, freq="1h",
                            order=[1, 0, 1], seasonal_order=[1, 0, 1, 12],
                        ),
                        ETSGenerator(
                            min_length=length, max_length=length,
                            seed=45, freq="1h",
                            error_type="add", trend_type="add",
                            seasonal_type="add", seasonal_period=12,
                        ),
                        GARCHGenerator(
                            min_length=length, max_length=length,
                            seed=46, freq="1h",
                            p=1, q=1, omega=0.1, alpha=[0.15], beta=[0.75],
                        ),
                        OrnsteinUhlenbeckGenerator(
                            min_length=length, max_length=length,
                            seed=47, freq="1h",
                            theta=0.7, mu=5.0, sigma=0.3,
                        ),
                    ]
                )

            # Threaded path
            set_batch_mode(False)
            ss = make_synset()
            t_threaded = time_fn(lambda: ss.generate(n_series_per_generator=n))

            # Batch path
            set_batch_mode(True)
            ss = make_synset()
            t_batch = time_fn(lambda: ss.generate(n_series_per_generator=n))

            results_threaded[length][n] = t_threaded
            results_batch[length][n] = t_batch
            speedup = t_threaded / t_batch if t_batch > 0 else float("inf")
            results_speedup[length][n] = speedup

    for length in synset_lengths:
        print(f"  Series length = {length}, 6 generators")
        print(
            f"  {'N/gen':>8} {'Total series':>14} {'Threaded (s)':>14}"
            f" {'Batch (s)':>14} {'Speedup':>10}"
        )
        print("  " + "-" * 66)
        for n in synset_n_counts:
            t_t = results_threaded[length][n]
            t_b = results_batch[length][n]
            sp = results_speedup[length][n]
            marker = " **" if sp > 1.5 else ""
            print(
                f"  {n:>8} {n * 6:>14} {t_t:>14.5f}"
                f" {t_b:>14.5f} {sp:>9.2f}x{marker}"
            )
        print()

    # Compact speedup table
    speedup_display = {}
    for n in synset_n_counts:
        speedup_display[n] = {}
        for length in synset_lengths:
            speedup_display[n][length] = f"{results_speedup[length][n]:.2f}x"

    print_table(
        "Speedup (Threaded / Batch) — >1.0x means batch is faster",
        "N/gen \\ L",
        synset_lengths,
        synset_n_counts,
        speedup_display,
        "s",
    )

    return results_threaded, results_batch


# ---------------------------------------------------------------------------
# 3. Balanced pool benchmark
# ---------------------------------------------------------------------------


def benchmark_balanced_pool():
    print("=" * 78)
    print("  BALANCED POOL: Rayon Multi-Batch vs Python Threaded")
    print("  (full preset, n_series_per_generator scaling)")
    print("=" * 78)
    print()

    pool_n_counts = [4, 8, 16, 32, 64]
    pool_lengths = [256, 1024, 4096]

    for length in pool_lengths:
        # Create SynSet from balanced_pool preset
        gens = balanced_pool(
            min_length=length, max_length=length, freq="1d", seed=42,
        )
        n_gens = len(gens)

        print(f"  Series length = {length}, {n_gens} generators (balanced_pool)")
        print(
            f"  {'N/gen':>8} {'Total series':>14} {'Threaded (s)':>14}"
            f" {'Batch (s)':>14} {'Speedup':>10}"
        )
        print("  " + "-" * 66)

        for n in pool_n_counts:
            # Threaded path
            set_batch_mode(False)
            ss = SynSet(generators=gens)
            t_threaded = time_fn(
                lambda: ss.generate(n_series_per_generator=n), repeats=1,
            )

            # Batch path
            set_batch_mode(True)
            ss = SynSet(generators=gens)
            t_batch = time_fn(
                lambda: ss.generate(n_series_per_generator=n), repeats=1,
            )

            speedup = t_threaded / t_batch if t_batch > 0 else float("inf")
            marker = " **" if speedup > 1.5 else ""
            print(
                f"  {n:>8} {n * n_gens:>14} {t_threaded:>14.5f}"
                f" {t_batch:>14.5f} {speedup:>9.2f}x{marker}"
            )

        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print()
    print("  Rayon Batch Parallelization Benchmark")
    print("  Comparing: Python ThreadPoolExecutor vs Rust rayon batch")
    print(f"  Repeats: {REPEATS} (reporting minimum), Warmup: {WARMUP}")
    print()

    benchmark_single_generator()
    benchmark_synset()
    benchmark_balanced_pool()

    # Restore batch mode
    set_batch_mode(True)

    print("=" * 78)
    print("  Done. Batch mode restored.")
    print("=" * 78)


if __name__ == "__main__":
    main()
