"""Benchmark the native MAR, MBB, DBA, and feature-computation paths.

The comparisons isolate the paths replaced by Rust:

* MAR: public ``generate()`` through Rayon batch dispatch versus the Python
  ``generate_single_series`` fallback, including DataFrame construction.
* MBB: native decomposition and block sampling versus the former NumPy recipe.
* DBA: native banded DTW/barycenter updates versus the former Python recipe.
* Features: the native targeting feature tuple versus its NumPy equivalent,
  at a power-of-two length and at an odd length that exercises RealFFT.
* Pairwise DTW: the parallel native distance matrix versus per-pair Python.

Timings are descriptive and are never asserted in CI. The minimum of repeated
runs is reported to reduce scheduler noise, and saved results include the full
environment and dirty-tree fingerprint.

Usage:
    uv run maturin develop --release
    uv run python benchmarks/benchmark_native_paths.py --quick
    uv run python benchmarks/benchmark_native_paths.py \
        --save benchmarks/data/native_paths_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from _env import environment_metadata

import synforecast.base as base_module
from synforecast._lib import augmentation as native
from synforecast.generators import MARGenerator


def _time_call(function: Callable[[], object], repeats: int) -> float:
    """Return the minimum runtime after one untimed warm-up."""
    function()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def _moving_average(values: np.ndarray, period: int | None) -> np.ndarray:
    """Former NumPy centered-moving-average implementation."""
    n = len(values)
    if period is None:
        window = max(3, (n // 10) | 1)
        if window > n:
            window = n if n % 2 else n - 1
        weights = np.full(window, 1.0 / window)
    elif period % 2:
        weights = np.full(period, 1.0 / period)
    else:
        weights = np.concatenate(
            ([0.5 / period], np.full(period - 1, 1.0 / period), [0.5 / period])
        )
    if len(weights) > n:
        window = n if n % 2 else n - 1
        weights = np.full(window, 1.0 / window)
    computed = np.convolve(values, weights, mode="valid")
    left = (len(weights) - 1) // 2
    right = n - len(computed) - left
    return np.pad(computed, (left, right), mode="edge")


def _classical_decompose(
    values: np.ndarray, period: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Former NumPy classical decomposition implementation."""
    usable_period = period if period is not None and len(values) >= 2 * period else None
    trend = _moving_average(values, usable_period)
    seasonal = np.zeros_like(values)
    if usable_period is not None:
        detrended = values - trend
        phase_means = np.asarray(
            [detrended[phase::usable_period].mean() for phase in range(usable_period)]
        )
        phase_means -= phase_means.mean()
        seasonal = np.resize(phase_means, len(values))
    return trend, seasonal, values - trend - seasonal


def _python_mbb(
    values: np.ndarray, period: int | None, block_size: int, seed: int
) -> np.ndarray:
    """Former NumPy moving-block-bootstrap numeric kernel."""
    trend, seasonal, remainder = _classical_decompose(values, period)
    rng = np.random.default_rng(seed)
    n_blocks = len(values) // block_size + 2
    starts = rng.integers(0, len(values) - block_size + 1, n_blocks)
    sampled = np.concatenate(
        [remainder[start : start + block_size] for start in starts]
    )
    offset = int(rng.integers(0, block_size))
    return trend + seasonal + sampled[offset : offset + len(values)]


def _python_dtw_alignment(
    first: np.ndarray, second: np.ndarray, band: int | None
) -> tuple[float, np.ndarray]:
    """Former Python banded-DTW implementation."""
    n, m = len(first), len(second)
    width = max(n, m) if band is None else max(band, abs(n - m))
    previous_costs = {0: 0.0}
    parents: dict[tuple[int, int], int] = {}
    for i in range(1, n + 1):
        current_costs: dict[int, float] = {}
        for j in range(max(1, i - width), min(m, i + width) + 1):
            options = (
                previous_costs.get(j - 1, np.inf),
                previous_costs.get(j, np.inf),
                current_costs.get(j - 1, np.inf),
            )
            direction = int(np.argmin(options))
            current_costs[j] = (first[i - 1] - second[j - 1]) ** 2 + options[direction]
            parents[(i - 1, j - 1)] = direction
        previous_costs = current_costs

    i, j = n - 1, m - 1
    path: list[tuple[int, int]] = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        direction = parents[(i, j)]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(np.sqrt(previous_costs[m])), np.asarray(path)


def _python_dba(
    reference: np.ndarray,
    neighbors: list[np.ndarray],
    weights: np.ndarray,
    n_iterations: int,
    band: int | None,
) -> np.ndarray:
    """Former Python weighted-DBA implementation."""
    series = [reference, *neighbors]
    barycenter = reference.copy()
    for _ in range(n_iterations):
        sums = np.zeros_like(barycenter)
        totals = np.zeros_like(barycenter)
        for values, weight in zip(series, weights, strict=True):
            _, path = _python_dtw_alignment(barycenter, values, band)
            for barycenter_index, values_index in path:
                sums[barycenter_index] += weight * values[values_index]
                totals[barycenter_index] += weight
        observed = totals > 0
        barycenter[observed] = sums[observed] / totals[observed]
    return barycenter


def _python_features(
    values: np.ndarray, period: int
) -> tuple[float, float, float, float]:
    """Former NumPy implementation of the targeting feature tuple."""
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered))[1:] ** 2
    probabilities = power / power.sum()
    positive = probabilities > 0
    entropy = -float(
        np.sum(probabilities[positive] * np.log(probabilities[positive]))
    ) / np.log(len(power))
    trend, seasonal, remainder = _classical_decompose(values, period)
    trend_denominator = float(np.var(trend + remainder))
    trend_strength = np.clip(
        1.0 - float(np.var(remainder)) / trend_denominator, 0.0, 1.0
    )
    seasonal_denominator = float(np.var(seasonal + remainder))
    seasonal_strength = np.clip(
        1.0 - float(np.var(remainder)) / seasonal_denominator, 0.0, 1.0
    )
    variance = float(np.var(values))
    acf1 = (
        float(np.sum(centered[:-1] * centered[1:]) / len(values) / variance)
        if variance > 0
        else 0.0
    )
    return float(entropy), float(trend_strength), float(seasonal_strength), acf1


def _python_pairwise_dtw(
    series: list[np.ndarray], window_fraction: float
) -> np.ndarray:
    """Per-pair Python loop over the former DTW implementation."""
    n = len(series)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            longest = max(len(series[i]), len(series[j]))
            band = max(
                int(np.ceil(window_fraction * longest)),
                abs(len(series[i]) - len(series[j])) + 1,
            )
            distance, _ = _python_dtw_alignment(series[i], series[j], band)
            matrix[i, j] = matrix[j, i] = distance
    return matrix


def _python_nearest_dtw(series, window_fraction, n_neighbors):
    matrix = _python_pairwise_dtw(series, window_fraction)
    return [
        sorted(
            [(j, float(matrix[i, j])) for j in range(len(series)) if j != i],
            key=lambda item: (item[1], item[0]),
        )[:n_neighbors]
        for i in range(len(series))
    ]


def _mar_generate(native_path: bool, n_series: int, length: int, workers: int) -> int:
    """Generate a MAR panel through native batch or Python fallback."""
    original_type = base_module._GEN_TYPE_MAP.pop("MARGenerator", None)
    try:
        if native_path and original_type is not None:
            base_module._GEN_TYPE_MAP["MARGenerator"] = original_type
        generator = MARGenerator(
            min_length=length,
            max_length=length,
            freq="h",
            seasonal_period=24,
            seed=42,
        )
        return len(generator.generate(n_series=n_series, n_jobs=workers))
    finally:
        base_module._GEN_TYPE_MAP.pop("MARGenerator", None)
        if original_type is not None:
            base_module._GEN_TYPE_MAP["MARGenerator"] = original_type


def _benchmark_case(
    name: str,
    native_call: Callable[[], object],
    python_call: Callable[[], object],
    repeats: int,
) -> dict[str, float | str]:
    native_seconds = _time_call(native_call, repeats)
    python_seconds = _time_call(python_call, repeats)
    speedup = python_seconds / native_seconds
    print(
        f"{name:<20} {native_seconds * 1e3:>11.3f} ms "
        f"{python_seconds * 1e3:>11.3f} ms {speedup:>9.2f}x"
    )
    return {
        "name": name,
        "native_seconds": native_seconds,
        "python_seconds": python_seconds,
        "speedup": speedup,
    }


def run_benchmarks(quick: bool, repeats: int, workers: int) -> dict:
    """Run all comparisons and return a serializable result dictionary."""
    config = {
        "mar_n_series": 32 if quick else 256,
        "mar_length": 256 if quick else 1_024,
        "mbb_length": 512 if quick else 8_192,
        "dba_length": 64 if quick else 256,
        "dba_iterations": 2 if quick else 5,
        "feature_length": 512 if quick else 4_096,
        "feature_length_odd": 511 if quick else 4_095,
        "pairwise_n_series": 8 if quick else 24,
        "pairwise_length": 64 if quick else 256,
        "repeats": repeats,
        "workers": workers,
        "quick": quick,
    }
    rng = np.random.default_rng(7)
    mbb_time = np.arange(config["mbb_length"], dtype=float)
    mbb_values = (
        0.002 * mbb_time
        + np.sin(2 * np.pi * mbb_time / 24)
        + rng.normal(0.0, 0.2, len(mbb_time))
    )
    dba_reference = rng.normal(size=config["dba_length"]).cumsum()
    dba_neighbors = [
        np.roll(dba_reference, shift) + rng.normal(0.0, 0.1, len(dba_reference))
        for shift in (1, 2, 3)
    ]
    dba_weights = np.asarray([0.55, 0.2, 0.15, 0.1])
    dba_band = max(1, config["dba_length"] // 10)
    feature_values = rng.normal(size=config["feature_length"])
    feature_values_odd = rng.normal(size=config["feature_length_odd"])
    pairwise_series = [
        rng.normal(size=config["pairwise_length"]).cumsum()
        for _ in range(config["pairwise_n_series"])
    ]

    print(f"{'case':<20} {'native':>14} {'python':>14} {'speedup':>10}")
    print("-" * 72)
    cases = [
        _benchmark_case(
            "mar_generate",
            lambda: _mar_generate(
                True,
                config["mar_n_series"],
                config["mar_length"],
                workers,
            ),
            lambda: _mar_generate(
                False,
                config["mar_n_series"],
                config["mar_length"],
                workers,
            ),
            repeats,
        ),
        _benchmark_case(
            "mbb_kernel",
            lambda: native.moving_block_bootstrap(mbb_values, 48, 11, 24),
            lambda: _python_mbb(mbb_values, 24, 48, 11),
            repeats,
        ),
        _benchmark_case(
            "dba_kernel",
            lambda: native.dba_barycenter(
                dba_reference,
                dba_neighbors,
                dba_weights,
                config["dba_iterations"],
                dba_band,
            ),
            lambda: _python_dba(
                dba_reference,
                dba_neighbors,
                dba_weights,
                config["dba_iterations"],
                dba_band,
            ),
            repeats,
        ),
        _benchmark_case(
            "feature_kernel",
            lambda: native.compute_features(feature_values, 24),
            lambda: _python_features(feature_values, 24),
            repeats,
        ),
        _benchmark_case(
            "feature_kernel_odd",
            lambda: native.compute_features(feature_values_odd, 24),
            lambda: _python_features(feature_values_odd, 24),
            repeats,
        ),
        _benchmark_case(
            "nearest_dtw",
            lambda: native.nearest_dtw_neighbors(pairwise_series, 0.1, 3),
            lambda: _python_nearest_dtw(pairwise_series, 0.1, 3),
            repeats,
        ),
    ]
    return {
        "benchmark": "native_paths",
        "environment": environment_metadata(),
        "configuration": config,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a fast smoke grid")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    results = run_benchmarks(args.quick, args.repeats, args.workers)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()
