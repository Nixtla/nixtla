"""Measure search, FFT and panel paths with fixed workloads.

Run against either checkout using PYTHONPATH; build that checkout's extension
first. Use OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4 for comparable runs.
Search uses exactly 16 candidate evaluations per call (no early stopping).
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import polars as pl
from _env import environment_metadata

import synforecast
from synforecast import SynAugment
from synforecast._lib import augmentation
from synforecast.generators import MARGenerator


def panel(count, length):
    return pl.DataFrame(
        {
            "unique_id": np.repeat(np.arange(count), length),
            "ds": np.tile(np.arange(length), count),
            "y": np.random.default_rng(42).normal(size=count * length),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    small = panel(400, 96)
    short = panel(24, 96)
    long = panel(16, 1024)
    coefficients = [
        MARGenerator._apply_seasonal_factor(np.array([0.3, 0.1]), 24, phi)
        for phi in [0.6, 0.5]
    ]
    cases = {
        "target_period12": lambda: MARGenerator.tune_to_features(
            {"acf1": 0.7},
            64,
            128,
            1,
            seasonal_period=12,
            n_generations=2,
            population_size=8,
            n_draws_per_candidate=3,
            tolerance=1e-12,
            seed=42,
        ),
        "target_period24": lambda: MARGenerator.tune_to_features(
            {"acf1": 0.7},
            64,
            128,
            1,
            seasonal_period=24,
            n_generations=2,
            population_size=8,
            n_draws_per_candidate=3,
            tolerance=1e-12,
            seed=42,
        ),
        "stationarity_order26": lambda: MARGenerator._is_mixture_stationary(
            np.array([0.4, 0.6]), coefficients
        ),
        "mbb_400x96_3copies": lambda: SynAugment(seed=42).mbb(
            small, n_augment=3, seasonal_period=12
        ),
        "dba_24x96_5copies": lambda: SynAugment(seed=42).dba(
            short, n_augment=5, n_iterations=5
        ),
        "dba_16x1024_4copies": lambda: SynAugment(seed=42).dba(
            long, n_augment=4, n_iterations=5
        ),
    }
    for length in [4095, 4096]:
        values = np.random.default_rng(42).normal(size=length)
        cases[f"features_{length}_200calls"] = lambda v=values: [
            augmentation.compute_features(v, 12) for _ in range(200)
        ]
    measurements = {}
    for name, run in cases.items():
        samples = []
        for _ in range(args.repeats):
            # Cold validation cache per call; repeated elites within search
            # still benefit from memoization, as in a real invocation.
            for cache in ["_component_stationary_cached", "_mixture_stationary_cached"]:
                if hasattr(MARGenerator, cache):
                    getattr(MARGenerator, cache).cache_clear()
            start = perf_counter()
            result = run()
            samples.append(perf_counter() - start)
            if name.startswith("target_"):
                assert result.tuning_diagnostics["candidates_evaluated"] == 16
        measurements[name] = {
            "median_seconds": median(samples),
            "samples_seconds": samples,
        }
        print(name, median(samples), flush=True)
    result = {
        "label": args.label,
        "environment": environment_metadata(),
        "package_path": synforecast.__file__,
        "thread_env": {
            k: os.environ.get(k) for k in ["RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "measurements": measurements,
    }
    args.save.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
