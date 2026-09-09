"""Compare evolutionary MAR search with random search at equal evaluation budget.

Run: python benchmarks/benchmark_mar_targeting.py --save /tmp/mar-targeting.json
Each method scores three training draws per candidate. Evaluation uses 24 fresh
native draws, with a shared held-out seed, at lengths in [64, 128]. Lower L2
distance is better. This is an experiment, not a guaranteed-improvement test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from _env import environment_metadata

from synforecast._features import compute_features
from synforecast.generators.mar import MARGenerator

TARGETS = [
    {"acf1": 0.7},
    {"spectral_entropy": 0.5},
    {"trend_strength": 0.5},
    {"seasonal_strength": 0.5},
]
SEEDS = [31, 32, 33, 34, 35]
CONFIG = {"min_length": 64, "max_length": 128, "freq": 1, "seasonal_period": 12}


def random_search(target, seed, budget):
    rng = np.random.default_rng(seed)
    best, best_distance = None, np.inf
    for _ in range(budget):
        candidate = MARGenerator._random_candidate(rng, 12)
        distance = MARGenerator._candidate_fitness(
            candidate, target, np.array([64, 96, 128]), 1, 12, rng
        )
        if distance < best_distance:
            best, best_distance = candidate, distance
    if best is None:
        raise RuntimeError("random search found no valid candidate")
    weights, coefficients, intercepts, scales = MARGenerator._candidate_to_fixed(
        best, 12
    )
    return MARGenerator(
        **CONFIG,
        seed=seed,
        weights=weights,
        ar_coefficients=coefficients,
        intercepts=intercepts,
        noise_scales=scales,
    ), best_distance


def held_out_distance(generator, target, seed):
    fresh = MARGenerator(
        **CONFIG,
        seed=seed,
        engine="polars",
        weights=generator.weights,
        ar_coefficients=generator.ar_coefficients,
        intercepts=generator.intercepts,
        noise_scales=generator.noise_scales,
    )
    frame = fresh.generate(n_series=24, n_jobs=1)
    features = [
        compute_features(part["y"].to_numpy(), 12)
        for part in frame.partition_by("unique_id", maintain_order=True)
    ]
    return float(
        np.linalg.norm(
            [
                np.mean([row[name] for row in features]) - value
                for name, value in target.items()
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            evolved = MARGenerator.tune_to_features(
                target,
                **CONFIG,
                n_generations=5,
                population_size=12,
                n_draws_per_candidate=3,
                tolerance=1e-12,
                seed=seed,
            )
            diagnostics = evolved.tuning_diagnostics
            baseline, random_training = random_search(
                target, seed, diagnostics["candidates_evaluated"]
            )
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "candidate_budget": diagnostics["candidates_evaluated"],
                    "evolution_training": diagnostics["best_distance"],
                    "random_training": random_training,
                    "evolution_held_out": held_out_distance(
                        evolved, target, 10000 + seed
                    ),
                    "random_held_out": held_out_distance(
                        baseline, target, 10000 + seed
                    ),
                }
            )
    result = {
        "environment": environment_metadata(),
        "config": CONFIG,
        "training_draws": 3,
        "held_out_draws": 24,
        "rows": rows,
    }
    output = json.dumps(result, indent=2) + "\n"
    print(output, end="")
    if args.save:
        args.save.write_text(output)


if __name__ == "__main__":
    main()
