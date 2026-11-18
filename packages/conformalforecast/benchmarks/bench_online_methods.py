#!/usr/bin/env python
"""Time the online methods' ``fit`` and ``update`` paths.

Not part of the installed package. Used to record the pre-vectorization baseline and
the post-vectorization result:

    uv run python benchmarks/bench_online_methods.py --json /tmp/bench_before.json

``SplitConformal`` is the control — it was already loop-free and must not change.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from conformalforecast.methods import (
    AdaptiveConformalInference,
    AggregatedAdaptiveConformalInference,
    NonexchangeableConformalPrediction,
    SplitConformal,
)
from conformalforecast.methods.base import ConformalMethod

ALPHA = 0.1
N_UPDATE_STEPS = 20

# (label, n_series, n_windows, horizon, ragged)
PANELS: list[tuple[str, int, int, int, bool]] = [
    ("small", 1, 50, 1, False),
    ("medium", 5, 200, 12, False),
    ("large", 20, 500, 12, False),
    ("ragged-medium", 5, 200, 12, True),
]

METHODS: list[tuple[str, type[ConformalMethod[Any]], dict[str, Any]]] = [
    ("SplitConformal", SplitConformal, {}),
    ("AdaptiveConformalInference", AdaptiveConformalInference, {"gamma": 0.01}),
    ("AggregatedAdaptiveConformalInference", AggregatedAdaptiveConformalInference, {}),
    ("NonexchangeableConformalPrediction", NonexchangeableConformalPrediction, {"rho": 0.99}),
]


def make_panel(
    n_series: int, n_windows: int, horizon: int, ragged: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(predictions, truths, next_prediction)`` for one panel shape."""
    rng = np.random.default_rng(0)
    predictions = np.zeros((n_series, n_windows, horizon))
    truths = rng.normal(size=(n_series, n_windows, horizon))
    if ragged:
        for i in range(n_series):
            prefix = (i * n_windows) // (2 * n_series)
            if prefix:
                predictions[i, :prefix, :] = np.nan
                truths[i, :prefix, :] = np.nan
    return predictions, truths, np.zeros((n_series, 1, horizon))


def best_of(fn: Any, repeats: int = 3) -> float:
    """Best wall-clock seconds over ``repeats`` runs."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="Write raw timings here.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip the 20x500x12 panel (slow before the loops were vectorized).",
    )
    args = parser.parse_args()

    results: dict[str, float] = {}
    rows: list[tuple[str, str, str]] = []

    for panel_label, n_series, n_windows, horizon, ragged in PANELS:
        if args.skip_large and panel_label == "large":
            continue
        predictions, truths, next_pred = make_panel(n_series, n_windows, horizon, ragged)
        for method_label, method_cls, kwargs in METHODS:

            def fit_once(
                cls: Any = method_cls,
                kw: Any = kwargs,
                preds: Any = predictions,
                ys: Any = truths,
            ) -> None:
                cls(alpha=ALPHA, **kw).fit(preds, ys)

            fit_key = f"fit/{method_label}/{panel_label}"
            fit_time = best_of(fit_once, args.repeats)
            results[fit_key] = fit_time

            fitted = method_cls(alpha=ALPHA, **kwargs).fit(predictions, truths)
            truth_step = next_pred + 1.0

            def update_loop(
                k: Any = fitted,
                pred: Any = next_pred,
                truth: Any = truth_step,
            ) -> None:
                for _ in range(N_UPDATE_STEPS):
                    k.update(pred, truth)

            update_key = f"update{N_UPDATE_STEPS}/{method_label}/{panel_label}"
            update_time = best_of(update_loop, 1)
            results[update_key] = update_time

            rows.append((f"{method_label}", panel_label, f"{fit_time:.4f} | {update_time:.4f}"))

    print()
    print(f"| method | panel | fit (s) | {N_UPDATE_STEPS} updates (s) |")
    print("|---|---|---|---|")
    for method_label, panel_label, timings in rows:
        fit_s, update_s = timings.split(" | ")
        print(f"| {method_label} | {panel_label} | {fit_s} | {update_s} |")
    print()

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"raw timings -> {args.json}")


if __name__ == "__main__":
    main()
