"""Guards against reintroducing per-cell Python loops.

Budgets are ~20x the timings measured on the development machine
(Darwin/arm64, numpy 2.3.5) so that a slower, noisier shared CI runner cannot
trip them, while still failing loudly if a ``for s in range(n_series)`` loop or
a per-step re-sort of the whole history comes back — the unvectorized code
misses these budgets even on the fast machine:

    AdaptiveConformalInference.fit            10x400x12  measured 0.09 s
        (before vectorization: ~2 s, extrapolated from 0.25 s at 5x200x12,
        quadratic in n_windows)
    AggregatedAdaptiveConformalInference.fit  10x400x12  measured 0.10 s
        (before vectorization: ~11 s)
    NonexchangeableConformalPrediction.fit    200x50x12  measured 0.006 s
        (before vectorization: 0.022 s)

The NexCP case is a gross-regression guard rather than a loop detector: its
per-cell loop was only ``n_series * horizon`` iterations of vectorized work, so
vectorizing it buys panel-width scaling (3.5x at 2400 cells), not orders of
magnitude.

``fit`` is measured at two values of ``n_windows``, because the window loop's
insertion step is quadratic in ``n_windows`` and only the larger panel notices a
regression in its constant factor:

    AdaptiveConformalInference.fit            10x100x12  measured 0.007 s
    AdaptiveConformalInference.fit            10x400x12  measured 0.053 s
    AggregatedAdaptiveConformalInference.fit  10x100x12  measured 0.010 s
    AggregatedAdaptiveConformalInference.fit  10x400x12  measured 0.062 s

``interval`` and ``update`` are covered too. Neither was guarded before, so a
per-cell loop reintroduced in either would have been invisible here:

    AdaptiveConformalInference.interval            10x400x12  measured 0.0007 s
    AggregatedAdaptiveConformalInference.interval  10x400x12  measured 0.0008 s
    20x AdaptiveConformalInference.update          10x400x12  measured 0.015 s
    20x NonexchangeableConformalPrediction.update  100x200x12 measured 0.21 s
        -- the outlier, because update re-argsorts the whole panel per step (a
        known cost, not a regression); the budget guards against it getting
        worse.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conformalforecast.methods import (
    AdaptiveConformalInference,
    AggregatedAdaptiveConformalInference,
    NonexchangeableConformalPrediction,
)


def _panel(n_series: int, n_windows: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        np.zeros((n_series, n_windows, horizon)),
        rng.normal(size=(n_series, n_windows, horizon)),
    )


@pytest.mark.perf
@pytest.mark.parametrize(
    "method_cls", [AdaptiveConformalInference, AggregatedAdaptiveConformalInference]
)
@pytest.mark.parametrize(("n_windows", "budget_seconds"), [(100, 0.5), (400, 2.0)])
def test_online_fit_stays_batched(method_cls: type, n_windows: int, budget_seconds: float) -> None:
    predictions, truths = _panel(10, n_windows, 12)
    method = method_cls(alpha=0.1)
    start = time.perf_counter()
    method.fit(predictions, truths)
    elapsed = time.perf_counter() - start
    assert elapsed < budget_seconds, (
        f"{method_cls.__name__}.fit took {elapsed:.2f}s on a 10x{n_windows}x12 panel "
        f"(budget {budget_seconds}s) — a per-cell Python loop or a per-step "
        "re-sort of the full history has likely come back."
    )


@pytest.mark.perf
@pytest.mark.parametrize(
    "method_cls", [AdaptiveConformalInference, AggregatedAdaptiveConformalInference]
)
def test_interval_stays_batched(method_cls: type) -> None:
    predictions, truths = _panel(10, 400, 12)
    method = method_cls(alpha=0.1)
    fitted = method.fit(predictions, truths)
    next_step = predictions[:, -1:, :]

    start = time.perf_counter()
    fitted.interval(next_step)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.20, (
        f"{method_cls.__name__}.interval took {elapsed:.3f}s for one step against a "
        "10x400x12 history (budget 0.20s) — the threshold is likely being computed "
        "per cell rather than in one gather."
    )


@pytest.mark.perf
@pytest.mark.parametrize(
    ("method_cls", "shape", "budget_seconds"),
    [
        (AdaptiveConformalInference, (10, 400, 12), 1.0),
        (AggregatedAdaptiveConformalInference, (10, 400, 12), 1.0),
        # NexCP re-argsorts the whole panel per step. That is a known cost, not a
        # regression, so the budget is set against it rather than against zero.
        (NonexchangeableConformalPrediction, (100, 200, 12), 6.0),
    ],
)
def test_update_stays_batched(
    method_cls: type, shape: tuple[int, int, int], budget_seconds: float
) -> None:
    predictions, truths = _panel(*shape)
    method = method_cls(alpha=0.1)
    fitted = method.fit(predictions, truths)
    next_pred = predictions[:, -1:, :]
    next_truth = truths[:, -1:, :]

    start = time.perf_counter()
    for _ in range(20):
        fitted.update(next_pred, next_truth)
    elapsed = time.perf_counter() - start
    assert elapsed < budget_seconds, (
        f"20 {method_cls.__name__}.update steps took {elapsed:.2f}s on a "
        f"{shape[0]}x{shape[1]}x{shape[2]} panel (budget {budget_seconds}s) — a "
        "per-cell Python loop in the online update path has likely come back."
    )


@pytest.mark.perf
def test_weighted_quantile_scales_with_panel_width() -> None:
    predictions, truths = _panel(200, 50, 12)
    method = NonexchangeableConformalPrediction(alpha=0.1)
    start = time.perf_counter()
    method.fit(predictions, truths)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.30, (
        f"NonexchangeableConformalPrediction.fit took {elapsed:.2f}s on a 200x50x12 "
        "panel (budget 0.30s) — the weighted quantile is likely iterating the "
        "(series x horizon) grid again."
    )
