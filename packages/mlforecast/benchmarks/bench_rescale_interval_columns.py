"""Benchmark _rescale_interval_columns variants.

Run with: uv run tmp/bench_rescale.py
"""
import timeit

import numpy as np
import pandas as pd
import polars as pl

from mlforecast.conformal_prediction import (
    _rescale_interval_columns,
    _rescale_interval_columns_narwhals,
    _rescale_interval_columns_per_model,
    _rescale_interval_columns_single_call,
)

N_SERIES = 10_000
HORIZON = 10
N_ROWS = N_SERIES * HORIZON
MODEL_NAMES = ["m1", "m2", "m3"]
LEVELS = [80, 90, 95]
REPEATS = 20

rng = np.random.default_rng(42)

# Build base data dict
base = {"unique_id": np.repeat(np.arange(N_SERIES), HORIZON)}
for m in MODEL_NAMES:
    base[m] = rng.standard_normal(N_ROWS)
    for lv in LEVELS:
        half = rng.uniform(0.1, 1.0, N_ROWS)
        base[f"{m}-lo-{lv}"] = base[m] - half
        base[f"{m}-hi-{lv}"] = base[m] + half

sigma_tgt = rng.uniform(0.5, 2.0, N_ROWS)

df_pd = pd.DataFrame(base)
df_pl = pl.DataFrame(base)

FUNS = {
    "triple_loop (original)": _rescale_interval_columns,
    "single_call":            _rescale_interval_columns_single_call,
    "per_model":              _rescale_interval_columns_per_model,
    "narwhals":               _rescale_interval_columns_narwhals,
}

header = f"{'function':<30} {'pandas ms':>12} {'polars ms':>12}"
print(header)
print("-" * len(header))

for name, fn in FUNS.items():
    t_pd = timeit.timeit(
        lambda fn=fn: fn(df_pd.copy(), MODEL_NAMES, LEVELS, sigma_tgt),
        number=REPEATS,
    ) / REPEATS * 1000

    t_pl = timeit.timeit(
        lambda fn=fn: fn(df_pl, MODEL_NAMES, LEVELS, sigma_tgt),
        number=REPEATS,
    ) / REPEATS * 1000

    print(f"{name:<30} {t_pd:>11.2f}  {t_pl:>11.2f}")
