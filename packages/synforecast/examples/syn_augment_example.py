"""Example usage of SynAugment for time series data augmentation."""

import numpy as np
import polars as pl

from synforecast.dataset import SynAugment
from synforecast.generators import RandomWalkGenerator, SeasonalGenerator

# Basic augmentation of a random walk with drift
n = 100
np.random.seed(42)
df = pl.DataFrame(
    {
        "unique_id": ["0"] * n,
        "ds": pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        ),
        "y": np.cumsum(np.random.randn(n) * 0.5 + 0.01),
    }
)

print(f"Original dataset: {len(df)} rows, {df['unique_id'].n_unique()} series")
print(df.head(5))

augmenter = SynAugment(seed=42)
augmented_df = augmenter.augment(df, n_augment=3)

print(
    f"\nAugmented dataset: {len(augmented_df)} rows, "
    f"{augmented_df['unique_id'].n_unique()} series"
)
print(f"Series IDs: {sorted(augmented_df['unique_id'].unique().to_list())}")


# Analyze series to see which generator would be selected for each pattern
n = 150
t = np.arange(n)

seasonal_values = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 2
random_walk_values = np.cumsum(np.random.randn(n))
intermittent_values = np.zeros(n)
demand_times = np.random.choice(n, size=30, replace=False)
intermittent_values[demand_times] = np.random.randint(1, 20, 30)

multi_df = pl.DataFrame(
    {
        "unique_id": ["seasonal"] * n + ["random_walk"] * n + ["intermittent"] * n,
        "ds": list(
            pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            )
        )
        * 3,
        "y": list(seasonal_values)
        + list(random_walk_values)
        + list(intermittent_values),
    }
)

augmenter = SynAugment(seed=42)
analysis = augmenter.analyze(multi_df)

print("\nAnalysis results per series:")
for series_id, info in analysis.items():
    props = info["properties"]
    print(f"  {series_id}:")
    print(f"    Recommended generator: {info['recommended_generator']}")
    print(f"    Has seasonality: {props['seasonality']['has_seasonality']}")
    print(f"    Has trend: {props['trend']['has_trend']}")
    print(f"    Is stationary: {props['stationarity']['is_stationary']}")
    print(f"    Is intermittent: {props['intermittency']['is_intermittent']}")


# generator_override forces a specific generator for selected series
augmented_override = augmenter.augment(
    multi_df,
    n_augment=1,
    generator_override={
        "seasonal": "SARIMAGenerator",
        "random_walk": "FractionalBrownianMotionGenerator",
    },
)

print(
    f"\nAugmented with overrides: {augmented_override['unique_id'].n_unique()} series"
)
print(f"Series IDs: {sorted(augmented_override['unique_id'].unique().to_list())}")


# Synthetic series preserve the statistics of the original
augmented = augmenter.augment(df, n_augment=5)

original = augmented.filter(pl.col("unique_id") == "0")
print("\nOriginal series (id 0):")
print(f"  mean={original['y'].mean():.4f}, std={original['y'].std():.4f}")

print("Synthetic series:")
for i in range(5):
    aug = augmented.filter(pl.col("unique_id") == f"0_aug_{i}")
    print(f"  0_aug_{i}: mean={aug['y'].mean():.4f}, std={aug['y'].std():.4f}")


# Augmenting a dataset produced by SynForecast generators
rw_gen = RandomWalkGenerator(
    min_length=100,
    max_length=100,
    freq="h",
    engine="polars",
    drift=0.05,
    volatility=1.0,
    seed=42,
)
seasonal_gen = SeasonalGenerator(
    min_length=100,
    max_length=100,
    freq="h",
    engine="polars",
    seasonality_period=24,
    seasonality_amplitude=5.0,
    trend=0.01,
    seed=43,
)

rw_df = rw_gen.generate(n_series=2)
seasonal_df = seasonal_gen.generate(n_series=2, start_id=2)
combined_df = pl.concat([rw_df, seasonal_df])

print(f"\nGenerated dataset: {combined_df['unique_id'].n_unique()} series")
print(f"Series IDs: {sorted(combined_df['unique_id'].unique().to_list())}")

augmenter = SynAugment(seed=42)
augmented_combined = augmenter.augment(combined_df, n_augment=2)

print(f"After augmentation: {augmented_combined['unique_id'].n_unique()} series")
print(f"Series IDs: {sorted(augmented_combined['unique_id'].unique().to_list())}")


# Low-level API: augment a single series from raw arrays
single_series = df.filter(pl.col("unique_id") == "0")
values = single_series["y"].to_numpy()
timestamps = single_series["ds"].to_numpy()

augmenter = SynAugment(seed=42)
augmented_tuples = augmenter.augment_single_series(
    series_id="my_series",
    values=values,
    timestamps=timestamps,
    n_augment=2,
    generator_name=None,  # Auto-detect
)

print(f"\nGenerated {len(augmented_tuples)} augmented series via low-level API:")
for aug_id, aug_values, _aug_ts in augmented_tuples:
    print(f"  {aug_id}: length={len(aug_values)}, mean={np.mean(aug_values):.4f}")
