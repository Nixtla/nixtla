"""Example usage of SynSet dataset class."""

import polars as pl

from synforecast import SynSet
from synforecast.generators import RandomWalkGenerator, SeasonalGenerator


def main() -> None:
    """Generate and display a dataset with multiple generator types."""
    rw_gen = RandomWalkGenerator(
        min_length=100,
        max_length=150,
        freq="h",
        engine="polars",
        drift=0.1,
        volatility=1.5,
        start_value=100.0,
        seed=42,
    )
    seasonal_gen = SeasonalGenerator(
        min_length=100,
        max_length=150,
        freq="h",
        engine="polars",
        seasonality_period=24,
        seasonality_amplitude=15.0,
        trend=0.05,
        noise_level=2.0,
        base_level=50.0,
        seed=123,
    )

    # Combine both generators; ids are assigned across generators, so the
    # random walk series get ids 0-2 and the seasonal series get ids 3-5
    dataset = SynSet([rw_gen, seasonal_gen])
    df = dataset.generate(n_series_per_generator=3)

    print(f"Generated {df['unique_id'].n_unique()} time series")
    print(f"Total observations: {len(df)}")
    print(df.head(10))

    stats = (
        df.group_by("unique_id")
        .agg(
            [
                pl.col("y").count().alias("count"),
                pl.col("y").min().alias("min_value"),
                pl.col("y").max().alias("max_value"),
                pl.col("y").mean().alias("mean_value"),
                pl.col("y").std().alias("std_value"),
            ]
        )
        .sort("unique_id")
    )
    print(f"\nStatistics by series:\n{stats}")

    print("\nSeries 0 (random walk, first 10 rows):")
    print(df.filter(pl.col("unique_id") == "0").head(10))

    print("\nSeries 3 (seasonal, first 10 rows):")
    print(df.filter(pl.col("unique_id") == "3").head(10))


if __name__ == "__main__":
    main()
