"""Example usage of RandomWalkGenerator."""

import polars as pl

from synforecast.generators import RandomWalkGenerator


def main() -> None:
    """Generate and display random walk time series."""
    generator = RandomWalkGenerator(
        min_length=100,
        max_length=150,
        freq="h",
        engine="polars",
        drift=0.1,
        volatility=1.5,
        start_value=100.0,
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} time series")
    print(f"Total observations: {len(df)}")
    print(df.head(10))

    stats = df.group_by("unique_id").agg(
        [
            pl.col("y").count().alias("count"),
            pl.col("y").min().alias("min_value"),
            pl.col("y").max().alias("max_value"),
            pl.col("y").mean().alias("mean_value"),
            pl.col("y").std().alias("std_value"),
        ]
    )
    print("\nStatistics by series:")
    print(stats)

    output_file = "random_walk_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
