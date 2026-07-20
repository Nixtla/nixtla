"""Example usage of SeasonalGenerator."""

import polars as pl

from synforecast.generators import SeasonalGenerator


def main() -> None:
    """Generate and display seasonal time series."""
    generator = SeasonalGenerator(
        min_length=168,  # One week of hourly data
        max_length=336,  # Two weeks of hourly data
        freq="h",
        engine="polars",
        seasonality_period=24,  # Daily seasonality
        seasonality_amplitude=15.0,
        trend=0.05,
        noise_level=2.0,
        base_level=50.0,
        seed=123,
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

    print("\nSeries 0, first 24 hours:")
    print(df.filter(pl.col("unique_id") == "0").head(24))

    output_file = "seasonal_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
