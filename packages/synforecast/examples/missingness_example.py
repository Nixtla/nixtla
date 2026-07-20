"""Example usage of missing data patterns in generators."""

import numpy as np
import polars as pl

from synforecast.generators import RandomWalkGenerator, SeasonalGenerator


def main() -> None:
    """Generate and display time series with missing data patterns."""
    # Random missing pattern: each observation independently missing
    gen_random = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.15,
        seed=42,
    )
    df_random = gen_random.generate(n_series=1)

    print("Random missing pattern:")
    print(df_random.head(20))

    values = df_random["y"].to_numpy()
    nan_count = np.sum(np.isnan(values))
    print(f"Missing values: {nan_count} ({nan_count / len(values):.1%})")
    non_nan = values[~np.isnan(values)]
    print(f"Non-missing mean: {non_nan.mean():.2f}, std: {non_nan.std():.2f}")

    # Block missing pattern: consecutive gaps, e.g. sensor outages
    gen_block = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        missing_data=True,
        missing_pattern="block",
        missing_rate=0.2,
        missing_block_size=5,
        seed=123,
    )
    df_block = gen_block.generate(n_series=1)

    values_block = df_block["y"].to_numpy()
    nan_count_block = np.sum(np.isnan(values_block))

    max_consecutive = 0
    current_consecutive = 0
    for val in values_block:
        if np.isnan(val):
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    print(
        f"\nBlock pattern (block size 5): {nan_count_block} missing "
        f"({nan_count_block / len(values_block):.1%}), "
        f"longest gap: {max_consecutive} days"
    )

    # Seasonal missing pattern: missingness follows a weekly cycle
    gen_seasonal = SeasonalGenerator(
        min_length=365,
        max_length=365,
        freq="D",
        engine="polars",
        missing_data=True,
        missing_pattern="seasonal",
        missing_rate=0.12,
        missing_seasonal_period=7,
        seed=456,
    )
    df_seasonal = gen_seasonal.generate(n_series=1)

    week_stats = (
        df_seasonal.with_columns(
            (pl.col("ds").dt.ordinal_day() % 7).alias("day_of_week"),
            pl.col("y").is_nan().alias("is_missing"),
        )
        .group_by("day_of_week")
        .agg(
            pl.col("is_missing").sum().alias("missing_count"),
            pl.col("is_missing").count().alias("total_count"),
        )
        .with_columns(
            (pl.col("missing_count") / pl.col("total_count") * 100).alias(
                "missing_rate_pct"
            )
        )
        .sort("day_of_week")
    )
    print("\nSeasonal pattern (weekly), missing rate by day of week:")
    print(week_stats)

    # Actual missing rate tracks the target rate
    print("\nTarget vs actual missing rate:")
    for rate in [0.05, 0.15, 0.30]:
        gen = RandomWalkGenerator(
            min_length=300,
            max_length=300,
            freq="D",
            engine="polars",
            missing_data=True,
            missing_pattern="random",
            missing_rate=rate,
            seed=789,
        )
        df = gen.generate(n_series=1)
        values = df["y"].to_numpy()
        actual = np.sum(np.isnan(values)) / len(values)
        print(f"  target={rate:.0%}, actual={actual:.1%}")

    # Multiple series each get an independent missing pattern
    gen_multi = RandomWalkGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.2,
        seed=1234,
    )
    df_multi = gen_multi.generate(n_series=3)

    print("\nMissing data by series:")
    for series_id in df_multi["unique_id"].unique().sort():
        series_values = df_multi.filter(pl.col("unique_id") == series_id)[
            "y"
        ].to_numpy()
        nan_count = np.sum(np.isnan(series_values))
        print(
            f"  {series_id}: {nan_count} missing ({nan_count / len(series_values):.1%})"
        )

    df_random.write_csv("missing_data_random.csv")
    df_block.write_csv("missing_data_block.csv")
    df_seasonal.write_csv("missing_data_seasonal.csv")
    df_multi.write_csv("missing_data_multiple.csv")


if __name__ == "__main__":
    main()
