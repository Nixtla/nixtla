"""Example usage of IntermittentDemandGenerator."""

import numpy as np
import polars as pl

from synforecast.generators import IntermittentDemandGenerator


def main() -> None:
    """Generate and display intermittent demand time series."""
    # Random intermittent pattern with Poisson-distributed demand sizes
    gen_random = IntermittentDemandGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        intermittent_pattern="random",
        demand_probability=0.2,
        demand_distribution="poisson",
        demand_mean=5.0,
        seed=42,
    )
    df_random = gen_random.generate(n_series=1)

    print("Random intermittent pattern:")
    print(df_random.head(20))

    zero_count = (df_random["y"] == 0).sum()
    zero_ratio = zero_count / len(df_random)
    non_zero_demands = df_random.filter(pl.col("y") > 0)["y"].to_numpy()
    print(f"Zero demand days: {zero_count} ({zero_ratio:.1%})")
    if len(non_zero_demands) > 0:
        print(
            f"Demand when non-zero: mean={non_zero_demands.mean():.2f}, "
            f"std={non_zero_demands.std():.2f}, "
            f"min={non_zero_demands.min():.0f}, max={non_zero_demands.max():.0f}"
        )

    # Clustered pattern: demand occurs in runs of consecutive days (lumpy demand)
    gen_clustered = IntermittentDemandGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        intermittent_pattern="clustered",
        demand_probability=0.15,
        cluster_size=5,
        demand_distribution="negative_binomial",
        demand_mean=8.0,
        demand_std=4.0,
        seed=123,
    )
    df_clustered = gen_clustered.generate(n_series=1)

    zero_ratio = (df_clustered["y"] == 0).sum() / len(df_clustered)
    print(
        f"\nClustered pattern (cluster size 5): "
        f"{(df_clustered['y'] > 0).sum()} demand days, {zero_ratio:.1%} zero days"
    )

    # Seasonal pattern: demand probability peaks periodically
    gen_seasonal = IntermittentDemandGenerator(
        min_length=365,
        max_length=365,
        freq="D",
        engine="polars",
        intermittent_pattern="seasonal",
        demand_probability=0.1,
        seasonal_period=30,
        seasonal_peak_prob=0.4,
        demand_distribution="lognormal",
        demand_mean=10.0,
        demand_std=5.0,
        seed=456,
    )
    df_seasonal = gen_seasonal.generate(n_series=1)

    period_stats = (
        df_seasonal.with_columns(
            ((pl.col("ds").dt.ordinal_day() - 1) // 30).alias("period")
        )
        .group_by("period")
        .agg(
            (pl.col("y") > 0).sum().alias("demand_days"),
            pl.col("y").sum().alias("total_demand"),
        )
        .sort("period")
    )
    print("\nSeasonal pattern, demand by 30-day period:")
    print(period_stats.head(12))

    # Compare demand size distributions
    results = {}
    for dist in ["poisson", "negative_binomial", "lognormal", "gamma"]:
        gen = IntermittentDemandGenerator(
            min_length=300,
            max_length=300,
            freq="D",
            engine="polars",
            demand_probability=0.3,
            demand_distribution=dist,
            demand_mean=7.0,
            demand_std=3.0,
            seed=789,
        )
        df = gen.generate(n_series=1)
        non_zero = df.filter(pl.col("y") > 0)["y"].to_numpy()
        if len(non_zero) > 0:
            results[dist] = non_zero

    print("\nDistribution comparison (demand > 0):")
    print(f"{'Distribution':<20} {'Count':<8} {'Mean':<8} {'Std':<8}")
    for dist, non_zero in results.items():
        print(
            f"{dist:<20} {len(non_zero):<8} "
            f"{non_zero.mean():<8.2f} {non_zero.std():<8.2f}"
        )

    # Bulk orders: min_demand enforces a minimum order quantity
    gen_bulk = IntermittentDemandGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        demand_probability=0.1,
        demand_distribution="gamma",
        demand_mean=50.0,
        demand_std=20.0,
        min_demand=20,
        seed=999,
    )
    df_bulk = gen_bulk.generate(n_series=1)

    non_zero_bulk = df_bulk.filter(pl.col("y") > 0)["y"].to_numpy()
    if len(non_zero_bulk) > 0:
        print(
            f"\nBulk orders: {len(non_zero_bulk)} orders, "
            f"mean size {non_zero_bulk.mean():.2f}, "
            f"all >= 20: {np.all(non_zero_bulk >= 20)}"
        )

    # Multiple series
    gen_multi = IntermittentDemandGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        demand_probability=0.25,
        demand_distribution="poisson",
        demand_mean=6.0,
        seed=1234,
    )
    df_multi = gen_multi.generate(n_series=3)

    series_stats = (
        df_multi.group_by("unique_id")
        .agg(
            (pl.col("y") == 0).sum().alias("zero_days"),
            (pl.col("y") > 0).sum().alias("demand_days"),
            pl.col("y").sum().alias("total_demand"),
            pl.col("y").filter(pl.col("y") > 0).mean().alias("avg_demand_when_nonzero"),
        )
        .sort("unique_id")
    )
    print("\nStatistics for 3 generated series:")
    print(series_stats)

    df_random.write_csv("intermittent_demand_random.csv")
    df_clustered.write_csv("intermittent_demand_clustered.csv")
    df_seasonal.write_csv("intermittent_demand_seasonal.csv")
    df_bulk.write_csv("intermittent_demand_bulk.csv")
    df_multi.write_csv("intermittent_demand_multiple.csv")


if __name__ == "__main__":
    main()
