"""Example usage of the Poisson Process generator for modeling random event
occurrences over time, as cumulative or per-period counts."""

import polars as pl

from synforecast.generators import PoissonProcessGenerator


def main() -> None:
    """Generate and display Poisson Process examples."""
    basic_gen = PoissonProcessGenerator(
        min_length=50,
        max_length=50,
        freq="h",
        engine="polars",
        lambda_rate=3.0,  # Average 3 events per hour
        cumulative=False,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print("Hourly event counts (lambda=3.0):")
    print(basic_df.head(10))
    print(
        f"Statistics: Mean={basic_df['y'].mean():.4f}, "
        f"Total Events={basic_df['y'].sum():.0f}"
    )

    cumulative_gen = PoissonProcessGenerator(
        min_length=50,
        max_length=50,
        freq="h",
        engine="polars",
        lambda_rate=3.0,
        cumulative=True,
        seed=42,
    )
    cumulative_df = cumulative_gen.generate(n_series=1)
    print("\nCumulative counts:")
    print(cumulative_df.head(10))
    print(f"Final cumulative count: {cumulative_df['y'].tail(1).item():.0f}")

    high_rate_gen = PoissonProcessGenerator(
        min_length=50,
        max_length=50,
        freq="h",
        engine="polars",
        lambda_rate=10.0,
        cumulative=False,
        seed=42,
    )
    high_rate_df = high_rate_gen.generate(n_series=1)
    print(
        f"\nHigh rate (lambda=10.0): Mean={high_rate_df['y'].mean():.4f}, "
        f"Total Events={high_rate_df['y'].sum():.0f}"
    )

    low_rate_gen = PoissonProcessGenerator(
        min_length=100,
        max_length=100,
        freq="h",
        engine="polars",
        lambda_rate=0.5,
        cumulative=False,
        seed=42,
    )
    low_rate_df = low_rate_gen.generate(n_series=1)
    print(
        f"Low rate (lambda=0.5): Mean={low_rate_df['y'].mean():.4f}, "
        f"Total Events={low_rate_df['y'].sum():.0f}"
    )

    daily_gen = PoissonProcessGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        lambda_rate=5.0,
        cumulative=False,
        seed=42,
    )
    daily_df = daily_gen.generate(n_series=1)
    print(
        f"Daily counts (lambda=5.0): Mean={daily_df['y'].mean():.4f}, "
        f"Total Events={daily_df['y'].sum():.0f}"
    )

    multi_gen = PoissonProcessGenerator(
        min_length=50,
        max_length=50,
        freq="h",
        engine="polars",
        lambda_rate=3.0,
        cumulative=False,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=4)
    print(f"\nGenerated 4 processes with {len(multi_df)} total observations")
    print("First process preview:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("poisson_process_basic_example.csv")
    cumulative_df.write_csv("poisson_process_cumulative_example.csv")
    high_rate_df.write_csv("poisson_process_high_rate_example.csv")
    low_rate_df.write_csv("poisson_process_low_rate_example.csv")
    daily_df.write_csv("poisson_process_daily_example.csv")
    multi_df.write_csv("poisson_process_multiple_series_example.csv")
    print("\nSaved CSV outputs for all examples.")


if __name__ == "__main__":
    main()
