"""Example usage of the Ornstein-Uhlenbeck generator, a mean-reverting process
used for interest rates, commodity prices, and similar variables."""

import polars as pl

from synforecast.generators import OrnsteinUhlenbeckGenerator


def main() -> None:
    """Generate and display Ornstein-Uhlenbeck examples."""
    basic_gen = OrnsteinUhlenbeckGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        theta=0.5,  # Speed of mean reversion
        mu=100.0,  # Long-term mean
        sigma=5.0,
        initial_value=80.0,  # Start below the mean
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print("Basic mean-reverting process (mu=100):")
    print(basic_df.head(10))
    print(f"Statistics: Mean={basic_df['y'].mean():.4f}, Std={basic_df['y'].std():.4f}")

    fast_gen = OrnsteinUhlenbeckGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        theta=1.5,  # Euler discretization requires theta * dt < 2
        mu=50.0,
        sigma=3.0,
        initial_value=100.0,  # Start far from the mean
        seed=42,
    )
    fast_df = fast_gen.generate(n_series=1)
    print(
        f"\nFast reversion (theta=1.5): Mean={fast_df['y'].mean():.4f}, "
        f"Std={fast_df['y'].std():.4f}"
    )

    slow_gen = OrnsteinUhlenbeckGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        theta=0.1,
        mu=100.0,
        sigma=5.0,
        initial_value=80.0,
        seed=42,
    )
    slow_df = slow_gen.generate(n_series=1)
    print(
        f"Slow reversion (theta=0.1): Mean={slow_df['y'].mean():.4f}, "
        f"Std={slow_df['y'].std():.4f}"
    )

    high_vol_gen = OrnsteinUhlenbeckGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        theta=0.5,
        mu=100.0,
        sigma=15.0,
        initial_value=100.0,
        seed=42,
    )
    high_vol_df = high_vol_gen.generate(n_series=1)
    print(
        f"High volatility (sigma=15): Mean={high_vol_df['y'].mean():.4f}, "
        f"Std={high_vol_df['y'].std():.4f}"
    )

    multi_gen = OrnsteinUhlenbeckGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        theta=0.5,
        mu=100.0,
        sigma=5.0,
        initial_value=80.0,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print("First series preview:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("ornstein_uhlenbeck_basic_example.csv")
    fast_df.write_csv("ornstein_uhlenbeck_fast_reversion_example.csv")
    slow_df.write_csv("ornstein_uhlenbeck_slow_reversion_example.csv")
    high_vol_df.write_csv("ornstein_uhlenbeck_high_volatility_example.csv")
    multi_df.write_csv("ornstein_uhlenbeck_multiple_series_example.csv")
    print("\nSaved CSV outputs for all examples.")


if __name__ == "__main__":
    main()
