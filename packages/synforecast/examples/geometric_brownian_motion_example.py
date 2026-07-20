"""Example usage of the Geometric Brownian Motion (GBM) generator, the standard
model for stock prices with exponential growth and random fluctuations."""

import polars as pl

from synforecast.generators import GeometricBrownianMotionGenerator


def main() -> None:
    """Generate and display Geometric Brownian Motion examples."""
    basic_gen = GeometricBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,  # 5% annual drift
        sigma=0.2,  # 20% annual volatility
        initial_value=100.0,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print("Basic stock price simulation:")
    print(basic_df.head(10))
    print(
        f"Statistics: Mean={basic_df['y'].mean():.4f}, "
        f"Min={basic_df['y'].min():.4f}, Max={basic_df['y'].max():.4f}"
    )

    high_growth_gen = GeometricBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.15,
        sigma=0.3,
        initial_value=50.0,
        seed=42,
    )
    high_growth_df = high_growth_gen.generate(n_series=1)
    print(
        f"\nHigh growth (mu=0.15): Mean={high_growth_df['y'].mean():.4f}, "
        f"Min={high_growth_df['y'].min():.4f}, Max={high_growth_df['y'].max():.4f}"
    )

    declining_gen = GeometricBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=-0.05,
        sigma=0.2,
        initial_value=100.0,
        seed=42,
    )
    declining_df = declining_gen.generate(n_series=1)
    print(
        f"Negative drift (mu=-0.05): Mean={declining_df['y'].mean():.4f}, "
        f"Min={declining_df['y'].min():.4f}, Max={declining_df['y'].max():.4f}"
    )

    high_vol_gen = GeometricBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.5,
        initial_value=100.0,
        seed=42,
    )
    high_vol_df = high_vol_gen.generate(n_series=1)
    print(
        f"High volatility (sigma=0.5): Mean={high_vol_df['y'].mean():.4f}, "
        f"Min={high_vol_df['y'].min():.4f}, Max={high_vol_df['y'].max():.4f}"
    )

    multi_gen = GeometricBrownianMotionGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.2,
        initial_value=100.0,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=5)
    print(f"\nGenerated 5 simulations with {len(multi_df)} total observations")
    print("First simulation preview:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("gbm_basic_example.csv")
    high_growth_df.write_csv("gbm_high_growth_example.csv")
    declining_df.write_csv("gbm_declining_example.csv")
    high_vol_df.write_csv("gbm_high_volatility_example.csv")
    multi_df.write_csv("gbm_multiple_simulations_example.csv")
    print("\nSaved CSV outputs for all examples.")


if __name__ == "__main__":
    main()
