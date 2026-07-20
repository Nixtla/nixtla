"""Example usage of the Jump Diffusion (Merton) generator: GBM with random jumps
for modeling assets that experience sudden price changes."""

import polars as pl

from synforecast.generators import JumpDiffusionGenerator


def main() -> None:
    """Generate and display Jump Diffusion examples."""
    basic_gen = JumpDiffusionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.15,
        lambda_jump=0.2,  # Average 0.2 jumps per time unit
        jump_mean=0.0,
        jump_std=0.1,
        initial_value=100.0,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print("Basic jump diffusion:")
    print(basic_df.head(10))
    print(f"Statistics: Mean={basic_df['y'].mean():.4f}, Std={basic_df['y'].std():.4f}")

    frequent_gen = JumpDiffusionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.1,
        lambda_jump=1.0,  # More frequent jumps
        jump_mean=0.0,
        jump_std=0.05,
        initial_value=100.0,
        seed=42,
    )
    frequent_df = frequent_gen.generate(n_series=1)
    print(
        f"\nFrequent jumps: Mean={frequent_df['y'].mean():.4f}, "
        f"Std={frequent_df['y'].std():.4f}"
    )

    # Positive jump_mean models upward news shocks
    positive_jump_gen = JumpDiffusionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.15,
        lambda_jump=0.3,
        jump_mean=0.1,
        jump_std=0.05,
        initial_value=100.0,
        seed=42,
    )
    positive_jump_df = positive_jump_gen.generate(n_series=1)
    print(
        f"Positive jumps: Mean={positive_jump_df['y'].mean():.4f}, "
        f"Std={positive_jump_df['y'].std():.4f}"
    )

    # Negative jump_mean models market crashes
    negative_jump_gen = JumpDiffusionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.15,
        lambda_jump=0.2,
        jump_mean=-0.15,
        jump_std=0.08,
        initial_value=100.0,
        seed=42,
    )
    negative_jump_df = negative_jump_gen.generate(n_series=1)
    print(
        f"Negative jumps: Mean={negative_jump_df['y'].mean():.4f}, "
        f"Std={negative_jump_df['y'].std():.4f}"
    )

    high_vol_jump_gen = JumpDiffusionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.15,
        lambda_jump=0.3,
        jump_mean=0.0,
        jump_std=0.2,  # High jump size volatility
        initial_value=100.0,
        seed=42,
    )
    high_vol_jump_df = high_vol_jump_gen.generate(n_series=1)
    print(
        f"High jump volatility: Mean={high_vol_jump_df['y'].mean():.4f}, "
        f"Std={high_vol_jump_df['y'].std():.4f}"
    )

    multi_gen = JumpDiffusionGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        mu=0.05,
        sigma=0.15,
        lambda_jump=0.2,
        jump_mean=0.0,
        jump_std=0.1,
        initial_value=100.0,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print("First series preview:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("jump_diffusion_basic_example.csv")
    frequent_df.write_csv("jump_diffusion_frequent_jumps_example.csv")
    positive_jump_df.write_csv("jump_diffusion_positive_jumps_example.csv")
    negative_jump_df.write_csv("jump_diffusion_negative_jumps_example.csv")
    high_vol_jump_df.write_csv("jump_diffusion_high_volatility_example.csv")
    multi_df.write_csv("jump_diffusion_multiple_series_example.csv")
    print("\nSaved CSV outputs for all examples.")


if __name__ == "__main__":
    main()
