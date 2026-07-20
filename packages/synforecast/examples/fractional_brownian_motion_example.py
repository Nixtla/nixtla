"""Example usage of FractionalBrownianMotionGenerator."""

import polars as pl

from synforecast.generators import FractionalBrownianMotionGenerator


def main() -> None:
    """Generate and display fractional Brownian motion time series."""
    # The Hurst exponent controls long-range dependence:
    # H < 0.5 is anti-persistent, H = 0.5 is standard BM, H > 0.5 is persistent
    hurst_values = [0.2, 0.5, 0.8]
    behaviors = [
        "Anti-persistent (mean-reverting)",
        "Standard BM",
        "Persistent (trending)",
    ]

    for hurst, behavior in zip(hurst_values, behaviors, strict=True):
        generator = FractionalBrownianMotionGenerator(
            min_length=200,
            max_length=200,
            freq="D",
            engine="polars",
            hurst=hurst,
            sigma=1.0,
            method="cholesky",
            seed=42,
        )
        df = generator.generate(n_series=1)

        # Recover the Hurst exponent from the generated series
        estimated_h = generator.estimate_hurst(df["y"].to_numpy(), method="rs")

        print(f"\nHurst = {hurst}: {behavior}")
        print(f"Estimated Hurst (R/S): {estimated_h:.3f}")

        stats = df.group_by("unique_id").agg(
            [
                pl.col("y").min().alias("min"),
                pl.col("y").max().alias("max"),
                pl.col("y").mean().alias("mean"),
                pl.col("y").std().alias("std"),
            ]
        )
        print(f"Statistics: {stats.to_dicts()[0]}")

    # Compare fBm (cumulative) vs fGn (increments)
    fbm_gen = FractionalBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        hurst=0.7,
        return_increments=False,
        seed=42,
    )
    fbm_df = fbm_gen.generate(n_series=1)

    fgn_gen = FractionalBrownianMotionGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        hurst=0.7,
        return_increments=True,
        seed=42,
    )
    fgn_df = fgn_gen.generate(n_series=1)

    print(f"\nfBm mean: {fbm_df['y'].mean():.3f}, std: {fbm_df['y'].std():.3f}")
    print(f"fGn mean: {fgn_df['y'].mean():.3f}, std: {fgn_df['y'].std():.3f}")

    print("\nModel information:")
    for key, value in fbm_gen.get_model_info().items():
        print(f"{key}: {value}")

    output_file = "fractional_brownian_motion_data.csv"
    fbm_df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
