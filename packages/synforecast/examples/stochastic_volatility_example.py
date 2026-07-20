"""Example usage of StochasticVolatilityGenerator (Heston and SABR models)."""

import numpy as np
import polars as pl

from synforecast.generators import StochasticVolatilityGenerator


def main() -> None:
    """Generate and display stochastic volatility time series."""
    # Heston model, one year of daily data
    generator = StochasticVolatilityGenerator(
        min_length=252,
        max_length=252,
        freq="D",
        engine="polars",
        model="heston",
        initial_price=100.0,
        initial_vol=0.04,  # 20% annualized vol
        drift=0.05,
        mean_vol=0.04,
        vol_mean_reversion=2.0,
        vol_of_vol=0.3,
        correlation=-0.7,  # Strong leverage effect
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} Heston price paths")

    stats = df.group_by("unique_id").agg(
        [
            pl.col("y").first().alias("start_price"),
            pl.col("y").last().alias("end_price"),
            pl.col("y").min().alias("min_price"),
            pl.col("y").max().alias("max_price"),
        ]
    )
    print(f"\nPrice statistics:\n{stats}")

    # Joint price and volatility paths
    prices, vols, ids = generator.generate_with_volatility(n_series=1)
    print(f"\nPrice range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"Volatility range: [{vols.min():.3f}, {vols.max():.3f}]")
    print(
        f"Mean volatility: {vols.mean():.3f} "
        f"({vols.mean() * np.sqrt(252) * 100:.1f}% annualized)"
    )

    # Leverage effect: volatility rises when prices fall
    returns = np.diff(np.log(prices))
    vol_changes = np.diff(vols)
    corr = np.corrcoef(returns, vol_changes)[0, 1]
    print(f"Return-VolChange correlation: {corr:.3f} (negative = leverage effect)")

    sabr_gen = StochasticVolatilityGenerator(
        min_length=252,
        max_length=252,
        freq="D",
        engine="polars",
        model="sabr",
        initial_price=100.0,
        beta=0.5,  # CEV exponent
        correlation=-0.3,
        seed=42,
    )
    sabr_df = sabr_gen.generate(n_series=1)
    print(f"\nSABR price range: [{sabr_df['y'].min():.2f}, {sabr_df['y'].max():.2f}]")

    print("\nModel information (Heston):")
    info = generator.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    strikes = np.array([80, 90, 95, 100, 105, 110, 120])
    impl_vols = sabr_gen.implied_volatility_smile(strikes, maturity=1.0)

    print("\nImplied volatility smile (SABR):")
    print("Strike | Implied Vol")
    for k, iv in zip(strikes, impl_vols, strict=True):
        print(f"  {k:3d}  |   {iv * 100:.2f}%")

    print("\nOutput types:")
    for output_type in ["price", "returns", "volatility"]:
        gen = StochasticVolatilityGenerator(
            min_length=100,
            max_length=100,
            freq="D",
            engine="polars",
            output_type=output_type,
            seed=42,
        )
        out_df = gen.generate(n_series=1)
        vals = out_df["y"].to_numpy()
        print(f"{output_type:12s}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    output_file = "stochastic_volatility_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
