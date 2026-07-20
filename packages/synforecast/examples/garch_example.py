"""Example usage of GARCHGenerator, which models volatility clustering
as seen in financial time series."""

import polars as pl

from synforecast.generators import GARCHGenerator


def main() -> None:
    """Generate and display GARCH examples."""
    # GARCH(1,1)
    basic_gen = GARCHGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=1,
        q=1,
        omega=0.1,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print(f"GARCH(1,1): {len(basic_df)} observations")
    print(basic_df.head(10))
    print(f"Mean={basic_df['y'].mean():.4f}, Std={basic_df['y'].std():.4f}")

    # Higher base volatility
    high_vol_gen = GARCHGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=1,
        q=1,
        omega=0.5,
        seed=42,
    )
    high_vol_df = high_vol_gen.generate(n_series=1)
    print(f"\nHigh volatility (omega=0.5): {len(high_vol_df)} observations")
    print(high_vol_df.head(10))
    print(f"Mean={high_vol_df['y'].mean():.4f}, Std={high_vol_df['y'].std():.4f}")

    # Multiple series
    multi_gen = GARCHGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        p=1,
        q=1,
        omega=0.1,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))
    print(f"Overall: Mean={multi_df['y'].mean():.4f}, Std={multi_df['y'].std():.4f}")

    basic_df.write_csv("garch_basic_example.csv")
    print("\nSaved garch_basic_example.csv")

    high_vol_df.write_csv("garch_high_volatility_example.csv")
    print("Saved garch_high_volatility_example.csv")

    multi_df.write_csv("garch_multiple_series_example.csv")
    print("Saved garch_multiple_series_example.csv")


if __name__ == "__main__":
    main()
