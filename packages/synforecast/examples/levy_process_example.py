"""Example usage of LevyProcessGenerator, which produces heavy-tailed
increments from alpha-stable distributions (alpha=2 is Gaussian)."""

import polars as pl

from synforecast.generators import LevyProcessGenerator


def main() -> None:
    """Generate and display Levy process examples."""
    # Gaussian case (alpha=2)
    gauss_gen = LevyProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        alpha=2.0,
        cumulative=True,
        initial_value=100.0,
        seed=42,
    )
    gauss_df = gauss_gen.generate(n_series=1)
    print(f"Gaussian (alpha=2.0): {len(gauss_df)} observations")
    print(gauss_df.head(10))
    print(f"Mean={gauss_df['y'].mean():.4f}, Std={gauss_df['y'].std():.4f}")

    # Heavy-tailed process (alpha=1.5)
    heavy_gen = LevyProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        alpha=1.5,
        cumulative=True,
        initial_value=100.0,
        seed=42,
    )
    heavy_df = heavy_gen.generate(n_series=1)
    print(f"\nHeavy-tailed (alpha=1.5): {len(heavy_df)} observations")
    print(heavy_df.head(10))
    print(f"Mean={heavy_df['y'].mean():.4f}, Std={heavy_df['y'].std():.4f}")

    # Cauchy-like process (alpha=1.0), non-cumulative increments
    cauchy_gen = LevyProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        alpha=1.0,
        scale=0.5,
        cumulative=False,
        seed=42,
    )
    cauchy_df = cauchy_gen.generate(n_series=1)
    print(f"\nCauchy-like (alpha=1.0): {len(cauchy_df)} observations")
    print(cauchy_df.head(10))
    print(f"Mean={cauchy_df['y'].mean():.4f}, Std={cauchy_df['y'].std():.4f}")

    # Multiple series
    multi_gen = LevyProcessGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        alpha=1.8,
        cumulative=True,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    gauss_df.write_csv("levy_gaussian_example.csv")
    print("\nSaved levy_gaussian_example.csv")

    heavy_df.write_csv("levy_heavy_tailed_example.csv")
    print("Saved levy_heavy_tailed_example.csv")


if __name__ == "__main__":
    main()
