"""Example usage of GaussianProcessGenerator, where roughness and
periodicity are controlled by kernel functions (RBF, Matern, periodic)."""

import polars as pl

from synforecast.generators import GaussianProcessGenerator


def main() -> None:
    """Generate and display Gaussian process examples."""
    # RBF kernel (smooth)
    rbf_gen = GaussianProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        kernel="rbf",
        length_scale=20.0,
        amplitude=1.0,
        seed=42,
    )
    rbf_df = rbf_gen.generate(n_series=1)
    print(f"RBF kernel: {len(rbf_df)} observations")
    print(rbf_df.head(10))
    print(f"Mean={rbf_df['y'].mean():.4f}, Std={rbf_df['y'].std():.4f}")

    # Matern 0.5 kernel (rough, Ornstein-Uhlenbeck-like)
    matern_gen = GaussianProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        kernel="matern_0.5",
        length_scale=10.0,
        amplitude=1.0,
        seed=42,
    )
    matern_df = matern_gen.generate(n_series=1)
    print(f"\nMatern 0.5 kernel: {len(matern_df)} observations")
    print(matern_df.head(10))
    print(f"Mean={matern_df['y'].mean():.4f}, Std={matern_df['y'].std():.4f}")

    # Periodic kernel
    periodic_gen = GaussianProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        kernel="periodic",
        length_scale=5.0,
        amplitude=1.0,
        period=30.0,
        seed=42,
    )
    periodic_df = periodic_gen.generate(n_series=1)
    print(f"\nPeriodic kernel: {len(periodic_df)} observations")
    print(periodic_df.head(10))
    print(f"Mean={periodic_df['y'].mean():.4f}, Std={periodic_df['y'].std():.4f}")

    # Multiple series
    multi_gen = GaussianProcessGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        kernel="matern_2.5",
        length_scale=15.0,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    rbf_df.write_csv("gp_rbf_example.csv")
    print("\nSaved gp_rbf_example.csv")

    periodic_df.write_csv("gp_periodic_example.csv")
    print("Saved gp_periodic_example.csv")


if __name__ == "__main__":
    main()
