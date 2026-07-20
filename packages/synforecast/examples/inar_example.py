"""Example usage of INARGenerator, which produces count-valued time series
via binomial thinning, suitable for event counts with memory."""

import polars as pl

from synforecast.generators import INARGenerator


def main() -> None:
    """Generate and display INAR examples."""
    # INAR(1) with Poisson innovations
    basic_gen = INARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=1,
        alpha=[0.5],
        innovation_type="poisson",
        innovation_mean=3.0,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print(f"INAR(1) Poisson: {len(basic_df)} count-valued observations")
    print(basic_df.head(10))
    print(f"Mean={basic_df['y'].mean():.4f}, Std={basic_df['y'].std():.4f}")

    # INAR(1) with negative binomial innovations (overdispersed)
    nb_gen = INARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=1,
        alpha=[0.6],
        innovation_type="negative_binomial",
        innovation_mean=5.0,
        innovation_dispersion=2.0,
        seed=42,
    )
    nb_df = nb_gen.generate(n_series=1)
    print(f"\nINAR(1) NegBin: {len(nb_df)} observations")
    print(nb_df.head(10))
    print(f"Mean={nb_df['y'].mean():.4f}, Std={nb_df['y'].std():.4f}")

    # INAR(2) with higher-order dependence
    inar2_gen = INARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=2,
        alpha=[0.3, 0.2],
        innovation_type="poisson",
        innovation_mean=2.0,
        seed=42,
    )
    inar2_df = inar2_gen.generate(n_series=1)
    print(f"\nINAR(2): {len(inar2_df)} observations")
    print(inar2_df.head(10))
    print(f"Mean={inar2_df['y'].mean():.4f}, Std={inar2_df['y'].std():.4f}")

    # Multiple series
    multi_gen = INARGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        p=1,
        alpha=[0.5],
        innovation_mean=3.0,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("inar_basic_example.csv")
    print("\nSaved inar_basic_example.csv")

    nb_df.write_csv("inar_negbin_example.csv")
    print("Saved inar_negbin_example.csv")


if __name__ == "__main__":
    main()
