"""Example usage of BoundedProcessGenerator for series constrained to a
finite interval, such as proportions, rates, and percentages."""

import polars as pl

from synforecast.generators import BoundedProcessGenerator


def main() -> None:
    """Generate and display bounded process examples."""
    # Beta-AR model (values in [0, 1])
    beta_gen = BoundedProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        model="beta_ar",
        phi=0.8,
        omega=0.1,
        kappa=20.0,
        seed=42,
    )
    beta_df = beta_gen.generate(n_series=1)
    print(f"Beta-AR: {len(beta_df)} observations in [0, 1]")
    print(beta_df.head(10))
    print(f"Mean={beta_df['y'].mean():.4f}, Std={beta_df['y'].std():.4f}")
    print(f"Min={beta_df['y'].min():.4f}, Max={beta_df['y'].max():.4f}")

    # Logit-normal model
    logit_gen = BoundedProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        model="logit_normal",
        phi=0.9,
        sigma=0.5,
        seed=42,
    )
    logit_df = logit_gen.generate(n_series=1)
    print(f"\nLogit-normal: {len(logit_df)} observations")
    print(logit_df.head(10))
    print(f"Mean={logit_df['y'].mean():.4f}, Std={logit_df['y'].std():.4f}")
    print(f"Min={logit_df['y'].min():.4f}, Max={logit_df['y'].max():.4f}")

    # Custom bounds, e.g. temperature in [15, 35]
    temp_gen = BoundedProcessGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        model="beta_ar",
        phi=0.85,
        omega=0.1,
        kappa=30.0,
        lower=15.0,
        upper=35.0,
        seed=42,
    )
    temp_df = temp_gen.generate(n_series=1)
    print(f"\nCustom bounds: {len(temp_df)} observations in [15, 35]")
    print(temp_df.head(10))
    print(f"Mean={temp_df['y'].mean():.4f}, Std={temp_df['y'].std():.4f}")
    print(f"Min={temp_df['y'].min():.4f}, Max={temp_df['y'].max():.4f}")

    # Multiple series
    multi_gen = BoundedProcessGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        model="beta_ar",
        phi=0.7,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    beta_df.write_csv("bounded_beta_ar_example.csv")
    print("\nSaved bounded_beta_ar_example.csv")

    temp_df.write_csv("bounded_temperature_example.csv")
    print("Saved bounded_temperature_example.csv")


if __name__ == "__main__":
    main()
