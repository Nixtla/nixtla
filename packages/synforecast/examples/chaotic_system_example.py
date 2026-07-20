"""Example usage of ChaoticSystemGenerator: Lorenz attractors, logistic
maps, and Mackey-Glass delay equations."""

import polars as pl

from synforecast.generators import ChaoticSystemGenerator


def main() -> None:
    """Generate and display chaotic system examples."""
    # Lorenz attractor (x-component)
    lorenz_gen = ChaoticSystemGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        system="lorenz",
        sigma=10.0,
        rho=28.0,
        lorenz_beta=8.0 / 3.0,
        observation_noise=0.1,
        seed=42,
    )
    lorenz_df = lorenz_gen.generate(n_series=1)
    print(f"Lorenz system: {len(lorenz_df)} observations")
    print(lorenz_df.head(10))
    print(f"Mean={lorenz_df['y'].mean():.4f}, Std={lorenz_df['y'].std():.4f}")

    # Logistic map in the chaotic regime (r=3.9)
    logistic_gen = ChaoticSystemGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        system="logistic",
        logistic_r=3.9,
        observation_noise=0.01,
        seed=42,
    )
    logistic_df = logistic_gen.generate(n_series=1)
    print(f"\nLogistic map (r=3.9): {len(logistic_df)} observations")
    print(logistic_df.head(10))
    print(f"Mean={logistic_df['y'].mean():.4f}, Std={logistic_df['y'].std():.4f}")

    # Mackey-Glass delay equation
    mg_gen = ChaoticSystemGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        system="mackey_glass",
        mg_tau=17,
        observation_noise=0.01,
        seed=42,
    )
    mg_df = mg_gen.generate(n_series=1)
    print(f"\nMackey-Glass system: {len(mg_df)} observations")
    print(mg_df.head(10))
    print(f"Mean={mg_df['y'].mean():.4f}, Std={mg_df['y'].std():.4f}")

    # Multiple series
    multi_gen = ChaoticSystemGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        system="lorenz",
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series with {len(multi_df)} total observations")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    lorenz_df.write_csv("chaotic_lorenz_example.csv")
    print("\nSaved chaotic_lorenz_example.csv")

    logistic_df.write_csv("chaotic_logistic_example.csv")
    print("Saved chaotic_logistic_example.csv")


if __name__ == "__main__":
    main()
