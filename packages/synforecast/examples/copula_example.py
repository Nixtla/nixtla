"""Example usage of CopulaGenerator for multivariate series with dependencies."""

import numpy as np
import polars as pl

from synforecast.generators import CopulaGenerator


def main() -> None:
    """Generate and display Copula-based multivariate time series."""
    # Gaussian copula with correlated normal marginals
    corr_matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])

    gen_gaussian = CopulaGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        copula_type="gaussian",
        correlation_matrix=corr_matrix,
        marginal_distributions=[
            {"type": "normal", "loc": 100.0, "scale": 10.0},
            {"type": "normal", "loc": 50.0, "scale": 5.0},
            {"type": "normal", "loc": 200.0, "scale": 20.0},
        ],
        seed=42,
    )
    df_gaussian = gen_gaussian.generate(n_series=3)

    print("Gaussian copula with 3 correlated variables:")
    print(df_gaussian.head(10))

    # Compare empirical correlations against the specified matrix
    df_wide = df_gaussian.pivot(on="unique_id", index="ds", values="y")
    series_cols = sorted(col for col in df_wide.columns if col != "ds")

    if len(series_cols) >= 3:
        corr_0_1 = np.corrcoef(
            df_wide[series_cols[0]].to_numpy(), df_wide[series_cols[1]].to_numpy()
        )[0, 1]
        corr_0_2 = np.corrcoef(
            df_wide[series_cols[0]].to_numpy(), df_wide[series_cols[2]].to_numpy()
        )[0, 1]
        corr_1_2 = np.corrcoef(
            df_wide[series_cols[1]].to_numpy(), df_wide[series_cols[2]].to_numpy()
        )[0, 1]

        print("\nCorrelations (compared to specified):")
        print(f"  series 0 vs 1: {corr_0_1:.3f} (specified: 0.800)")
        print(f"  series 0 vs 2: {corr_0_2:.3f} (specified: 0.600)")
        print(f"  series 1 vs 2: {corr_1_2:.3f} (specified: 0.700)")

    # t-copula: heavier tail dependence than the Gaussian copula
    gen_t = CopulaGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        copula_type="t",
        df=5.0,  # Degrees of freedom (lower = heavier tails)
        marginal_distributions=[
            {"type": "normal", "loc": 0.0, "scale": 1.0},
            {"type": "normal", "loc": 0.0, "scale": 1.0},
        ],
        seed=123,
    )
    df_t = gen_t.generate(n_series=2)

    df_t_wide = df_t.pivot(on="unique_id", index="ds", values="y")
    t_cols = sorted(col for col in df_t_wide.columns if col != "ds")
    if len(t_cols) >= 2:
        corr_t = np.corrcoef(
            df_t_wide[t_cols[0]].to_numpy(), df_t_wide[t_cols[1]].to_numpy()
        )[0, 1]
        print(f"\nt-copula correlation: {corr_t:.3f}")

    # Mixed marginal distributions under a Gaussian copula
    gen_mixed = CopulaGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        copula_type="gaussian",
        marginal_distributions=[
            {"type": "normal", "loc": 100.0, "scale": 15.0},  # Stock price
            {"type": "lognormal", "mean": 4.0, "sigma": 0.3},  # Trading volume
            {"type": "gamma", "shape": 2.0, "scale": 10.0},  # Volatility measure
            {"type": "uniform", "low": 0.0, "high": 100.0},  # Bounded metric
        ],
        seed=456,
    )
    df_mixed = gen_mixed.generate(n_series=4)

    print("\nMixed marginals, summary statistics by series:")
    stats = df_mixed.group_by("unique_id").agg(
        pl.col("y").mean().alias("mean"), pl.col("y").std().alias("std")
    )
    print(stats.sort("unique_id"))

    # Multiple correlated series
    gen_multi = CopulaGenerator(
        min_length=100,
        max_length=100,
        freq="h",
        engine="polars",
        copula_type="gaussian",
        seed=789,
    )
    df_multi = gen_multi.generate(n_series=3)

    print(
        f"\nMultiple series, unique IDs: {df_multi['unique_id'].unique().sort().to_list()}"
    )
    print("Sample from series 0:")
    print(df_multi.filter(pl.col("unique_id") == "0").head(5))

    df_gaussian.write_csv("copula_gaussian_example.csv")
    df_t.write_csv("copula_t_example.csv")
    df_mixed.write_csv("copula_mixed_marginals_example.csv")
    df_multi.write_csv("copula_multiple_series_example.csv")
    print("\nSaved CSV outputs for all examples.")


if __name__ == "__main__":
    main()
