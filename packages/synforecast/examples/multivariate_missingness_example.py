"""Example usage of missing data patterns with multivariate generators."""

import numpy as np
import polars as pl

from synforecast.generators import CopulaGenerator, VARGenerator


def main() -> None:
    """Generate and display multivariate time series with missing data patterns."""
    # VAR with random missing data
    var_gen = VARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        lag_order=1,
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.15,
        seed=42,
    )
    df_var = var_gen.generate(n_series=3)

    print("VAR(1) with random missing data:")
    print(df_var.head(30))

    print("\nMissing data by series:")
    for series_id in df_var["unique_id"].unique().sort():
        values = df_var.filter(pl.col("unique_id") == series_id)["y"].to_numpy()
        nan_count = np.sum(np.isnan(values))
        print(f"  {series_id}: {nan_count} missing ({nan_count / len(values):.1%})")

    # Copula with block missing data (week-long outages)
    correlation_matrix = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]])
    copula_gen = CopulaGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        copula_type="gaussian",
        correlation_matrix=correlation_matrix,
        missing_data=True,
        missing_pattern="block",
        missing_rate=0.2,
        missing_block_size=7,
        seed=123,
    )
    df_copula = copula_gen.generate(n_series=3)

    print("\nGaussian copula with block missing data (block size 7):")
    for series_id in df_copula["unique_id"].unique().sort():
        values = df_copula.filter(pl.col("unique_id") == series_id)["y"].to_numpy()
        nan_count = np.sum(np.isnan(values))

        max_consecutive = 0
        current_consecutive = 0
        for val in values:
            if np.isnan(val):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        print(
            f"  {series_id}: {nan_count} missing ({nan_count / len(values):.1%}), "
            f"max block: {max_consecutive} days"
        )

    # VAR with seasonal missing data (weekly reporting gaps)
    var_seasonal_gen = VARGenerator(
        min_length=365,
        max_length=365,
        freq="D",
        engine="polars",
        lag_order=2,
        missing_data=True,
        missing_pattern="seasonal",
        missing_rate=0.12,
        missing_seasonal_period=7,
        seed=456,
    )
    df_var_seasonal = var_seasonal_gen.generate(n_series=2)

    print("\nVAR(2) with seasonal missing data (weekly pattern):")
    for series_id in df_var_seasonal["unique_id"].unique().sort():
        values = df_var_seasonal.filter(pl.col("unique_id") == series_id)[
            "y"
        ].to_numpy()
        nan_count = np.sum(np.isnan(values))
        print(f"  {series_id}: {nan_count} missing ({nan_count / len(values):.1%})")

    # Missing data leaves cross-series correlation approximately intact
    gen_complete = VARGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        lag_order=1,
        missing_data=False,
        seed=789,
    )
    df_complete = gen_complete.generate(n_series=2)

    # Same seed so the underlying process matches the complete version
    gen_missing = VARGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        lag_order=1,
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.25,
        seed=789,
    )
    df_missing = gen_missing.generate(n_series=2)

    series_0_complete = df_complete.filter(pl.col("unique_id") == "0")["y"].to_numpy()
    series_1_complete = df_complete.filter(pl.col("unique_id") == "1")["y"].to_numpy()
    corr_complete = np.corrcoef(series_0_complete, series_1_complete)[0, 1]

    series_0_missing = df_missing.filter(pl.col("unique_id") == "0")["y"].to_numpy()
    series_1_missing = df_missing.filter(pl.col("unique_id") == "1")["y"].to_numpy()
    mask = ~(np.isnan(series_0_missing) | np.isnan(series_1_missing))
    corr_missing = np.corrcoef(series_0_missing[mask], series_1_missing[mask])[0, 1]

    print(f"\nCorrelation (complete data): {corr_complete:.3f}")
    print(f"Correlation (25% missing):   {corr_missing:.3f}")
    print(f"Correlation preserved:       {abs(corr_complete - corr_missing) < 0.1}")

    df_var.write_csv("multivariate_missing_var_random.csv")
    df_copula.write_csv("multivariate_missing_copula_block.csv")
    df_var_seasonal.write_csv("multivariate_missing_var_seasonal.csv")


if __name__ == "__main__":
    main()
