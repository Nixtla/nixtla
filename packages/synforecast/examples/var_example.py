"""Example usage of VARGenerator (Vector Autoregression)."""

import numpy as np
import polars as pl

from synforecast.generators import VARGenerator


def main() -> None:
    """Generate and display VAR (Vector Autoregression) time series."""
    # VAR(1) with auto-generated stable coefficients;
    # n_series=3 generates 3 correlated variables
    gen_auto = VARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        lag_order=1,
        seed=42,
    )
    df_auto = gen_auto.generate(n_series=3)

    print("VAR(1) with auto-generated coefficients:")
    print(df_auto.head(10))

    stats = df_auto.group_by("unique_id").agg(
        pl.col("y").mean().alias("mean"),
        pl.col("y").std().alias("std"),
    )
    print("\nSummary statistics by series:")
    print(stats)

    df_wide = df_auto.pivot(on="unique_id", index="ds", values="y")
    series_cols = [col for col in df_wide.columns if col != "ds"]
    corr_01 = np.corrcoef(
        df_wide[series_cols[0]].to_numpy(), df_wide[series_cols[1]].to_numpy()
    )[0, 1]
    print(f"\nCorrelation between series 0 and 1: {corr_01:.3f}")

    # Custom coefficient matrix with strong cross-effects:
    # var1[t] = 0.3*var1[t-1] + 0.6*var2[t-1] + innovation
    # var2[t] = 0.5*var1[t-1] + 0.2*var2[t-1] + innovation
    coef_matrix = np.array(
        [
            [0.3, 0.6],
            [0.5, 0.2],
        ]
    )
    gen_custom = VARGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        lag_order=1,
        coef_matrices=[coef_matrix],
        intercept=np.array([1.0, 2.0]),
        seed=123,
    )
    df_custom = gen_custom.generate(n_series=2)

    df_custom_wide = df_custom.pivot(on="unique_id", index="ds", values="y")
    custom_cols = [col for col in df_custom_wide.columns if col != "ds"]
    corr_custom = np.corrcoef(
        df_custom_wide[custom_cols[0]].to_numpy(),
        df_custom_wide[custom_cols[1]].to_numpy(),
    )[0, 1]
    print(
        f"\nCustom coefficients with strong cross-effects, "
        f"series correlation: {corr_custom:.3f}"
    )

    # Higher-order VAR(3) uses 3 lags of each variable
    gen_var3 = VARGenerator(
        min_length=200,
        max_length=200,
        freq="h",
        engine="polars",
        lag_order=3,
        seed=456,
    )
    df_var3 = gen_var3.generate(n_series=2)
    print(f"\nVAR(3): generated {len(df_var3)} hourly observations")
    print(df_var3.head(10))

    # Correlated innovations add contemporaneous correlation
    innov_cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    gen_corr_innov = VARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        lag_order=1,
        innovation_covariance=innov_cov,
        seed=789,
    )
    df_corr_innov = gen_corr_innov.generate(n_series=2)

    df_corr_wide = df_corr_innov.pivot(on="unique_id", index="ds", values="y")
    corr_cols = [col for col in df_corr_wide.columns if col != "ds"]
    corr_innov = np.corrcoef(
        df_corr_wide[corr_cols[0]].to_numpy(), df_corr_wide[corr_cols[1]].to_numpy()
    )[0, 1]
    print(f"\nCorrelated innovations (cov 0.7), series correlation: {corr_innov:.3f}")

    # Multiple series
    gen_multi = VARGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        lag_order=1,
        seed=999,
    )
    df_multi = gen_multi.generate(n_series=3)

    print(f"\nGenerated 3 series ({len(df_multi)} total rows)")
    print(f"Unique series IDs: {df_multi['unique_id'].unique().sort().to_list()}")
    print(df_multi.filter(pl.col("unique_id") == "0").head(5))

    df_auto.write_csv("var_auto_coefficients.csv")
    df_custom.write_csv("var_custom_coefficients.csv")
    df_var3.write_csv("var_higher_order.csv")
    df_corr_innov.write_csv("var_correlated_innovations.csv")
    df_multi.write_csv("var_multiple_series.csv")


if __name__ == "__main__":
    main()
