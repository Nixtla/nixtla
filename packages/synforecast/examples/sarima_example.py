"""Example usage of SARIMAGenerator."""

import polars as pl

from synforecast.generators import SARIMAGenerator


def main() -> None:
    """Generate and display SARIMA time series."""
    # SARIMA(2,1,1)(1,1,1)_7 with automatically generated coefficients
    generator = SARIMAGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=2,
        d=1,
        q=1,
        P=1,
        D=1,
        Q=1,
        seasonal_period=7,
        noise_std=2.0,
        drift=0.5,
        seed=42,
    )

    model_info = generator.get_model_info()
    print(f"Model: {model_info['model']}")
    print(f"AR parameters: {model_info['ar_params']}")
    print(f"MA parameters: {model_info['ma_params']}")
    print(f"Seasonal AR parameters: {model_info['seasonal_ar_params']}")
    print(f"Seasonal MA parameters: {model_info['seasonal_ma_params']}")
    print(f"Expanded AR polynomial lags: {model_info['full_ar_polynomial_lags']}")
    print(f"Expanded AR polynomial coeffs: {model_info['full_ar_polynomial_coeffs']}")

    df = generator.generate(n_series=3)

    print(f"\nGenerated {df['unique_id'].n_unique()} time series")
    print(df.head(10))

    stats = (
        df.group_by("unique_id")
        .agg(
            [
                pl.col("y").count().alias("count"),
                pl.col("y").min().alias("min_value"),
                pl.col("y").max().alias("max_value"),
                pl.col("y").mean().alias("mean_value"),
                pl.col("y").std().alias("std_value"),
            ]
        )
        .sort("unique_id")
    )
    print("\nStatistics by series:")
    print(stats)

    # Stationary ARMA(1,1) with custom coefficients
    generator_arma = SARIMAGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        p=1,
        d=0,
        q=1,
        P=0,
        D=0,
        Q=0,
        ar_params=[0.7],
        ma_params=[0.3],
        mean=50.0,  # Process mean, only meaningful for stationary models
        noise_std=1.0,
        seed=123,
    )
    df_arma = generator_arma.generate(n_series=1)

    print(f"\nModel: {generator_arma.get_model_info()['model']}")
    print(f"Specified mean: {generator_arma.get_model_info()['mean']}")
    print(f"Series mean: {df_arma['y'].mean():.2f}")
    print(f"Series std: {df_arma['y'].std():.2f}")

    # Pure seasonal ARIMA(0,0,0)(1,1,1)_12 on monthly data
    generator_seasonal = SARIMAGenerator(
        min_length=100,
        max_length=100,
        freq="MS",
        engine="polars",
        p=0,
        d=0,
        q=0,
        P=1,
        D=1,
        Q=1,
        seasonal_period=12,
        seasonal_ar_params=[0.5],
        seasonal_ma_params=[0.3],
        noise_std=1.0,
        seed=456,
    )
    df_seasonal = generator_seasonal.generate(n_series=1)

    print(f"\nModel: {generator_seasonal.get_model_info()['model']}")
    print("First 24 months:")
    print(df_seasonal.head(24))

    output_file = "sarima_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
