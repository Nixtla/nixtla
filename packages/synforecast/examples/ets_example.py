"""Example usage of ETSGenerator."""

import polars as pl

from synforecast.generators import ETSGenerator


def main() -> None:
    """Generate and display ETS time series."""
    # ETS(A,Ad,A): damped additive Holt-Winters with weekly seasonality
    generator = ETSGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        error_type="add",
        trend_type="add",
        seasonal_type="add",
        seasonal_period=7,
        level=100.0,
        trend=0.5,
        alpha=0.3,  # Level smoothing
        beta=0.1,  # Trend smoothing
        gamma=0.1,  # Seasonal smoothing
        damped=True,
        phi=0.95,  # Damping parameter
        noise_std=2.0,
        seed=123,
    )

    model_info = generator.get_model_info()
    print(f"Model: {model_info['model']}")
    print(
        f"alpha={model_info['alpha']}, beta={model_info['beta']}, "
        f"gamma={model_info['gamma']}, phi={model_info['phi']}"
    )

    df = generator.generate(n_series=3)

    print(f"\nGenerated {df['unique_id'].n_unique()} series, {len(df)} observations")
    print(df.head(10))

    stats = (
        df.group_by("unique_id")
        .agg(
            pl.col("y").count().alias("count"),
            pl.col("y").min().alias("min_value"),
            pl.col("y").max().alias("max_value"),
            pl.col("y").mean().alias("mean_value"),
            pl.col("y").std().alias("std_value"),
        )
        .sort("unique_id")
    )
    print("\nStatistics by series:")
    print(stats)

    # generate_with_states also returns level, trend, and seasonal components
    generator_states = ETSGenerator(
        min_length=50,
        max_length=50,
        freq="D",
        engine="polars",
        error_type="add",
        trend_type="add",
        seasonal_type="add",
        seasonal_period=7,
        level=100.0,
        trend=1.0,
        alpha=0.3,
        beta=0.1,
        gamma=0.1,
        noise_std=1.0,
        seed=42,
    )
    obs_df, states_df = generator_states.generate_with_states(n_series=1)

    print("\nObservations:")
    print(obs_df.head(10))
    print("\nStates (level, trend, seasonal components):")
    print(states_df.head(10))

    # ETS(M,A,M): multiplicative Holt-Winters with monthly seasonality
    generator_mul = ETSGenerator(
        min_length=200,
        max_length=200,
        freq="MS",
        engine="polars",
        error_type="mul",
        trend_type="add",
        seasonal_type="mul",
        seasonal_period=12,
        level=100.0,
        trend=1.0,
        alpha=0.3,
        beta=0.1,
        gamma=0.1,
        noise_std=0.1,  # Smaller for multiplicative
        seed=456,
    )
    df_mul = generator_mul.generate(n_series=1)

    print(f"\nModel: {generator_mul.get_model_info()['model']}")
    print(df_mul.head(12))
    # Multiplicative models require positive values
    print(f"All values positive: {(df_mul['y'] > 0).all()}")

    # ETS(A,N,N): simple exponential smoothing
    generator_ses = ETSGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        error_type="add",
        trend_type=None,
        seasonal_type=None,
        level=50.0,
        alpha=0.2,
        noise_std=5.0,
        seed=789,
    )
    df_ses = generator_ses.generate(n_series=1)

    print(f"\nModel: {generator_ses.get_model_info()['model']}")
    print(
        f"Mean={df_ses['y'].mean():.2f}, Std={df_ses['y'].std():.2f}, "
        f"Min={df_ses['y'].min():.2f}, Max={df_ses['y'].max():.2f}"
    )

    # Box-Cox transformation (lambda=0.5 is a square root transformation)
    generator_boxcox = ETSGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        error_type="add",
        trend_type="add",
        seasonal_type=None,
        level=100.0,
        trend=0.5,
        alpha=0.3,
        beta=0.1,
        noise_std=0.5,
        box_cox_lambda=0.5,
        seed=111,
    )
    df_boxcox = generator_boxcox.generate(n_series=1)

    print(f"\nModel: {generator_boxcox.get_model_info()['model']}")
    print(f"Box-Cox lambda: {generator_boxcox.get_model_info()['box_cox_lambda']}")
    print(
        f"After inverse Box-Cox: Mean={df_boxcox['y'].mean():.2f}, "
        f"Std={df_boxcox['y'].std():.2f}"
    )

    df.write_csv("ets_data.csv")
    df_mul.write_csv("ets_multiplicative_data.csv")


if __name__ == "__main__":
    main()
