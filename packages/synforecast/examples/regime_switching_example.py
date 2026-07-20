"""Example usage of RegimeSwitchingGenerator."""

import polars as pl

from synforecast.generators import RegimeSwitchingGenerator


def main() -> None:
    """Generate and display regime-switching time series."""
    # Bull/bear market model: bull regime has positive drift and is
    # persistent, bear regime has negative drift and higher volatility
    generator = RegimeSwitchingGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        n_regimes=2,
        regime_means=[0.05, -0.03],
        regime_variances=[1.0, 4.0],
        regime_ar_coeffs=[0.1, 0.2],
        transition_matrix=[
            [0.98, 0.02],
            [0.10, 0.90],
        ],
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} series, {len(df)} observations")
    print(df.head(10))

    info = generator.get_model_info()
    print(f"\nNumber of regimes: {info['n_regimes']}")
    print(f"Regime means: {info['regime_means']}")
    print(f"Regime variances: {info['regime_variances']}")
    print(
        "Stationary distribution: "
        f"{[f'{p:.3f}' for p in info['stationary_distribution']]}"
    )

    # Generate with regime labels
    values, regimes, ids = generator.generate_with_regimes(n_series=1)
    print(f"\nRegime 0 (Bull) observations: {(regimes == 0).sum()}")
    print(f"Regime 1 (Bear) observations: {(regimes == 1).sum()}")

    stats = df.group_by("unique_id").agg(
        [
            pl.col("y").count().alias("count"),
            pl.col("y").min().alias("min_value"),
            pl.col("y").max().alias("max_value"),
            pl.col("y").mean().alias("mean_value"),
            pl.col("y").std().alias("std_value"),
        ]
    )
    print(f"\nStatistics by series:\n{stats}")

    output_file = "regime_switching_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
