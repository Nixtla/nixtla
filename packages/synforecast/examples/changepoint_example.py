"""Examples of changepoint injection: structural breaks in level, trend, or
variance, available on any generator via the changepoint parameters."""

import polars as pl

from synforecast.generators import (
    RandomWalkGenerator,
    SeasonalGenerator,
    VARGenerator,
)


def main() -> None:
    """Generate and display examples with changepoint injection."""
    # Level shift changepoints at fixed locations with fixed magnitudes
    level_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        changepoints=True,
        num_changepoints=3,
        changepoint_type="level",
        changepoint_level_changes=[50.0, -30.0, 40.0],
        changepoint_locations=[0.2, 0.5, 0.8],
        seed=42,
    )
    level_df = level_gen.generate(n_series=1)
    print("Level shifts at 20%, 50%, 80% with changes +50, -30, +40:")
    print(level_df.head(10))
    print(
        f"Mean={level_df['y'].mean():.4f}, "
        f"Min={level_df['y'].min():.4f}, Max={level_df['y'].max():.4f}"
    )

    # Trend changepoints on a seasonal series
    trend_gen = SeasonalGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        seasonality_period=7,
        seasonality_amplitude=10.0,
        base_level=100.0,
        changepoints=True,
        num_changepoints=2,
        changepoint_type="trend",
        changepoint_trend_changes=[0.3, -0.2],
        changepoint_locations=[0.3, 0.7],
        seed=42,
    )
    trend_df = trend_gen.generate(n_series=1)
    print("\nSeasonal series with trend changes +0.3 at 30%, -0.2 at 70%:")
    print(trend_df.head(10))
    print(
        f"Mean={trend_df['y'].mean():.4f}, "
        f"Min={trend_df['y'].min():.4f}, Max={trend_df['y'].max():.4f}"
    )

    # Variance changepoints (volatility regime shifts)
    variance_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.05,
        volatility=2.0,
        changepoints=True,
        num_changepoints=2,
        changepoint_type="variance",
        changepoint_variance_changes=[2.0, 0.5],
        changepoint_locations=[0.33, 0.67],
        seed=42,
    )
    variance_df = variance_gen.generate(n_series=1)
    print("\nVariance multipliers 2.0x at 33%, 0.5x at 67%:")
    print(f"Mean={variance_df['y'].mean():.4f}, Std={variance_df['y'].std():.4f}")

    # Mixed changepoint types: random combination of level, trend, and variance
    mixed_gen = RandomWalkGenerator(
        min_length=400,
        max_length=400,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        changepoints=True,
        num_changepoints=4,
        changepoint_type="mixed",
        changepoint_locations=[0.2, 0.4, 0.6, 0.8],
        seed=42,
    )
    mixed_df = mixed_gen.generate(n_series=1)
    print("\nMixed changepoint types at 20%, 40%, 60%, 80%:")
    print(
        f"Mean={mixed_df['y'].mean():.4f}, "
        f"Min={mixed_df['y'].min():.4f}, Max={mixed_df['y'].max():.4f}"
    )

    # Omitting changepoint_locations places changepoints randomly
    random_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.05,
        volatility=2.0,
        changepoints=True,
        num_changepoints=3,
        changepoint_type="level",
        seed=42,
    )
    random_df = random_gen.generate(n_series=1)
    print("\nRandomly placed level shifts:")
    print(f"Mean={random_df['y'].mean():.4f}, Std={random_df['y'].std():.4f}")

    # Changepoints also work with multivariate generators
    var_gen = VARGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        lag_order=1,
        changepoints=True,
        num_changepoints=2,
        changepoint_type="level",
        changepoint_level_changes=[30.0, -20.0],
        changepoint_locations=[0.3, 0.7],
        seed=42,
    )
    var_df = var_gen.generate(n_series=3)
    print(f"\nVAR: 3 correlated series with changepoints ({len(var_df)} rows):")
    print(var_df.filter(pl.col("unique_id") == "0").head(10))
    print(
        f"Mean={var_df['y'].mean():.4f}, "
        f"Min={var_df['y'].min():.4f}, Max={var_df['y'].max():.4f}"
    )

    # Changepoints combined with anomalies
    combined_anomalies_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        changepoints=True,
        num_changepoints=2,
        changepoint_type="level",
        changepoint_level_changes=[40.0, -30.0],
        changepoint_locations=[0.33, 0.67],
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike", "dip"],
        spike_magnitude=35.0,
        dip_magnitude=-35.0,
        seed=42,
    )
    combined_anomalies_df = combined_anomalies_gen.generate(n_series=1)
    print("\nChangepoints combined with anomalies:")
    print(
        f"Mean={combined_anomalies_df['y'].mean():.4f}, "
        f"Min={combined_anomalies_df['y'].min():.4f}, "
        f"Max={combined_anomalies_df['y'].max():.4f}"
    )

    # Changepoints combined with missing data
    combined_missing_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        changepoints=True,
        num_changepoints=2,
        changepoint_type="level",
        changepoint_level_changes=[35.0, -25.0],
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.1,
        seed=42,
    )
    combined_missing_df = combined_missing_gen.generate(n_series=1)
    null_count = combined_missing_df["y"].null_count()
    print(f"\nChangepoints with missing data ({null_count} missing values):")
    print(
        f"Mean={combined_missing_df['y'].mean():.4f}, "
        f"Min={combined_missing_df['y'].min():.4f}, "
        f"Max={combined_missing_df['y'].max():.4f}"
    )

    # Changepoints, anomalies, and missing data combined
    all_features_gen = RandomWalkGenerator(
        min_length=400,
        max_length=400,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        changepoints=True,
        num_changepoints=3,
        changepoint_type="mixed",
        changepoint_locations=[0.25, 0.5, 0.75],
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike", "dip", "level_shift"],
        spike_magnitude=40.0,
        dip_magnitude=-40.0,
        level_shift_magnitude=30.0,
        level_shift_duration=15,
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.08,
        seed=42,
    )
    all_features_df = all_features_gen.generate(n_series=1)
    null_count_all = all_features_df["y"].null_count()
    print(f"\nAll features combined ({null_count_all} missing values):")
    print(
        f"Mean={all_features_df['y'].mean():.4f}, "
        f"Min={all_features_df['y'].min():.4f}, "
        f"Max={all_features_df['y'].max():.4f}"
    )

    outputs = {
        "changepoint_level_example.csv": level_df,
        "changepoint_trend_example.csv": trend_df,
        "changepoint_variance_example.csv": variance_df,
        "changepoint_mixed_example.csv": mixed_df,
        "changepoint_random_example.csv": random_df,
        "changepoint_var_example.csv": var_df,
        "changepoint_with_anomalies_example.csv": combined_anomalies_df,
        "changepoint_with_missing_example.csv": combined_missing_df,
        "changepoint_all_features_example.csv": all_features_df,
    }
    for filename, df in outputs.items():
        df.write_csv(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
