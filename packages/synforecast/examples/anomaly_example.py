"""Example usage of anomaly injection: spikes, dips, and level shifts can be
added to any generator via the anomaly parameters."""

import polars as pl

from synforecast.generators import (
    RandomWalkGenerator,
    SeasonalGenerator,
    VARGenerator,
)


def main() -> None:
    """Generate and display examples with anomaly injection."""
    # Spike anomalies on a random walk
    spike_gen = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike"],
        spike_magnitude=20.0,
        seed=42,
    )
    spike_df = spike_gen.generate(n_series=1)
    print("Random walk with ~5% spike anomalies:")
    print(spike_df.head(10))
    print(
        f"Mean={spike_df['y'].mean():.4f}, "
        f"Min={spike_df['y'].min():.4f}, Max={spike_df['y'].max():.4f}"
    )

    # Dip anomalies on a seasonal series
    dip_gen = SeasonalGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        seasonality_period=7,
        seasonality_amplitude=10.0,
        base_level=100.0,
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["dip"],
        dip_magnitude=-30.0,
        seed=42,
    )
    dip_df = dip_gen.generate(n_series=1)
    print(
        f"\nSeasonal series with dip anomalies: Mean={dip_df['y'].mean():.4f}, "
        f"Min={dip_df['y'].min():.4f}, Max={dip_df['y'].max():.4f}"
    )

    # Level shifts persist for a configurable duration
    level_shift_gen = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        drift=0.05,
        volatility=1.5,
        anomalies=True,
        anomaly_fraction=0.03,
        anomaly_types=["level_shift"],
        level_shift_magnitude=25.0,
        level_shift_duration=15,
        seed=42,
    )
    level_shift_df = level_shift_gen.generate(n_series=1)
    print(
        f"\nLevel shift anomalies (duration 15): "
        f"Mean={level_shift_df['y'].mean():.4f}, Std={level_shift_df['y'].std():.4f}"
    )

    # Mixed anomaly types in one series
    mixed_gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        anomalies=True,
        anomaly_fraction=0.08,
        anomaly_types=["spike", "dip", "level_shift"],
        spike_magnitude=30.0,
        dip_magnitude=-30.0,
        level_shift_magnitude=20.0,
        level_shift_duration=10,
        seed=42,
    )
    mixed_df = mixed_gen.generate(n_series=1)
    print(
        f"\nMixed anomaly types: Mean={mixed_df['y'].mean():.4f}, "
        f"Min={mixed_df['y'].min():.4f}, Max={mixed_df['y'].max():.4f}"
    )

    # Anomalies also work with multivariate generators
    var_gen = VARGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        lag_order=1,
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike", "dip"],
        spike_magnitude=15.0,
        dip_magnitude=-15.0,
        seed=42,
    )
    var_df = var_gen.generate(n_series=3)
    print("\nVAR with anomalies, 3 correlated series, first series:")
    print(var_df.filter(pl.col("unique_id") == "0").head(10))

    # Anomalies combined with missing data
    combined_gen = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        drift=0.1,
        volatility=2.0,
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike", "dip"],
        spike_magnitude=25.0,
        dip_magnitude=-25.0,
        missing_data=True,
        missing_pattern="random",
        missing_rate=0.1,
        seed=42,
    )
    combined_df = combined_gen.generate(n_series=1)
    null_count = combined_df["y"].null_count()
    print(f"\nAnomalies plus missing data ({null_count} missing values):")
    print(combined_df.head(15))

    spike_df.write_csv("anomaly_spike_example.csv")
    dip_df.write_csv("anomaly_dip_example.csv")
    level_shift_df.write_csv("anomaly_level_shift_example.csv")
    mixed_df.write_csv("anomaly_mixed_example.csv")
    var_df.write_csv("anomaly_var_example.csv")
    combined_df.write_csv("anomaly_combined_example.csv")


if __name__ == "__main__":
    main()
