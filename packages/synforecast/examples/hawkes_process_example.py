"""Example usage of HawkesProcessGenerator."""

import polars as pl

from synforecast.generators import HawkesProcessGenerator


def main() -> None:
    """Generate and display Hawkes process time series."""
    # Event counts (default output)
    generator = HawkesProcessGenerator(
        min_length=200,
        max_length=200,
        freq="h",
        engine="polars",
        baseline_intensity=1.0,  # 1 event per hour baseline
        excitation_amplitude=0.5,
        decay_rate=2.0,
        output_type="counts",
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} series, {len(df)} observations")
    print(f"Total events: {df['y'].sum()}")

    stats = df.group_by("unique_id").agg(
        [
            pl.col("y").sum().alias("total_events"),
            pl.col("y").mean().alias("mean_per_hour"),
            pl.col("y").max().alias("max_in_hour"),
        ]
    )
    print(stats)

    # Intensity output
    intensity_gen = HawkesProcessGenerator(
        min_length=100,
        max_length=100,
        freq="h",
        engine="polars",
        baseline_intensity=0.5,
        excitation_amplitude=0.3,
        decay_rate=1.0,
        output_type="intensity",
        seed=42,
    )
    intensity_df = intensity_gen.generate(n_series=1)
    print(
        f"\nIntensity range: "
        f"[{intensity_df['y'].min():.3f}, {intensity_df['y'].max():.3f}], "
        f"mean: {intensity_df['y'].mean():.3f}"
    )

    # Model info and stability
    info = generator.get_model_info()
    print(f"\nBaseline intensity (mu): {info['baseline_intensity']}")
    print(f"Excitation amplitude (alpha): {info['excitation_amplitude']}")
    print(f"Decay rate (beta): {info['decay_rate']}")
    print(f"Branching ratio (alpha/beta): {info['branching_ratio']:.3f}")
    print(f"Expected cluster size: {info['expected_cluster_size']:.3f}")
    print(f"Process is stable: {info['is_stable']}")

    # Raw event simulation
    event_times, intensities = generator.simulate_with_events(time_horizon=50.0)
    print(f"\nSimulated {len(event_times)} events over 50 time units")
    print(f"Event rate: {len(event_times) / 50:.2f} events/unit time")
    print(f"First 10 event times: {event_times[:10].round(3)}")

    # Power-law kernel
    power_law_gen = HawkesProcessGenerator(
        min_length=100,
        max_length=100,
        freq="h",
        engine="polars",
        kernel="power_law",
        power_law_exponent=1.5,
        baseline_intensity=0.5,
        excitation_amplitude=0.2,
        seed=42,
    )
    power_law_df = power_law_gen.generate(n_series=1)
    print(f"\nPower-law kernel total events: {power_law_df['y'].sum()}")

    output_file = "hawkes_process_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
