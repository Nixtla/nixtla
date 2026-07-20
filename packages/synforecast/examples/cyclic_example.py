"""Example usage of CyclicGenerator: irregular business cycles combining trend,
multiple overlapping cycles with varying periods and amplitudes, and noise."""

import polars as pl

from synforecast.generators import CyclicGenerator


def main() -> None:
    """Generate and display Cyclic generator examples."""
    # Basic business cycle: upward trend with ~60-day cycles
    basic_gen = CyclicGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        base_level=100.0,
        trend=0.05,
        cycle_period_mean=60.0,
        cycle_amplitude_mean=20.0,
        num_cycles=3,
        seed=42,
    )
    basic_df = basic_gen.generate(n_series=1)
    print("Basic business cycle:")
    print(basic_df.head(10))
    print(f"Mean={basic_df['y'].mean():.4f}, Std={basic_df['y'].std():.4f}")

    # Fast cycles (~30 days)
    fast_gen = CyclicGenerator(
        min_length=400,
        max_length=400,
        freq="D",
        engine="polars",
        base_level=50.0,
        trend=0.02,
        cycle_period_mean=30.0,
        cycle_amplitude_mean=10.0,
        num_cycles=5,
        seed=42,
    )
    fast_df = fast_gen.generate(n_series=1)
    print(
        f"\nFast cycles (~30 days): Mean={fast_df['y'].mean():.4f}, "
        f"Std={fast_df['y'].std():.4f}"
    )

    # Slow cycles (~90 days)
    slow_gen = CyclicGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        base_level=100.0,
        trend=0.03,
        cycle_period_mean=90.0,
        cycle_amplitude_mean=30.0,
        num_cycles=2,
        seed=42,
    )
    slow_df = slow_gen.generate(n_series=1)
    print(
        f"\nSlow cycles (~90 days): Mean={slow_df['y'].mean():.4f}, "
        f"Std={slow_df['y'].std():.4f}"
    )

    # High amplitude cycles
    high_amp_gen = CyclicGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        base_level=100.0,
        trend=0.01,
        cycle_period_mean=60.0,
        cycle_amplitude_mean=50.0,
        num_cycles=3,
        seed=42,
    )
    high_amp_df = high_amp_gen.generate(n_series=1)
    print(
        f"\nHigh amplitude cycles: Mean={high_amp_df['y'].mean():.4f}, "
        f"Std={high_amp_df['y'].std():.4f}"
    )

    # Negative trend with cycles (declining market)
    declining_gen = CyclicGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        base_level=150.0,
        trend=-0.05,
        cycle_period_mean=50.0,
        cycle_amplitude_mean=20.0,
        num_cycles=3,
        seed=42,
    )
    declining_df = declining_gen.generate(n_series=1)
    print(
        f"\nDeclining trend with cycles: Mean={declining_df['y'].mean():.4f}, "
        f"Std={declining_df['y'].std():.4f}"
    )

    # Many overlapping cycles produce a complex pattern
    complex_gen = CyclicGenerator(
        min_length=500,
        max_length=500,
        freq="D",
        engine="polars",
        base_level=100.0,
        trend=0.02,
        cycle_period_mean=60.0,
        cycle_amplitude_mean=15.0,
        num_cycles=5,
        seed=42,
    )
    complex_df = complex_gen.generate(n_series=1)
    print(
        f"\n5 overlapping cycles: Mean={complex_df['y'].mean():.4f}, "
        f"Std={complex_df['y'].std():.4f}"
    )

    # Multiple series
    multi_gen = CyclicGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        base_level=100.0,
        trend=0.03,
        cycle_period_mean=60.0,
        cycle_amplitude_mean=20.0,
        num_cycles=3,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series ({len(multi_df)} total observations), first series:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    basic_df.write_csv("cyclic_basic_example.csv")
    fast_df.write_csv("cyclic_fast_cycles_example.csv")
    slow_df.write_csv("cyclic_slow_cycles_example.csv")
    high_amp_df.write_csv("cyclic_high_amplitude_example.csv")
    declining_df.write_csv("cyclic_declining_trend_example.csv")
    complex_df.write_csv("cyclic_complex_pattern_example.csv")
    multi_df.write_csv("cyclic_multiple_series_example.csv")


if __name__ == "__main__":
    main()
