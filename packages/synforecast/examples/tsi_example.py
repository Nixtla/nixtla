"""Example usage of TSIGenerator's trend/seasonality/irregularity composition."""

import numpy as np
from scipy import signal

from synforecast.generators.tsi import TSIGenerator


def diversity_metrics(values: np.ndarray) -> tuple[float, float, float]:
    """(roughness, |lag-12 acf|, spectral entropy) of a standardized series."""
    z = (values - values.mean()) / (values.std() + 1e-9)
    roughness = np.std(np.diff(z)) / (np.std(z) + 1e-9)
    _, p = signal.periodogram(z)
    p = p[1:]
    p = p / p.sum()
    entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    demeaned = z - z.mean()
    acf12 = abs(float(demeaned[:-12] @ demeaned[12:] / (demeaned @ demeaned)))
    return roughness, acf12, entropy


def main() -> None:
    """Generate and display TSI-composed time series."""
    generator = TSIGenerator(
        min_length=200,
        max_length=400,
        freq="h",
        engine="polars",
        seed=42,
    )
    df = generator.generate(n_series=5)
    print(f"Generated {df['unique_id'].n_unique()} series, {len(df)} observations")
    print(df.head(10))

    # Descriptive diversity metrics for a small generated pool.
    pool_gen = TSIGenerator(min_length=512, max_length=512, freq="h", seed=0)
    rows = np.array(
        [diversity_metrics(pool_gen.generate_single_series(512)) for _ in range(100)]
    )
    print("\nPool diversity (median over 100 series of length 512):")
    print(f"  roughness:        {np.median(rows[:, 0]):.3f}")
    print(f"  |lag-12 acf|:     {np.median(rows[:, 1]):.3f}")
    print(f"  spectral entropy: {np.median(rows[:, 2]):.3f}")

    output_file = "tsi_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
