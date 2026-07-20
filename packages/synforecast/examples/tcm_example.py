"""Example usage of TCMGenerator (temporal causal model)."""

import numpy as np

from synforecast.generators.tcm import TCMGenerator


def main() -> None:
    """Generate and display temporal causal model time series."""
    # Each series comes from a freshly sampled random causal graph over
    # latent variables; the observed series is one node of the rollout
    generator = TCMGenerator(
        min_length=256,
        max_length=512,
        freq="h",
        engine="polars",
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} time series")
    print(f"Total observations: {len(df)}")
    print(df.head(10))

    # Diversity audit on a small pool: roughness, spectral entropy, lag-12
    # autocorrelation of the standardized series
    pool_gen = TCMGenerator(min_length=512, max_length=512, freq="h", seed=0)
    roughness, entropy, acf12 = [], [], []
    for _ in range(50):
        values = pool_gen.generate_single_series(512)
        z = (values - values.mean()) / (values.std() + 1e-9)
        roughness.append(np.std(np.diff(z)) / (np.std(z) + 1e-9))
        power = np.abs(np.fft.rfft(z)[1:]) ** 2
        power = power / power.sum()
        entropy.append(-(power * np.log(power + 1e-12)).sum() / np.log(len(power)))
        demeaned = z - z.mean()
        acf12.append(demeaned[:-12] @ demeaned[12:] / (demeaned @ demeaned))

    print("\nPool diversity metrics (50 series, medians):")
    print(f"Roughness:           {np.median(roughness):.3f}")
    print(f"Spectral entropy:    {np.median(entropy):.3f}")
    print(f"|Lag-12 autocorr|:   {np.median(np.abs(acf12)):.3f}")


if __name__ == "__main__":
    main()
