"""Example usage of Multivariatizer (cross-series coupling wrapper)."""

import numpy as np

from synforecast import Multivariatizer
from synforecast.generators import TSIGenerator


def main() -> None:
    """Wrap a univariate TSI generator into correlated channels."""
    base = TSIGenerator(min_length=256, max_length=256, freq="h")

    # Default recipe: cotemporaneous mixing + sequential lead-lag coupling
    mv = Multivariatizer(base=base, seed=42)
    df = mv.generate(n_series=4)

    print("Multivariatized TSI (4 coupled channels):")
    print(df.head(10))

    recipe = mv.last_recipe
    print(f"\nSampled coupling recipe: {recipe['couplings']}")
    print(f"Mixing strength: {recipe['mixing']['strength']:.3f}")
    for pair in recipe["leadlag"]:
        print(
            f"Lead-lag: channel {pair['dst']} = "
            f"{pair['sign']:+.0f} x channel {pair['src']} "
            f"lagged {pair['lag']} steps (noise {pair['noise']:.3f})"
        )

    wide = df.pivot_table(index="ds", columns="unique_id", values="y", observed=True)
    corr = np.corrcoef(wide.to_numpy().T)
    print("\nCotemporaneous cross-correlation matrix:")
    with np.printoptions(precision=3, suppress=True):
        print(corr)

    # Mixing only, with strong coupling
    mv_mix = Multivariatizer(
        base=base,
        couplings=["mixing"],
        mixing_strength_range=(0.7, 0.9),
        seed=7,
    )
    df_mix = mv_mix.generate(n_series=3)
    wide_mix = df_mix.pivot_table(
        index="ds", columns="unique_id", values="y", observed=True
    )
    corr_mix = np.corrcoef(wide_mix.to_numpy().T)
    print("\nMixing-only (strength 0.7-0.9) cross-correlation matrix:")
    with np.printoptions(precision=3, suppress=True):
        print(corr_mix)


if __name__ == "__main__":
    main()
