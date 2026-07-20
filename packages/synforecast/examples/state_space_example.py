"""Example usage of StateSpaceGenerator: hidden states evolving over time with
observations that depend on them (local level, local trend, custom dynamics)."""

import numpy as np
import polars as pl

from synforecast.generators import StateSpaceGenerator


def main() -> None:
    """Generate and display State Space Model examples."""
    # Local level model (random walk plus observation noise)
    local_level_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=1,
        obs_dim=1,
        seed=42,
    )
    local_level_df = local_level_gen.generate(n_series=1)
    print("Local level model:")
    print(local_level_df.head(10))
    print(f"Mean={local_level_df['y'].mean():.4f}, Std={local_level_df['y'].std():.4f}")

    # 2D state: level plus trend
    two_dim_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        seed=42,
    )
    two_dim_df = two_dim_gen.generate(n_series=1)
    print(
        f"\n2D state space (level + trend): Mean={two_dim_df['y'].mean():.4f}, "
        f"Std={two_dim_df['y'].std():.4f}"
    )

    # Custom transition matrix produces AR-like dynamics
    transition_matrix = np.array([[0.9, 0.1], [0.0, 0.8]])
    custom_transition_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        transition_matrix=transition_matrix.tolist(),
        seed=42,
    )
    custom_transition_df = custom_transition_gen.generate(n_series=1)
    print(
        f"\nCustom transition matrix: Mean={custom_transition_df['y'].mean():.4f}, "
        f"Std={custom_transition_df['y'].std():.4f}"
    )

    # Custom observation matrix: observe a weighted combination of states
    observation_matrix = np.array([[1.0, 0.5]])
    custom_obs_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        observation_matrix=observation_matrix.tolist(),
        seed=42,
    )
    custom_obs_df = custom_obs_gen.generate(n_series=1)
    print(
        f"\nCustom observation matrix: Mean={custom_obs_df['y'].mean():.4f}, "
        f"Std={custom_obs_df['y'].std():.4f}"
    )

    # High process noise gives volatile state evolution
    high_process_noise = np.array([[5.0, 0.0], [0.0, 5.0]])
    high_noise_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        process_covariance=high_process_noise.tolist(),
        seed=42,
    )
    high_noise_df = high_noise_gen.generate(n_series=1)
    print(
        f"\nHigh process noise: Mean={high_noise_df['y'].mean():.4f}, "
        f"Std={high_noise_df['y'].std():.4f}"
    )

    # High observation noise gives noisy measurements of a smooth state
    high_obs_noise = np.array([[10.0]])
    noisy_obs_gen = StateSpaceGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        observation_covariance=high_obs_noise.tolist(),
        seed=42,
    )
    noisy_obs_df = noisy_obs_gen.generate(n_series=1)
    print(
        f"\nHigh observation noise: Mean={noisy_obs_df['y'].mean():.4f}, "
        f"Std={noisy_obs_df['y'].std():.4f}"
    )

    # generate_with_states also returns the hidden states
    with_states_gen = StateSpaceGenerator(
        min_length=100,
        max_length=100,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        seed=42,
    )
    obs_df, states_df = with_states_gen.generate_with_states(n_series=1)
    print("\nObservations:")
    print(obs_df.head(10))
    print("\nHidden states:")
    print(states_df.head(10))

    # Multiple series
    multi_gen = StateSpaceGenerator(
        min_length=150,
        max_length=150,
        freq="D",
        engine="polars",
        state_dim=2,
        obs_dim=1,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=3)
    print(f"\nGenerated 3 series ({len(multi_df)} total observations), first series:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))

    local_level_df.write_csv("state_space_local_level_example.csv")
    two_dim_df.write_csv("state_space_2d_example.csv")
    custom_transition_df.write_csv("state_space_custom_transition_example.csv")
    custom_obs_df.write_csv("state_space_custom_observation_example.csv")
    high_noise_df.write_csv("state_space_high_process_noise_example.csv")
    noisy_obs_df.write_csv("state_space_high_obs_noise_example.csv")
    obs_df.write_csv("state_space_with_states_observations_example.csv")
    states_df.write_csv("state_space_with_states_hidden_example.csv")
    multi_df.write_csv("state_space_multiple_series_example.csv")


if __name__ == "__main__":
    main()
