"""Core accelerated functions using pure NumPy (optimized vectorized version)"""

import numpy as np

try:
    from synforecast._lib import pattern_injection as _rs_pi

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _add_changepoints(
    values: np.ndarray,
    rng: np.random.Generator,
    num_changepoints: int,
    changepoint_locations: np.ndarray,
    changepoint_type: str,
    changepoint_level_changes: np.ndarray,
    changepoint_trend_changes: np.ndarray,
    changepoint_variance_changes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Add changepoints (structural breaks) to a time series.

    Args:
        values (np.ndarray): Array of time series values
        rng (np.random.Generator): NumPy random number generator
        num_changepoints (int): Number of changepoints to add
        changepoint_locations (np.ndarray): Locations of changepoints (relative, 0 to 1)
        changepoint_type (str): Type of changepoints ("level", "trend", "variance", "mixed")
        changepoint_level_changes (np.ndarray): Level changes at changepoints
        changepoint_trend_changes (np.ndarray): Trend changes at changepoints
        changepoint_variance_changes (np.ndarray): Variance changes at changepoints

    Returns:
        tuple: (modified values array, metadata dict with 'changepoint_indices')
    """
    if _HAS_RUST:
        seed = int(rng.integers(0, 2**63))
        return _rs_pi.add_changepoints(
            values,
            seed,
            num_changepoints,
            changepoint_locations,
            changepoint_type,
            changepoint_level_changes,
            changepoint_trend_changes,
            changepoint_variance_changes,
        )

    length = len(values)
    n_cp = num_changepoints

    # Generate changepoint locations if not provided; pad with random draws
    # when fewer than num_changepoints are given, truncate extras (matches
    # the Rust implementation's semantics).
    if changepoint_locations.size == 0:
        # Random locations, sorted, avoid edges
        locations = np.sort(rng.uniform(0.1, 0.9, n_cp))
    elif changepoint_locations.size < n_cp:
        pad = rng.uniform(0.1, 0.9, n_cp - changepoint_locations.size)
        locations = np.concatenate(
            [np.asarray(changepoint_locations, dtype=float), pad]
        )
    else:
        locations = np.asarray(changepoint_locations[:n_cp], dtype=float)

    # Convert relative locations to indices; clip so a location of 1.0 maps
    # to the last valid index instead of out of bounds.
    changepoint_indices = np.clip(
        (locations * length).astype(np.int64), 0, max(length - 1, 0)
    )

    def _resolve_changes(
        provided: np.ndarray,
        active: bool,
        low: float,
        high: float,
        inactive_fill: float,
    ) -> np.ndarray:
        """Resolve per-changepoint change sizes to exactly n_cp entries."""
        if not active:
            return np.full(n_cp, inactive_fill, dtype=float)
        provided = np.asarray(provided, dtype=float)
        if provided.size >= n_cp:
            return provided[:n_cp]
        pad = rng.uniform(low, high, n_cp - provided.size)
        return np.concatenate([provided, pad])

    level_changes = _resolve_changes(
        changepoint_level_changes,
        changepoint_type in ("level", "mixed"),
        -20.0,
        20.0,
        0.0,
    )
    trend_changes = _resolve_changes(
        changepoint_trend_changes,
        changepoint_type in ("trend", "mixed"),
        -0.5,
        0.5,
        0.0,
    )
    variance_changes = _resolve_changes(
        changepoint_variance_changes,
        changepoint_type in ("variance", "mixed"),
        0.5,
        2.0,
        1.0,
    )

    # Apply changepoints using incremental changes per changepoint
    for cp_idx in range(num_changepoints):
        changepoint_pos = changepoint_indices[cp_idx]

        # Create index range from changepoint to end
        indices = np.arange(changepoint_pos, length)

        # Level change: shift the values (vectorized)
        values[changepoint_pos:] += level_changes[cp_idx]

        # Trend change: add time-dependent component (vectorized)
        time_since_changepoint = indices - changepoint_pos
        values[changepoint_pos:] += trend_changes[cp_idx] * time_since_changepoint

        # Variance change: scale the deviation from mean
        # This requires element-wise processing due to rolling mean dependency
        if variance_changes[cp_idx] != 1.0:
            for t in range(changepoint_pos, length):
                # Calculate mean at this point
                mean_val = np.mean(values[max(0, t - 10) : t + 1])
                deviation = values[t] - mean_val
                values[t] = mean_val + deviation * variance_changes[cp_idx]

    metadata = {"changepoint_indices": changepoint_indices}
    return values, metadata


def _add_missingness(
    values: np.ndarray,
    rng: np.random.Generator,
    missing_pattern: str,
    missing_rate: float,
    missing_block_size: int,
    missing_seasonal_period: int,
) -> tuple[np.ndarray, dict]:
    """Add missing data patterns to a time series.

    Args:
        values (np.ndarray): Array of time series values
        rng (np.random.Generator): NumPy random number generator
        missing_pattern (str): Pattern: 'random', 'block', 'seasonal'
        missing_rate (float): Proportion of missing values 0-1
        missing_block_size (int): Size of missing blocks for 'block' pattern
        missing_seasonal_period (int): Period for 'seasonal' pattern

    Returns:
        tuple: (modified values array, metadata dict with 'missing_indices')
    """
    if _HAS_RUST:
        seed = int(rng.integers(0, 2**63))
        return _rs_pi.add_missingness(
            values,
            seed,
            missing_pattern,
            missing_rate,
            missing_block_size,
            missing_seasonal_period,
        )

    if missing_pattern == "block" and missing_block_size < 1:
        raise ValueError("missing_block_size must be >= 1 for block pattern")
    if missing_pattern == "seasonal" and missing_seasonal_period < 1:
        raise ValueError("missing_seasonal_period must be >= 1 for seasonal pattern")

    length = len(values)

    # Endpoint semantics (all patterns): rate 0 is a no-op, rate 1 marks
    # every point missing with exact metadata. Mirrors the Rust path.
    if missing_rate <= 0:
        return values, {"missing_indices": np.array([], dtype=np.int64)}
    if missing_rate >= 1:
        values[:] = np.nan
        return values, {"missing_indices": np.arange(length, dtype=np.int64)}

    all_missing_indices = []

    if missing_pattern == "random":
        # Random missing: floor(length * missing_rate) distinct points,
        # sampled without replacement so the NaN fraction matches the rate.
        n_missing = int(length * missing_rate)
        if n_missing > 0:
            missing_indices = rng.choice(length, size=n_missing, replace=False)
            values[missing_indices] = np.nan
            all_missing_indices.append(missing_indices)

    elif missing_pattern == "block":
        # Block missing: consecutive blocks of missing values. Blocks may
        # overlap, so the realized NaN fraction can fall below the target.
        n_missing = int(length * missing_rate)
        n_blocks = max(1, n_missing // missing_block_size)
        max_start = max(0, length - missing_block_size)

        for _ in range(n_blocks):
            # Choose random start position (max_start inclusive, so blocks
            # can reach the end of the series; when the series is shorter
            # than the block size, a truncated block starts at 0)
            start_idx = int(rng.integers(0, max_start + 1))
            end_idx = min(start_idx + missing_block_size, length)
            if end_idx > start_idx:
                values[start_idx:end_idx] = np.nan
                all_missing_indices.append(np.arange(start_idx, end_idx))

    elif missing_pattern == "seasonal":
        # Seasonal missing: missing values at regular intervals with some randomness
        # Higher probability of missingness at certain phases of the seasonal period
        seasonal_indices = []
        for i in range(length):
            phase = i % missing_seasonal_period
            # Sinusoidal missingness probability over the period (peak at
            # phase = period/4); averages to missing_rate over a full period
            phase_prob = missing_rate * (
                1 + np.sin(2 * np.pi * phase / missing_seasonal_period)
            )
            if rng.random() < phase_prob:
                values[i] = np.nan
                seasonal_indices.append(i)
        if seasonal_indices:
            all_missing_indices.append(np.array(seasonal_indices, dtype=np.int64))

    if all_missing_indices:
        # Sorted unique indices so the metadata matches the injected
        # positions exactly even when blocks overlap
        combined = np.unique(np.concatenate(all_missing_indices)).astype(np.int64)
    else:
        combined = np.array([], dtype=np.int64)

    metadata = {"missing_indices": combined}
    return values, metadata


def _add_anomalies(
    values: np.ndarray,
    rng: np.random.Generator,
    anomaly_types: list[str],
    anomaly_fraction: float,
    spike_magnitude: float,
    dip_magnitude: float,
    level_shift_magnitude: float,
    level_shift_duration: int,
) -> tuple[np.ndarray, dict]:
    """Add anomalies to a time series.

    Args:
        values (np.ndarray): Array of time series values
        rng (np.random.Generator): NumPy random number generator
        anomaly_types (list[str]): Types: 'spike', 'dip', 'level_shift'
        anomaly_fraction (float): Fraction of points that are anomalies
        spike_magnitude (float): Magnitude of spikes
        dip_magnitude (float): Magnitude of dips
        level_shift_magnitude (float): Magnitude of level shifts
        level_shift_duration (int): Duration of level shifts in time steps

    Returns:
        tuple: (modified values array, metadata dict with 'anomaly_indices')
    """
    if _HAS_RUST:
        seed = int(rng.integers(0, 2**63))
        return _rs_pi.add_anomalies(
            values,
            seed,
            anomaly_types,
            anomaly_fraction,
            spike_magnitude,
            dip_magnitude,
            level_shift_magnitude,
            level_shift_duration,
        )

    length = len(values)

    # Determine number of anomalies
    num_anomalies = int(length * anomaly_fraction)

    if num_anomalies > 0:
        # Generate random anomaly locations without replacement so exactly
        # num_anomalies distinct points are affected and magnitudes never
        # stack at a duplicated location
        anomaly_locations = rng.choice(length, size=num_anomalies, replace=False)

        for location in anomaly_locations:
            # Randomly select anomaly type
            anomaly_type = anomaly_types[int(rng.integers(0, len(anomaly_types)))]

            if anomaly_type == "spike":
                # Single point spike
                values[location] += spike_magnitude

            elif anomaly_type == "dip":
                # Single point dip
                values[location] += dip_magnitude

            elif anomaly_type == "level_shift":
                # Level shift for a duration
                end_location = min(location + level_shift_duration, length)
                values[location:end_location] += level_shift_magnitude

        metadata = {"anomaly_indices": anomaly_locations}
    else:
        metadata = {"anomaly_indices": np.array([], dtype=np.int64)}

    return values, metadata
