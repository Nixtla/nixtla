"""Pattern-injection functions backed by the native extension."""

import numpy as np

from synforecast._lib import pattern_injection as _rs_pi


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
    """Add structural breaks to a time series."""
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


def _add_missingness(
    values: np.ndarray,
    rng: np.random.Generator,
    missing_pattern: str,
    missing_rate: float,
    missing_block_size: int,
    missing_seasonal_period: int,
) -> tuple[np.ndarray, dict]:
    """Add a missing-value pattern to a time series."""
    seed = int(rng.integers(0, 2**63))
    return _rs_pi.add_missingness(
        values,
        seed,
        missing_pattern,
        missing_rate,
        missing_block_size,
        missing_seasonal_period,
    )


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
    """Add anomalies to a time series."""
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
