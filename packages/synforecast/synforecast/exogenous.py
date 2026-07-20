"""Exogenous variable generation for synthetic time series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorrelatedExogConfig(BaseModel):
    """Configuration for a single correlated exogenous variable."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Column name for this exogenous variable")
    method: Literal["correlated_noise", "lagged_copy", "trend_following"] = Field(
        default="correlated_noise",
        description="Method for generating correlated exogenous",
    )
    # For correlated_noise
    correlation: float = Field(
        default=0.7, ge=-1, le=1, description="Target correlation with the series"
    )
    # For lagged_copy
    lag: int = Field(default=1, ge=1, description="Lag for lagged_copy method")
    noise_std: float = Field(
        default=0.1, ge=0, description="Noise std for lagged_copy method"
    )
    # For trend_following
    smoothing_window: int = Field(
        default=10, ge=1, description="Window size for trend_following method"
    )
    trend_noise_std: float = Field(
        default=0.1, ge=0, description="Noise std for trend_following method"
    )


class ExogenousConfig(BaseModel):
    """Configuration for exogenous variable generation.

    Controls which exogenous columns are added to the output DataFrame.
    All options are off by default for backward compatibility.

    Args:
        datetime_features: Add calendar features (year, month, day_of_week, etc.)
        datetime_cyclical: Add sin/cos cyclical encodings of datetime features
        anomaly_flags: Add binary column indicating anomaly positions
        changepoint_flags: Add binary column indicating changepoint positions
        missing_flags: Add binary column indicating missing data positions
        correlated: List of correlated exogenous variables to generate
    """

    model_config = ConfigDict(extra="forbid")

    datetime_features: bool = Field(default=False, description="Add calendar features")
    datetime_cyclical: bool = Field(
        default=False, description="Add sin/cos cyclical encodings"
    )
    anomaly_flags: bool = Field(
        default=False, description="Add anomaly indicator column"
    )
    changepoint_flags: bool = Field(
        default=False, description="Add changepoint indicator column"
    )
    missing_flags: bool = Field(
        default=False, description="Add missing data indicator column"
    )
    correlated: list[CorrelatedExogConfig] = Field(
        default_factory=list,
        description="Correlated exogenous variables to generate",
    )

    @model_validator(mode="after")
    def validate_unique_names(self) -> ExogenousConfig:
        """Reject duplicate output columns before dataframe construction."""
        names = [config.name for config in self.correlated]
        if len(names) != len(set(names)):
            raise ValueError("correlated exogenous variable names must be unique")
        return self


@dataclass
class SeriesMetadata:
    """Accumulated metadata for a single generated series.

    Carries per-series data through the generation pipeline to
    the DataFrame construction step.
    """

    values: np.ndarray
    timestamps: np.ndarray
    series_id: int
    length: int
    anomaly_indices: np.ndarray | None = None
    changepoint_indices: np.ndarray | None = None
    missing_indices: np.ndarray | None = None
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)


def extract_datetime_features(
    timestamps: np.ndarray,
    freq: str,
    include_basic: bool = True,
    include_cyclical: bool = False,
) -> dict[str, np.ndarray]:
    """Extract datetime features from a numpy datetime64 array.

    Features are filtered based on the frequency to avoid constant columns.
    For example, hourly features are omitted for daily frequency data.

    Args:
        timestamps: Array of numpy datetime64 values
        freq: Frequency as a pandas offset alias (e.g., 'h', 'D', 'MS')
        include_basic: Include calendar features (year, month, etc.)
        include_cyclical: Include sin/cos cyclical encodings

    Returns:
        Dict mapping column names to numpy arrays
    """
    ts = pd.DatetimeIndex(timestamps)

    features: dict[str, np.ndarray] = {}

    # Classify the step size to skip features that would be constant
    # (e.g. hour-of-day for daily data). Non-fixed offsets (monthly and
    # coarser) have no fixed Timedelta and only keep the coarse features.
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except ValueError:
        step = None
    sub_monthly = step is not None and step <= pd.Timedelta(days=1)
    sub_daily = step is not None and step < pd.Timedelta(days=1)
    sub_hourly = step is not None and step < pd.Timedelta(hours=1)

    if include_basic:
        features["year"] = np.asarray(ts.year, dtype=np.int32)
        features["quarter"] = np.asarray(ts.quarter, dtype=np.int8)
        features["month"] = np.asarray(ts.month, dtype=np.int8)
        features["day_of_year"] = np.asarray(ts.dayofyear, dtype=np.int16)

        if sub_monthly:
            features["day_of_week"] = np.asarray(ts.dayofweek, dtype=np.int8)
            features["day_of_month"] = np.asarray(ts.day, dtype=np.int8)
            features["is_weekend"] = np.asarray(ts.dayofweek >= 5, dtype=np.int8)

        if sub_daily:
            features["hour"] = np.asarray(ts.hour, dtype=np.int8)

        if sub_hourly:
            features["minute"] = np.asarray(ts.minute, dtype=np.int8)

    if include_cyclical:
        hour = np.asarray(ts.hour)
        dow = np.asarray(ts.dayofweek)
        month = np.asarray(ts.month)
        doy = np.asarray(ts.dayofyear)

        if sub_daily:
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

        if sub_monthly:
            features["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
            features["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

        features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12).astype(np.float32)
        features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12).astype(np.float32)

        features["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25).astype(np.float32)
        features["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25).astype(np.float32)

    return features


def build_flag_column(
    series_metadata: list[SeriesMetadata],
    flag_type: str,
) -> np.ndarray:
    """Build a binary flag array from series metadata.

    Args:
        series_metadata: List of SeriesMetadata with index arrays
        flag_type: One of 'anomaly', 'changepoint', 'missing'

    Returns:
        Concatenated binary int8 array (1 = flagged, 0 = not)
    """
    attr_name = f"{flag_type}_indices"
    all_flags = []
    for meta in series_metadata:
        flags = np.zeros(meta.length, dtype=np.int8)
        indices = getattr(meta, attr_name, None)
        if indices is not None and len(indices) > 0:
            valid = indices[(indices >= 0) & (indices < meta.length)]
            flags[valid] = 1
        all_flags.append(flags)
    return np.concatenate(all_flags)


def generate_correlated_exog(
    series_metadata: list[SeriesMetadata],
    flat_values: np.ndarray,
    config: CorrelatedExogConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a correlated exogenous variable for all series.

    Args:
        series_metadata: List of SeriesMetadata
        flat_values: Concatenated target values
        config: Configuration for this exogenous variable
        rng: Random number generator

    Returns:
        Concatenated exogenous values array
    """
    all_exog = []
    offset = 0
    for meta in series_metadata:
        target = flat_values[offset : offset + meta.length]
        exog = _generate_single_correlated(target, config, rng)
        all_exog.append(exog)
        offset += meta.length
    return np.concatenate(all_exog)


def _generate_single_correlated(
    target: np.ndarray,
    config: CorrelatedExogConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate correlated exogenous values for a single series.

    Args:
        target: Target series values
        config: Configuration for correlation method
        rng: Random number generator

    Returns:
        Exogenous values array of same length as target
    """
    length = len(target)

    if config.method == "correlated_noise":
        rho = config.correlation
        # Standardize target (handle NaN)
        target_mean = np.nanmean(target)
        target_std = np.nanstd(target)
        z1 = (target - target_mean) / (target_std + 1e-10)
        z1 = np.where(np.isnan(z1), 0.0, z1)
        z2 = rng.standard_normal(length)
        return rho * z1 + np.sqrt(1 - rho**2) * z2

    elif config.method == "lagged_copy":
        lag = config.lag
        exog = np.roll(target, lag)
        exog[:lag] = target[:lag]
        noise = rng.normal(0, config.noise_std, length)
        return exog + noise

    elif config.method == "trend_following":
        kernel = np.ones(config.smoothing_window) / config.smoothing_window
        target_clean = np.where(np.isnan(target), 0, target)
        smoothed = np.convolve(target_clean, kernel, mode="same")
        noise = rng.normal(0, config.trend_noise_std, length)
        return smoothed + noise

    else:
        raise ValueError(f"Unknown correlated exogenous method: {config.method}")
