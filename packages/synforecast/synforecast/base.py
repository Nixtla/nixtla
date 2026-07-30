"""Base classes for time series generators."""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ._core import _add_anomalies, _add_changepoints, _add_missingness
from ._lib import batch as _rs_batch
from .exogenous import (
    ExogenousConfig,
    SeriesMetadata,
    build_flag_column,
    extract_datetime_features,
    generate_correlated_exog,
)

# Match Rayon's own default: all logical cores.
# Override with RAYON_NUM_THREADS (e.g. when running inside multiprocessing.Pool).
_default_n_workers: int = int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1))

# Mapping from generator class names to Rust batch type IDs.
# Must match the constants in rust/src/batch.rs.
#
# IDs 17, 18, 23, 26 are intentionally absent — the four generators that are
# not yet in the batch path:
#   17 — CopulaGenerator
#   18 — VARGenerator
#   23 — DailyActiveUsersGenerator
#   26 — IoTSensorGenerator
# Add _get_batch_params() to any of these generators to enable batch generation.
_GEN_TYPE_MAP: dict[str, int] = {
    "RandomWalkGenerator": 0,
    "SeasonalGenerator": 1,
    "SARIMAGenerator": 2,
    "ETSGenerator": 3,
    "INARGenerator": 4,
    "OrnsteinUhlenbeckGenerator": 5,
    "GeometricBrownianMotionGenerator": 6,
    "JumpDiffusionGenerator": 7,
    "PoissonProcessGenerator": 8,
    "CyclicGenerator": 9,
    "GARCHGenerator": 10,
    "HawkesProcessGenerator": 11,
    "ChaoticSystemGenerator": 12,
    "BoundedProcessGenerator": 13,
    "LevyProcessGenerator": 14,
    "StochasticVolatilityGenerator": 15,
    "RegimeSwitchingGenerator": 16,
    "StateSpaceGenerator": 19,
    "FractionalBrownianMotionGenerator": 20,
    "GaussianProcessGenerator": 21,
    "IntermittentDemandGenerator": 22,
    "EnergyLoadGenerator": 24,
    "VitalSignsGenerator": 25,
    "ClickstreamGenerator": 27,
    "TSIGenerator": 28,
    "TCMGenerator": 29,
}


def _categorize_ids(df: nw.DataFrame, id_col: str) -> nw.DataFrame:
    """Cast the id column to a categorical dtype.

    pandas-like backends keep integer categories (matching utilsforecast);
    other backends only support string categoricals, so ids become string
    categories there. Backends without categorical support keep plain ids.
    """
    if df.implementation.is_pandas_like():
        return df.with_columns(nw.col(id_col).cast(nw.Categorical()))
    try:
        return df.with_columns(nw.col(id_col).cast(nw.String()).cast(nw.Categorical()))
    except Exception:
        return df


def _batch_results_to_metadata(
    gen: "BaseGenerator",
    results: list[dict],
    lengths: np.ndarray,
    start_id: int,
) -> list["SeriesMetadata"]:
    """Convert raw Rust batch results into a list of SeriesMetadata.

    Shared by BaseGenerator._generate_parallel and SynSet._generate_parallel
    to avoid duplicating this logic.
    """
    has_exog = gen.exogenous is not None
    all_metadata: list[SeriesMetadata] = []
    for i, r in enumerate(results):
        length = int(lengths[i])
        timestamps = gen._timestamps(length)

        cp_idx = (
            np.asarray(r["cp_indices"])
            if has_exog and gen.exogenous.changepoint_flags
            else None
        )
        anom_idx = (
            np.asarray(r["anom_indices"])
            if has_exog and gen.exogenous.anomaly_flags
            else None
        )
        miss_idx = (
            np.asarray(r["miss_indices"])
            if has_exog and gen.exogenous.missing_flags
            else None
        )

        all_metadata.append(
            SeriesMetadata(
                values=np.asarray(r["values"]),
                timestamps=timestamps,
                series_id=start_id + i,
                length=length,
                anomaly_indices=anom_idx,
                changepoint_indices=cp_idx,
                missing_indices=miss_idx,
            )
        )
    return all_metadata


class BaseGenerator(BaseModel, ABC):
    """Base class for all time series generators.

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data. Either a pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS', 'W-MON') or an integer for an
            integer time index
        engine (str): Output dataframe library (default: 'pandas').
            Options are 'pandas', 'polars', 'cudf', 'modin', 'pyarrow'
        alias (str | None): Name of the generator (default: class name)
        id_col (str): Name of the ID column (default: 'unique_id')
        time_col (str): Name of the timestamp column (default: 'ds')
        target_col (str): Name of the value column (default: 'y')
        start_datetime (str): First timestamp of every series, in any format
            accepted by pandas.Timestamp (default: '2000-01-01'). Ignored
            when freq is an integer
        seed (int | None): Random seed for reproducibility (default: None)

        Exogenous parameters:
        exogenous (ExogenousConfig | None): Configuration for exogenous variable
            generation. None = no exogenous columns (default: None)

        Missingness parameters:
        missing_data (bool): Enable missing data patterns (default: False)
        missing_pattern (str): Pattern: 'random', 'block', 'seasonal' (default: 'random')
        missing_rate (float): Proportion of missing values 0-1 (default: 0.1)
        missing_block_size (int): Size of missing blocks for 'block' pattern (default: 3)
        missing_seasonal_period (int): Period for 'seasonal' pattern (default: 7)

        Anomaly parameters:
        anomalies (bool): Enable anomaly injection (default: False)
        anomaly_fraction (float): Fraction of points that are anomalies (default: 0.05)
        anomaly_types (list[str]): Types: 'spike', 'dip', 'level_shift'
            (default: ['spike', 'dip'])
        spike_magnitude (float): Magnitude of spikes (default: 10.0)
        dip_magnitude (float): Magnitude of dips (default: -10.0)
        level_shift_magnitude (float): Magnitude of level shifts (default: 20.0)
        level_shift_duration (int): Duration of level shifts in time steps (default: 10)

        Changepoint parameters:
        changepoints (bool): Enable changepoint injection (default: False)
        num_changepoints (int): Number of changepoints (default: 2)
        changepoint_type (str): Type: 'level', 'trend', 'variance', 'mixed' (default: 'level')
        changepoint_level_changes (list[float] | None): Size of level changes (default: random)
        changepoint_trend_changes (list[float] | None): Size of trend changes (default: random)
        changepoint_variance_changes (list[float] | None): Size of variance changes
            (default: random)
        changepoint_locations (list[float] | None): Relative positions 0-1 (default: random)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Required fields
    min_length: int = Field(..., gt=0, description="Minimum length of each series")
    max_length: int = Field(..., description="Maximum length of each series")
    freq: str | int = Field(
        ..., description="Frequency of the data (pandas offset alias or integer)"
    )

    # Optional fields with defaults
    engine: Literal["pandas", "polars", "cudf", "modin", "pyarrow"] = Field(
        default="pandas", description="Output dataframe library"
    )
    alias: str | None = Field(default=None, description="Name of the generator")
    id_col: str = Field(default="unique_id", description="Name of the ID column")
    time_col: str = Field(default="ds", description="Name of the timestamp column")
    target_col: str = Field(default="y", description="Name of the value column")
    start_datetime_str: str = Field(
        default="2000-01-01",
        alias="start_datetime",
        description="First timestamp of every series",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    # Innovation distribution
    innovation_distribution: Literal[
        "normal", "t", "laplace", "uniform", "skew_normal"
    ] = Field(
        default="normal",
        description=(
            "Distribution for innovation/noise terms. Options: "
            "'normal' (Gaussian), 't' (Student's t, heavy tails), "
            "'laplace' (double exponential, sharp peak), "
            "'uniform' (bounded), 'skew_normal' (asymmetric)"
        ),
    )
    innovation_params: dict[str, float] | None = Field(
        default=None,
        description=(
            "Distribution-specific parameters. "
            "For 't': {'df': 5} (degrees of freedom, must be > 2). "
            "For 'skew_normal': {'alpha': 5} (skewness parameter)."
        ),
    )

    # Exogenous configuration
    exogenous: ExogenousConfig | None = Field(
        default=None,
        description="Configuration for exogenous variable generation",
    )

    # Missingness parameters
    missing_data: bool = Field(
        default=False, description="Enable missing data patterns"
    )
    missing_pattern: Literal["random", "block", "seasonal"] = Field(
        default="random", description="Missing data pattern"
    )
    missing_rate: float = Field(
        default=0.1, ge=0, le=1, description="Proportion of missing values"
    )
    missing_block_size: int = Field(
        default=3, ge=1, description="Size of missing blocks for 'block' pattern"
    )
    missing_seasonal_period: int = Field(
        default=7, ge=1, description="Period for 'seasonal' pattern"
    )

    # Anomaly parameters
    anomalies: bool = Field(default=False, description="Enable anomaly injection")
    anomaly_fraction: float = Field(
        default=0.05, ge=0, le=1, description="Fraction of points that are anomalies"
    )
    anomaly_types: list[str] = Field(
        default=["spike", "dip"], description="Types of anomalies to inject"
    )
    spike_magnitude: float = Field(default=10.0, description="Magnitude of spikes")
    dip_magnitude: float = Field(default=-10.0, description="Magnitude of dips")
    level_shift_magnitude: float = Field(
        default=20.0, description="Magnitude of level shifts"
    )
    level_shift_duration: int = Field(
        default=10, ge=1, description="Duration of level shifts in time steps"
    )

    # Changepoint parameters
    changepoints: bool = Field(
        default=False, description="Enable changepoint injection"
    )
    num_changepoints: int = Field(default=2, ge=0, description="Number of changepoints")
    changepoint_type: Literal["level", "trend", "variance", "mixed"] = Field(
        default="level", description="Type of changepoint"
    )
    changepoint_level_changes: list[float] | None = Field(
        default=None, description="Size of level changes"
    )
    changepoint_trend_changes: list[float] | None = Field(
        default=None, description="Size of trend changes"
    )
    changepoint_variance_changes: list[float] | None = Field(
        default=None, description="Size of variance changes"
    )
    changepoint_locations: list[float] | None = Field(
        default=None, description="Relative positions 0-1"
    )

    # Runtime state is private: it is not configuration and must not appear in
    # the constructor or serialized model.
    _start_datetime: pd.Timestamp = PrivateAttr()
    _rng: np.random.Generator = PrivateAttr()
    _ts_cache: np.ndarray | None = PrivateAttr(default=None)

    @property
    def start_datetime(self) -> pd.Timestamp:
        """Parsed first timestamp for datetime-indexed series."""
        return self._start_datetime

    @property
    def rng(self) -> np.random.Generator:
        """Generator-local random number stream."""
        return self._rng

    @field_validator("freq")
    @classmethod
    def validate_freq(cls, v: str | int) -> str | int:
        """Validate that freq is a pandas offset alias or a positive integer."""
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("Integer freq must be positive")
            return v
        try:
            pd.tseries.frequencies.to_offset(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid frequency '{v}'. Use a pandas offset alias "
                "(e.g. 'D', 'h', '5min', 'MS') or an integer."
            ) from e
        return v

    @field_validator("anomaly_types")
    @classmethod
    def validate_anomaly_types(cls, v: list[str]) -> list[str]:
        """Validate anomaly types."""
        valid_anomaly_types = {"spike", "dip", "level_shift"}
        for anomaly_type in v:
            if anomaly_type not in valid_anomaly_types:
                raise ValueError(
                    f"anomaly_type must be one of {valid_anomaly_types}, got '{anomaly_type}'"
                )
        return v

    @model_validator(mode="after")
    def validate_and_compute_fields(self) -> "BaseGenerator":
        """Validate cross-field constraints and compute derived fields."""
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")

        columns = (self.id_col, self.time_col, self.target_col)
        if any(not column.strip() for column in columns):
            raise ValueError("output column names must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError("id_col, time_col, and target_col must be distinct")

        params = self.innovation_params or {}
        allowed_params = {
            "normal": set(),
            "t": {"df"},
            "laplace": set(),
            "uniform": set(),
            "skew_normal": {"alpha"},
        }[self.innovation_distribution]
        unknown_params = set(params) - allowed_params
        if unknown_params:
            names = ", ".join(sorted(unknown_params))
            raise ValueError(
                f"unsupported innovation_params for "
                f"{self.innovation_distribution!r}: {names}"
            )
        if self.innovation_distribution == "t":
            df = params.get("df", 5.0)
            if not df > 2:
                raise ValueError("innovation_params['df'] must be > 2")

        if self.changepoint_locations is not None and any(
            not np.isfinite(location) or not 0.0 <= location <= 1.0
            for location in self.changepoint_locations
        ):
            raise ValueError("changepoint_locations must contain values in [0, 1]")
        if self.changepoint_variance_changes is not None and any(
            not np.isfinite(change) or change <= 0
            for change in self.changepoint_variance_changes
        ):
            raise ValueError("changepoint_variance_changes must be positive")

        if self.exogenous is not None:
            reserved = set(columns)
            if self.exogenous.anomaly_flags:
                reserved.add("anomaly_flag")
            if self.exogenous.changepoint_flags:
                reserved.add("changepoint_flag")
            if self.exogenous.missing_flags:
                reserved.add("missing_flag")
            collisions = reserved.intersection(
                config.name for config in self.exogenous.correlated
            )
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(
                    f"correlated exogenous names collide with output columns: {names}"
                )

        self._start_datetime = pd.Timestamp(self.start_datetime_str)
        self._rng = np.random.default_rng(self.seed)

        if self.alias is None:
            object.__setattr__(self, "alias", self.__class__.__name__)

        return self

    def __repr__(self) -> str:
        return self.alias

    def _timestamps(self, length: int) -> np.ndarray:
        """Timestamps (or integer time indices) for a series of `length` steps.

        All series share the same start, so a single range of max_length is
        cached and sliced per series.
        """
        cache = self._ts_cache
        if cache is None or len(cache) < length:
            periods = max(length, self.max_length)
            if isinstance(self.freq, int):
                cache = np.arange(periods, dtype=np.int64) * self.freq
            else:
                cache = pd.date_range(
                    self.start_datetime, periods=periods, freq=self.freq
                ).values
            self._ts_cache = cache
        return cache[:length]

    def _freq_timedelta(self) -> pd.Timedelta | None:
        """Duration of one time step, or None when freq is an integer or a
        non-fixed offset (e.g. 'MS', 'ME', 'YS')."""
        if isinstance(self.freq, int):
            return None
        try:
            return pd.Timedelta(pd.tseries.frequencies.to_offset(self.freq))
        except ValueError:
            return None

    def _step_hours(self, default: float = 24.0) -> float:
        """Hours per time step; `default` when freq is an integer or a
        non-fixed offset."""
        td = self._freq_timedelta()
        if td is None:
            return default
        return td / pd.Timedelta(hours=1)

    def _sample_innovations(
        self,
        size: int | tuple[int, ...],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Sample innovation terms from the configured distribution.

        All distributions are standardized to have approximately mean 0
        and standard deviation equal to ``scale``.

        Args:
            size: Shape of the output array.
            scale: Desired standard deviation of the innovations.

        Returns:
            Innovation samples with the requested shape.
        """
        dist = self.innovation_distribution
        params = self.innovation_params or {}

        if dist == "normal":
            return self.rng.normal(0.0, scale, size)

        if dist == "t":
            df = params.get("df", 5.0)
            if df <= 2:
                raise ValueError(
                    "Degrees of freedom (df) must be > 2 for finite variance."
                )
            # standard_t has variance df/(df-2); rescale to desired sigma
            t_scale = scale * np.sqrt((df - 2) / df)
            return self.rng.standard_t(df, size) * t_scale

        if dist == "laplace":
            # Laplace with scale b has variance 2*b^2, so b = sigma/sqrt(2)
            laplace_scale = scale / np.sqrt(2.0)
            return self.rng.laplace(0.0, laplace_scale, size)

        if dist == "uniform":
            # Uniform[-a, a] has variance a^2/3, so a = sigma*sqrt(3)
            a = scale * np.sqrt(3.0)
            return self.rng.uniform(-a, a, size)

        if dist == "skew_normal":
            alpha = params.get("alpha", 5.0)
            delta = alpha / np.sqrt(1.0 + alpha**2)
            u0 = self.rng.normal(0.0, 1.0, size)
            v = self.rng.normal(0.0, 1.0, size)
            raw = delta * np.abs(u0) + np.sqrt(1.0 - delta**2) * v
            # Centre and rescale: skew-normal has mean delta*sqrt(2/pi)
            # and variance 1 - 2*delta^2/pi
            mean = delta * np.sqrt(2.0 / np.pi)
            std = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
            return (raw - mean) * (scale / std) if std > 0 else raw * scale

        raise ValueError(f"Unknown innovation_distribution: {dist!r}")

    @property
    def _rs_innov_dist(self) -> int:
        """Map innovation_distribution name to Rust enum int."""
        return {"normal": 0, "t": 1, "laplace": 2, "uniform": 3, "skew_normal": 4}[
            self.innovation_distribution
        ]

    @property
    def _rs_innov_param(self) -> float:
        """Extract the single extra parameter needed by the Rust sampler."""
        params = self.innovation_params or {}
        if self.innovation_distribution == "t":
            return float(params.get("df", 5.0))
        if self.innovation_distribution == "skew_normal":
            return float(params.get("alpha", 5.0))
        return 0.0

    @property
    def _batch_gen_type(self) -> int | None:
        """Return Rust batch generator type ID, or None if not supported."""
        return _GEN_TYPE_MAP.get(self.__class__.__name__)

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Return (scalar_params, array_params) for Rust batch generation.

        Override in subclasses to enable the batch code path. The parameter
        order must match the ``dispatch_generator`` match arm in
        ``rust/src/batch.rs``.

        Returns None by default (batch path disabled).
        """
        return None

    def _build_pi_config_dict(self) -> dict:
        """Build pattern injection config dict for Rust batch functions."""
        cfg: dict = {
            "changepoints": self.changepoints,
            "num_changepoints": self.num_changepoints,
            "changepoint_type": self.changepoint_type,
            "anomalies": self.anomalies,
            "anomaly_types": self.anomaly_types,
            "anomaly_fraction": self.anomaly_fraction,
            "spike_magnitude": self.spike_magnitude,
            "dip_magnitude": self.dip_magnitude,
            "level_shift_magnitude": self.level_shift_magnitude,
            "level_shift_duration": self.level_shift_duration,
            "missing_data": self.missing_data,
            "missing_pattern": self.missing_pattern,
            "missing_rate": self.missing_rate,
            "missing_block_size": self.missing_block_size,
            "missing_seasonal_period": self.missing_seasonal_period,
        }
        if self.changepoint_locations is not None:
            cfg["changepoint_locations"] = self.changepoint_locations
        if self.changepoint_level_changes is not None:
            cfg["changepoint_level_changes"] = self.changepoint_level_changes
        if self.changepoint_trend_changes is not None:
            cfg["changepoint_trend_changes"] = self.changepoint_trend_changes
        if self.changepoint_variance_changes is not None:
            cfg["changepoint_variance_changes"] = self.changepoint_variance_changes
        return cfg

    def _add_missingness(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add missing data patterns to a time series.

        Args:
            values: Array of time series values

        Returns:
            Tuple of (modified values, missing indices or None)
        """
        if not self.missing_data:
            return values, None

        values, metadata = _add_missingness(
            values=values,
            rng=self.rng,
            missing_pattern=self.missing_pattern,
            missing_rate=self.missing_rate,
            missing_block_size=self.missing_block_size,
            missing_seasonal_period=self.missing_seasonal_period,
        )

        if self.exogenous and self.exogenous.missing_flags:
            return values, metadata["missing_indices"]
        return values, None

    def _add_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add anomalies to a time series.

        Args:
            values: Array of time series values

        Returns:
            Tuple of (modified values, anomaly indices or None)
        """
        if not self.anomalies:
            return values, None

        values, metadata = _add_anomalies(
            values=values,
            rng=self.rng,
            anomaly_types=self.anomaly_types,
            anomaly_fraction=self.anomaly_fraction,
            spike_magnitude=self.spike_magnitude,
            dip_magnitude=self.dip_magnitude,
            level_shift_magnitude=self.level_shift_magnitude,
            level_shift_duration=self.level_shift_duration,
        )

        if self.exogenous and self.exogenous.anomaly_flags:
            return values, metadata["anomaly_indices"]
        return values, None

    def _add_changepoints(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add changepoints (structural breaks) to a time series.

        Args:
            values: Array of time series values

        Returns:
            Tuple of (modified values, changepoint indices or None)
        """
        if not self.changepoints or self.num_changepoints == 0:
            return values, None

        changepoint_locations = (
            np.array(self.changepoint_locations)
            if self.changepoint_locations is not None
            else np.array([])
        )
        changepoint_level_changes = (
            np.array(self.changepoint_level_changes)
            if self.changepoint_level_changes is not None
            else np.array([])
        )
        changepoint_trend_changes = (
            np.array(self.changepoint_trend_changes)
            if self.changepoint_trend_changes is not None
            else np.array([])
        )
        changepoint_variance_changes = (
            np.array(self.changepoint_variance_changes)
            if self.changepoint_variance_changes is not None
            else np.array([])
        )

        values, metadata = _add_changepoints(
            values=values,
            rng=self.rng,
            num_changepoints=self.num_changepoints,
            changepoint_locations=changepoint_locations,
            changepoint_type=self.changepoint_type,
            changepoint_level_changes=changepoint_level_changes,
            changepoint_trend_changes=changepoint_trend_changes,
            changepoint_variance_changes=changepoint_variance_changes,
        )

        if self.exogenous and self.exogenous.changepoint_flags:
            return values, metadata["changepoint_indices"]
        return values, None

    def _apply_pattern_injection(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Apply the full pattern injection pipeline.

        Runs changepoints → anomalies → missingness in order.

        Args:
            values: Array of time series values

        Returns:
            Tuple of (values, changepoint_indices, anomaly_indices, missing_indices). Index arrays are None when the corresponding flag is not requested.
        """
        values, cp_indices = self._add_changepoints(values)
        values, anom_indices = self._add_anomalies(values)
        values, miss_indices = self._add_missingness(values)
        return values, cp_indices, anom_indices, miss_indices

    def _build_dataframe(
        self,
        series_metadata: list[SeriesMetadata],
    ) -> IntoDataFrameT:
        """Build the output DataFrame from accumulated series metadata.

        Handles flattening values/timestamps/IDs, adding exogenous columns
        (pattern flags, datetime features, correlated variables), and merging
        any generator-specific extra columns.

        Args:
            series_metadata: List of SeriesMetadata for all generated series
        Returns:
            DataFrame in the configured engine.
        """
        # Flatten core columns
        flat_values = np.concatenate([m.values for m in series_metadata])
        flat_timestamps = np.concatenate([m.timestamps for m in series_metadata])
        flat_ids = np.repeat(
            np.array([m.series_id for m in series_metadata], dtype=np.int64),
            [m.length for m in series_metadata],
        )

        result: dict[str, Any] = {
            self.id_col: flat_ids,
            self.time_col: flat_timestamps,
            self.target_col: flat_values,
        }

        # Merge generator-specific extra columns (e.g., DAU event_col)
        extra_col_names: set[str] = set()
        for meta in series_metadata:
            if meta.extra_columns:
                extra_col_names.update(meta.extra_columns.keys())
        for col_name in sorted(extra_col_names):
            result[col_name] = np.concatenate(
                [
                    meta.extra_columns.get(col_name, np.full(meta.length, np.nan))
                    for meta in series_metadata
                ]
            )

        # Add exogenous columns
        if self.exogenous is not None:
            # Pattern injection flag columns
            if self.exogenous.anomaly_flags:
                result["anomaly_flag"] = build_flag_column(series_metadata, "anomaly")
            if self.exogenous.changepoint_flags:
                result["changepoint_flag"] = build_flag_column(
                    series_metadata, "changepoint"
                )
            if self.exogenous.missing_flags:
                result["missing_flag"] = build_flag_column(series_metadata, "missing")

            # Datetime features
            if self.exogenous.datetime_features or self.exogenous.datetime_cyclical:
                if isinstance(self.freq, int):
                    raise ValueError(
                        "Datetime features require a datetime frequency; "
                        "got an integer freq."
                    )
                dt_cols = extract_datetime_features(
                    flat_timestamps,
                    self.freq,
                    include_basic=self.exogenous.datetime_features,
                    include_cyclical=self.exogenous.datetime_cyclical,
                )
                result.update(dt_cols)

            # Correlated exogenous
            for exog_config in self.exogenous.correlated:
                result[exog_config.name] = generate_correlated_exog(
                    series_metadata, flat_values, exog_config, self.rng
                )

        df_nw = nw.DataFrame.from_dict(result, backend=self.engine)
        df = _categorize_ids(df_nw, self.id_col).to_native()

        return df

    def _generate_one_series(
        self,
        length: int,
        series_id: int,
        thread_rng: np.random.Generator | None = None,
        thread_gen_rng: np.random.Generator | None = None,
    ) -> SeriesMetadata:
        """Generate a single series with timestamps and pattern injection.

        Args:
            length: Length of the series to generate
            series_id: ID for this series
            thread_rng: Per-thread RNG for pattern injection (parallel mode).
                If None, uses self.rng (sequential mode).
            thread_gen_rng: Per-thread RNG for series generation (parallel mode).
                If None, uses self.rng (sequential mode).

        Returns:
            SeriesMetadata for the generated series
        """
        timestamps = self._timestamps(length)

        if thread_gen_rng is not None:
            # Thread-safe: use a shallow copy with thread-local RNG
            local_copy = self.model_copy()
            local_copy._rng = thread_gen_rng
            values = local_copy.generate_single_series(length=length)
        else:
            values = self.generate_single_series(length=length)

        # Pattern injection uses the provided RNG (thread-local or shared)
        rng = thread_rng if thread_rng is not None else self.rng
        values, cp_indices, anom_indices, miss_indices = (
            self._apply_pattern_injection_with_rng(values, rng)
        )

        return SeriesMetadata(
            values=values,
            timestamps=timestamps,
            series_id=series_id,
            length=length,
            anomaly_indices=anom_indices,
            changepoint_indices=cp_indices,
            missing_indices=miss_indices,
        )

    def _apply_pattern_injection_with_rng(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Apply pattern injection using a specific RNG (for thread safety)."""
        # Use a shallow copy with the given RNG to avoid mutating shared state
        local_copy = self.model_copy()
        local_copy._rng = rng
        return local_copy._apply_pattern_injection(values)

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,
    ) -> IntoDataFrameT:
        """Generate synthetic time series data.

        Args:
            n_series (int): Number of time series to generate
            start_id (int, optional): Starting ID for the series numbering (default: 0)
                Series will be numbered from start_id to start_id + n_series - 1
            n_jobs (int): Number of parallel workers. -1 (default) uses
                ``RAYON_NUM_THREADS`` if set, otherwise all logical cores.
                Results are seed-deterministic and do not depend on n_jobs.

        Returns:
            DataFrame in long format with columns [id_col, time_col, target_col] (default ['unique_id', 'ds', 'y']), plus any exogenous or flag columns.
        """
        if n_series < 1:
            raise ValueError(f"n_series must be a positive integer, got {n_series}.")
        if n_jobs == -1:
            n_jobs = _default_n_workers
        all_metadata = self._generate_parallel(n_series, start_id, n_jobs)

        return self._build_dataframe(all_metadata)

    def _generate_parallel(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Generate series in parallel via Rust rayon batch path.

        Generators without a Rust batch implementation use the threaded path.
        """
        gen_type = self._batch_gen_type
        batch_params = self._get_batch_params() if gen_type is not None else None
        if batch_params is None:
            return self._generate_parallel_threaded(n_series, start_id, n_jobs)

        scalar_params, array_params = batch_params

        # Pre-sample lengths and seeds deterministically from master RNG
        lengths = np.array(
            [
                int(self.rng.integers(self.min_length, self.max_length + 1))
                for _ in range(n_series)
            ],
            dtype=np.int32,
        )
        gen_seeds = self.rng.integers(0, 2**63, size=n_series, dtype=np.uint64)
        # 3 PI seeds per series: changepoint, anomaly, missingness
        pi_seeds = self.rng.integers(0, 2**63, size=n_series * 3, dtype=np.uint64)

        pi_config = self._build_pi_config_dict()

        results = _rs_batch.generate_batch(
            gen_type,
            scalar_params,
            array_params,
            lengths,
            gen_seeds,
            pi_seeds,
            pi_config,
            n_workers=n_jobs,
        )

        return _batch_results_to_metadata(self, results, lengths, start_id)

    def _generate_parallel_threaded(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Generate series in parallel using ThreadPoolExecutor.

        Pre-samples all random state from self.rng sequentially for
        reproducibility, then dispatches generation to threads.
        """
        # Pre-sample lengths and seeds deterministically from master RNG
        lengths = [
            int(self.rng.integers(self.min_length, self.max_length + 1))
            for _ in range(n_series)
        ]
        gen_seeds = [int(self.rng.integers(0, 2**63)) for _ in range(n_series)]
        pi_seeds = [int(self.rng.integers(0, 2**63)) for _ in range(n_series)]

        workers = max(1, min(n_series, n_jobs))

        def gen_one(i: int) -> SeriesMetadata:
            thread_gen_rng = np.random.default_rng(gen_seeds[i])
            thread_pi_rng = np.random.default_rng(pi_seeds[i])
            return self._generate_one_series(
                length=lengths[i],
                series_id=start_id + i,
                thread_rng=thread_pi_rng,
                thread_gen_rng=thread_gen_rng,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_metadata = list(pool.map(gen_one, range(n_series)))

        return all_metadata

    @abstractmethod
    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        pass
