"""Base classes for time series generators."""

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
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

from ._lib import batch as _rs_batch
from ._lib import overlays as _rs_ov
from .exogenous import (
    ExogenousConfig,
    SeriesMetadata,
    build_flag_column,
    extract_datetime_features,
    generate_correlated_exog,
)
from .overlays import (
    LEGACY_DEFAULTS,
    OverlayConfig,
    absorb_legacy_kwargs,
    cycle_phase,
    cycle_supported,
    legacy_value,
)

# Match Rayon's own default: all logical cores.
# Override with RAYON_NUM_THREADS (e.g. when running inside multiprocessing.Pool).
_default_n_workers: int = int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1))

# The native batch kernels take a plan even when no overlay is on.
_NO_OVERLAYS = _rs_ov.OverlayPlan({"base_weight": 1.0})

# Sentinel: build the overlay plan inside _apply_pattern_injection.
_PLAN_PER_CALL: Any = object()

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
    "MARGenerator": 30,
    "KernelSynthGenerator": 31,
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
                timestamps=None,
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

        Overlay parameters:
        overlays (OverlayConfig): Calendar cycles, waveforms, changepoints,
            anomalies, observation effects and missing data drawn on every
            series, in that order (default: none). The flat missing_*,
            anomaly_*, changepoint_* and overlay_scale arguments still work
            but are deprecated.
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

    # Overlays drawn on every series
    overlays: OverlayConfig = Field(
        default_factory=OverlayConfig,
        description=(
            "Calendar cycles, waveforms, changepoints, anomalies, observation "
            "effects and missing data, applied in that order"
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _absorb_legacy_overlays(cls, data: Any) -> Any:
        """Move deprecated flat overlay arguments into ``overlays``."""
        return absorb_legacy_kwargs(data, stacklevel=4)

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

        calendar = self.overlays.calendar
        if calendar is not None:
            unsupported = [
                cycle.phase
                for cycle in calendar.cycles
                if not cycle_supported(cycle, self.freq, self._start_datetime)
            ]
            if unsupported:
                message = (
                    f"calendar cycles {unsupported} cannot show at freq "
                    f"{self.freq!r} (a cycle needs two steps per period) and are skipped"
                )
                if calendar.on_unsupported == "raise":
                    raise ValueError(message)
                if calendar.on_unsupported == "warn":
                    warnings.warn(message, UserWarning, stacklevel=3)

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

    def _overlay_plan_dict(self, length: int) -> dict | None:
        """Overlay plan for the native executor covering series up to ``length`` steps, or None with every stage off."""
        config = self.overlays
        if not config.active:
            return None
        plan: dict[str, Any] = {"base_weight": config.base_weight}
        cp = config.changepoints
        # Zero breaks draw nothing, as before.
        if cp is not None and cp.count > 0:
            plan["changepoints"] = {
                "count": cp.count,
                "kind": cp.kind,
                "locations": cp.locations or [],
                "level_changes": cp.level_changes or [],
                "trend_changes": cp.trend_changes or [],
                "variance_changes": cp.variance_changes or [],
                "relative": cp.scale == "relative",
            }
        an = config.anomalies
        if an is not None:
            plan["anomalies"] = {
                "types": list(an.types),
                "fraction": an.fraction,
                "spike_magnitude": an.spike_magnitude,
                "dip_magnitude": an.dip_magnitude,
                "level_shift_magnitude": an.level_shift_magnitude,
                "level_shift_duration": an.level_shift_duration,
                "relative": an.scale == "relative",
            }
        miss = config.missing
        if miss is not None:
            plan["missing"] = {
                "pattern": miss.pattern,
                "rate": miss.rate,
                "block_size": miss.block_size,
                "seasonal_period": miss.seasonal_period,
            }
        calendar = config.calendar
        if calendar is not None:
            cycles = [
                cycle
                for cycle in calendar.cycles
                if cycle_supported(cycle, self.freq, self.start_datetime)
            ]
            if cycles:
                timestamps = (
                    None
                    if all(cycle.phase == "step" for cycle in cycles)
                    else self._timestamps(length)
                )
                plan["calendar"] = {
                    "cycles": [
                        {
                            "profile": cycle.profile,
                            "harmonics": cycle.harmonics,
                            "bins": cycle.n_bins,
                            "values": cycle.values,
                            "jitter": cycle.jitter,
                            "weight": cycle.weight,
                            "phase": cycle_phase(cycle, timestamps, length),
                        }
                        for cycle in cycles
                    ],
                    "mode": calendar.mode,
                    "amplitude": calendar.amplitude,
                    "swing": calendar.swing,
                    "relative": calendar.scale == "relative",
                    "fallback_additive": calendar.fallback == "additive",
                }
        wave = config.waveform
        if wave is not None:
            plan["waveform"] = {
                **wave.model_dump(exclude={"scale", "fallback"}),
                "relative": wave.scale == "relative",
                "fallback_additive": wave.fallback == "additive",
            }
        if config.observation is not None:
            plan["observation"] = config.observation.model_dump()
        return plan

    def _overlay_plan(self, length: int) -> Any:
        """Parsed native overlay plan for series up to ``length`` steps, or None with every stage off."""
        plan = self._overlay_plan_dict(length)
        return None if plan is None else _rs_ov.OverlayPlan(plan)

    def _draw_pi_seeds(self, n_series: int) -> np.ndarray:
        """Draw the three overlay seeds per series, as in the Rust batch path."""
        return self.rng.integers(0, 2**63, size=(n_series, 3), dtype=np.uint64)

    def _apply_pattern_injection(
        self,
        values: np.ndarray,
        pi_seeds: Sequence[int] | np.ndarray,
        plan: Any = _PLAN_PER_CALL,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Run the overlay pipeline on one series outside the batch kernels.

        The released stages draw from NumPy redraws of the seeds, as this path
        always has; the newer stages derive their streams from the seeds
        themselves, as on the batch path, so both paths agree on them.

        Args:
            values: Array of time series values
            pi_seeds: One row of :meth:`_draw_pi_seeds`.
            plan: Result of :meth:`_overlay_plan`, to build it once for many
                series; built here by default.

        Returns:
            Tuple of (values, changepoint_indices, anomaly_indices, missing_indices). Index arrays are None when the corresponding flag is not requested.
        """
        if plan is _PLAN_PER_CALL:
            plan = self._overlay_plan(len(values))
        if plan is None:
            return values, None, None, None
        raw = tuple(int(s) for s in pi_seeds)
        legacy = tuple(int(np.random.default_rng(s).integers(0, 2**63)) for s in raw)
        values, cp_idx, anom_idx, miss_idx = _rs_ov.apply_overlays(
            np.asarray(values, dtype=np.float64), plan, raw, legacy
        )
        flags = self.exogenous
        return (
            values,
            cp_idx if flags is not None and flags.changepoint_flags else None,
            anom_idx if flags is not None and flags.anomaly_flags else None,
            miss_idx if flags is not None and flags.missing_flags else None,
        )

    def _build_dataframe(
        self,
        series_metadata: list[SeriesMetadata],
    ) -> IntoDataFrameT:
        """Build the output DataFrame from accumulated series metadata.

        Handles flattening values/timestamps/IDs, adding exogenous columns
        (pattern flags, datetime features, correlated variables), and merging
        any generator-specific extra columns.

        Args:
            series_metadata: List of SeriesMetadata for all generated series;
                series without timestamps get the generator's calendar.
        Returns:
            DataFrame in the configured engine.
        """
        # Flatten core columns
        flat_values = np.concatenate([m.values for m in series_metadata])
        flat_timestamps = np.concatenate(
            [
                self._timestamps(m.length) if m.timestamps is None else m.timestamps
                for m in series_metadata
            ]
        )
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
        gen_seed: int,
        pi_seeds: Sequence[int] | np.ndarray,
        plan: Any = _PLAN_PER_CALL,
    ) -> SeriesMetadata:
        """Generate a single series with timestamps and pattern injection.

        Args:
            length: Length of the series to generate
            series_id: ID for this series
            gen_seed: Seed for the series' generation stream.
            pi_seeds: One row of :meth:`_draw_pi_seeds`.
            plan: Overlay plan shared by the batch (see :meth:`_overlay_plan`).

        Returns:
            SeriesMetadata for the generated series, without timestamps.
        """
        local_copy = self.model_copy()
        local_copy._rng = np.random.default_rng(gen_seed)
        values = local_copy.generate_single_series(length=length)

        values, cp_indices, anom_indices, miss_indices = self._apply_pattern_injection(
            values, pi_seeds, plan
        )

        return SeriesMetadata(
            values=values,
            timestamps=None,
            series_id=series_id,
            length=length,
            anomaly_indices=anom_indices,
            changepoint_indices=cp_indices,
            missing_indices=miss_indices,
        )

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
        return self._build_dataframe(
            self._generate_metadata(n_series, start_id, n_jobs)
        )

    def generate_values(self, n_series: int, n_jobs: int = -1) -> list[np.ndarray]:
        """Generate series as value arrays, without building a DataFrame.

        Draws exactly the series :meth:`generate` would, with the same overlays
        (missing points as NaN), and leaves the generator in the same state.
        No DataFrame, exogenous column, or calendar is built, so lengths beyond
        pandas' datetime range work. Generators with their own generation
        scheme override :meth:`_generate_metadata`, which both methods share.

        Args:
            n_series (int): Number of time series to generate.
            n_jobs (int): Number of parallel workers. -1 (default) uses
                ``RAYON_NUM_THREADS`` if set, otherwise all logical cores.
                Results are seed-deterministic and do not depend on n_jobs.

        Returns:
            One 1-D float array per series in id order (``n_series * n_sensors``
            for ``IoTSensorGenerator``).
        """
        if n_series < 1:
            raise ValueError(f"n_series must be a positive integer, got {n_series}.")
        metadata = self._generate_metadata(n_series, 0, n_jobs)
        # Correlated exogenous draws still advance self.rng, as in generate().
        if self.exogenous is not None and self.exogenous.correlated:
            flat_values = np.concatenate([m.values for m in metadata])
            for exog_config in self.exogenous.correlated:
                generate_correlated_exog(metadata, flat_values, exog_config, self.rng)
        return [np.asarray(m.values, dtype=np.float64) for m in metadata]

    def _generate_metadata(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Draw the series of :meth:`generate` as metadata, without timestamps.

        Both :meth:`generate` and :meth:`generate_values` go through this hook,
        so a generator with its own generation scheme overrides this method
        rather than :meth:`generate`. Timestamps are left as None and filled in
        by :meth:`_build_dataframe`.
        """
        if n_jobs == -1:
            n_jobs = _default_n_workers
        return self._generate_parallel(n_series, start_id, n_jobs)

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
        pi_seeds = self._draw_pi_seeds(n_series).ravel()

        plan = self._overlay_plan(int(lengths.max())) or _NO_OVERLAYS

        results = _rs_batch.generate_batch(
            gen_type,
            scalar_params,
            array_params,
            lengths,
            gen_seeds,
            pi_seeds,
            plan,
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
        pi_seeds = self._draw_pi_seeds(n_series)
        plan = self._overlay_plan(max(lengths))

        workers = max(1, min(n_series, n_jobs))

        def gen_one(i: int) -> SeriesMetadata:
            return self._generate_one_series(
                length=lengths[i],
                series_id=start_id + i,
                gen_seed=gen_seeds[i],
                pi_seeds=pi_seeds[i],
                plan=plan,
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


def _legacy_property(name: str) -> property:
    def getter(self: BaseGenerator) -> Any:
        warnings.warn(
            f"{name} is deprecated; read it from the generator's overlays",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_value(self.overlays, name)

    return property(
        getter,
        doc=f"Deprecated: the released ``{name}`` setting, read from ``overlays``.",
    )


for _name in LEGACY_DEFAULTS:
    setattr(BaseGenerator, _name, _legacy_property(_name))
