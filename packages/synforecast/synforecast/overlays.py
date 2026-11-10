"""Overlay specifications: structure drawn on top of every generated series.

A generator's ``overlays`` holds one optional spec per stage. Stages run on
each series in a fixed order, inside the native kernel on every generation
path:

    base weight -> calendar -> waveform -> changepoints -> anomalies
    -> observation -> missing

Relative magnitudes are in standard deviations of the untouched base series.
Each stage draws from its own stream of the series' seeds, so switching one
stage on never moves another, and a config with every stage off draws exactly
what the generator alone draws.
"""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Scale = Literal["absolute", "relative"]
AnomalyType = Literal["spike", "dip", "level_shift"]
WaveShape = Literal["pulse", "spike", "sawtooth", "trapezoid", "levels", "staircase"]
WaveTiming = Literal["periodic", "jitter", "renewal"]

_PHASE_PERIODS = {
    "day": pd.Timedelta(days=1),
    "week": pd.Timedelta(days=7),
    "year": pd.Timedelta(days=365.2425),
}
_DEFAULT_BINS = {"day": 24, "week": 7, "year": 12}


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Changepoints(_Spec):
    """Structural breaks at random or given positions.

    Args:
        count: Number of breaks.
        kind: 'level', 'trend', 'variance' or 'mixed'.
        locations: Relative positions in [0, 1]; drawn from U(0.1, 0.9) where missing.
        level_changes: Level jumps; drawn where missing, from U(-20, 20), or
            U(-3, 3) standard deviations with ``scale='relative'``.
        trend_changes: Slope changes; drawn where missing, from U(-0.5, 0.5),
            or as a total drift of U(-3, 3) standard deviations over the rest
            of the series with ``scale='relative'``.
        variance_changes: Positive factors on the deviation from a rolling
            mean; drawn from U(0.5, 2) where missing.
        scale: 'absolute' adds the changes as given; 'relative' measures them
            in standard deviations of the base series.
    """

    count: int = Field(default=2, ge=0)
    kind: Literal["level", "trend", "variance", "mixed"] = "level"
    locations: list[float] | None = None
    level_changes: list[float] | None = None
    trend_changes: list[float] | None = None
    variance_changes: list[float] | None = None
    scale: Scale = "absolute"

    @model_validator(mode="after")
    def _check_values(self) -> "Changepoints":
        if self.locations is not None and any(
            not 0.0 <= x <= 1.0 for x in self.locations
        ):
            raise ValueError("changepoint locations must contain values in [0, 1]")
        if self.variance_changes is not None and any(
            x <= 0 for x in self.variance_changes
        ):
            raise ValueError("changepoint variance changes must be positive")
        return self


class Anomalies(_Spec):
    """Spikes, dips and temporary level shifts at random positions.

    Args:
        types: Non-empty subset of 'spike', 'dip' and 'level_shift'.
        fraction: Share of steps that start an anomaly.
        spike_magnitude: Size of a spike.
        dip_magnitude: Size of a dip (negative for a drop).
        level_shift_magnitude: Size of a level shift.
        level_shift_duration: Steps a level shift lasts.
        scale: 'absolute' adds the magnitudes as given; 'relative' measures
            them in standard deviations of the base series.
    """

    types: list[AnomalyType] = Field(
        default_factory=lambda: list[AnomalyType](["spike", "dip"]), min_length=1
    )
    fraction: float = Field(default=0.05, ge=0, le=1)
    spike_magnitude: float = 10.0
    dip_magnitude: float = -10.0
    level_shift_magnitude: float = 20.0
    level_shift_duration: int = Field(default=10, ge=1)
    scale: Scale = "absolute"


class Missing(_Spec):
    """Missing values, as NaN.

    Args:
        pattern: 'random' points, 'block' runs, or 'seasonal' (more likely in
            part of each period).
        rate: Share of missing steps.
        block_size: Run length for 'block'.
        seasonal_period: Period for 'seasonal'.
    """

    pattern: Literal["random", "block", "seasonal"] = "random"
    rate: float = Field(default=0.1, ge=0, le=1)
    block_size: int = Field(default=3, ge=1)
    seasonal_period: int = Field(default=7, ge=1)


class CalendarCycle(_Spec):
    """One periodic profile aligned to the timestamps (or to the step index).

    Args:
        phase: What the cycle repeats over: 'day' (time of day), 'week' (day
            and time of week), 'year' (month and time of year, so monthly
            data lines up with calendar months) or 'step' (every ``period``
            steps, independent of freq; works with integer frequencies).
        profile: 'fourier' for a smooth shape of ``harmonics`` harmonics, or
            'levels' for a constant level per bin (hour of day, weekday,
            month, or step within the period).
        harmonics: Harmonics of a Fourier profile.
        bins: Bins of a levels profile; defaults to 24 (day), 7 (week),
            12 (year) or ``period`` (step).
        period: Cycle length in steps, for ``phase='step'`` only.
        values: A fixed profile instead of a random one: ``2 * harmonics``
            Fourier coefficients (cosines, then sines) or one value per bin.
            Random profiles draw coefficients N(0, 1/k) or levels N(0, 1).
        jitter: Standard deviation of per-series normal noise added to fixed
            ``values``, so series share a shape without being identical.
        weight: Weight of this cycle in the calendar's weighted mean.

    Each cycle is normalized to peak magnitude 1 before weighting.
    """

    phase: Literal["day", "week", "year", "step"]
    profile: Literal["fourier", "levels"] = "fourier"
    harmonics: int = Field(default=3, ge=1, le=12)
    bins: int | None = Field(default=None, ge=2, le=10_000)
    period: int | None = Field(default=None, ge=2)
    values: list[float] | None = None
    jitter: float = Field(default=0.0, ge=0)
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_shape(self) -> "CalendarCycle":
        if (self.phase == "step") != (self.period is not None):
            raise ValueError(
                "period is required for phase='step' and only allowed there"
            )
        if self.values is not None and len(self.values) != self.n_values:
            raise ValueError(
                f"values must have {self.n_values} entries for this profile, "
                f"got {len(self.values)}"
            )
        return self

    @property
    def n_bins(self) -> int:
        """Bins of a levels profile."""
        if self.bins is not None:
            return self.bins
        if self.period is not None:
            return self.period
        return _DEFAULT_BINS[self.phase]

    @property
    def n_values(self) -> int:
        """Length of a fixed profile."""
        return 2 * self.harmonics if self.profile == "fourier" else self.n_bins


class Calendar(_Spec):
    """Periodic profiles aligned to the series' timestamps.

    Args:
        cycles: One or more cycles, combined as their weighted mean.
        mode: 'additive' adds ``amplitude`` times the profile; 'multiplicative'
            scales a non-negative series by ``1 + swing * profile``.
        amplitude: Additive peak, in standard deviations of the base series
            with ``scale='relative'`` (the default), else in series units.
        swing: Multiplicative peak, as a fraction of the level, below 1.
        scale: Units of ``amplitude``.
        fallback: What a multiplicative calendar does on a series with
            negative values: 'additive' adds the additive profile instead,
            'none' leaves the series unchanged.
        on_unsupported: What to do at construction when a cycle cannot show
            at the generator's frequency (a daily cycle on daily data, any
            date cycle on an integer freq): 'warn' or 'skip' drops it,
            'raise' fails.

    Phases follow the timestamps the generator emits (UTC for an offset-aware
    start). A cycle needs at least two steps per period.
    """

    cycles: list[CalendarCycle] = Field(min_length=1)
    mode: Literal["additive", "multiplicative"] = "additive"
    amplitude: float = Field(default=1.0, ge=0)
    swing: float = Field(default=0.3, ge=0, lt=1)
    scale: Scale = "relative"
    fallback: Literal["additive", "none"] = "additive"
    on_unsupported: Literal["warn", "skip", "raise"] = "warn"


class Waveform(_Spec):
    """A sharp-edged repeating template whose period is in steps.

    Args:
        shape: 'pulse' (a rectangle of duty ``duty``), 'spike' (a pulse
            ``width`` steps wide), 'sawtooth' (ramp and reset each cycle),
            'trapezoid' (a pulse with linear ramps), 'levels' (a jump to a
            random other level each cycle) or 'staircase' (a one-level step in
            a persistent direction each cycle).
        period: Cycle length in steps; mean dwell for 'levels' and 'staircase'.
        duty: On-fraction of a cycle for 'pulse' and 'trapezoid'.
        width: Spike width in steps.
        ramp: Trapezoid ramp per side, as a fraction of the on-time.
        levels: Number of levels for 'levels' and 'staircase'.
        timing: 'periodic' repeats every ``period`` steps from a random phase;
            'jitter' moves each cycle start by a normal offset of standard
            deviation ``jitter * period``; 'renewal' draws on and off
            durations (whole cycles for sawtooth and level shapes) from
            ``renewal`` with the nominal means.
        jitter: Cycle-start jitter, as a fraction of the period.
        renewal: 'gamma' or 'lognormal' renewal durations.
        cv: Coefficient of variation of renewal durations.
        direction: 'peak' rises, 'dip' falls.
        mode: 'additive' adds ``amplitude`` times the template;
            'multiplicative' scales a non-negative series by
            ``1 +/- swing * template``.
        amplitude: Additive peak, in standard deviations of the base series
            with ``scale='relative'`` (the default).
        swing: Multiplicative peak, as a fraction of the level; at most 1 for dips.
        scale: Units of ``amplitude``.
        fallback: Multiplicative behaviour on a series with negative values,
            as for :class:`Calendar`.
    """

    shape: WaveShape = "pulse"
    period: int = Field(default=96, ge=2, le=10_000_000)
    duty: float = Field(default=0.1, gt=0, lt=1)
    width: int = Field(default=1, ge=1)
    ramp: float = Field(default=0.25, gt=0, le=0.5)
    levels: int = Field(default=4, ge=2, le=1000)
    timing: WaveTiming = "periodic"
    jitter: float = Field(default=0.05, ge=0, le=0.5)
    renewal: Literal["gamma", "lognormal"] = "gamma"
    cv: float = Field(default=0.5, gt=0, le=10)
    direction: Literal["peak", "dip"] = "peak"
    mode: Literal["additive", "multiplicative"] = "additive"
    amplitude: float = Field(default=3.0, ge=0)
    swing: float = Field(default=0.5, ge=0)
    scale: Scale = "relative"
    fallback: Literal["additive", "none"] = "additive"

    @model_validator(mode="after")
    def _check_swing(self) -> "Waveform":
        if self.mode == "multiplicative" and self.direction == "dip" and self.swing > 1:
            raise ValueError("waveform swing must be <= 1 for multiplicative dips")
        return self

    @property
    def on_steps(self) -> int:
        """Nominal on-time of one cycle in steps; the whole cycle for sawtooth and level shapes."""
        if self.shape == "spike":
            return min(self.width, self.period - 1)
        if self.shape in ("pulse", "trapezoid"):
            return int(np.clip(round(self.duty * self.period), 1, self.period - 1))
        return self.period


class Observation(_Spec):
    """An observation process between the signal and the recorded readings.

    Args:
        floor: Saturation floor, as a fraction of the way from the series'
            minimum (0) to its maximum (1).
        cap: Saturation cap, on the same scale; above ``floor``.
        quantum: Value-grid step, in standard deviations of the series
            entering this stage; 0 disables quantization.
        hold: Sample-and-hold block in steps (each block repeats its first
            reading); 1 disables it.
        stuck_fraction: Share of steps inside windows where the reading freezes.
        stuck_length: Mean length of a frozen window in steps.

    Missing values stay missing and never spread.
    """

    floor: float | None = Field(default=None, ge=0, lt=1)
    cap: float | None = Field(default=None, gt=0, le=1)
    quantum: float = Field(default=0.0, ge=0)
    hold: int = Field(default=1, ge=1, le=1_000_000)
    stuck_fraction: float = Field(default=0.0, ge=0, lt=1)
    stuck_length: int = Field(default=20, ge=2)

    @model_validator(mode="after")
    def _check_bounds(self) -> "Observation":
        if self.floor is not None and self.cap is not None and self.floor >= self.cap:
            raise ValueError("observation floor must be below cap")
        return self


class OverlayConfig(_Spec):
    """Every overlay of a generator; ``None`` switches a stage off.

    Args:
        base_weight: Weight of the base series around its mean before any
            overlay; small values let a calendar or waveform dominate, 0
            leaves them exact.
        calendar: Timestamp-aligned periodic profiles.
        waveform: A sharp-edged template with its period in steps.
        changepoints: Structural breaks.
        anomalies: Spikes, dips and level shifts.
        observation: Saturation, quantization, sample-and-hold, frozen readings.
        missing: Missing values.
    """

    base_weight: float = Field(default=1.0, ge=0, le=1)
    calendar: Calendar | None = None
    waveform: Waveform | None = None
    changepoints: Changepoints | None = None
    anomalies: Anomalies | None = None
    observation: Observation | None = None
    missing: Missing | None = None

    @property
    def active(self) -> bool:
        """Whether any stage changes the series."""
        return self.base_weight != 1.0 or any(
            getattr(self, stage) is not None for stage in _STAGES
        )

    def fill(self, other: "OverlayConfig") -> "OverlayConfig":
        """A copy with every stage that is off here taken from ``other``."""
        update = {
            stage: getattr(other, stage)
            for stage in _STAGES
            if getattr(self, stage) is None and getattr(other, stage) is not None
        }
        if self.base_weight == 1.0 and other.base_weight != 1.0:
            update["base_weight"] = other.base_weight
        return self.model_copy(update=update)


_STAGES = (
    "calendar",
    "waveform",
    "changepoints",
    "anomalies",
    "observation",
    "missing",
)


# ---------------------------------------------------------------------------
# Calendar phases
# ---------------------------------------------------------------------------


def approximate_step(freq: str | int, start: pd.Timestamp) -> pd.Timedelta | None:
    """Average duration of one step from ``start``; None for an integer freq.

    Works for fixed and calendar offsets alike ('h', 'B', 'W-MON', 'MS', 'QS').
    """
    if isinstance(freq, int):
        return None
    offset = pd.tseries.frequencies.to_offset(freq)
    return (start + 12 * offset - start) / 12


def cycle_supported(cycle: CalendarCycle, freq: str | int, start: pd.Timestamp) -> bool:
    """Whether a cycle shows at this frequency: at least two steps per period."""
    if cycle.phase == "step":
        return True
    step = approximate_step(freq, start)
    return step is not None and 2 * step <= _PHASE_PERIODS[cycle.phase]


def cycle_phase(
    cycle: CalendarCycle, timestamps: np.ndarray | None, length: int
) -> np.ndarray:
    """Phase in [0, 1) of each of ``length`` steps for one cycle."""
    if cycle.period is not None:
        return (np.arange(length) % cycle.period) / cycle.period
    if timestamps is None:
        raise ValueError(f"a {cycle.phase!r} cycle needs timestamps")
    ts = pd.DatetimeIndex(timestamps[:length])
    day = np.asarray((ts - ts.normalize()) / pd.Timedelta(days=1), dtype=np.float64)
    if cycle.phase == "day":
        phase = day
    elif cycle.phase == "week":
        phase = (np.asarray(ts.dayofweek) + day) / 7.0
    else:
        within_month = (np.asarray(ts.day) - 1 + day) / np.asarray(ts.days_in_month)
        phase = (np.asarray(ts.month) - 1 + within_month) / 12.0
    return np.minimum(phase, np.nextafter(1.0, 0.0))


# ---------------------------------------------------------------------------
# Released flat keyword arguments
# ---------------------------------------------------------------------------

LEGACY_DEFAULTS: dict[str, Any] = {
    "missing_data": False,
    "missing_pattern": "random",
    "missing_rate": 0.1,
    "missing_block_size": 3,
    "missing_seasonal_period": 7,
    "anomalies": False,
    "anomaly_fraction": 0.05,
    "anomaly_types": ["spike", "dip"],
    "spike_magnitude": 10.0,
    "dip_magnitude": -10.0,
    "level_shift_magnitude": 20.0,
    "level_shift_duration": 10,
    "changepoints": False,
    "num_changepoints": 2,
    "changepoint_type": "level",
    "changepoint_level_changes": None,
    "changepoint_trend_changes": None,
    "changepoint_variance_changes": None,
    "changepoint_locations": None,
    "overlay_scale": "absolute",
}

_CHANGEPOINT_KEYS = {
    "num_changepoints": "count",
    "changepoint_type": "kind",
    "changepoint_locations": "locations",
    "changepoint_level_changes": "level_changes",
    "changepoint_trend_changes": "trend_changes",
    "changepoint_variance_changes": "variance_changes",
}
_ANOMALY_KEYS = {
    "anomaly_types": "types",
    "anomaly_fraction": "fraction",
    "spike_magnitude": "spike_magnitude",
    "dip_magnitude": "dip_magnitude",
    "level_shift_magnitude": "level_shift_magnitude",
    "level_shift_duration": "level_shift_duration",
}
_MISSING_KEYS = {
    "missing_pattern": "pattern",
    "missing_rate": "rate",
    "missing_block_size": "block_size",
    "missing_seasonal_period": "seasonal_period",
}


def _legacy_spec(
    spec: type[_Spec], keys: dict[str, str], kwargs: dict[str, Any], **extra: Any
) -> Any:
    """Build one stage's spec from flat arguments, naming them in errors."""
    try:
        return spec(
            **extra, **{new: kwargs[old] for old, new in keys.items() if old in kwargs}
        )
    except ValidationError as error:
        names = {new: old for old, new in keys.items()} | {"scale": "overlay_scale"}
        messages = [
            f"{names.get(str(e['loc'][0]), e['loc'][0]) if e['loc'] else spec.__name__}: {e['msg']}"
            for e in error.errors()
        ]
        raise ValueError("; ".join(messages)) from None


def legacy_overlays(kwargs: dict[str, Any]) -> OverlayConfig:
    """The stages that released flat keyword arguments describe.

    Settings are validated as before even for a stage that is not switched
    on, which they leave unused.
    """
    scale = kwargs.get("overlay_scale", "absolute")
    if scale not in ("absolute", "relative"):
        raise ValueError(
            f"overlay_scale must be 'absolute' or 'relative', got {scale!r}"
        )
    stages: dict[str, Any] = {}
    for stage, flag, spec, keys, extra in (
        (
            "changepoints",
            "changepoints",
            Changepoints,
            _CHANGEPOINT_KEYS,
            {"scale": scale},
        ),
        ("anomalies", "anomalies", Anomalies, _ANOMALY_KEYS, {"scale": scale}),
        ("missing", "missing_data", Missing, _MISSING_KEYS, {}),
    ):
        if kwargs.get(flag) or any(key in kwargs for key in keys):
            built = _legacy_spec(spec, keys, kwargs, **extra)
            if kwargs.get(flag):
                stages[stage] = built
    return OverlayConfig(**stages)


def absorb_legacy_kwargs(data: Any, stacklevel: int) -> Any:
    """Move released flat overlay keyword arguments of a generator into ``overlays``.

    A stage set both ways is an error; otherwise the flat arguments fill the
    stages ``overlays`` leaves off.
    """
    if not isinstance(data, dict):
        return data
    legacy = {key: data[key] for key in LEGACY_DEFAULTS if key in data}
    if not legacy:
        return data
    warnings.warn(
        f"{', '.join(sorted(legacy))}: flat overlay arguments are deprecated; "
        "pass overlays=OverlayConfig(...) instead (see synforecast.overlays).",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    data = {key: value for key, value in data.items() if key not in legacy}
    from_legacy = legacy_overlays(legacy)
    given = data.get("overlays")
    if given is None:
        data["overlays"] = from_legacy
        return data
    given = OverlayConfig.model_validate(given)
    clash = [
        stage
        for stage in _STAGES
        if getattr(given, stage) is not None and getattr(from_legacy, stage) is not None
    ]
    if clash:
        raise ValueError(
            f"{', '.join(clash)} set both in overlays and as flat arguments"
        )
    data["overlays"] = given.fill(from_legacy)
    return data


def legacy_value(config: OverlayConfig, name: str) -> Any:
    """Value a released flat attribute had for this config."""
    cp, an, miss = config.changepoints, config.anomalies, config.missing
    if name == "changepoints":
        return cp is not None
    if name == "anomalies":
        return an is not None
    if name == "missing_data":
        return miss is not None
    if name == "overlay_scale":
        scaled = cp or an
        return scaled.scale if scaled is not None else "absolute"
    stages: tuple[tuple[dict[str, str], _Spec | None], ...] = (
        (_CHANGEPOINT_KEYS, cp),
        (_ANOMALY_KEYS, an),
        (_MISSING_KEYS, miss),
    )
    for keys, spec in stages:
        if name in keys:
            return LEGACY_DEFAULTS[name] if spec is None else getattr(spec, keys[name])
    raise AttributeError(name)
