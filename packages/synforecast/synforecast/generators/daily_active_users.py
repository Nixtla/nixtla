"""Daily Active Users (DAU) generator with event-driven jumps."""

from typing import Any, Literal

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

try:
    from synforecast._lib import domain as _rs_dom

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class DailyActiveUsersGenerator(BaseGenerator):
    """Generate Daily Active Users time series with event-driven jumps.

    The DAU level is ``base_users * (1 + growth_rate)^day`` with a weekend
    multiplier, plus a decaying boost from random events: with probability
    ``event_probability`` per step an event adds ``(impact - 1) * base`` to a
    boost that decays geometrically at rate ``event_decay_rate``. Proportional
    Gaussian noise is added and the result is clipped at zero.

    The day index is derived from the step position relative to the series
    start using the step size implied by ``freq`` (an integer ``freq`` is
    treated as daily). Weekends are day indices 5 and 6 of each 7-day block.

    Also outputs an exogenous column marking the steps where events occur,
    usable as a feature for forecasting models.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data, e.g. 'D', 'h'.
        base_users (float): Base number of daily active users
            (default: 10000.0).
        growth_rate (float): Daily organic growth rate (default: 0.0005).
        growth_rate_std (float): Std dev of a per-series perturbation of
            growth_rate (default: 0.0).
        app_type (str): 'consumer', 'business' or 'gaming'
            (default: 'consumer').
        weekly_pattern (bool): Enable weekly seasonality (default: True).
        weekend_factor (float | None): Multiplier for weekend activity
            (default: 1.2 for gaming, 0.8 otherwise).
        event_probability (float): Per-step probability of an event
            (default: 0.02).
        event_impact_min (float): Minimum event impact multiplier
            (default: 1.2).
        event_impact_max (float): Maximum event impact multiplier
            (default: 2.0).
        event_decay_rate (float): Per-step decay rate of the event boost
            (default: 0.1).
        noise_std (float): Std of noise relative to the current level
            (default: 0.05).
        event_col (str): Name of the event indicator column
            (default: 'event').
        seed (int | None): Random seed for reproducibility (default: None).
    """

    base_users: float = Field(
        default=10000.0, gt=0, description="Base number of daily active users"
    )
    growth_rate: float = Field(default=0.0005, description="Daily organic growth rate")
    growth_rate_std: float = Field(
        default=0.0,
        ge=0,
        description="Std dev for per-series growth rate perturbation",
    )
    app_type: Literal["consumer", "business", "gaming"] = Field(
        default="consumer", description="Type of application"
    )

    weekly_pattern: bool = Field(default=True, description="Enable weekly seasonality")
    weekend_factor: float | None = Field(
        default=None, gt=0, description="Multiplier for weekend activity"
    )

    event_probability: float = Field(
        default=0.02, ge=0, le=1, description="Daily probability of an event"
    )
    event_impact_min: float = Field(
        default=1.2, ge=1, description="Minimum event impact multiplier"
    )
    event_impact_max: float = Field(
        default=2.0, description="Maximum event impact multiplier"
    )
    event_decay_rate: float = Field(
        default=0.1, gt=0, le=1, description="Rate at which event impact decays"
    )

    noise_std: float = Field(default=0.05, description="Standard deviation of noise")

    event_col: str = Field(
        default="event", description="Name of the event indicator column"
    )

    # Events of the most recently generated series (excluded from serialization)
    current_events: Any = Field(default=None, init=False, exclude=True)

    @model_validator(mode="after")
    def validate_dau_params(self) -> "DailyActiveUsersGenerator":
        """Validate DAU parameters and set defaults."""
        if self.weekend_factor is None:
            default_weekend = 1.2 if self.app_type == "gaming" else 0.8
            object.__setattr__(self, "weekend_factor", default_weekend)

        if self.event_impact_max < self.event_impact_min:
            raise ValueError("event_impact_max must be >= event_impact_min")

        object.__setattr__(self, "current_events", np.array([]))
        return self

    def _get_day_index(self, t: int) -> int:
        """Day index at step ``t``, relative to the series start."""
        return int(t * self._step_hours()) // 24

    def _get_day_of_week(self, t: int) -> int:
        """Day of week (0-6) at step ``t``, relative to the series start."""
        return self._get_day_index(t) % 7

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single DAU time series.

        Also populates ``self.current_events`` with event indicators.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of DAU values.
        """
        if self.growth_rate_std > 0:
            series_growth_rate = float(
                self.rng.normal(self.growth_rate, self.growth_rate_std)
            )
        else:
            series_growth_rate = self.growth_rate

        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            dau, events = _rs_dom.daily_active_users(
                length,
                self.base_users,
                series_growth_rate,
                self.weekend_factor,
                self.weekly_pattern,
                self.event_probability,
                self.event_impact_min,
                self.event_impact_max,
                self.event_decay_rate,
                self.noise_std,
                self._step_hours(),
                seed,
            )
            object.__setattr__(self, "current_events", events.astype(np.int32))
            return dau

        dau = np.zeros(length)
        events = np.zeros(length, dtype=np.int32)
        event_boost = 0.0

        for t in range(length):
            base = self.base_users * (1 + series_growth_rate) ** self._get_day_index(t)

            if self.weekly_pattern and self._get_day_of_week(t) >= 5:
                base *= self.weekend_factor

            if self.rng.random() < self.event_probability:
                events[t] = 1
                impact = self.rng.uniform(self.event_impact_min, self.event_impact_max)
                event_boost += (impact - 1) * base

            dau[t] = base + event_boost
            event_boost *= 1 - self.event_decay_rate

            dau[t] += self.rng.normal(0, self.noise_std * dau[t])

        dau = np.maximum(dau, 0)
        self.current_events = events
        return dau

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,  # noqa: ARG002
    ) -> IntoDataFrameT:
        """Generate synthetic DAU time series data with event indicators.

        Args:
            n_series (int): Number of time series to generate.
            start_id (int): Starting ID for the series numbering (default: 0).
            n_jobs (int): Ignored; generation is sequential.

        Returns:
            DataFrame in long format with columns [id_col, time_col,
            target_col, event_col], where event_col is 1 at steps where an
            event occurred.
        """
        all_metadata: list[SeriesMetadata] = []

        for i in range(n_series):
            length = self.rng.integers(self.min_length, self.max_length + 1)
            timestamps = self._timestamps(length)

            values = self.generate_single_series(length=length)
            # Capture events before pattern injection modifies anything
            events = self.current_events.copy()

            values, cp_indices, anom_indices, miss_indices = (
                self._apply_pattern_injection(values)
            )

            all_metadata.append(
                SeriesMetadata(
                    values=values,
                    timestamps=timestamps,
                    series_id=start_id + i,
                    length=length,
                    anomaly_indices=anom_indices,
                    changepoint_indices=cp_indices,
                    missing_indices=miss_indices,
                    extra_columns={self.event_col: events},
                )
            )

        return self._build_dataframe(all_metadata)
