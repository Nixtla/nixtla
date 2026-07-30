"""Clickstream (Web Analytics) time series generator."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import domain as _rs_dom
from synforecast.base import BaseGenerator

# Typical web traffic profile by hour of day (0-23): low at night, ramps up
# in the morning, evening peak around 8 PM. Normalized to mean 1.
_HOURLY_PATTERN = np.array(
    [
        0.3, 0.2, 0.15, 0.1, 0.1, 0.15,  # 0-5 AM
        0.3, 0.5, 0.7, 0.9, 1.0, 1.1,  # 6-11 AM
        1.0, 0.95, 0.9, 0.85, 0.9, 1.0,  # 12-5 PM
        1.1, 1.2, 1.3, 1.2, 0.9, 0.5,  # 6-11 PM
    ]
)  # fmt: skip
_HOURLY_PATTERN = _HOURLY_PATTERN / _HOURLY_PATTERN.mean()

# B2C day-of-week profile (Mon-Sun): higher on weekends. Normalized to mean 1.
_DAILY_PATTERN = np.array([0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1])
_DAILY_PATTERN = _DAILY_PATTERN / _DAILY_PATTERN.mean()

# Traffic source characteristics: multipliers on conversion rate, bounce
# rate, session depth, and seasonality amplitude
_SOURCE_CONFIGS = {
    "organic": {  # higher intent
        "conversion_mult": 1.2,
        "bounce_mult": 0.8,
        "depth_mult": 1.3,
        "seasonality_amp": 1.0,
    },
    "paid": {  # targeted traffic, more consistent
        "conversion_mult": 1.5,
        "bounce_mult": 0.7,
        "depth_mult": 1.1,
        "seasonality_amp": 0.6,
    },
    "direct": {  # returning users
        "conversion_mult": 1.8,
        "bounce_mult": 0.5,
        "depth_mult": 1.5,
        "seasonality_amp": 0.8,
    },
    "referral": {  # browsing
        "conversion_mult": 0.8,
        "bounce_mult": 1.2,
        "depth_mult": 0.9,
        "seasonality_amp": 1.2,
    },
    "mixed": {
        "conversion_mult": 1.0,
        "bounce_mult": 1.0,
        "depth_mult": 1.0,
        "seasonality_amp": 1.0,
    },
}

_OUTPUT_TYPE_IDS = {"sessions": 0, "pageviews": 1, "conversions": 2, "bounce_rate": 3}


class ClickstreamGenerator(BaseGenerator):
    """Generate web clickstream/session time series for analytics applications.

    Human sessions per time bin are Poisson-distributed around
    ``base_sessions`` modulated by hour-of-day/day-of-week seasonality and a
    slow log-random-walk trend. Bot traffic (flatter profile plus occasional
    crawl spikes) can be added on top. Bounces, pageviews (geometric page
    depth for engaged sessions) and conversions are derived per bin from the
    human sessions, with multipliers depending on ``traffic_source``.

    Output types:
    - 'sessions': total session counts (human + bot) per time bin
    - 'pageviews': total pageviews per time bin
    - 'conversions': conversion counts per time bin
    - 'bounce_rate': bounced fraction of total sessions per time bin

    Note:
        Seasonality assumes hourly frequency (freq='h'). Other frequencies
        produce incorrect day/night and weekday patterns.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data; use 'h' for correct
            seasonality patterns.
        base_sessions (float): Baseline sessions per time bin (default: 100).
        traffic_source (str): 'organic', 'paid', 'direct', 'referral' or
            'mixed' (default: 'mixed').
        conversion_rate (float): Base conversion rate for engaged sessions
            (default: 0.03).
        bounce_rate (float): Base rate of single-page sessions (default: 0.40).
        avg_session_depth (float): Average pages per engaged session
            (default: 3.5).
        include_seasonality (bool): Include time-of-day and day-of-week
            patterns (default: True).
        include_bots (bool): Include bot traffic (default: True).
        bot_fraction (float): Fraction of traffic from bots, < 1.0
            (default: 0.15).
        output_type (str): Metric to output (default: 'sessions').
        seed (int | None): Random seed for reproducibility (default: None).

    Example:
        >>> gen = ClickstreamGenerator(
        ...     min_length=168,  # 1 week of hourly data
        ...     max_length=168,
        ...     freq="h",
        ...     base_sessions=500,
        ...     output_type="sessions",
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    base_sessions: float = Field(
        default=100.0, gt=0.0, description="Baseline sessions per time unit"
    )

    traffic_source: Literal["organic", "paid", "direct", "referral", "mixed"] = Field(
        default="mixed", description="Traffic source type or mix"
    )

    conversion_rate: float = Field(
        default=0.03, ge=0.0, le=1.0, description="Base conversion rate"
    )
    bounce_rate: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Base bounce rate (single-page sessions)",
    )
    avg_session_depth: float = Field(
        default=3.5, gt=0.0, description="Average pages per non-bounced session"
    )

    include_seasonality: bool = Field(
        default=True, description="Include time-of-day and day-of-week patterns"
    )

    include_bots: bool = Field(default=True, description="Include bot traffic")
    bot_fraction: float = Field(
        default=0.15,
        ge=0.0,
        lt=1.0,  # < 1.0 to avoid division by zero in bot traffic calculation
        description="Fraction of traffic from bots",
    )

    output_type: Literal["sessions", "pageviews", "conversions", "bounce_rate"] = Field(
        default="sessions", description="Metric to output"
    )

    _source_params: dict = {}

    @model_validator(mode="after")
    def setup_source_parameters(self) -> "ClickstreamGenerator":
        """Initialize traffic source specific parameters."""
        object.__setattr__(self, "_source_params", _SOURCE_CONFIGS[self.traffic_source])
        return self

    def _generate_seasonality(self, length: int) -> np.ndarray:
        """Combined hour-of-day and day-of-week multipliers (assumes hourly).

        The source-specific amplitude scales how far the multipliers deviate
        from 1.
        """
        if not self.include_seasonality:
            return np.ones(length)

        t = np.arange(length)
        combined = _HOURLY_PATTERN[t % 24] * _DAILY_PATTERN[(t // 24) % 7]
        amp = self._source_params["seasonality_amp"]
        return 1 + amp * (combined - 1)

    def _generate_trend(self, length: int) -> np.ndarray:
        """Slow multiplicative trend: random walk in log space.

        The drift is drawn once per series from U(-0.0005, 0.001) (slight
        upward bias), so realized trend levels vary widely between series.
        """
        drift = self.rng.uniform(-0.0005, 0.001)
        noise = self.rng.normal(0, 0.002, length)
        log_trend = np.cumsum(drift + noise)
        return np.exp(log_trend - log_trend[0])  # Start at 1

    def _generate_bot_traffic(
        self, length: int, base_sessions: np.ndarray
    ) -> np.ndarray:
        """Bot sessions: flatter than human traffic, with occasional crawl spikes.

        Scaled so bots make up ~bot_fraction of total traffic.
        """
        if not self.include_bots:
            return np.zeros(length)

        bot_baseline = base_sessions * self.bot_fraction / (1 - self.bot_fraction)
        bot_sessions = bot_baseline * (0.8 + 0.4 * self.rng.uniform(size=length))

        n_spikes = self.rng.integers(0, max(1, length // 100) + 1)
        for _ in range(n_spikes):
            spike_time = self.rng.integers(0, length)
            spike_duration = self.rng.integers(1, 6)
            spike_magnitude = self.rng.uniform(2, 10)
            end = min(spike_time + spike_duration, length)
            bot_sessions[spike_time:end] *= spike_magnitude

        return bot_sessions

    def _simulate_sessions(
        self, length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate complete session data.

        Returns:
            (sessions, pageviews, conversions, bounces) arrays.
        """
        seasonality = self._generate_seasonality(length)
        trend = self._generate_trend(length)

        # Sessions are count data: Poisson around the modulated baseline
        human_sessions = self.rng.poisson(
            np.maximum(self.base_sessions * seasonality * trend, 0.1)
        )
        bot_sessions = self._generate_bot_traffic(length, human_sessions.astype(float))
        total_sessions = human_sessions + bot_sessions.astype(int)

        source_params = self._source_params

        # Only human sessions bounce meaningfully
        bounce_rate_actual = np.clip(
            self.bounce_rate * source_params["bounce_mult"], 0, 1
        )
        bounces = self.rng.binomial(human_sessions, bounce_rate_actual)
        engaged_sessions = human_sessions - bounces

        # Pageviews: geometric page depth per engaged session (mean = depth)
        depth = self.avg_session_depth * source_params["depth_mult"]
        pageviews_engaged = np.zeros(length)
        total_engaged = int(engaged_sessions.sum())
        if total_engaged > 0:
            depths = self.rng.geometric(1 / depth, total_engaged)
            splits = np.cumsum(engaged_sessions)[:-1]
            pageviews_engaged = np.array(
                [seg.sum() for seg in np.split(depths, splits)], dtype=float
            )
        # Bounces contribute 1 page each; bots average 5 pages
        pageviews = bounces + pageviews_engaged + bot_sessions * 5

        conv_rate = np.clip(
            self.conversion_rate * source_params["conversion_mult"], 0, 1
        )
        conversions = self.rng.binomial(engaged_sessions, conv_rate)

        return (
            total_sessions.astype(float),
            pageviews,
            conversions.astype(float),
            bounces.astype(float),
        )

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        sp = self._source_params
        return (
            np.array(
                [
                    float(self.base_sessions),
                    sp["conversion_mult"],
                    sp["bounce_mult"],
                    sp["depth_mult"],
                    sp["seasonality_amp"],
                    self.conversion_rate,
                    self.bounce_rate,
                    self.avg_session_depth,
                    1.0 if self.include_seasonality else 0.0,
                    1.0 if self.include_bots else 0.0,
                    self.bot_fraction,
                    float(_OUTPUT_TYPE_IDS[self.output_type]),
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single clickstream time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of metric values.
        """
        seed = int(self.rng.integers(0, 2**63))
        sp = self._source_params
        return _rs_dom.clickstream(
            length,
            self.base_sessions,
            sp["conversion_mult"],
            sp["bounce_mult"],
            sp["depth_mult"],
            sp["seasonality_amp"],
            self.conversion_rate,
            self.bounce_rate,
            self.avg_session_depth,
            self.include_seasonality,
            self.include_bots,
            self.bot_fraction,
            _OUTPUT_TYPE_IDS[self.output_type],
            seed,
        )

    def generate_full_metrics(
        self, n_series: int = 1, start_id: int = 0
    ) -> dict[str, np.ndarray]:
        """Generate all clickstream metrics for complete analytics.

        Args:
            n_series (int): Number of series to generate.
            start_id (int): Starting ID for series naming.

        Returns:
            All metrics as flat arrays, keyed by metric name plus 'series_id'.
        """
        all_data: dict[str, list] = {
            "series_id": [],
            "sessions": [],
            "pageviews": [],
            "conversions": [],
            "bounces": [],
            "bounce_rate": [],
            "conversion_rate": [],
            "pages_per_session": [],
        }

        for i in range(n_series):
            length = self.rng.integers(self.min_length, self.max_length + 1)
            sessions, pageviews, conversions, bounces = self._simulate_sessions(length)

            all_data["series_id"].extend([start_id + i] * length)
            all_data["sessions"].extend(sessions)
            all_data["pageviews"].extend(pageviews)
            all_data["conversions"].extend(conversions)
            all_data["bounces"].extend(bounces)
            all_data["bounce_rate"].extend(
                np.where(sessions > 0, bounces / sessions, 0.0)
            )
            all_data["conversion_rate"].extend(
                np.where(sessions > 0, conversions / sessions, 0.0)
            )
            all_data["pages_per_session"].extend(
                np.where(sessions > 0, pageviews / sessions, 0.0)
            )

        return {k: np.array(v) for k, v in all_data.items()}

    def generate_funnel(
        self, n_sessions: int = 1000, stages: list[str] | None = None
    ) -> dict[str, int]:
        """Generate a conversion funnel with stage-by-stage drop-off.

        Retention between stages rises from ~0.4 to ~0.7 (committed users
        drop off less), adjusted by the traffic source's conversion
        multiplier.

        Args:
            n_sessions (int): Number of sessions entering the funnel.
            stages (list[str] | None): Funnel stage names (default: standard
                e-commerce funnel).

        Returns:
            Stage name -> session count at that stage.
        """
        if stages is None:
            stages = [
                "visit",
                "product_view",
                "add_to_cart",
                "checkout_start",
                "checkout_complete",
            ]

        base_retention = np.linspace(0.4, 0.7, len(stages) - 1)
        base_retention = np.clip(
            base_retention * self._source_params["conversion_mult"], 0.1, 0.95
        )

        funnel = {stages[0]: n_sessions}
        current = n_sessions

        for i, stage in enumerate(stages[1:]):
            retention = np.clip(base_retention[i] + self.rng.normal(0, 0.05), 0.1, 0.95)
            current = self.rng.binomial(current, retention)
            funnel[stage] = current

        return funnel

    def get_model_info(self) -> dict:
        """Get information about the clickstream model.

        Returns:
            Model parameters and traffic characteristics.
        """
        return {
            "base_sessions": self.base_sessions,
            "traffic_source": self.traffic_source,
            "conversion_rate": self.conversion_rate,
            "bounce_rate": self.bounce_rate,
            "avg_session_depth": self.avg_session_depth,
            "include_seasonality": self.include_seasonality,
            "include_bots": self.include_bots,
            "bot_fraction": self.bot_fraction,
            "source_params": self._source_params,
            "output_type": self.output_type,
        }
