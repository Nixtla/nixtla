"""Intermittent Demand generator for sparse time series patterns."""

from typing import Literal

import numpy as np
from pydantic import Field

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import domain as _rs_dom

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class IntermittentDemandGenerator(BaseGenerator):
    """Generate intermittent demand time series with sparse patterns.

    Demand is a two-part process: a binary occurrence process decides at
    which periods demand happens, and a size distribution draws the demand
    quantity for those periods (all other periods are zero). Common in
    retail, spare parts, and inventory contexts (Croston-style demand).

    Occurrence patterns:
        - 'random': i.i.d. Bernoulli(demand_probability) per period, so the
          long-run fraction of non-zero periods equals demand_probability.
        - 'clustered': runs of `cluster_size` consecutive demand periods
          separated by Geometric(demand_probability) gaps. The overall
          demand fraction is cluster_size / (1/demand_probability +
          cluster_size), not demand_probability. demand_probability == 0
          means infinite gaps, i.e. an all-zero series.
        - 'seasonal': per-period Bernoulli with probability
          p(t) = demand_probability + (seasonal_peak_prob -
          demand_probability) * (cos(2*pi*(t mod P)/P) + 1) / 2,
          which peaks at seasonal_peak_prob at the start of each cycle
          (t mod P == 0) and falls to demand_probability mid-cycle.

    Size distributions are moment-matched to (demand_mean, demand_std):
        - 'poisson': Poisson(demand_mean); demand_std is ignored.
        - 'negative_binomial': p = mean/var, n = mean*p/(1-p); falls back
          to Poisson when demand_std**2 <= demand_mean.
        - 'lognormal': mu = ln(mean^2 / sqrt(var + mean^2)),
          sigma^2 = ln(1 + var/mean^2).
        - 'gamma': shape = (mean/std)^2, scale = var/mean.
    Sizes are clipped from below at min_demand.

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data (e.g. 'D', 'h', '5min') or int
        demand_probability (float): Probability of non-zero demand per period
            (occurrence-pattern dependent, see above) (default: 0.2)
        demand_distribution (str): Distribution for non-zero demand sizes
            (default: 'poisson')
        demand_mean (float): Mean of demand when non-zero (default: 5.0)
        demand_std (float): Std of demand when non-zero (default: 2.0)
        intermittent_pattern (str): Occurrence pattern: 'random', 'clustered'
            or 'seasonal' (default: 'random')
        cluster_size (int): Size of demand clusters (default: 3)
        seasonal_period (int): Period for seasonal intermittency (default: 12)
        seasonal_peak_prob (float): Peak occurrence probability at the start
            of each seasonal cycle (default: 0.4)
        min_demand (int): Minimum non-zero demand value (default: 1)
        seed (int | None): Random seed for reproducibility (default: None)
    """

    demand_probability: float = Field(
        default=0.2, ge=0, le=1, description="Probability of non-zero demand"
    )
    demand_distribution: Literal[
        "poisson", "negative_binomial", "lognormal", "gamma"
    ] = Field(default="poisson", description="Distribution for non-zero demand")
    demand_mean: float = Field(
        default=5.0, gt=0, description="Mean of demand when non-zero"
    )
    demand_std: float = Field(default=2.0, ge=0, description="Std of demand")
    intermittent_pattern: Literal["random", "clustered", "seasonal"] = Field(
        default="random", description="Pattern type"
    )
    cluster_size: int = Field(default=3, ge=1, description="Size of demand clusters")
    seasonal_period: int = Field(
        default=12, ge=1, description="Period for seasonal intermittency"
    )
    seasonal_peak_prob: float = Field(
        default=0.4, ge=0, le=1, description="Peak probability in seasonal pattern"
    )
    min_demand: int = Field(
        default=1, ge=1, description="Minimum non-zero demand value"
    )

    def _generate_demand_occurrence(self, length: int) -> np.ndarray:
        """Generate a binary array indicating the periods where demand occurs."""
        if self.intermittent_pattern == "random":
            return self.rng.binomial(1, self.demand_probability, length)

        elif self.intermittent_pattern == "clustered":
            occurrence = np.zeros(length, dtype=int)
            i = 0
            while i < length:
                # Geometric gap (>= 1) before each cluster of ones
                gap = self.rng.geometric(self.demand_probability)
                i += gap
                if i < length:
                    cluster_length = min(self.cluster_size, length - i)
                    occurrence[i : i + cluster_length] = 1
                    i += cluster_length
            return occurrence

        elif self.intermittent_pattern == "seasonal":
            occurrence = np.zeros(length, dtype=int)
            for t in range(length):
                season_position = (t % self.seasonal_period) / self.seasonal_period
                # cos(2*pi*pos) = +1 at the cycle start and -1 mid-cycle, so
                # the probability peaks at seasonal_peak_prob at t mod P == 0
                # and bottoms out at demand_probability at t mod P == P/2.
                seasonal_factor = np.cos(2 * np.pi * season_position)
                prob = (
                    self.demand_probability
                    + (self.seasonal_peak_prob - self.demand_probability)
                    * (seasonal_factor + 1)
                    / 2
                )
                occurrence[t] = self.rng.binomial(1, prob)
            return occurrence

        return np.zeros(length, dtype=int)

    def _generate_demand_sizes(self, num_demands: int) -> np.ndarray:
        """Generate demand sizes for the non-zero demand periods."""
        if num_demands == 0:
            return np.array([])

        if self.demand_distribution == "poisson":
            demands = self.rng.poisson(self.demand_mean, num_demands)

        elif self.demand_distribution == "negative_binomial":
            # Moment matching: p = mean/var, n = mean*p/(1-p)
            variance = self.demand_std**2
            if variance <= self.demand_mean:
                # NB requires var > mean; fall back to Poisson
                demands = self.rng.poisson(self.demand_mean, num_demands)
            else:
                p = self.demand_mean / variance
                n = self.demand_mean * p / (1 - p)
                demands = self.rng.negative_binomial(n, p, num_demands)

        elif self.demand_distribution == "lognormal":
            # Moment matching: mu = ln(mean^2/sqrt(var + mean^2)),
            # sigma^2 = ln(1 + var/mean^2)
            mean_sq = self.demand_mean**2
            variance = self.demand_std**2
            mu = np.log(mean_sq / np.sqrt(variance + mean_sq))
            sigma = np.sqrt(np.log(1 + variance / mean_sq))
            demands = self.rng.lognormal(mu, sigma, num_demands)

        elif self.demand_distribution == "gamma":
            if self.demand_std > 0:
                # Moment matching: shape = (mean/std)^2, scale = var/mean
                shape = (self.demand_mean / self.demand_std) ** 2
                scale = (self.demand_std**2) / self.demand_mean
                demands = self.rng.gamma(shape, scale, num_demands)
            else:
                # Degenerate to constant
                demands = np.full(num_demands, self.demand_mean)

        else:
            demands = np.full(num_demands, self.demand_mean)

        return np.maximum(demands, self.min_demand).astype(float)

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        if self.demand_probability == 0.0 and self.intermittent_pattern == "clustered":
            # The Rust geometric sampler overflows at p=0; disable the batch
            # path so generate_single_series returns the exact all-zero limit.
            return None
        dist_t = {"poisson": 0, "negative_binomial": 1, "lognormal": 2, "gamma": 3}[
            self.demand_distribution
        ]
        pattern_t = {"random": 0, "clustered": 1, "seasonal": 2}[
            self.intermittent_pattern
        ]
        return (
            np.array(
                [
                    self.demand_probability,
                    float(dist_t),
                    self.demand_mean,
                    self.demand_std,
                    float(pattern_t),
                    float(self.cluster_size),
                    float(self.seasonal_period),
                    self.seasonal_peak_prob,
                    float(self.min_demand),
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single intermittent demand time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of values (mostly zeros with intermittent demand)
        """
        if self.demand_probability == 0.0 and self.intermittent_pattern == "clustered":
            # Geometric(p=0) gaps are infinite: no cluster ever starts, so the
            # exact result is an all-zero series. Short-circuit here because
            # numpy raises on geometric(0) and the Rust sampler overflows.
            return np.zeros(length, dtype=float)

        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            dist_t = {"poisson": 0, "negative_binomial": 1, "lognormal": 2, "gamma": 3}[
                self.demand_distribution
            ]
            pattern_t = {"random": 0, "clustered": 1, "seasonal": 2}[
                self.intermittent_pattern
            ]
            return _rs_dom.intermittent_demand(
                length,
                self.demand_probability,
                dist_t,
                self.demand_mean,
                self.demand_std,
                pattern_t,
                self.cluster_size,
                self.seasonal_period,
                self.seasonal_peak_prob,
                float(self.min_demand),
                seed,
            )

        occurrence = self._generate_demand_occurrence(length)
        num_demands = int(np.sum(occurrence))

        series = np.zeros(length, dtype=float)
        if num_demands > 0:
            series[occurrence == 1] = self._generate_demand_sizes(num_demands)
        return series
