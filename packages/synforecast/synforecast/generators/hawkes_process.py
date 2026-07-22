"""Hawkes Process (Self-Exciting Point Process) time series generator."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import stochastic as _rs_stoch
from synforecast.base import BaseGenerator


class HawkesProcessGenerator(BaseGenerator):
    """Generate time series using Hawkes (self-exciting) point processes.

    Hawkes processes model events where past occurrences increase the
    probability of future events. The conditional intensity at time t is:

        lambda(t) = mu + sum_{t_i <= t} g(t - t_i)

    with baseline intensity mu and excitation kernel g. Supported kernels:

        - exponential: g(t) = alpha * exp(-beta * t), branching ratio
          n = alpha / beta
        - power_law: g(t) = alpha / (1 + beta * t)^p with p > 1, branching
          ratio n = alpha / (beta * (p - 1))

    Stability requires n < 1; the long-run event rate is then mu / (1 - n)
    events per time step, and each event spawns on average 1 / (1 - n)
    events (itself included) in its cluster. Time is measured in steps of
    `freq`, so mu and beta are per-step quantities.

    Applications: order arrivals in high-frequency trading, earthquake
    aftershock sequences, viral cascades, clustered fraud events.

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data (e.g. 'D', 'h', '5min') or int
        baseline_intensity (float): Background event rate mu per time step
            (default: 1.0)
        excitation_amplitude (float): Jump in intensity per event alpha
            (default: 0.5)
        decay_rate (float): Rate of intensity decay beta (default: 1.0)
        kernel (str): Excitation kernel, 'exponential' or 'power_law'
            (default: 'exponential')
        power_law_exponent (float): Exponent p for the power-law kernel,
            must be > 1 (default: 1.5)
        output_type (str): 'counts' (events per bin), 'intensity'
            (lambda at bin midpoints), or 'events' (0/1 indicator per bin)
            (default: 'counts')
        max_events (int): Maximum events to simulate per series
            (default: 10000)
        seed (int | None): Random seed for reproducibility (default: None)

    Example:
        >>> gen = HawkesProcessGenerator(
        ...     min_length=100,
        ...     max_length=200,
        ...     freq="h",
        ...     baseline_intensity=0.5,
        ...     excitation_amplitude=0.3,
        ...     decay_rate=2.0,
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    baseline_intensity: float = Field(
        default=1.0, gt=0.0, description="Background event rate μ"
    )
    excitation_amplitude: float = Field(
        default=0.5, ge=0.0, description="Jump in intensity per event α"
    )
    decay_rate: float = Field(
        default=1.0, gt=0.0, description="Rate of intensity decay β"
    )
    kernel: Literal["exponential", "power_law"] = Field(
        default="exponential", description="Excitation kernel type"
    )
    power_law_exponent: float = Field(
        default=1.5, gt=1.0, description="Exponent for power-law kernel (must be > 1)"
    )
    output_type: Literal["counts", "intensity", "events"] = Field(
        default="counts",
        description="Output type: event counts, intensity, or raw event times",
    )
    max_events: int = Field(
        default=10000, gt=0, description="Maximum events to simulate"
    )

    @model_validator(mode="after")
    def validate_stability(self) -> "HawkesProcessGenerator":
        """Require branching ratio < 1 (subcritical process).

        Exponential kernel: alpha/beta < 1. Power-law kernel:
        alpha / (beta * (p - 1)) < 1 (the integral of the kernel).
        """
        if self.kernel == "exponential":
            branching_ratio = self.excitation_amplitude / self.decay_rate
            if branching_ratio >= 1.0:
                raise ValueError(
                    f"Process is unstable: branching ratio α/β = {branching_ratio:.3f} >= 1. "
                    f"Reduce excitation_amplitude or increase decay_rate."
                )
        else:  # power_law
            branching_ratio = self.excitation_amplitude / (
                self.decay_rate * (self.power_law_exponent - 1)
            )
            if branching_ratio >= 1.0:
                raise ValueError(
                    f"Process is unstable: branching ratio α/(β(p-1)) = {branching_ratio:.3f} >= 1. "
                    f"Reduce excitation_amplitude, increase decay_rate, or increase power_law_exponent."
                )
        return self

    def _exponential_kernel(self, dt: np.ndarray) -> np.ndarray:
        """Exponential excitation kernel: g(t) = alpha * exp(-beta * t)."""
        return self.excitation_amplitude * np.exp(-self.decay_rate * dt)

    def _power_law_kernel(self, dt: np.ndarray) -> np.ndarray:
        """Power-law excitation kernel: g(t) = alpha / (1 + beta * t)^p."""
        return (
            self.excitation_amplitude
            / (1 + self.decay_rate * dt) ** self.power_law_exponent
        )

    def _compute_intensity(self, t: float, event_times: np.ndarray) -> float:
        """Compute the intensity lambda(t) given past events (t_i <= t)."""
        past_events = event_times[event_times <= t]
        if len(past_events) == 0:
            return self.baseline_intensity

        dt = t - past_events
        if self.kernel == "exponential":
            excitation = np.sum(self._exponential_kernel(dt))
        else:  # power_law
            excitation = np.sum(self._power_law_kernel(dt))
        return self.baseline_intensity + excitation

    def _simulate_hawkes(self, time_horizon: float) -> np.ndarray:
        """Simulate event times with Ogata's thinning algorithm.

        Both kernels decay monotonically, so between events the intensity
        can only decrease: lambda evaluated at the current time (including
        an event just accepted at that instant) is a valid upper bound
        until the next candidate.

        Args:
            time_horizon: Total time to simulate

        Returns:
            np.ndarray: Array of event times
        """
        event_times: list[float] = []
        t = 0.0

        while t < time_horizon and len(event_times) < self.max_events:
            events = np.asarray(event_times)
            # Small safety factor guards against float rounding in the bound
            lambda_bar = 1.01 * self._compute_intensity(t, events)

            t = t + self.rng.exponential(1.0 / lambda_bar)
            if t >= time_horizon:
                break

            lambda_t = self._compute_intensity(t, events)
            if self.rng.uniform() * lambda_bar <= lambda_t:
                event_times.append(t)

        return np.array(event_times)

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        kernel_t = 0 if self.kernel == "exponential" else 1
        output_t = {"counts": 0, "intensity": 1, "events": 2}[self.output_type]
        return (
            np.array(
                [
                    self.baseline_intensity,
                    self.excitation_amplitude,
                    self.decay_rate,
                    float(kernel_t),
                    self.power_law_exponent,
                    float(output_t),
                    float(self.max_events),
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single Hawkes process series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of values (counts, intensity, or event indicator)
        """
        seed = int(self.rng.integers(0, 2**63))
        kernel_t = 0 if self.kernel == "exponential" else 1
        output_t = {"counts": 0, "intensity": 1, "events": 2}[self.output_type]
        return _rs_stoch.hawkes_process(
            length,
            self.baseline_intensity,
            self.excitation_amplitude,
            self.decay_rate,
            kernel_t,
            self.power_law_exponent,
            output_t,
            self.max_events,
            seed,
        )

    def simulate_with_events(
        self, time_horizon: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate and return both event times and intensity at those times.

        Args:
            time_horizon: Total time to simulate

        Returns:
            tuple: (event_times, intensity_at_events)
        """
        event_times = self._simulate_hawkes(time_horizon)
        intensity_at_events = np.array(
            [
                self._compute_intensity(t, event_times[:i])
                for i, t in enumerate(event_times)
            ]
        )
        return event_times, intensity_at_events

    def get_model_info(self) -> dict:
        """Get information about the Hawkes process model.

        Returns:
            dict: Model parameters and characteristics
        """
        if self.kernel == "exponential":
            branching_ratio = self.excitation_amplitude / self.decay_rate
            expected_cluster_size = 1 / (1 - branching_ratio)
        else:
            branching_ratio = self.excitation_amplitude / (
                self.decay_rate * (self.power_law_exponent - 1)
            )
            expected_cluster_size = 1 / (1 - min(branching_ratio, 0.99))

        return {
            "baseline_intensity": self.baseline_intensity,
            "excitation_amplitude": self.excitation_amplitude,
            "decay_rate": self.decay_rate,
            "kernel": self.kernel,
            "branching_ratio": branching_ratio,
            "expected_cluster_size": expected_cluster_size,
            "output_type": self.output_type,
            "is_stable": branching_ratio < 1.0,
        }

    def estimate_parameters(
        self, event_times: np.ndarray, _method: str = "mle"
    ) -> dict:
        """Estimate Hawkes process parameters from observed event times.

        Heuristic moment-based estimation: the coefficient of variation of
        inter-arrival times proxies the branching ratio (CV = 1 for a
        Poisson process, larger under clustering), and the mean rate
        identifies mu via rate = mu / (1 - n).

        Args:
            event_times: Array of observed event times
            _method: Estimation method (currently only 'mle' supported)

        Returns:
            dict: Estimated parameters
        """
        if len(event_times) < 2:
            return {
                "baseline_intensity": self.baseline_intensity,
                "excitation_amplitude": 0.0,
                "decay_rate": self.decay_rate,
            }

        mean_rate = len(event_times) / event_times[-1]
        inter_arrivals = np.diff(event_times)
        cv = (
            np.std(inter_arrivals) / np.mean(inter_arrivals)
            if len(inter_arrivals) > 1
            else 1.0
        )

        estimated_branching = min(0.9, max(0.0, 1 - 1 / (cv**2 + 1)))
        estimated_mu = mean_rate * (1 - estimated_branching)
        estimated_alpha = estimated_branching * self.decay_rate

        return {
            "baseline_intensity": estimated_mu,
            "excitation_amplitude": estimated_alpha,
            "decay_rate": self.decay_rate,  # Keep original (hard to estimate)
            "branching_ratio": estimated_branching,
        }
