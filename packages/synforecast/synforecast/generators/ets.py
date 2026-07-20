"""ETS (Error, Trend, Seasonal) time series generator with state export."""

from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import statistical as _rs_stat

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class ETSGenerator(BaseGenerator):
    """Generate time series based on ETS (Error, Trend, Seasonal) models.

    Creates time series from the innovations state space form of exponential
    smoothing (Hyndman, Koehler, Ord & Snyder, 2008). Each component is
    additive (A), multiplicative (M), or absent (N):

    - y_t = μ_t + ε_t (additive error) or y_t = μ_t (1 + ε_t) (multiplicative)
    - μ_t combines level l, trend b (optionally damped by φ), and seasonal s,
      e.g. ETS(A,A,A): μ_t = l_{t-1} + φ b_{t-1} + s_{t-m}
    - States update per the standard taxonomy, e.g. ETS(A,A,A):
      l_t = l_{t-1} + φ b_{t-1} + α ε_t; b_t = φ b_{t-1} + β ε_t;
      s_t = s_{t-m} + γ ε_t

    Common models: ETS(A,N,N) simple exponential smoothing, ETS(A,A,N) Holt,
    ETS(A,A,A) additive Holt-Winters, ETS(M,A,M) multiplicative Holt-Winters,
    ETS(A,Ad,A) damped Holt-Winters.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency, a pandas offset alias (e.g. 'D', 'h',
            '5min', 'MS') or an integer time step.
        error_type (str): Error component, 'add' or 'mul' (default: 'add').
        trend_type (str | None): Trend component, 'add', 'mul', or None
            (default: 'add').
        seasonal_type (str | None): Seasonal component, 'add', 'mul', or None
            (default: 'add').
        seasonal_period (int): Seasonal period m (default: 12).
        level (float): Initial level l_0 (default: 100.0).
        trend (float): Initial trend b_0 (default: 0.0; reset to 1.0 for
            multiplicative trend when <= 0).
        seasonal (list[float] | None): Initial seasonal states, one per season
            (default: random, zero-sum for additive / unit-mean for
            multiplicative).
        alpha (float): Level smoothing parameter in [0, 1] (default: 0.3).
        beta (float): Trend smoothing parameter in [0, 1] (default: 0.1).
        gamma (float): Seasonal smoothing parameter in [0, 1] (default: 0.1).
        phi (float): Damping parameter in [0, 1], used when damped=True
            (default: 0.98).
        damped (bool): Whether to damp the trend (default: False).
        noise_std (float): Standard deviation of the innovations ε
            (default: 1.0).
        box_cox_lambda (float | None): If set, apply the inverse Box-Cox
            transform with this λ to the generated series (default: None).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp of every series
            (default: '2000-01-01').
    """

    # ETS component types
    error_type: Literal["add", "mul"] = Field(
        default="add", description="Type of error: 'add' (A) or 'mul' (M)"
    )
    trend_type: Literal["add", "mul"] | None = Field(
        default="add", description="Type of trend: 'add' (A), 'mul' (M), or None (N)"
    )
    seasonal_type: Literal["add", "mul"] | None = Field(
        default="add",
        description="Type of seasonality: 'add' (A), 'mul' (M), or None (N)",
    )
    seasonal_period: int = Field(default=12, ge=1, description="Seasonal period m")

    # Initial state values
    level: float = Field(default=100.0, description="Initial level l_0")
    trend: float = Field(default=0.0, description="Initial trend b_0")
    seasonal: list[float] | None = Field(
        default=None, description="Initial seasonal factors s_{1-m}, ..., s_0"
    )

    # Smoothing parameters
    alpha: float = Field(
        default=0.3, ge=0, le=1, description="Level smoothing parameter α"
    )
    beta: float = Field(
        default=0.1, ge=0, le=1, description="Trend smoothing parameter β"
    )
    gamma: float = Field(
        default=0.1, ge=0, le=1, description="Seasonal smoothing parameter γ"
    )
    phi: float = Field(default=0.98, ge=0, le=1, description="Damping parameter φ")
    damped: bool = Field(default=False, description="Whether to use damped trend")

    # Noise parameter
    noise_std: float = Field(
        default=1.0, gt=0, description="Standard deviation of noise ε"
    )

    # Box-Cox transformation
    box_cox_lambda: float | None = Field(
        default=None, description="Box-Cox λ parameter (None = no transformation)"
    )

    # Bounds for numerical stability
    _MIN_LEVEL: float = 1e-6
    _MAX_LEVEL: float = 1e12
    _MIN_SEASONAL_MUL: float = 0.01
    _MAX_SEASONAL_MUL: float = 100.0
    _MAX_TREND_ADD: float = 1e6
    _MIN_TREND_MUL: float = 0.01
    _MAX_TREND_MUL: float = 100.0

    # Computed field for numpy seasonal array
    _seasonal_array: Any = None

    @model_validator(mode="after")
    def validate_and_compute(self) -> "ETSGenerator":
        """Validate parameters and compute initial seasonal components."""
        self._validate_model_combination()

        if self.seasonal is None and self.seasonal_type is not None:
            seasonal_array = self._generate_seasonal_components()
            object.__setattr__(self, "_seasonal_array", seasonal_array)
        elif self.seasonal is not None:
            if len(self.seasonal) != self.seasonal_period:
                raise ValueError(
                    f"seasonal has {len(self.seasonal)} elements but "
                    f"seasonal_period={self.seasonal_period}"
                )
            object.__setattr__(self, "_seasonal_array", np.array(self.seasonal))

        return self

    def _validate_model_combination(self) -> None:
        """Validate that the ETS model combination is valid."""
        # Multiplicative components require a positive level
        if self.level <= 0 and (
            self.error_type == "mul"
            or self.trend_type == "mul"
            or self.seasonal_type == "mul"
        ):
            raise ValueError("Multiplicative components require level > 0")

        # Multiplicative trend needs a positive growth factor
        if self.trend_type == "mul" and self.trend <= 0:
            object.__setattr__(self, "trend", 1.0)

    def _generate_seasonal_components(self) -> np.ndarray:
        """Generate random initial seasonal components.

        Additive: zero-sum. Multiplicative: unit mean.

        Returns:
            np.ndarray: Initial seasonal components.
        """
        m = self.seasonal_period

        if self.seasonal_type == "add":
            seasonal = self.rng.uniform(-5, 5, m)
            seasonal = seasonal - seasonal.mean()
        else:
            seasonal = self.rng.uniform(0.8, 1.2, m)
            seasonal = seasonal / seasonal.mean()

        return seasonal

    def _get_model_notation(self) -> str:
        """Get standard ETS model notation.

        Returns:
            str: Model notation like 'ETS(A,A,A)' or 'ETS(M,Ad,M)'
        """
        error_char = "A" if self.error_type == "add" else "M"

        if self.trend_type is None:
            trend_char = "N"
        elif self.trend_type == "add":
            trend_char = "Ad" if self.damped else "A"
        else:
            trend_char = "Md" if self.damped else "M"

        if self.seasonal_type is None:
            seasonal_char = "N"
        else:
            seasonal_char = "A" if self.seasonal_type == "add" else "M"

        return f"ETS({error_char},{trend_char},{seasonal_char})"

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        error_t = 0 if self.error_type == "add" else 1
        trend_t = (
            0 if self.trend_type is None else (1 if self.trend_type == "add" else 2)
        )
        seasonal_t = (
            0
            if self.seasonal_type is None
            else (1 if self.seasonal_type == "add" else 2)
        )
        s_init = (
            np.asarray(self._seasonal_array, dtype=np.float64)
            if self.seasonal_type is not None
            else np.zeros(self.seasonal_period)
        )
        return (
            np.array(
                [
                    float(error_t),
                    float(trend_t),
                    float(seasonal_t),
                    float(self.seasonal_period),
                    self.level,
                    self.trend,
                    self.alpha,
                    self.beta,
                    self.gamma,
                    self.phi,
                    1.0 if self.damped else 0.0,
                    self.noise_std,
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [s_init],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single ETS time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            error_t = 0 if self.error_type == "add" else 1
            trend_t = (
                0 if self.trend_type is None else (1 if self.trend_type == "add" else 2)
            )
            seasonal_t = (
                0
                if self.seasonal_type is None
                else (1 if self.seasonal_type == "add" else 2)
            )
            s_init = (
                self._seasonal_array
                if self.seasonal_type is not None
                else np.zeros(self.seasonal_period)
            )
            return _rs_stat.ets(
                length,
                error_t,
                trend_t,
                seasonal_t,
                self.seasonal_period,
                self.level,
                self.trend,
                s_init,
                self.alpha,
                self.beta,
                self.gamma,
                self.phi,
                self.damped,
                self.noise_std,
                seed,
                self._rs_innov_dist,
                self._rs_innov_param,
            )

        l_t = self.level
        b_t = self.trend if self.trend_type is not None else 0.0
        s = self._seasonal_array.copy() if self.seasonal_type is not None else None

        m = self.seasonal_period
        series = np.zeros(length)

        for t in range(length):
            if s is not None:
                s_idx = t % m
                s_t = s[s_idx]
            else:
                s_t = 0.0

            y_hat = self._forecast(l_t, b_t, s_t)

            eps = self._sample_innovations(1, scale=self.noise_std)[0]
            if self.error_type == "add":
                y_t = np.clip(y_hat + eps, -self._MAX_LEVEL, self._MAX_LEVEL)
            else:
                y_t = np.clip(y_hat * (1 + eps), self._MIN_LEVEL, self._MAX_LEVEL)

            series[t] = y_t

            l_new, b_new, s_new = self._update_state(l_t, b_t, s_t, eps)

            # Bounds for numerical stability
            if self.error_type == "mul":
                l_t = np.clip(l_new, self._MIN_LEVEL, self._MAX_LEVEL)
            else:
                l_t = np.clip(l_new, -self._MAX_LEVEL, self._MAX_LEVEL)

            if self.trend_type == "add":
                b_t = np.clip(b_new, -self._MAX_TREND_ADD, self._MAX_TREND_ADD)
            elif self.trend_type == "mul":
                b_t = np.clip(b_new, self._MIN_TREND_MUL, self._MAX_TREND_MUL)
            else:
                b_t = b_new

            if s is not None:
                if self.seasonal_type == "mul":
                    s_new = np.clip(
                        s_new, self._MIN_SEASONAL_MUL, self._MAX_SEASONAL_MUL
                    )
                s[s_idx] = s_new

        # Apply inverse Box-Cox transformation if specified
        if self.box_cox_lambda is not None:
            series = self._inverse_box_cox(series)

        return series

    def _forecast(self, l_t: float, b_t: float, s_t: float) -> float:
        """One-step-ahead point forecast μ_t.

        Args:
            l_t: Current level.
            b_t: Current trend.
            s_t: Current seasonal state.

        Returns:
            float: Point forecast.
        """
        phi = self.phi if self.damped else 1.0

        if self.trend_type == "add":
            level_plus_trend = l_t + phi * b_t  # l + φb
        elif self.trend_type == "mul":
            level_plus_trend = l_t * (b_t**phi)  # l · b^φ
        else:
            level_plus_trend = l_t

        if self.seasonal_type == "add":
            return level_plus_trend + s_t
        if self.seasonal_type == "mul":
            return level_plus_trend * s_t
        return level_plus_trend

    def _update_state(
        self, l_t: float, b_t: float, s_t: float, eps: float
    ) -> tuple[float, float, float]:
        """Update the ETS state per Hyndman et al. (2008), Tables 2.2/2.3.

        All 30 model variants reduce to one scheme in terms of the additive
        one-step error e_t = y_t - μ_t (for multiplicative error,
        e_t = μ_t ε_t):

        - l_t = trend_base + α e_t / s_div
        - b_t = φ b_{t-1} + β e_t / s_div          (additive trend)
        - b_t = b_{t-1}^φ + β e_t / (s_div l_{t-1}) (multiplicative trend)
        - s_t = s_{t-m} + γ e_t                     (additive seasonal)
        - s_t = s_{t-m} + γ e_t / trend_base        (multiplicative seasonal)

        where trend_base is l + φb (add), l·b^φ (mul), or l (none), and
        s_div is s_{t-m} for multiplicative seasonality, else 1.

        Args:
            l_t: Current level.
            b_t: Current trend.
            s_t: Current seasonal state s_{t-m}.
            eps: Innovation ε_t.

        Returns:
            tuple: (new_level, new_trend, new_seasonal)
        """
        phi = self.phi if self.damped else 1.0

        if self.trend_type == "add":
            trend_base = l_t + phi * b_t
        elif self.trend_type == "mul":
            trend_base = l_t * (b_t**phi)
        else:
            trend_base = l_t

        e = eps if self.error_type == "add" else self._forecast(l_t, b_t, s_t) * eps

        s_div = s_t if self.seasonal_type == "mul" and s_t != 0.0 else 1.0

        l_new = trend_base + self.alpha * e / s_div

        if self.trend_type == "add":
            b_new = phi * b_t + self.beta * e / s_div
        elif self.trend_type == "mul":
            denom = s_div * l_t
            b_new = b_t**phi + (self.beta * e / denom if denom != 0.0 else 0.0)
        else:
            b_new = b_t

        if self.seasonal_type == "add":
            s_new = s_t + self.gamma * e
        elif self.seasonal_type == "mul":
            s_new = s_t + (self.gamma * e / trend_base if trend_base != 0.0 else 0.0)
        else:
            s_new = s_t

        return l_new, b_new, s_new

    def _inverse_box_cox(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse Box-Cox transformation.

        Args:
            y: Transformed values

        Returns:
            np.ndarray: Original scale values
        """
        lam = self.box_cox_lambda
        if lam == 0:
            return np.exp(y)
        else:
            # y = (x^λ - 1) / λ => x = (λy + 1)^(1/λ)
            return np.power(np.maximum(lam * y + 1, 1e-10), 1 / lam)

    def generate_with_states(
        self, n_series: int = 1, start_id: int = 0
    ) -> tuple[IntoDataFrameT, IntoDataFrameT]:
        """Generate series and return both observations and hidden states.

        This is useful for analyzing the underlying ETS state evolution.

        Args:
            n_series (int): Number of series to generate (default: 1)
            start_id (int): Starting ID for series naming (default: 0)

        Returns:
            tuple[DataFrame, DataFrame]:
                - DataFrame with observations (id_col, time_col, target_col)
                - DataFrame with states (id_col, time_col, level, trend, seasonal_*)
        """
        m = self.seasonal_period

        # Collect data for all series
        all_values = []
        all_lengths = []
        all_ids = []
        all_timestamps = []
        all_states_data = []

        for i in range(n_series):
            series_id = start_id + i

            length = self.rng.integers(self.min_length, self.max_length + 1)

            timestamps = self._timestamps(length)

            l_t = self.level
            b_t = self.trend if self.trend_type is not None else 0.0
            s = self._seasonal_array.copy() if self.seasonal_type is not None else None

            series = np.zeros(length)
            levels = np.zeros(length)
            trends = np.zeros(length)
            seasonals = np.zeros((length, m)) if s is not None else None

            for t in range(length):
                if s is not None:
                    s_idx = t % m
                    s_t = s[s_idx]
                else:
                    s_t = 0.0

                levels[t] = l_t
                trends[t] = b_t
                if seasonals is not None:
                    seasonals[t] = s.copy()

                y_hat = self._forecast(l_t, b_t, s_t)

                eps = self._sample_innovations(1, scale=self.noise_std)[0]
                if self.error_type == "add":
                    y_t = np.clip(y_hat + eps, -self._MAX_LEVEL, self._MAX_LEVEL)
                else:
                    y_t = np.clip(y_hat * (1 + eps), self._MIN_LEVEL, self._MAX_LEVEL)

                series[t] = y_t

                l_new, b_new, s_new = self._update_state(l_t, b_t, s_t, eps)

                if self.error_type == "mul":
                    l_t = np.clip(l_new, self._MIN_LEVEL, self._MAX_LEVEL)
                else:
                    l_t = np.clip(l_new, -self._MAX_LEVEL, self._MAX_LEVEL)

                if self.trend_type == "add":
                    b_t = np.clip(b_new, -self._MAX_TREND_ADD, self._MAX_TREND_ADD)
                elif self.trend_type == "mul":
                    b_t = np.clip(b_new, self._MIN_TREND_MUL, self._MAX_TREND_MUL)
                else:
                    b_t = b_new

                if s is not None:
                    if self.seasonal_type == "mul":
                        s_new = np.clip(
                            s_new, self._MIN_SEASONAL_MUL, self._MAX_SEASONAL_MUL
                        )
                    s[s_idx] = s_new

            if self.box_cox_lambda is not None:
                series = self._inverse_box_cox(series)

            # Apply pattern injection
            series, _ = self._add_changepoints(series)
            series, _ = self._add_anomalies(series)
            series, _ = self._add_missingness(series)

            all_values.append(series)
            all_lengths.append(length)
            all_ids.append(series_id)
            all_timestamps.append(timestamps)

            all_states_data.append(
                {
                    "series_id": series_id,
                    "timestamps": timestamps,
                    "levels": levels,
                    "trends": trends,
                    "seasonals": seasonals,
                }
            )

        # Create observations DataFrame
        flat_values = np.concatenate(all_values)
        flat_timestamps = np.concatenate(all_timestamps)
        flat_ids = np.concatenate(
            [
                np.full(length, f"series_{series_id}")
                for length, series_id in zip(all_lengths, all_ids, strict=False)
            ]
        )

        obs_result = {
            self.id_col: flat_ids,
            self.time_col: flat_timestamps,
            self.target_col: flat_values,
        }
        obs_df = nw.DataFrame.from_dict(obs_result, backend=self.engine).to_native()

        # Create states DataFrame
        states_flat_ids = []
        states_flat_timestamps = []
        states_flat_levels = []
        states_flat_trends = []
        states_flat_seasonals: dict[str, list[float]] = {
            f"seasonal_{j}": [] for j in range(m)
        }

        for state_data in all_states_data:
            series_id = state_data["series_id"]
            ts = state_data["timestamps"]
            length = len(ts)

            states_flat_ids.extend([f"series_{series_id}"] * length)
            states_flat_timestamps.extend(ts)
            states_flat_levels.extend(state_data["levels"])
            states_flat_trends.extend(state_data["trends"])

            if state_data["seasonals"] is not None:
                for j in range(m):
                    states_flat_seasonals[f"seasonal_{j}"].extend(
                        state_data["seasonals"][:, j]
                    )

        states_result = {
            self.id_col: np.array(states_flat_ids),
            self.time_col: np.concatenate([d["timestamps"] for d in all_states_data]),
            "level": np.array(states_flat_levels),
            "trend": np.array(states_flat_trends),
        }

        # Add seasonal columns if applicable
        if self.seasonal_type is not None:
            for j in range(m):
                states_result[f"seasonal_{j}"] = np.array(
                    states_flat_seasonals[f"seasonal_{j}"]
                )

        states_df = nw.DataFrame.from_dict(
            states_result, backend=self.engine
        ).to_native()

        return obs_df, states_df

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the ETS model configuration.

        Returns:
            dict: Model information including type, parameters, and state
        """
        return {
            "model": self._get_model_notation(),
            "error_type": self.error_type,
            "trend_type": self.trend_type,
            "seasonal_type": self.seasonal_type,
            "seasonal_period": self.seasonal_period,
            "damped": self.damped,
            "level": self.level,
            "trend": self.trend,
            "seasonal": (
                self._seasonal_array.tolist()
                if self._seasonal_array is not None
                else None
            ),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "phi": self.phi,
            "noise_std": self.noise_std,
            "box_cox_lambda": self.box_cox_lambda,
        }
