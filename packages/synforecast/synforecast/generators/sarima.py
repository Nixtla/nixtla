"""Seasonal ARIMA time series generator with SARIMAX support."""

from typing import Any

import numpy as np
from pydantic import Field, model_validator

from synforecast._lib import statistical as _rs_stat
from synforecast.base import BaseGenerator


class SARIMAGenerator(BaseGenerator):
    """Generate time series based on Seasonal ARIMA (SARIMAX) processes.

    Creates time series using a Seasonal AutoRegressive Integrated Moving Average
    model with optional eXogenous regressors. The model is defined by (p,d,q)x(P,D,Q,s).

    The SARIMA model uses multiplicative seasonal structure:
    - AR polynomial: φ(B)Φ(B^s) where B is the backshift operator
    - MA polynomial: θ(B)Θ(B^s)
    - Differencing: (1-B)^d (1-B^s)^D

    For SARIMA(1,1,1)(1,1,1)_12, this creates dependencies at lags:
    - AR: 1, 12, 13 (from φ₁, Φ₁, φ₁Φ₁)
    - MA: 1, 12, 13 (from θ₁, Θ₁, θ₁Θ₁)

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency, a pandas offset alias (e.g. 'D', 'h',
            '5min', 'MS') or an integer time step.
        p (int): AR order (default: 1).
        d (int): Differencing order, 0-2 (default: 0).
        q (int): MA order (default: 1).
        P (int): Seasonal AR order (default: 1).
        D (int): Seasonal differencing order, 0-2 (default: 0).
        Q (int): Seasonal MA order (default: 1).
        seasonal_period (int): Seasonal period s (default: 12).
        ar_params (list[float] | None): AR coefficients φ₁,...,φ_p
            (default: random stable).
        ma_params (list[float] | None): MA coefficients θ₁,...,θ_q
            (default: random in (-0.5, 0.5)).
        seasonal_ar_params (list[float] | None): Seasonal AR coefficients
            Φ₁,...,Φ_P (default: random stable).
        seasonal_ma_params (list[float] | None): Seasonal MA coefficients
            Θ₁,...,Θ_Q (default: random in (-0.5, 0.5)).
        mean (float): Process mean for stationary models (d=0, D=0)
            (default: 0.0).
        drift (float): Constant added to the differenced series for integrated
            models (d>0 or D>0); yields slope `drift` per step when d=1
            (default: 0.0).
        noise_std (float): Standard deviation of innovation noise
            (default: 1.0).
        burn_in (int | None): Burn-in period; None computes it from model
            order and AR persistence (default: None).
        validate_stationarity (bool): Validate AR parameters for stationarity
            (default: True).
        exog_coefficients (list[float] | None): Coefficients for exogenous
            regressors (default: None).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp of every series
            (default: '2000-01-01').
    """

    # SARIMA orders
    p: int = Field(default=1, ge=0, description="AR order")
    d: int = Field(default=0, ge=0, le=2, description="Differencing order")
    q: int = Field(default=1, ge=0, description="MA order")
    P: int = Field(default=1, ge=0, description="Seasonal AR order")
    D: int = Field(default=0, ge=0, le=2, description="Seasonal differencing order")
    Q: int = Field(default=1, ge=0, description="Seasonal MA order")
    seasonal_period: int = Field(default=12, ge=1, description="Seasonal period")

    # Model parameters
    mean: float = Field(
        default=0.0, description="Process mean (for stationary models with d=0, D=0)"
    )
    drift: float = Field(
        default=0.0, description="Drift term (for integrated models with d>0 or D>0)"
    )
    noise_std: float = Field(
        default=1.0, gt=0, description="Standard deviation of innovation noise"
    )

    # Optional burn-in override
    burn_in: int | None = Field(
        default=None, ge=0, description="Burn-in period; None for automatic"
    )

    # Validation flag
    validate_stationarity: bool = Field(
        default=True, description="Validate AR parameters for stationarity"
    )

    # Optional model coefficients (computed if not provided)
    ar_params: list[float] | None = Field(
        default=None, description="AR coefficients φ₁,...,φₚ"
    )
    ma_params: list[float] | None = Field(
        default=None, description="MA coefficients θ₁,...,θq"
    )
    seasonal_ar_params: list[float] | None = Field(
        default=None, description="Seasonal AR coefficients Φ₁,...,Φₚ"
    )
    seasonal_ma_params: list[float] | None = Field(
        default=None, description="Seasonal MA coefficients Θ₁,...,Θq"
    )

    # Exogenous regressors (SARIMAX)
    exog_coefficients: list[float] | None = Field(
        default=None, description="Coefficients for exogenous regressors"
    )

    # Internal computed arrays
    _ar_params_array: Any = None
    _ma_params_array: Any = None
    _seasonal_ar_params_array: Any = None
    _seasonal_ma_params_array: Any = None
    _full_ar_poly: Any = None  # Expanded AR polynomial coefficients
    _full_ma_poly: Any = None  # Expanded MA polynomial coefficients
    _exog_coefficients_array: Any = None
    _computed_burn_in: int = 100

    @model_validator(mode="after")
    def compute_params(self) -> "SARIMAGenerator":
        """Generate random stable parameters if not provided and validate."""
        # AR params
        if self.ar_params is None and self.p > 0:
            ar_array = self._generate_stable_ar_params(self.p)
            object.__setattr__(self, "_ar_params_array", ar_array)
        elif self.ar_params is not None:
            if len(self.ar_params) != self.p:
                raise ValueError(
                    f"ar_params length {len(self.ar_params)} != p={self.p}"
                )
            object.__setattr__(self, "_ar_params_array", np.array(self.ar_params))
        else:
            object.__setattr__(self, "_ar_params_array", np.array([]))

        # MA params
        if self.ma_params is None and self.q > 0:
            ma_array = self.rng.uniform(-0.5, 0.5, self.q)
            object.__setattr__(self, "_ma_params_array", ma_array)
        elif self.ma_params is not None:
            if len(self.ma_params) != self.q:
                raise ValueError(
                    f"ma_params length {len(self.ma_params)} != q={self.q}"
                )
            object.__setattr__(self, "_ma_params_array", np.array(self.ma_params))
        else:
            object.__setattr__(self, "_ma_params_array", np.array([]))

        # Seasonal AR params
        if self.seasonal_ar_params is None and self.P > 0:
            sar_array = self._generate_stable_ar_params(self.P)
            object.__setattr__(self, "_seasonal_ar_params_array", sar_array)
        elif self.seasonal_ar_params is not None:
            if len(self.seasonal_ar_params) != self.P:
                raise ValueError(
                    f"seasonal_ar_params length {len(self.seasonal_ar_params)} != P={self.P}"
                )
            object.__setattr__(
                self, "_seasonal_ar_params_array", np.array(self.seasonal_ar_params)
            )
        else:
            object.__setattr__(self, "_seasonal_ar_params_array", np.array([]))

        # Seasonal MA params
        if self.seasonal_ma_params is None and self.Q > 0:
            sma_array = self.rng.uniform(-0.5, 0.5, self.Q)
            object.__setattr__(self, "_seasonal_ma_params_array", sma_array)
        elif self.seasonal_ma_params is not None:
            if len(self.seasonal_ma_params) != self.Q:
                raise ValueError(
                    f"seasonal_ma_params length {len(self.seasonal_ma_params)} != Q={self.Q}"
                )
            object.__setattr__(
                self, "_seasonal_ma_params_array", np.array(self.seasonal_ma_params)
            )
        else:
            object.__setattr__(self, "_seasonal_ma_params_array", np.array([]))

        # Exogenous coefficients
        if self.exog_coefficients is not None:
            object.__setattr__(
                self, "_exog_coefficients_array", np.array(self.exog_coefficients)
            )

        # Compute full AR and MA polynomials (multiplicative expansion)
        full_ar = self._compute_full_ar_polynomial()
        full_ma = self._compute_full_ma_polynomial()
        object.__setattr__(self, "_full_ar_poly", full_ar)
        object.__setattr__(self, "_full_ma_poly", full_ma)

        # Validate stationarity if requested
        if self.validate_stationarity:
            self._validate_ar_stationarity(full_ar)

        # Compute burn-in
        computed_burn_in = self._compute_burn_in()
        object.__setattr__(self, "_computed_burn_in", computed_burn_in)

        return self

    def _generate_stable_ar_params(self, order: int) -> np.ndarray:
        """Generate stable AR parameters via random partial autocorrelations.

        Draws partial autocorrelations in (-1, 1) (which guarantees
        stationarity) and converts them to AR coefficients with the
        Durbin-Levinson recursion. Conservative bounds keep the process from
        being nearly integrated.

        Args:
            order (int): Order of the AR process.

        Returns:
            Array of stable AR parameters.
        """
        has_seasonal = self.P > 0 or self.Q > 0
        bound = 0.4 if has_seasonal else 0.6
        pacf = self.rng.uniform(-bound, bound, order)

        # Durbin-Levinson: phi_{k,j} = phi_{k-1,j} - pacf_k * phi_{k-1,k-j}
        ar_coeffs = np.zeros(order)
        ar_coeffs[0] = pacf[0]

        for k in range(1, order):
            ar_prev = ar_coeffs[:k].copy()
            ar_coeffs[k] = pacf[k]
            for j in range(k):
                ar_coeffs[j] = ar_prev[j] - pacf[k] * ar_prev[k - 1 - j]

        return ar_coeffs

    def _compute_full_ar_polynomial(self) -> np.ndarray:
        """Compute the full AR polynomial from φ(B)Φ(B^s) multiplication.

        The AR polynomial is (1 - φ₁B - φ₂B² - ...) × (1 - Φ₁B^s - Φ₂B^{2s} - ...)

        Returns:
            Coefficients of the full AR polynomial (excluding lag 0)
                       where index i corresponds to lag (i+1)
        """
        s = self.seasonal_period

        # Non-seasonal AR polynomial: [1, -φ₁, -φ₂, ...]
        ar_poly = np.zeros(self.p + 1)
        ar_poly[0] = 1.0
        if self.p > 0 and self._ar_params_array is not None:
            ar_poly[1 : self.p + 1] = -self._ar_params_array

        # Seasonal AR polynomial: [1, 0, ..., 0, -Φ₁, 0, ..., 0, -Φ₂, ...]
        sar_poly_len = self.P * s + 1
        sar_poly = np.zeros(sar_poly_len)
        sar_poly[0] = 1.0
        if self.P > 0 and self._seasonal_ar_params_array is not None:
            for i in range(self.P):
                sar_poly[(i + 1) * s] = -self._seasonal_ar_params_array[i]

        full_poly = np.convolve(ar_poly, sar_poly)

        # Coefficients for lags 1, 2, ... (constant term dropped), negated so
        # that y_t = sum_i coeff_i * y_{t-i} + u_t
        return -full_poly[1:]

    def _compute_full_ma_polynomial(self) -> np.ndarray:
        """Compute the full MA polynomial from θ(B)Θ(B^s) multiplication.

        The MA polynomial is (1 + θ₁B + θ₂B² + ...) × (1 + Θ₁B^s + Θ₂B^{2s} + ...)

        Returns:
            Coefficients of the full MA polynomial (excluding lag 0)
                       where index i corresponds to lag (i+1)
        """
        s = self.seasonal_period

        # Non-seasonal MA polynomial: [1, θ₁, θ₂, ...]
        ma_poly = np.zeros(self.q + 1)
        ma_poly[0] = 1.0
        if self.q > 0 and self._ma_params_array is not None:
            ma_poly[1 : self.q + 1] = self._ma_params_array

        # Seasonal MA polynomial: [1, 0, ..., 0, Θ₁, 0, ..., 0, Θ₂, ...]
        sma_poly_len = self.Q * s + 1
        sma_poly = np.zeros(sma_poly_len)
        sma_poly[0] = 1.0
        if self.Q > 0 and self._seasonal_ma_params_array is not None:
            for i in range(self.Q):
                sma_poly[(i + 1) * s] = self._seasonal_ma_params_array[i]

        full_poly = np.convolve(ma_poly, sma_poly)

        # Coefficients for lags 1, 2, ... (constant term dropped)
        return full_poly[1:]

    def _validate_ar_stationarity(self, ar_coeffs: np.ndarray) -> None:
        """Validate that AR polynomial has roots outside unit circle (stationary).

        Args:
            ar_coeffs: AR coefficients for lags 1, 2, ... (positive convention)

        Raises:
            ValueError: If the AR process is non-stationary
        """
        if len(ar_coeffs) == 0:
            return

        # phi(z) = 1 - phi_1 z - ... - phi_p z^p must have all roots |z| > 1.
        # np.roots wants highest degree first: [-phi_p, ..., -phi_1, 1]
        poly_coeffs = np.concatenate([-ar_coeffs[::-1], [1.0]])
        roots = np.roots(poly_coeffs)
        min_root_magnitude = np.min(np.abs(roots)) if len(roots) > 0 else float("inf")

        if min_root_magnitude <= 1.0:
            raise ValueError(
                f"AR polynomial is non-stationary: smallest root magnitude = "
                f"{min_root_magnitude:.4f} <= 1.0. "
                f"Provide AR parameters that satisfy stationarity conditions or "
                f"set validate_stationarity=False."
            )

    def _compute_burn_in(self) -> int:
        """Compute appropriate burn-in period based on AR persistence.

        Returns:
            Recommended burn-in period
        """
        if self.burn_in is not None:
            return self.burn_in

        max_lag = max(len(self._full_ar_poly), len(self._full_ma_poly))

        # AR persistence close to 1 (near-integrated) needs longer burn-in
        if len(self._full_ar_poly) > 0:
            ar_persistence = np.sum(self._full_ar_poly)
            if ar_persistence > 0.9:
                persistence_multiplier = int(10 / (1 - ar_persistence + 0.01))
            else:
                persistence_multiplier = 5
        else:
            persistence_multiplier = 2

        return max(100, max_lag * persistence_multiplier)

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return (
            np.array(
                [
                    float(self.d),
                    float(self.D),
                    float(self.seasonal_period),
                    self.mean,
                    self.drift,
                    self.noise_std,
                    float(self._computed_burn_in),
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [
                np.asarray(self._full_ar_poly, dtype=np.float64),
                np.asarray(self._full_ma_poly, dtype=np.float64),
            ],
        )

    def generate_single_series(
        self, length: int, exog: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate values for a single SARIMA time series.

        The generation process:
        1. Generate white noise innovations
        2. Apply MA filtering to get MA component
        3. Apply AR filtering recursively
        4. Apply inverse differencing to get integrated process
        5. Add mean/drift and exogenous effects

        Args:
            length (int): The length of the series to generate
            exog (np.ndarray | None): Exogenous regressors of shape (length, n_exog)

        Returns:
            Array of time series values
        """
        if exog is None:
            seed = int(self.rng.integers(0, 2**63))
            return _rs_stat.sarima(
                length,
                self._full_ar_poly,
                self._full_ma_poly,
                self.d,
                self.D,
                self.seasonal_period,
                self.mean,
                self.drift,
                self.noise_std,
                self._computed_burn_in,
                seed,
                self._rs_innov_dist,
                self._rs_innov_param,
            )

        burn_in = self._computed_burn_in
        total_diff = self.d + self.D * self.seasonal_period
        total_length = length + burn_in + total_diff

        innovations = self._sample_innovations(total_length, scale=self.noise_std)
        ma_filtered = self._apply_ma_filter(innovations)
        arma_series = self._apply_ar_filter(ma_filtered)

        if self.d > 0 or self.D > 0:
            # Constant in the differenced equation -> trend after integration
            arma_series = arma_series + self.drift
        else:
            arma_series = arma_series + self.mean

        integrated_series = self._apply_inverse_differencing(arma_series)
        result = integrated_series[burn_in + total_diff : burn_in + total_diff + length]

        if exog is not None and self._exog_coefficients_array is not None:
            if exog.shape[0] != length:
                raise ValueError(
                    f"exog has {exog.shape[0]} rows but series length is {length}"
                )
            if exog.shape[1] != len(self._exog_coefficients_array):
                raise ValueError(
                    f"exog has {exog.shape[1]} columns but "
                    f"{len(self._exog_coefficients_array)} coefficients provided"
                )
            result = result + exog @ self._exog_coefficients_array

        return result

    def _apply_ma_filter(self, innovations: np.ndarray) -> np.ndarray:
        """Apply MA filtering: u_t = ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}.

        Args:
            innovations: White noise innovations.

        Returns:
            MA-filtered series.
        """
        if len(self._full_ma_poly) == 0:
            return innovations.copy()
        kernel = np.concatenate(([1.0], self._full_ma_poly))
        return np.convolve(innovations, kernel)[: len(innovations)]

    def _apply_ar_filter(self, ma_series: np.ndarray) -> np.ndarray:
        """Apply AR filtering: y_t = φ₁y_{t-1} + ... + φ_py_{t-p} + u_t.

        Args:
            ma_series: MA-filtered series (u_t).

        Returns:
            ARMA series (y_t).
        """
        if len(self._full_ar_poly) == 0:
            return ma_series.copy()

        ar_coeffs = np.asarray(self._full_ar_poly)
        result = np.empty_like(
            ma_series, dtype=np.result_type(ma_series, ar_coeffs, np.float64)
        )
        for t, value in enumerate(ma_series):
            lag_count = min(t, len(ar_coeffs))
            if lag_count:
                value += np.dot(ar_coeffs[:lag_count], result[t - lag_count : t][::-1])
            result[t] = value
        return result

    def _apply_inverse_differencing(self, series: np.ndarray) -> np.ndarray:
        """Apply inverse differencing (cumulative sums) to recover integrated process.

        For d=1: y_t = y_{t-1} + Δy_t (cumsum)
        For D=1, s=12: y_t = y_{t-12} + Δ_12 y_t (seasonal cumsum)

        The order of operations:
        1. Apply seasonal inverse differencing D times
        2. Apply regular inverse differencing d times

        Args:
            series: Differenced series (ARMA output)

        Returns:
            Integrated series
        """
        result = series.copy()
        s = self.seasonal_period

        for _ in range(self.D):
            # y_t = y_{t-s} + x_t with zero initial conditions
            integrated = result.copy()
            for t in range(s, len(result)):
                integrated[t] += integrated[t - s]
            result = integrated

        for _ in range(self.d):
            result = np.cumsum(result)

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the SARIMA model configuration.

        Returns:
            Model information including orders, parameters, and polynomial structure
        """
        # A model without seasonal terms is plain ARIMA; the seasonal orders
        # and period would be misleading in the label.
        if self.P == 0 and self.D == 0 and self.Q == 0:
            model = f"ARIMA({self.p},{self.d},{self.q})"
        else:
            model = (
                f"SARIMA({self.p},{self.d},{self.q})"
                f"({self.P},{self.D},{self.Q})[{self.seasonal_period}]"
            )
        return {
            "model": model,
            "ar_params": (
                self._ar_params_array.tolist()
                if self._ar_params_array is not None and len(self._ar_params_array) > 0
                else None
            ),
            "ma_params": (
                self._ma_params_array.tolist()
                if self._ma_params_array is not None and len(self._ma_params_array) > 0
                else None
            ),
            "seasonal_ar_params": (
                self._seasonal_ar_params_array.tolist()
                if self._seasonal_ar_params_array is not None
                and len(self._seasonal_ar_params_array) > 0
                else None
            ),
            "seasonal_ma_params": (
                self._seasonal_ma_params_array.tolist()
                if self._seasonal_ma_params_array is not None
                and len(self._seasonal_ma_params_array) > 0
                else None
            ),
            "full_ar_polynomial_lags": (
                list(range(1, len(self._full_ar_poly) + 1))
                if len(self._full_ar_poly) > 0
                else []
            ),
            "full_ar_polynomial_coeffs": (
                self._full_ar_poly.tolist() if len(self._full_ar_poly) > 0 else []
            ),
            "full_ma_polynomial_lags": (
                list(range(1, len(self._full_ma_poly) + 1))
                if len(self._full_ma_poly) > 0
                else []
            ),
            "full_ma_polynomial_coeffs": (
                self._full_ma_poly.tolist() if len(self._full_ma_poly) > 0 else []
            ),
            "mean": self.mean if (self.d == 0 and self.D == 0) else None,
            "drift": self.drift if (self.d > 0 or self.D > 0) else None,
            "noise_std": self.noise_std,
            "burn_in": self._computed_burn_in,
            "exog_coefficients": (
                self._exog_coefficients_array.tolist()
                if self._exog_coefficients_array is not None
                else None
            ),
        }
