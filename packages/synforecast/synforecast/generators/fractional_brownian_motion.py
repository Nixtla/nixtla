"""Fractional Brownian Motion (fBm) time series generator."""

from typing import Literal

import numpy as np
from pydantic import Field

from synforecast._lib import multivariate as _rs_mv
from synforecast.base import BaseGenerator

# Non-FFT methods store an n x n covariance matrix; warn above this length.
_LENGTH_WARNING_THRESHOLD = 5000


class FractionalBrownianMotionGenerator(BaseGenerator):
    """Generate time series using Fractional Brownian Motion (fBm).

    fBm extends standard Brownian motion with a Hurst exponent H that
    controls long-range dependence:

    - H = 0.5: standard Brownian motion (independent increments)
    - H > 0.5: persistent/trending (positively correlated increments)
    - H < 0.5: anti-persistent/mean-reverting (negatively correlated)

    The increments (fractional Gaussian noise, fGn) are stationary with
    autocovariance ``gamma(k) = (sigma^2/2) * (|k+1|^{2H} - 2|k|^{2H} +
    |k-1|^{2H})``, and the path satisfies ``Var(B_H(t)) = sigma^2 * t^{2H}``.

    Warning:
        The 'cholesky' and 'hosking' methods have O(n^2) memory (an n x n
        covariance matrix); prefer the default 'fft' (Davies-Harte) method
        for long series.

    Args:
        hurst (float): Hurst exponent H in (0, 1) (default: 0.5).
        sigma (float): Volatility/scale of the increments (default: 1.0).
        method (str): Generation method: 'fft' (O(n log n), default),
            'cholesky' or 'hosking' (both exact, O(n^2) memory).
        return_increments (bool): Return fGn increments instead of the
            cumulative fBm path (default: False).
        initial_value (float): Starting value of the fBm path; ignored when
            return_increments=True (default: 0.0).

    Example:
        >>> gen = FractionalBrownianMotionGenerator(
        ...     min_length=100,
        ...     max_length=200,
        ...     freq="D",
        ...     hurst=0.8,  # H > 0.5: trending
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    hurst: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="Hurst exponent H in (0, 1)"
    )
    sigma: float = Field(default=1.0, gt=0.0, description="Volatility/scale parameter")
    method: Literal["cholesky", "hosking", "fft"] = Field(
        default="fft",
        description="Generation method ('fft' is O(n log n), others are O(n^2+))",
    )
    return_increments: bool = Field(
        default=False,
        description="Return fGn increments instead of cumulative fBm",
    )
    initial_value: float = Field(
        default=0.0, description="Starting value for the process"
    )

    def _autocovariance(self, k: int | np.ndarray) -> np.ndarray:
        """Autocovariance of fGn at lag(s) k.

        gamma(k) = (sigma^2/2) * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
        """
        H = self.hurst
        k = np.asarray(k, dtype=float)
        return (
            0.5
            * self.sigma**2
            * (
                np.abs(k + 1) ** (2 * H)
                - 2 * np.abs(k) ** (2 * H)
                + np.abs(k - 1) ** (2 * H)
            )
        )

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        method_map = {"cholesky": 0, "hosking": 1, "fft": 2}
        return (
            np.array(
                [
                    self.hurst,
                    self.sigma,
                    self.initial_value,
                    0.0 if self.return_increments else 1.0,
                    float(method_map[self.method]),
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate a single fBm series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            fBm path values (or fGn increments if
                return_increments=True).
        """
        seed = int(self.rng.integers(0, 2**63))
        method_map = {"cholesky": 0, "hosking": 1, "fft": 2}
        cumulative = not self.return_increments
        # The Rust kernel only applies initial_value in cumulative mode.
        return _rs_mv.fbm(
            length,
            self.hurst,
            self.sigma,
            self.initial_value,
            cumulative,
            method_map[self.method],
            seed,
        )

    def get_model_info(self) -> dict:
        """Return model parameters and qualitative behavior."""
        H = self.hurst
        if H < 0.5:
            behavior = "anti-persistent (mean-reverting)"
        elif H > 0.5:
            behavior = "persistent (trending)"
        else:
            behavior = "standard Brownian motion"

        return {
            "hurst_exponent": H,
            "sigma": self.sigma,
            "behavior": behavior,
            "long_range_dependence_exponent": 2 * H - 1 if H != 0.5 else 0.0,
            "method": self.method,
            "return_increments": self.return_increments,
        }

    def estimate_hurst(self, series: np.ndarray, method: str = "rs") -> float:
        """Estimate the Hurst exponent from an increment (fGn) series.

        Args:
            series: Time series of increments.
            method: 'rs' (rescaled range) or 'var' (variance of aggregates).

        Returns:
            Estimated Hurst exponent, clipped to [0.01, 0.99].
        """
        if method == "rs":
            return self._estimate_hurst_rs(series)
        elif method == "var":
            return self._estimate_hurst_variance(series)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_hurst_rs(self, series: np.ndarray) -> float:
        """Rescaled range (R/S) estimate: E[R/S](k) ~ k^H."""
        n = len(series)
        max_k = n // 4

        segment_sizes = []
        rs_values = []
        for k in range(10, max_k + 1, max(1, max_k // 20)):
            n_segments = n // k
            if n_segments < 2:
                continue

            rs_k = []
            for i in range(n_segments):
                segment = series[i * k : (i + 1) * k]
                cumdev = np.cumsum(segment - np.mean(segment))
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(segment, ddof=1)
                if S > 0:
                    rs_k.append(R / S)

            if rs_k:
                segment_sizes.append(k)
                rs_values.append(np.mean(rs_k))

        if len(segment_sizes) < 2:
            return 0.5

        slope, _ = np.polyfit(np.log(segment_sizes), np.log(rs_values), 1)
        return float(np.clip(slope, 0.01, 0.99))

    def _estimate_hurst_variance(self, series: np.ndarray) -> float:
        """Variance-of-aggregates estimate.

        Block sums of fGn satisfy Var(sum of m) = sigma^2 * m^{2H}, so the
        log-log slope of block-sum variance vs block size m is 2H.
        """
        max_m = len(series) // 10
        m_values = []
        var_values = []

        for m in range(1, max_m + 1, max(1, max_m // 20)):
            n_blocks = len(series) // m
            if n_blocks < 2:
                continue

            aggregated = np.array(
                [series[i * m : (i + 1) * m].sum() for i in range(n_blocks)]
            )
            var = np.var(aggregated, ddof=1)
            if var > 0:
                m_values.append(m)
                var_values.append(var)

        if len(m_values) < 2:
            return 0.5

        slope, _ = np.polyfit(np.log(m_values), np.log(var_values), 1)
        return float(np.clip(slope / 2, 0.01, 0.99))
