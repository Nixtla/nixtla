"""Copula generator for multivariate time series with dependency structures."""

from typing import Any, Literal

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast._distributions import (
    expon_ppf,
    gamma_ppf,
    lognorm_ppf,
    norm_cdf,
    norm_ppf,
    t_cdf,
    uniform_ppf,
)
from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

try:
    from synforecast._lib import multivariate as _rs_mv

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Clamp copula uniforms away from {0, 1} so inverse-CDF transforms stay
# finite (matches the Rust implementation in rust/src/generators/multivariate.rs).
_U_EPS = 1e-15


class CopulaGenerator(BaseGenerator):
    """Generate correlated time series using Gaussian or t copulas.

    Copulas model the dependence structure between variables independently
    of their marginal distributions. Sampling proceeds in two steps:

    1. Draw correlated uniforms from the copula. Gaussian copula:
       ``z ~ N(0, R)``, ``u_i = Phi(z_i)``. t copula: ``z ~ N(0, R)``,
       ``w ~ chi2(df)``, ``u_i = T_df(z_i * sqrt(df / w))`` (the chi-square
       mixing is shared across variables, which creates tail dependence).
    2. Map each uniform through the inverse CDF of its marginal:
       ``x_i = F_i^{-1}(u_i)``.

    For the Gaussian copula the rank correlations satisfy
    ``spearman = (6 / pi) * arcsin(rho / 2)`` and for both copulas
    ``kendall_tau = (2 / pi) * arcsin(rho)``.

    ``generate(n_series)`` creates ``n_series`` correlated variables sharing
    one length; each variable is one ``unique_id`` in the long-format output.
    Samples are i.i.d. over time (no serial dependence).

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data. A pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS') or an integer time index step.
        copula_type (str): 'gaussian' or 't' (default: 'gaussian').
        correlation_matrix (list[list[float]] | None): Correlation matrix.
            Must be symmetric positive definite with unit diagonal. When
            None, a random correlation matrix is generated. When smaller
            than n_series, it is padded with an identity block (extra
            variables are independent); when larger, the leading principal
            submatrix is used.
        df (float): Degrees of freedom for the t copula (default: 5.0).
        marginal_distributions (list[dict]): Marginal specs, cycled over
            variables. Types: 'normal' (loc, scale), 'lognormal' (mean,
            sigma), 'exponential' (scale), 'uniform' (low, high),
            'gamma' (shape, scale). Default: standard normal.
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp (default: '2000-01-01').
    """

    copula_type: Literal["gaussian", "t"] = Field(
        default="gaussian", description="Type of copula"
    )
    correlation_matrix: list[list[float]] | None = Field(
        default=None, description="Correlation matrix for n_series"
    )
    df: float = Field(default=5.0, gt=0, description="Degrees of freedom for t-copula")
    marginal_distributions: list[dict[str, Any]] = Field(
        default=[{"type": "normal", "loc": 0.0, "scale": 1.0}],
        description="Marginal distributions",
    )

    _correlation_matrix_array: Any = None

    @model_validator(mode="after")
    def validate_copula_params(self) -> "CopulaGenerator":
        """Validate copula parameters."""
        if self.correlation_matrix is not None:
            corr_array = np.array(self.correlation_matrix, dtype=np.float64)
            if corr_array.ndim != 2 or corr_array.shape[0] != corr_array.shape[1]:
                raise ValueError("correlation_matrix must be square")
            if not np.allclose(corr_array, corr_array.T, atol=1e-8):
                raise ValueError("correlation_matrix must be symmetric")
            if not np.allclose(np.diag(corr_array), 1.0, atol=1e-8):
                raise ValueError("correlation_matrix must have unit diagonal")
            if not self._is_positive_definite(corr_array):
                raise ValueError("correlation_matrix must be positive definite")
            object.__setattr__(self, "_correlation_matrix_array", corr_array)
        return self

    def _generate_correlation_matrix(self, n_variables: int) -> np.ndarray:
        """Generate a random positive definite correlation matrix.

        Args:
            n_variables (int): Size of the correlation matrix.

        Returns:
            np.ndarray: Correlation matrix of shape (n_variables, n_variables).
        """
        A = self.rng.uniform(-1, 1, (n_variables, n_variables))
        # A @ A.T is PSD; normalizing by the diagonal yields unit diagonal
        R = A @ A.T
        D = np.diag(1.0 / np.sqrt(np.diag(R)))
        return D @ R @ D

    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def _copula_to_uniform(self, length: int, n_variables: int) -> np.ndarray:
        """Draw uniform(0, 1) samples with the copula's dependence structure.

        Args:
            length (int): Number of samples to generate.
            n_variables (int): Number of variables (series) to generate.

        Returns:
            np.ndarray: Array of shape (length, n_variables) with uniform
                marginals, clamped to (0, 1) exclusive.
        """
        if self._correlation_matrix_array is None:
            corr_matrix = self._generate_correlation_matrix(n_variables)
        elif n_variables <= self._correlation_matrix_array.shape[0]:
            corr_matrix = self._correlation_matrix_array[:n_variables, :n_variables]
        else:
            # Pad with an identity block: extra variables are independent
            corr_matrix = np.eye(n_variables)
            size = self._correlation_matrix_array.shape[0]
            corr_matrix[:size, :size] = self._correlation_matrix_array

        mean = np.zeros(n_variables)
        if self.copula_type == "gaussian":
            samples = self.rng.multivariate_normal(mean, corr_matrix, size=length)
            uniform_samples = norm_cdf(samples)
        else:  # t copula
            chi2 = self.rng.chisquare(self.df, size=length)
            z = self.rng.multivariate_normal(mean, corr_matrix, size=length)
            # Shared mixing variable: t_i = z_i * sqrt(df / chi2) ~ t(df)
            samples = z * np.sqrt(self.df / chi2)[:, np.newaxis]
            uniform_samples = t_cdf(samples, df=self.df)

        # Clamp so extreme draws cannot saturate the CDF to exactly 0 or 1,
        # which would map to +/-inf under the inverse-CDF marginal transforms
        return np.clip(uniform_samples, _U_EPS, 1.0 - _U_EPS)

    def _transform_marginals(
        self, uniform_samples: np.ndarray, n_variables: int
    ) -> np.ndarray:
        """Transform uniform samples to the configured marginal distributions.

        Args:
            uniform_samples (np.ndarray): Array of shape (length, n_variables)
                with uniform(0, 1) marginals.
            n_variables (int): Number of variables to transform.

        Returns:
            np.ndarray: Array of shape (length, n_variables) with the
                requested marginals.
        """
        transformed = np.zeros_like(uniform_samples)

        for i in range(n_variables):
            marginal = self.marginal_distributions[i % len(self.marginal_distributions)]
            dist_type = marginal["type"]
            u = uniform_samples[:, i]

            if dist_type == "normal":
                transformed[:, i] = norm_ppf(
                    u, loc=marginal.get("loc", 0.0), scale=marginal.get("scale", 1.0)
                )
            elif dist_type == "lognormal":
                transformed[:, i] = lognorm_ppf(
                    u,
                    s=marginal.get("sigma", 1.0),
                    scale=np.exp(marginal.get("mean", 0.0)),
                )
            elif dist_type == "exponential":
                transformed[:, i] = expon_ppf(u, scale=marginal.get("scale", 1.0))
            elif dist_type == "uniform":
                low = marginal.get("low", 0.0)
                high = marginal.get("high", 1.0)
                transformed[:, i] = uniform_ppf(u, loc=low, scale=high - low)
            elif dist_type == "gamma":
                transformed[:, i] = gamma_ppf(
                    u,
                    shape=marginal.get("shape", 2.0),
                    scale=marginal.get("scale", 1.0),
                )
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

        return transformed

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single univariate series.

        Provided for compatibility with BaseGenerator; use generate(n_series)
        for multivariate output.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values.
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            copula_type = 0 if self.copula_type == "gaussian" else 1
            marginal = self.marginal_distributions[0]
            dist_type = marginal["type"]
            marginal_map = {
                "normal": 0,
                "lognormal": 1,
                "exponential": 2,
                "uniform": 3,
                "gamma": 4,
            }
            marginal_type = marginal_map.get(dist_type, 0)
            if dist_type == "normal":
                marginal_param1 = float(marginal.get("loc", 0.0))
                marginal_param2 = float(marginal.get("scale", 1.0))
            elif dist_type == "lognormal":
                marginal_param1 = float(marginal.get("sigma", 1.0))
                marginal_param2 = float(np.exp(marginal.get("mean", 0.0)))
            elif dist_type == "exponential":
                marginal_param1 = float(marginal.get("scale", 1.0))
                marginal_param2 = 0.0
            elif dist_type == "uniform":
                marginal_param1 = float(marginal.get("low", 0.0))
                marginal_param2 = float(
                    marginal.get("high", 1.0) - marginal.get("low", 0.0)
                )
            elif dist_type == "gamma":
                marginal_param1 = float(marginal.get("shape", 2.0))
                marginal_param2 = float(marginal.get("scale", 1.0))
            else:
                marginal_param1 = 0.0
                marginal_param2 = 1.0
            # A single variable only needs the 1x1 correlation matrix [[1.0]]
            corr_flat = np.array([1.0], dtype=np.float64)
            return _rs_mv.copula(
                length,
                1,
                copula_type,
                float(self.df),
                corr_flat,
                marginal_type,
                marginal_param1,
                marginal_param2,
                seed,
            )

        uniform_samples = self._copula_to_uniform(length, n_variables=1)
        samples = self._transform_marginals(uniform_samples, n_variables=1)
        return samples[:, 0]

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,  # noqa: ARG002
    ) -> IntoDataFrameT:
        """Generate n_series correlated series with copula dependence.

        Overrides the base generate() for multivariate output: n_series is
        the number of correlated variables, all sharing a single length.
        Generation is inherently joint, so n_jobs has no effect.

        Args:
            n_series (int): Number of correlated series (variables).
            start_id (int): Starting ID for series numbering (default: 0).
            n_jobs (int): Unused (accepted for API compatibility).

        Returns:
            DataFrame in long format with columns [id_col, time_col,
            target_col]; each unique_id is one correlated variable.
        """
        length = self.rng.integers(self.min_length, self.max_length + 1)

        uniform_samples = self._copula_to_uniform(length, n_series)
        samples = self._transform_marginals(uniform_samples, n_series)

        timestamps = self._timestamps(length)
        all_metadata: list[SeriesMetadata] = []

        for i in range(n_series):
            values = np.ascontiguousarray(samples[:, i])
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
                )
            )
        return self._build_dataframe(all_metadata)
