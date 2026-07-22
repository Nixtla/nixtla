"""VAR (Vector Autoregression) generator for multivariate time series."""

from typing import Any

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast._lib import multivariate as _rs_mv
from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

# Target companion spectral radius when rescaling random coefficients
_STABILITY_TARGET = 0.95
# Burn-in steps discarded so the process reaches its stationary distribution
_BURN_IN = 100


class VARGenerator(BaseGenerator):
    """Generate correlated time series using a Vector Autoregression model.

    A VAR(p) process models each variable as a linear function of past
    values of all variables:

        y[t] = c + A_1 y[t-1] + ... + A_p y[t-p] + e[t],  e[t] ~ (0, Sigma)

    The process is stable (stationary) iff the companion matrix

        [[A_1 ... A_p], [I 0 ... 0], ..., [0 ... I 0]]

    has spectral radius < 1, in which case the stationary mean is
    ``(I - A_1 - ... - A_p)^{-1} c``. Innovations are drawn from the
    configured innovation distribution and correlated via the Cholesky
    factor of Sigma. A burn-in of 100 steps is discarded.

    ``generate(n_series)`` creates ``n_series`` correlated variables sharing
    one length; each variable is one ``unique_id`` in the long-format output.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data. A pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS') or an integer time index step.
        lag_order (int): VAR lag order p (default: 1).
        coef_matrices (list[list[list[float]]] | None): One square
            coefficient matrix per lag, all the same size. Must define a
            stable VAR. When None, random stable coefficients are generated.
            When sized differently from n_series, the leading principal
            submatrices are used (or zero-padded).
        intercept (list[float] | None): Intercept vector c (default: zeros).
        innovation_covariance (list[list[float]] | None): Innovation
            covariance Sigma; symmetric positive definite (default: identity).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp (default: '2000-01-01').
    """

    lag_order: int = Field(default=1, ge=1, description="VAR lag order")
    coef_matrices: list[list[list[float]]] | None = Field(
        default=None, description="Coefficient matrices for each lag"
    )
    intercept: list[float] | None = Field(default=None, description="Intercept vector")
    innovation_covariance: list[list[float]] | None = Field(
        default=None, description="Covariance matrix of innovations"
    )

    _coef_matrices_arrays: Any = None
    _intercept_array: Any = None
    _innovation_covariance_array: Any = None

    @model_validator(mode="after")
    def validate_var_params(self) -> "VARGenerator":
        """Validate VAR parameters and convert to numpy arrays."""
        if self.coef_matrices is not None:
            coef_arrays = [
                np.array(mat, dtype=np.float64) for mat in self.coef_matrices
            ]
            if len(coef_arrays) != self.lag_order:
                raise ValueError(
                    f"Number of coefficient matrices {len(coef_arrays)} "
                    f"does not match lag_order {self.lag_order}"
                )
            for i, mat in enumerate(coef_arrays):
                if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                    raise ValueError(f"Coefficient matrix {i} must be square")
            sizes = [mat.shape[0] for mat in coef_arrays]
            if len(set(sizes)) > 1:
                raise ValueError("All coefficient matrices must have the same size")
            radius = self._companion_spectral_radius(coef_arrays)
            if radius >= 1.0:
                raise ValueError(
                    "coef_matrices define an unstable VAR: companion matrix "
                    f"spectral radius {radius:.4f} >= 1"
                )
            object.__setattr__(self, "_coef_matrices_arrays", coef_arrays)

        if self.intercept is not None:
            object.__setattr__(
                self, "_intercept_array", np.array(self.intercept, dtype=np.float64)
            )

        if self.innovation_covariance is not None:
            innov_cov = np.array(self.innovation_covariance, dtype=np.float64)
            if innov_cov.ndim != 2 or innov_cov.shape[0] != innov_cov.shape[1]:
                raise ValueError("innovation_covariance must be square")
            if not np.allclose(innov_cov, innov_cov.T, atol=1e-8):
                raise ValueError("innovation_covariance must be symmetric")
            try:
                np.linalg.cholesky(innov_cov)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    "innovation_covariance must be positive definite"
                ) from e
            object.__setattr__(self, "_innovation_covariance_array", innov_cov)

        return self

    @staticmethod
    def _companion_spectral_radius(coef_matrices: list[np.ndarray]) -> float:
        """Spectral radius of the VAR companion matrix.

        Args:
            coef_matrices (list[np.ndarray]): Coefficient matrices, one per lag.

        Returns:
            float: Maximum eigenvalue modulus; the VAR is stable iff < 1.
        """
        p = len(coef_matrices)
        k = coef_matrices[0].shape[0]
        companion = np.zeros((k * p, k * p))
        for i, mat in enumerate(coef_matrices):
            companion[0:k, i * k : (i + 1) * k] = mat
        if p > 1:
            companion[k:, 0:-k] = np.eye(k * (p - 1))
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def _generate_stable_coefficients(self, n_variables: int) -> list[np.ndarray]:
        """Generate random coefficient matrices that satisfy stability.

        Args:
            n_variables (int): Number of variables in the VAR system.

        Returns:
            list[np.ndarray]: Coefficient matrices, one per lag, with
                companion spectral radius <= 0.95.
        """
        coef_matrices = [
            self.rng.uniform(-0.3, 0.3, (n_variables, n_variables)) * 0.7**lag
            for lag in range(self.lag_order)
        ]
        radius = self._companion_spectral_radius(coef_matrices)
        if radius >= _STABILITY_TARGET:
            # Scaling A_l by s^l maps every companion eigenvalue z to s*z,
            # so this rescale sets the spectral radius exactly to the target
            s = _STABILITY_TARGET / radius
            coef_matrices = [
                mat * s ** (lag + 1) for lag, mat in enumerate(coef_matrices)
            ]
        return coef_matrices

    def _resolve_parameters(
        self, n_variables: int
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Resolve (coef_matrices, intercept, innovation_covariance) for
        n_variables, generating defaults and subsetting/padding user inputs.

        Args:
            n_variables (int): Number of variables in the VAR system.

        Returns:
            tuple: (coefficient matrices, intercept vector, innovation
                covariance matrix), all sized for n_variables.
        """
        if self._coef_matrices_arrays is None:
            coef_matrices = self._generate_stable_coefficients(n_variables)
        elif n_variables <= self._coef_matrices_arrays[0].shape[0]:
            coef_matrices = [
                mat[:n_variables, :n_variables] for mat in self._coef_matrices_arrays
            ]
        else:
            coef_matrices = []
            size = self._coef_matrices_arrays[0].shape[0]
            for mat in self._coef_matrices_arrays:
                new_mat = np.zeros((n_variables, n_variables))
                new_mat[:size, :size] = mat
                coef_matrices.append(new_mat)

        if self._intercept_array is None:
            intercept = np.zeros(n_variables)
        elif n_variables <= len(self._intercept_array):
            intercept = self._intercept_array[:n_variables]
        else:
            intercept = np.zeros(n_variables)
            intercept[: len(self._intercept_array)] = self._intercept_array

        if self._innovation_covariance_array is None:
            innovation_covariance = np.eye(n_variables)
        elif n_variables <= self._innovation_covariance_array.shape[0]:
            innovation_covariance = self._innovation_covariance_array[
                :n_variables, :n_variables
            ]
        else:
            innovation_covariance = np.eye(n_variables)
            size = self._innovation_covariance_array.shape[0]
            innovation_covariance[:size, :size] = self._innovation_covariance_array

        return coef_matrices, intercept, innovation_covariance

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single (univariate) VAR series.

        Provided for compatibility with BaseGenerator; use generate(n_series)
        for multivariate output.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Array of time series values (univariate).
        """
        seed = int(self.rng.integers(0, 2**63))
        coef_matrices, intercept, innovation_cov = self._resolve_parameters(1)
        coef_flat = np.concatenate(
            [np.asarray(m, dtype=np.float64).ravel() for m in coef_matrices]
        )
        return _rs_mv.var_process(
            length,
            1,
            self.lag_order,
            coef_flat,
            np.asarray(intercept, dtype=np.float64).ravel(),
            np.asarray(innovation_cov, dtype=np.float64).ravel(),
            seed,
            self._rs_innov_dist,
            self._rs_innov_param,
        )

    def _generate_multivariate(self, length: int, n_variables: int) -> np.ndarray:
        """Simulate the VAR process.

        Args:
            length (int): The length of the series to generate.
            n_variables (int): Number of variables in the VAR system.

        Returns:
            np.ndarray: Array of shape (length, n_variables).
        """
        coef_matrices, intercept, innovation_covariance = self._resolve_parameters(
            n_variables
        )

        total_length = length + _BURN_IN
        series = np.zeros((total_length, n_variables))
        series[: self.lag_order] = self.rng.normal(
            0, 0.1, (self.lag_order, n_variables)
        )

        # cov(z @ L.T) = L L' = Sigma for iid unit-variance rows z
        L = np.linalg.cholesky(innovation_covariance)
        raw_innovations = self._sample_innovations((total_length, n_variables))
        innovations = raw_innovations @ L.T

        for t in range(self.lag_order, total_length):
            value = intercept + innovations[t]
            for lag in range(self.lag_order):
                value += coef_matrices[lag] @ series[t - lag - 1]
            series[t] = value

        return series[_BURN_IN:]

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,  # noqa: ARG002
    ) -> IntoDataFrameT:
        """Generate n_series correlated time series using the VAR model.

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
        samples = self._generate_multivariate(length, n_series)

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
