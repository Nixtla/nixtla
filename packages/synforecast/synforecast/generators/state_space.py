"""State Space Model generator with custom transition and observation equations."""

from collections.abc import Callable
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, PrivateAttr, model_validator

from synforecast._lib import multivariate as _rs_mv
from synforecast.base import BaseGenerator, _categorize_ids
from synforecast.exogenous import SeriesMetadata


def _psd_factor(matrix: np.ndarray, name: str) -> np.ndarray:
    """Return A with A @ A.T == matrix for a symmetric PSD matrix.

    Uses Cholesky when the matrix is positive definite, otherwise an
    eigendecomposition with eigenvalues clamped at zero (allows singular
    covariances, e.g. deterministic state components). Raises ValueError
    for matrices with significantly negative eigenvalues.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        if eigvals.min() < -1e-8 * max(1.0, abs(eigvals.max())):
            raise ValueError(f"{name} must be positive semidefinite") from None
        return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


class StateSpaceGenerator(BaseGenerator):
    """Generate time series from a (linear-Gaussian or custom) state space model.

    The linear model is:

        x[t] = F x[t-1] + w[t],  w[t] ~ (0, Q)   (state equation)
        y[t] = H x[t] + v[t],    v[t] ~ N(0, R)  (observation equation)

    with ``x[0] ~ N(initial_state, initial_state_covariance)``. State noise
    ``w`` follows the configured innovation distribution (scaled by the
    Cholesky/PSD factor of Q); observation noise ``v`` is Gaussian. The
    univariate output is the first observation dimension ``y[t][0]``, with
    y[0] observing the initial state. Custom nonlinear dynamics can be
    supplied via ``transition_fn`` / ``observation_fn``, each called as
    ``fn(x, t, rng)``.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data. A pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS') or an integer time index step.
        state_dim (int): Dimension of the hidden state vector (default: 1).
        obs_dim (int): Dimension of the observation vector (default: 1).
        transition_matrix (list[list[float]] | None): State transition
            matrix F, shape (state_dim, state_dim). When None (and no
            transition_fn), a random stable matrix is generated.
        observation_matrix (list[list[float]] | None): Observation matrix H,
            shape (obs_dim, state_dim). Default observes the first state.
        state_covariance (list[list[float]] | None): State noise covariance
            Q, symmetric PSD (default: 0.1 * I).
        obs_covariance (list[list[float]] | None): Observation noise
            covariance R, symmetric PSD (default: 0.1 * I).
        transition_fn (Callable | None): Custom state transition function.
        observation_fn (Callable | None): Custom observation function.
        initial_state (list[float] | None): Initial state mean (default: zeros).
        initial_state_covariance (list[list[float]] | None): Initial state
            covariance, symmetric PSD (default: identity).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp (default: '2000-01-01').
    """

    state_dim: int = Field(
        default=1, ge=1, description="Dimension of hidden state vector"
    )
    obs_dim: int = Field(default=1, ge=1, description="Dimension of observation vector")

    transition_matrix: list[list[float]] | None = Field(
        default=None, description="State transition matrix F"
    )
    observation_matrix: list[list[float]] | None = Field(
        default=None, description="Observation matrix H"
    )
    state_covariance: list[list[float]] | None = Field(
        default=None, description="State noise covariance Q"
    )
    obs_covariance: list[list[float]] | None = Field(
        default=None, description="Observation noise covariance R"
    )

    transition_fn: Callable | None = Field(
        default=None, exclude=True, description="Custom state transition function"
    )
    observation_fn: Callable | None = Field(
        default=None, exclude=True, description="Custom observation function"
    )

    initial_state: list[float] | None = Field(
        default=None, description="Initial state x[0]"
    )
    initial_state_covariance: list[list[float]] | None = Field(
        default=None, description="Initial state uncertainty"
    )

    # Internal NumPy arrays are runtime state, not user configuration.
    _transition_matrix_array: Any = PrivateAttr(default=None)
    _observation_matrix_array: Any = PrivateAttr(default=None)
    _state_covariance_array: Any = PrivateAttr(default=None)
    _obs_covariance_array: Any = PrivateAttr(default=None)
    _initial_state_array: Any = PrivateAttr(default=None)
    _initial_state_covariance_array: Any = PrivateAttr(default=None)

    # PSD factor of Q, precomputed for state-noise sampling
    _state_cov_cholesky: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_matrices(self) -> "StateSpaceGenerator":
        """Initialize and validate model matrices."""
        if self.transition_matrix is None and self.transition_fn is None:
            trans_mat = self._generate_stable_transition_matrix()
        elif self.transition_matrix is not None:
            trans_mat = np.array(self.transition_matrix, dtype=np.float64)
            if trans_mat.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"transition_matrix must be shape ({self.state_dim}, {self.state_dim})"
                )
        else:
            trans_mat = None
        self._transition_matrix_array = trans_mat

        if self.observation_matrix is None and self.observation_fn is None:
            obs_mat = np.zeros((self.obs_dim, self.state_dim))
            obs_mat[0, 0] = 1.0
        elif self.observation_matrix is not None:
            obs_mat = np.array(self.observation_matrix, dtype=np.float64)
            if obs_mat.shape != (self.obs_dim, self.state_dim):
                raise ValueError(
                    f"observation_matrix must be shape ({self.obs_dim}, {self.state_dim})"
                )
        else:
            obs_mat = None
        self._observation_matrix_array = obs_mat

        state_cov = (
            np.array(self.state_covariance, dtype=np.float64)
            if self.state_covariance is not None
            else np.eye(self.state_dim) * 0.1
        )
        if state_cov.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"state_covariance must be shape ({self.state_dim}, {self.state_dim})"
            )
        self._state_covariance_array = state_cov
        self._state_cov_cholesky = _psd_factor(state_cov, "state_covariance")

        obs_cov = (
            np.array(self.obs_covariance, dtype=np.float64)
            if self.obs_covariance is not None
            else np.eye(self.obs_dim) * 0.1
        )
        if obs_cov.shape != (self.obs_dim, self.obs_dim):
            raise ValueError(
                f"obs_covariance must be shape ({self.obs_dim}, {self.obs_dim})"
            )
        _psd_factor(obs_cov, "obs_covariance")
        self._obs_covariance_array = obs_cov

        if self.initial_state is None:
            init_state = np.zeros(self.state_dim)
        else:
            init_state = np.array(self.initial_state, dtype=np.float64)
            if init_state.shape != (self.state_dim,):
                raise ValueError(f"initial_state must have shape ({self.state_dim},)")
        self._initial_state_array = init_state

        init_cov = (
            np.array(self.initial_state_covariance, dtype=np.float64)
            if self.initial_state_covariance is not None
            else np.eye(self.state_dim)
        )
        if init_cov.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                "initial_state_covariance must be shape "
                f"({self.state_dim}, {self.state_dim})"
            )
        _psd_factor(init_cov, "initial_state_covariance")
        self._initial_state_covariance_array = init_cov

        return self

    def _generate_stable_transition_matrix(self) -> np.ndarray:
        """Generate a random stable state transition matrix.

        Returns:
            np.ndarray: Transition matrix with spectral radius < 1.
        """
        F = self.rng.uniform(-0.3, 0.3, (self.state_dim, self.state_dim))
        max_eigenvalue: float = np.max(np.abs(np.linalg.eigvals(F)))
        if max_eigenvalue >= 1.0:
            F = F * (0.9 / max_eigenvalue)
        return F

    def _state_transition(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        """Apply the state transition x[t] = F x[t-1] + w[t].

        Args:
            x_prev (np.ndarray): Previous state.
            t (int): Time index.

        Returns:
            np.ndarray: Next state.
        """
        if self.transition_fn is not None:
            return self.transition_fn(x_prev, t, self.rng)
        raw = self._sample_innovations(self.state_dim)
        state_noise = self._state_cov_cholesky @ raw
        return self._transition_matrix_array @ x_prev + state_noise

    def _observation(self, x: np.ndarray, t: int) -> np.ndarray:
        """Generate the observation y[t] = H x[t] + v[t].

        Args:
            x (np.ndarray): Current state.
            t (int): Time index.

        Returns:
            np.ndarray: Observation.
        """
        if self.observation_fn is not None:
            return self.observation_fn(x, t, self.rng)
        obs_noise = self.rng.multivariate_normal(
            np.zeros(self.obs_dim), self._obs_covariance_array
        )
        return self._observation_matrix_array @ x + obs_noise

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        if self.transition_fn is not None or self.observation_fn is not None:
            return None
        return (
            np.array(
                [
                    float(self.state_dim),
                    float(self.obs_dim),
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [
                np.asarray(self._transition_matrix_array, dtype=np.float64).ravel(),
                np.asarray(self._observation_matrix_array, dtype=np.float64).ravel(),
                np.asarray(self._state_covariance_array, dtype=np.float64).ravel(),
                np.asarray(self._obs_covariance_array, dtype=np.float64).ravel(),
                np.asarray(self._initial_state_array, dtype=np.float64).ravel(),
                np.asarray(
                    self._initial_state_covariance_array, dtype=np.float64
                ).ravel(),
            ],
        )

    def _simulate(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the model, returning (observations, states).

        Args:
            length (int): Number of time steps.

        Returns:
            tuple[np.ndarray, np.ndarray]: Observations of shape
                (length, obs_dim) and states of shape (length, state_dim).
        """
        x = self._initial_state_array + self.rng.multivariate_normal(
            np.zeros(self.state_dim), self._initial_state_covariance_array
        )
        observations = np.zeros((length, self.obs_dim))
        states = np.zeros((length, self.state_dim))
        for t in range(length):
            states[t] = x
            observations[t] = self._observation(x, t)
            x = self._state_transition(x, t)
        return observations, states

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single state space series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            np.ndarray: Observed values (first observation dimension).
        """
        if self.transition_fn is None and self.observation_fn is None:
            seed = int(self.rng.integers(0, 2**63))
            F = np.asarray(self._transition_matrix_array, dtype=np.float64).ravel()
            H = np.asarray(self._observation_matrix_array, dtype=np.float64).ravel()
            Q = np.asarray(self._state_covariance_array, dtype=np.float64).ravel()
            R = np.asarray(self._obs_covariance_array, dtype=np.float64).ravel()
            x0 = np.asarray(self._initial_state_array, dtype=np.float64).ravel()
            P0 = np.asarray(
                self._initial_state_covariance_array, dtype=np.float64
            ).ravel()
            return _rs_mv.state_space(
                length,
                self.state_dim,
                self.obs_dim,
                F,
                H,
                Q,
                R,
                x0,
                P0,
                seed,
                self._rs_innov_dist,
                self._rs_innov_param,
            )

        observations, _ = self._simulate(length)
        return observations[:, 0]

    def generate_with_states(
        self, n_series: int = 1, start_id: int = 0
    ) -> tuple[IntoDataFrameT, IntoDataFrameT]:
        """Generate series and return both observations and hidden states.

        Only missingness is applied to the observations (changepoints and
        anomalies would desynchronize them from the returned states).

        Args:
            n_series (int): Number of series to generate (default: 1).
            start_id (int): Starting ID for series numbering (default: 0).
        Returns:
            tuple: (observations DataFrame in long format, states DataFrame
                with one ``state_j`` column per state dimension).
        """
        length = int(self.rng.integers(self.min_length, self.max_length + 1))
        timestamps = self._timestamps(length)

        all_metadata: list[SeriesMetadata] = []
        all_states: list[np.ndarray] = []

        for i in range(n_series):
            observations, states = self._simulate(length)
            values, miss_indices = self._add_missingness(observations[:, 0])
            all_metadata.append(
                SeriesMetadata(
                    values=values,
                    timestamps=timestamps,
                    series_id=start_id + i,
                    length=length,
                    missing_indices=miss_indices,
                )
            )
            all_states.append(states)

        obs_df: Any = self._build_dataframe(all_metadata)

        states_result: dict[str, Any] = {
            self.id_col: np.repeat(
                np.arange(start_id, start_id + n_series, dtype=np.int64), length
            ),
            self.time_col: np.tile(timestamps, n_series),
            **{
                f"state_{j}": np.concatenate([s[:, j] for s in all_states])
                for j in range(self.state_dim)
            },
        }
        states_nw = nw.DataFrame.from_dict(states_result, backend=self.engine)
        states_df = _categorize_ids(states_nw, self.id_col).to_native()

        return obs_df, states_df
