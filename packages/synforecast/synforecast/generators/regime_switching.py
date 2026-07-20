"""Regime Switching (Markov-Switching) time series generator."""

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

try:
    from synforecast._lib import volatility as _rs_vol

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class RegimeSwitchingGenerator(BaseGenerator):
    """Generate time series with Markov regime-switching dynamics.

    A hidden regime s_t follows a first-order Markov chain with transition
    matrix P (rows sum to 1). Conditional on the regime, values follow an
    AR(1) around the regime mean:

        y_t = mu_{s_t} + phi_{s_t} * (y_{t-1} - mu_{s_t}) + sigma_{s_t} * eps_t

    When no initial regime is given, s_0 is drawn from the stationary
    distribution pi of P (pi = pi P), so long-run regime occupancy matches pi.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias (e.g. 'D', 'h', '5min') or an
            integer time step.
        n_regimes (int): Number of regimes/states (default: 2).
        regime_means (list[float] | None): Mean per regime (default: spread
            across levels).
        regime_variances (list[float] | None): Variance per regime (default:
            linspace(0.5, 2.0)).
        regime_ar_coeffs (list[float] | None): AR(1) coefficient per regime,
            each |phi| < 1 (default: 0).
        transition_matrix (list[list[float]] | None): Row-stochastic regime
            transition matrix (default: 0.95 self-transition probability).
        initial_regime (int | None): Starting regime, 0-indexed (default:
            drawn from the stationary distribution).
        seed (int | None): Random seed for reproducibility (default: None).

    Example:
        >>> gen = RegimeSwitchingGenerator(
        ...     min_length=100,
        ...     max_length=200,
        ...     freq="D",
        ...     n_regimes=2,
        ...     regime_means=[0.0, 5.0],
        ...     regime_variances=[1.0, 4.0],
        ...     transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    n_regimes: int = Field(default=2, ge=2, description="Number of regimes/states")
    regime_means: list[float] | None = Field(
        default=None, description="Mean for each regime"
    )
    regime_variances: list[float] | None = Field(
        default=None, description="Variance for each regime"
    )
    regime_ar_coeffs: list[float] | None = Field(
        default=None, description="AR(1) coefficient for each regime"
    )
    transition_matrix: list[list[float]] | None = Field(
        default=None, description="Regime transition probability matrix"
    )
    initial_regime: int | None = Field(
        default=None, description="Starting regime (0-indexed)"
    )

    _regime_means_array: np.ndarray | None = None
    _regime_variances_array: np.ndarray | None = None
    _regime_ar_coeffs_array: np.ndarray | None = None
    _transition_matrix_array: np.ndarray | None = None

    @model_validator(mode="after")
    def setup_regime_parameters(self) -> "RegimeSwitchingGenerator":
        """Initialize and validate regime parameters."""
        n = self.n_regimes

        if self.regime_means is None:
            means = np.linspace(-2.0, 2.0, n) * (n / 2)
        else:
            means = np.array(self.regime_means)
            if len(means) != n:
                raise ValueError(f"regime_means must have {n} elements")
        object.__setattr__(self, "_regime_means_array", means)

        if self.regime_variances is None:
            variances = np.linspace(0.5, 2.0, n)
        else:
            variances = np.array(self.regime_variances)
            if len(variances) != n:
                raise ValueError(f"regime_variances must have {n} elements")
            if np.any(variances <= 0):
                raise ValueError("regime_variances must all be positive")
        object.__setattr__(self, "_regime_variances_array", variances)

        if self.regime_ar_coeffs is None:
            ar_coeffs = np.zeros(n)
        else:
            ar_coeffs = np.array(self.regime_ar_coeffs)
            if len(ar_coeffs) != n:
                raise ValueError(f"regime_ar_coeffs must have {n} elements")
            if np.any(np.abs(ar_coeffs) >= 1):
                raise ValueError("regime_ar_coeffs must all have |φ| < 1 for stability")
        object.__setattr__(self, "_regime_ar_coeffs_array", ar_coeffs)

        if self.transition_matrix is None:
            trans = np.full((n, n), 0.05 / (n - 1))
            np.fill_diagonal(trans, 0.95)
        else:
            trans = np.array(self.transition_matrix)
            if trans.shape != (n, n):
                raise ValueError(f"transition_matrix must be shape ({n}, {n})")
            if not np.allclose(trans.sum(axis=1), 1.0):
                raise ValueError("transition_matrix rows must sum to 1")
            if np.any(trans < 0) or np.any(trans > 1):
                raise ValueError("transition_matrix elements must be in [0, 1]")
        object.__setattr__(self, "_transition_matrix_array", trans)

        if self.initial_regime is not None and (
            self.initial_regime < 0 or self.initial_regime >= n
        ):
            raise ValueError(f"initial_regime must be in [0, {n - 1}]")

        return self

    def _get_stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution of the Markov chain.

        Returns:
            np.ndarray: Stationary probability distribution over regimes
        """
        P = self._transition_matrix_array
        n = self.n_regimes

        # Solve pi = pi P, i.e. (P^T - I) pi = 0 with sum(pi) = 1,
        # via least squares for numerical robustness
        A = np.vstack([P.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0

        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0, 1)
        pi = pi / pi.sum()

        return pi

    def _sample_next_regime(self, current_regime: int) -> int:
        """Sample the next regime given the current regime.

        Args:
            current_regime (int): Current regime index

        Returns:
            int: Next regime index
        """
        probs = self._transition_matrix_array[current_regime]
        return int(self.rng.choice(self.n_regimes, p=probs))

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        # initial_regime = -1 tells the Rust kernel to draw s_0 per series
        # from the stationary distribution using each series' own RNG.
        init_regime = -1 if self.initial_regime is None else self.initial_regime
        return (
            np.array(
                [
                    float(self.n_regimes),
                    float(init_regime),
                    float(self._rs_innov_dist),
                    self._rs_innov_param,
                ]
            ),
            [
                np.asarray(self._regime_means_array, dtype=np.float64),
                np.asarray(self._regime_variances_array, dtype=np.float64),
                np.asarray(self._regime_ar_coeffs_array, dtype=np.float64),
                np.asarray(self._transition_matrix_array.flatten(), dtype=np.float64),
                np.asarray(self._get_stationary_distribution(), dtype=np.float64),
            ],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single regime-switching time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            np.ndarray: Array of time series values
        """
        if _HAS_RUST:
            seed = int(self.rng.integers(0, 2**63))
            # initial_regime = -1: the kernel draws s_0 from the stationary
            # distribution with this series' RNG (same semantics as the
            # Python fallback below and as the batch path).
            init_regime = -1 if self.initial_regime is None else self.initial_regime
            return _rs_vol.regime_switching(
                length,
                self.n_regimes,
                self._regime_means_array,
                self._regime_variances_array,
                self._regime_ar_coeffs_array,
                self._transition_matrix_array.flatten(),
                self._get_stationary_distribution(),
                init_regime,
                seed,
                self._rs_innov_dist,
                self._rs_innov_param,
            )

        values = np.zeros(length)
        regimes = np.zeros(length, dtype=int)
        innovations = self._sample_innovations(length)

        if self.initial_regime is not None:
            current_regime = self.initial_regime
        else:
            pi = self._get_stationary_distribution()
            current_regime = int(self.rng.choice(self.n_regimes, p=pi))

        regimes[0] = current_regime
        mu = self._regime_means_array[current_regime]
        sigma = np.sqrt(self._regime_variances_array[current_regime])
        values[0] = mu + sigma * innovations[0]

        for t in range(1, length):
            current_regime = self._sample_next_regime(current_regime)
            regimes[t] = current_regime

            mu = self._regime_means_array[current_regime]
            sigma = np.sqrt(self._regime_variances_array[current_regime])
            phi = self._regime_ar_coeffs_array[current_regime]

            # y_t = mu + phi * (y_{t-1} - mu) + sigma * eps_t
            deviation = values[t - 1] - mu
            values[t] = mu + phi * deviation + sigma * innovations[t]

        return values

    def generate_with_regimes(
        self, n_series: int = 1, start_id: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate series and return both values and regime labels.

        Args:
            n_series (int): Number of series to generate (default: 1)
            start_id (int): Starting ID for series naming (default: 0)

        Returns:
            tuple: (values, regimes, series_ids) arrays
        """
        all_values = []
        all_regimes = []
        all_ids = []

        for i in range(n_series):
            length = self.rng.integers(self.min_length, self.max_length + 1)

            values = np.zeros(length)
            regimes = np.zeros(length, dtype=int)
            innovations = self._sample_innovations(length)

            if self.initial_regime is not None:
                current_regime = self.initial_regime
            else:
                pi = self._get_stationary_distribution()
                current_regime = int(self.rng.choice(self.n_regimes, p=pi))

            regimes[0] = current_regime
            mu = self._regime_means_array[current_regime]
            sigma = np.sqrt(self._regime_variances_array[current_regime])
            values[0] = mu + sigma * innovations[0]

            for t in range(1, length):
                current_regime = self._sample_next_regime(current_regime)
                regimes[t] = current_regime

                mu = self._regime_means_array[current_regime]
                sigma = np.sqrt(self._regime_variances_array[current_regime])
                phi = self._regime_ar_coeffs_array[current_regime]

                deviation = values[t - 1] - mu
                values[t] = mu + phi * deviation + sigma * innovations[t]

            all_values.append(values)
            all_regimes.append(regimes)
            all_ids.append(np.full(length, start_id + i))

        return (
            np.concatenate(all_values),
            np.concatenate(all_regimes),
            np.concatenate(all_ids),
        )

    def get_model_info(self) -> dict:
        """Get information about the regime-switching model.

        Returns:
            dict: Model parameters and characteristics
        """
        return {
            "n_regimes": self.n_regimes,
            "regime_means": self._regime_means_array.tolist(),
            "regime_variances": self._regime_variances_array.tolist(),
            "regime_ar_coeffs": self._regime_ar_coeffs_array.tolist(),
            "transition_matrix": self._transition_matrix_array.tolist(),
            "stationary_distribution": self._get_stationary_distribution().tolist(),
        }
