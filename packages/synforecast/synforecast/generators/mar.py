"""MAR (mixture autoregressive) generator from the GRATIS recipe."""

from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from synforecast._lib import augmentation as _rs_augmentation
from synforecast.base import BaseGenerator

_MAX_ABS = 1e8
_MIN_STD = 1e-8
_MAX_RETRIES = 8
_MAX_MIXTURE_CHECK_ORDER = 32
_SUPPORTED_FEATURES = {
    "spectral_entropy",
    "trend_strength",
    "seasonal_strength",
    "acf1",
}


class MARGenerator(BaseGenerator):
    """Generate GRATIS-style mixtures of autoregressive components.

    The mixture autoregressive model is due to Wong and Li (2000, "On a
    mixture autoregressive model", Journal of the Royal Statistical Society
    Series B 62(1), https://doi.org/10.1111/1467-9868.00222). This generator
    adapts the MAR simulation used for diverse series by GRATIS (Kang,
    Hyndman, and Li, 2020, "GRATIS: GeneRAting TIme Series with diverse and
    controllable characteristics", https://arxiv.org/abs/1903.02787). At each
    step a component ``k`` is drawn from the mixture weights, then
    ``y_t = c_k + sum_i phi[k, i] y[t-i] + sigma_k eps_t``. The parameter
    sampling, partial-autocorrelation reparameterization via Durbin-Levinson,
    defaults, and stability guards are SynForecast's own design rather than a
    reproduction of the reference ``gratis`` R package.

    Random mode samples one to ``max_components`` components for every series;
    ``seasonal_period`` only affects random mode, where it enables a seasonal
    AR factor for a random subset of components. Fixed mode supplies all four
    of ``weights``, ``ar_coefficients``, ``intercepts``, and ``noise_scales``
    and is used by feature targeting. Every fixed component must be
    stationary, and the mixture must be second-order stationary (Wong and Li
    2000). The numerical mixture check handles effective AR orders through 32,
    using covariance contraction certificates and a NumPy dense eigenvalue
    fallback. Solver failures are rejected. Validation results are cached.
    Larger mixtures are accepted when all components are identical and stable,
    or every component's sum of absolute AR coefficients is below one.
    Other large mixtures raise an error because stationarity could not be
    verified; this conservative policy can reject valid mixtures. Trailing
    zero coefficients do not count toward effective order. A fixed model
    that fails the finite, bounded, non-constant output guards raises a
    ``ValueError`` at generation time instead of silently substituting noise.
    Moment handling is explicit: ``standardize=True`` (the default) maps every
    accepted multi-observation draw to zero mean and unit standard deviation,
    rather than retaining the MAR model's component-implied level and scale.
    Set ``standardize=False`` to preserve those raw simulated moments. This
    output normalization is SynForecast's own design and is not source-series
    moment matching. Bulk :meth:`generate` calls use the native Rust batch
    path; :meth:`generate_single_series` remains the Python reference path, so
    their seeded RNG streams are not bit-for-bit equivalent.
    """

    max_components: int = Field(
        default=3, ge=1, le=64, description="Maximum number of AR mixture components"
    )
    max_ar_order: int = Field(
        default=5, ge=1, le=100, description="Maximum non-seasonal AR order"
    )
    seasonal_period: int | None = Field(
        default=None,
        description="Optional multiplicative seasonal AR period (random mode only)",
    )
    weights_concentration: float = Field(
        default=1.0, gt=0, description="Symmetric Dirichlet concentration"
    )
    intercept_scale: float = Field(
        default=1.0, ge=0, description="Standard deviation of component intercepts"
    )
    noise_scale_range: tuple[float, float] = Field(
        default=(0.1, 2.0), description="Minimum and maximum innovation scales"
    )
    burn_in: int = Field(
        default=100, ge=0, le=1_000_000, description="Discarded simulation steps"
    )
    standardize: bool = Field(
        default=True, description="Standardize each generated series"
    )
    weights: list[float] | None = Field(
        default=None, description="Fixed component probabilities"
    )
    ar_coefficients: list[list[float]] | None = Field(
        default=None, description="Fixed AR coefficients by component"
    )
    intercepts: list[float] | None = Field(
        default=None, description="Fixed component intercepts"
    )
    noise_scales: list[float] | None = Field(
        default=None, description="Fixed positive innovation scales"
    )
    _tuning_diagnostics: dict[str, float | int | bool] | None = PrivateAttr(
        default=None
    )

    @property
    def tuning_diagnostics(self) -> dict[str, float | int | bool] | None:
        """Search distance, convergence, and budget used, or None if untuned.

        ``best_distance`` is the selected candidate's observed training fitness,
        not a guarantee about new draws. The returned dictionary is a copy.
        """
        return (
            None
            if self._tuning_diagnostics is None
            else self._tuning_diagnostics.copy()
        )

    @model_validator(mode="after")
    def validate_mar_parameters(self) -> "MARGenerator":
        """Validate seasonal, noise-range, and fixed-mode configuration."""
        if self.seasonal_period is not None and not 2 <= self.seasonal_period <= 10_000:
            raise ValueError("seasonal_period must be in [2, 10000] when provided")
        if not np.isfinite(self.weights_concentration):
            raise ValueError("weights_concentration must be finite")
        if not np.isfinite(self.intercept_scale):
            raise ValueError("intercept_scale must be finite")
        lo, hi = self.noise_scale_range
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi):
            raise ValueError(
                "noise_scale_range must be finite and satisfy 0 < low <= high"
            )

        fixed = (
            self.weights,
            self.ar_coefficients,
            self.intercepts,
            self.noise_scales,
        )
        if all(value is None for value in fixed):
            return self
        if any(value is None for value in fixed):
            raise ValueError(
                "weights, ar_coefficients, intercepts, and noise_scales must be "
                "provided together"
            )

        weights = self.weights
        coefficients = self.ar_coefficients
        intercepts = self.intercepts
        scales = self.noise_scales
        assert weights is not None
        assert coefficients is not None
        assert intercepts is not None
        assert scales is not None
        counts = {len(weights), len(coefficients), len(intercepts), len(scales)}
        if len(counts) != 1 or not weights:
            raise ValueError("fixed parameter component counts must match and be >= 1")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("weights must be finite and positive")
        if any(not np.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("noise_scales must be finite and positive")
        if any(not np.isfinite(value) for value in intercepts):
            raise ValueError("intercepts must be finite")
        for component in coefficients:
            if not component or not np.all(np.isfinite(component)):
                raise ValueError(
                    "each AR coefficient list must be non-empty and finite"
                )
            if not self._is_stationary(np.asarray(component, dtype=float)):
                raise ValueError("every fixed AR component must be stationary")
        total = float(sum(weights))
        if not np.isfinite(total):
            raise ValueError("weights must have a finite sum")
        normalized = [float(weight / total) for weight in weights]
        if not self._is_mixture_stationary(
            np.asarray(normalized), [np.asarray(c, dtype=float) for c in coefficients]
        ):
            raise ValueError("the fixed MAR mixture must be second-order stationary")
        object.__setattr__(self, "weights", normalized)
        return self

    @staticmethod
    def _is_stationary(coefficients: np.ndarray) -> bool:
        return MARGenerator._component_stationary_cached(
            tuple(np.trim_zeros(coefficients, "b"))
        )

    @staticmethod
    @lru_cache(maxsize=2048)
    def _component_stationary_cached(coefficients: tuple[float, ...]) -> bool:
        if not coefficients:
            return True
        # A strict absolute-sum bound avoids root finding for simple components.
        if sum(abs(c) for c in coefficients) < 1.0 - 1e-10:
            return True
        return bool(
            np.max(np.abs(np.roots(np.r_[1.0, -np.asarray(coefficients)])))
            < 1.0 - 1e-10
        )

    @staticmethod
    def _is_mixture_stationary(
        weights: np.ndarray, coefficients: list[np.ndarray]
    ) -> bool:
        """Wong-Li second-moment check, cached by normalized model parameters."""
        return MARGenerator._mixture_stationary_cached(
            tuple(weights), tuple(tuple(np.trim_zeros(c, "b")) for c in coefficients)
        )

    @staticmethod
    @lru_cache(maxsize=2048)
    def _mixture_stationary_cached(
        weights: tuple[float, ...], coefficients: tuple[tuple[float, ...], ...]
    ) -> bool:
        order = max(map(len, coefficients))
        if order == 0:
            return True
        if all(c == coefficients[0] for c in coefficients):
            return MARGenerator._component_stationary_cached(coefficients[0])
        # A common contracting weighted max norm proves stability for every
        # switching sequence when all absolute coefficient sums are below one.
        if all(sum(abs(v) for v in c) < 1.0 - 1e-10 for c in coefficients):
            return True
        if order > _MAX_MIXTURE_CHECK_ORDER:
            raise ValueError(
                "could not verify second-order stationarity above effective AR order 32; components must be identical or each have sum(abs(AR coefficients)) < 1"
            )
        operator = MARGenerator._second_moment_operator(weights, coefficients)
        rows, cols = np.triu_indices(order)
        initial = (rows == cols).astype(float)
        # For a positive covariance map, T^n(I) < I is a stability
        # certificate. Bound its largest eigenvalue by the absolute row sums.
        # This inexpensive check handles seasonal models without asking an
        # eigensolver to separate eigenvalues on a nearly periodic circle.
        tail = initial.copy()
        for step in range(4 * order if order > 4 else 0):
            tail = operator(tail)
            if not np.isfinite(tail).all() or np.max(np.abs(tail)) > 1e12:
                break
            if (step + 1) % order == 0:
                matrix = np.empty((order, order))
                matrix[rows, cols] = tail
                matrix[cols, rows] = tail
                if np.max(np.abs(matrix).sum(axis=1)) < 1.0 - 1e-8:
                    return True
                # Conversely, T^n(I) > I certifies an unstable mixture.
                if (
                    np.min(matrix.diagonal()) > 1.0 + 1e-8
                    and np.linalg.eigvalsh(matrix)[0] > 1.0 + 1e-8
                ):
                    return False
        try:
            # Symmetric covariance coordinates reduce p² to p(p+1)/2, at
            # most 528 here. Use the full spectrum when the sufficient
            # certificate is inconclusive, including near-periodic models.
            basis = np.eye(len(rows))
            dense = np.column_stack([operator(column) for column in basis])
            spectrum = np.linalg.eigvals(dense)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "could not verify second-order stationarity: eigensolver did not converge"
            ) from error
        if not np.isfinite(spectrum).all():
            raise ValueError(
                "could not verify second-order stationarity: nonfinite eigenvalues"
            )
        # Reject the numerical boundary rather than accepting an uncertain model.
        margin = 1e-10 if order <= 4 else 1e-8
        return bool(max(abs(spectrum)) < 1.0 - margin)

    @staticmethod
    def _second_moment_operator(weights, coefficients):
        """Return a matvec for sum(w A X A.T) on upper-triangle coordinates."""
        order = max(map(len, coefficients))
        rows, cols = np.triu_indices(order)
        ar = np.zeros((len(coefficients), order))
        for i, c in enumerate(coefficients):
            ar[i, : len(c)] = c
        weights = np.asarray(weights)
        mean_ar = weights @ ar

        def apply(vector):
            vector = np.asarray(vector).reshape(-1)
            matrix = np.empty((order, order), dtype=vector.dtype)
            matrix[rows, cols] = vector
            matrix[cols, rows] = vector
            image = np.empty_like(matrix)
            image[1:, 1:] = weights.sum() * matrix[:-1, :-1]
            image[0, 1:] = mean_ar @ matrix[:, :-1]
            image[1:, 0] = image[0, 1:]
            image[0, 0] = np.sum(weights * np.sum((ar @ matrix) * ar, axis=1))
            return image[rows, cols]

        return apply

    @staticmethod
    def _pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
        """Map partial autocorrelations to stationary AR coefficients."""
        pacf = np.asarray(pacf, dtype=float)
        if np.any(np.abs(pacf) >= 1):
            raise ValueError(
                "partial autocorrelations must lie strictly inside (-1, 1)"
            )
        coefficients = np.empty(0, dtype=float)
        for reflection in pacf:
            previous = coefficients.copy()
            coefficients = np.empty(len(previous) + 1)
            if len(previous):
                coefficients[:-1] = previous - reflection * previous[::-1]
            coefficients[-1] = reflection
        return coefficients

    @staticmethod
    def _apply_seasonal_factor(
        coefficients: np.ndarray, seasonal_period: int, seasonal_phi: float
    ) -> np.ndarray:
        """Multiply an AR polynomial by one seasonal AR factor."""
        base = np.concatenate(([1.0], -coefficients))
        seasonal = np.zeros(seasonal_period + 1)
        seasonal[0] = 1.0
        seasonal[-1] = -seasonal_phi
        return -np.convolve(base, seasonal)[1:]

    def _sample_params(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Sample one stable random MAR configuration."""
        n_components = int(self.rng.integers(1, self.max_components + 1))
        weights = self.rng.dirichlet(np.full(n_components, self.weights_concentration))
        coefficients: list[np.ndarray] = []
        for _ in range(n_components):
            order = int(self.rng.integers(1, self.max_ar_order + 1))
            pacf = self.rng.uniform(-0.9, 0.9, order)
            component = self._pacf_to_ar(pacf)
            if self.seasonal_period is not None and self.rng.random() < 0.5:
                seasonal_phi = float(self.rng.uniform(-0.9, 0.9))
                component = self._apply_seasonal_factor(
                    component, self.seasonal_period, seasonal_phi
                )
            coefficients.append(component)
        intercepts = self.rng.normal(0.0, self.intercept_scale, n_components)
        low, high = np.log(self.noise_scale_range)
        scales = np.exp(self.rng.uniform(low, high, n_components))
        return weights, coefficients, intercepts, scales

    def _fixed_params(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Return fixed fields as simulation arrays."""
        assert self.weights is not None
        assert self.ar_coefficients is not None
        assert self.intercepts is not None
        assert self.noise_scales is not None
        return (
            np.asarray(self.weights, dtype=float),
            [np.asarray(values, dtype=float) for values in self.ar_coefficients],
            np.asarray(self.intercepts, dtype=float),
            np.asarray(self.noise_scales, dtype=float),
        )

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encode MAR configuration for the native batch kernel."""
        fixed_mode = self.weights is not None
        scalars = np.asarray(
            [
                self.max_components,
                self.max_ar_order,
                self.seasonal_period or 0,
                self.weights_concentration,
                self.intercept_scale,
                self.noise_scale_range[0],
                self.noise_scale_range[1],
                self.burn_in,
                float(self.standardize),
                float(fixed_mode),
                self._rs_innov_dist,
                self._rs_innov_param,
            ],
            dtype=np.float64,
        )
        if not fixed_mode:
            return scalars, []
        assert self.weights is not None
        assert self.ar_coefficients is not None
        assert self.intercepts is not None
        assert self.noise_scales is not None
        return scalars, [
            np.asarray(self.weights, dtype=np.float64),
            np.asarray(
                [len(values) for values in self.ar_coefficients], dtype=np.float64
            ),
            np.asarray(
                [value for component in self.ar_coefficients for value in component],
                dtype=np.float64,
            ),
            np.asarray(self.intercepts, dtype=np.float64),
            np.asarray(self.noise_scales, dtype=np.float64),
        ]

    def _simulate(
        self,
        length: int,
        params: tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Simulate a MAR path and discard burn-in observations."""
        weights, coefficients, intercepts, scales = params
        max_order = max(len(component) for component in coefficients)
        total = length + self.burn_in
        values = np.empty(total + max_order)
        values[:max_order] = self.rng.normal(0.0, 1.0, max_order)
        components = self.rng.choice(len(weights), size=total, p=weights)
        innovations = self._sample_innovations(total)
        for step, component_index in enumerate(components, start=max_order):
            component = coefficients[component_index]
            past = values[step - len(component) : step][::-1]
            values[step] = (
                intercepts[component_index]
                + float(component @ past)
                + scales[component_index] * innovations[step - max_order]
            )
        return values[max_order + self.burn_in :]

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate one finite MAR series of exactly ``length`` observations.

        Random mode falls back to standard normal noise after ``_MAX_RETRIES``
        rejected draws. Fixed mode raises ``ValueError`` instead.
        """
        fixed_mode = self.weights is not None
        for _ in range(_MAX_RETRIES):
            params = self._fixed_params() if fixed_mode else self._sample_params()
            values = self._simulate(length, params)
            if not np.all(np.isfinite(values)):
                continue
            if np.abs(values).max() >= _MAX_ABS:
                continue
            std = float(values.std()) if length > 1 else 1.0
            if length > 1 and std <= _MIN_STD:
                continue
            if self.standardize and length > 1:
                values = (values - values.mean()) / std
            return values
        if fixed_mode:
            raise ValueError(
                "fixed MAR configuration produced no finite, bounded, non-constant "
                f"series in {_MAX_RETRIES} attempts"
            )
        values = self.rng.normal(0.0, 1.0, length)
        if self.standardize and length > 1:
            values = (values - values.mean()) / values.std()
        return values

    @classmethod
    def tune_to_features(
        cls,
        target_features: dict[str, float],
        min_length: int,
        max_length: int,
        freq: str | int,
        seasonal_period: int | None = None,
        n_generations: int = 15,
        population_size: int = 30,
        n_draws_per_candidate: int = 3,
        tolerance: float = 0.05,
        seed: int | None = None,
        *,
        n_jobs: int = -1,
        **generator_kwargs: Any,
    ) -> "MARGenerator":
        """Tune fixed MAR parameters toward requested GRATIS-style features.

        Supported names are ``spectral_entropy``, ``trend_strength``,
        ``seasonal_strength``, and ``acf1``, as defined by
        :func:`synforecast._features.compute_features`. This implements
        GRATIS feature-targeted generation (Kang et al. 2020,
        https://arxiv.org/abs/1903.02787) as a seeded evolutionary search over
        MAR parameters. The elitism, PACF-space Gaussian mutation, and minimal
        feature set are SynForecast's own design rather than a reproduction of
        the GRATIS genetic algorithm. Candidate draw lengths are spaced evenly
        across ``min_length..max_length`` so the result targets the configured
        length range rather than one endpoint. The default budget evaluates at
        most ``15 * 30 * 3`` series; the search stops early once the best L2
        feature distance is at or below ``tolerance``. Infeasible targets
        return the best candidate after the fixed budget. Candidates whose
        mixture is not second-order stationary, or whose simulation fails the
        output guards, receive infinite fitness.
        The returned generator's ``tuning_diagnostics`` records the best observed
        distance, whether tolerance was reached, generations run, and candidates
        evaluated.

        Each generation scores its valid candidates together in native Rust,
        parallelized across candidates. Failed simulations affect only their
        own candidate. ``n_jobs=-1`` uses the default Rayon pool; a positive
        value selects the worker count. Results are independent of worker count.
        This native search uses a different seeded stream from the former
        Python simulation loop. Lengths are limited to 1,000,000 observations.

        Additional generator options are forwarded to candidate evaluation and
        the returned generator: ``standardize``, ``burn_in``,
        ``innovation_distribution``, ``innovation_params``, ``engine``, ``alias``,
        ``id_col``, ``time_col``, ``target_col``, and ``start_datetime``.
        Search-owned model parameters and pattern injection options are rejected.
        """
        cls._validate_targeting(
            target_features,
            min_length,
            max_length,
            seasonal_period,
            n_generations,
            population_size,
            n_draws_per_candidate,
            tolerance,
        )
        supported_options = {
            "engine",
            "alias",
            "id_col",
            "time_col",
            "target_col",
            "start_datetime",
            "standardize",
            "innovation_distribution",
            "innovation_params",
            "burn_in",
        }
        unsupported = generator_kwargs.keys() - supported_options
        if unsupported:
            raise ValueError(
                f"unsupported tuning generator options: {sorted(unsupported)}"
            )
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        # Validate options before any search, including invalid output engines.
        cls(
            min_length=min_length,
            max_length=max_length,
            freq=freq,
            seasonal_period=seasonal_period,
            seed=seed,
            **generator_kwargs,
        )
        rng = np.random.default_rng(seed)
        evaluation_lengths = cls._evaluation_lengths(
            min_length, max_length, n_draws_per_candidate
        )
        population = [
            cls._random_candidate(rng, seasonal_period) for _ in range(population_size)
        ]
        best_candidate: dict[str, Any] | None = None
        best_distance = np.inf
        candidates_evaluated = 0
        generations_run = 0
        for _ in range(n_generations):
            generations_run += 1
            distances = cls._population_fitness(
                population,
                target_features,
                evaluation_lengths,
                freq,
                seasonal_period,
                rng,
                generator_kwargs,
                0 if n_jobs == -1 else n_jobs,
            )
            ranked = list(zip(distances, population, strict=True))
            candidates_evaluated += len(population)
            ranked.sort(key=lambda item: item[0])
            if np.isfinite(ranked[0][0]) and ranked[0][0] < best_distance:
                best_distance, best_candidate = ranked[0]
            if best_distance <= tolerance:
                break
            n_elites = max(1, population_size // 4)
            elites = [candidate for _, candidate in ranked[:n_elites]]
            n_mutated = (population_size - n_elites + 1) // 2
            population = elites + [
                cls._mutate_candidate(elites[index % n_elites], rng)
                for index in range(n_mutated)
            ]
            while len(population) < population_size:
                population.append(cls._random_candidate(rng, seasonal_period))

        if best_candidate is None:
            raise ValueError(
                "feature targeting found no valid MAR candidate within the search "
                "budget"
            )
        fixed = cls._candidate_to_fixed(best_candidate, seasonal_period)
        generator = cls(
            min_length=min_length,
            max_length=max_length,
            freq=freq,
            seed=seed,
            seasonal_period=seasonal_period,
            weights=fixed[0],
            ar_coefficients=fixed[1],
            intercepts=fixed[2],
            noise_scales=fixed[3],
            **generator_kwargs,
        )
        generator._tuning_diagnostics = {
            "best_distance": float(best_distance),
            "converged": bool(best_distance <= tolerance),
            "generations_run": generations_run,
            "candidates_evaluated": candidates_evaluated,
        }
        return generator

    @classmethod
    def _validate_targeting(
        cls,
        target_features: dict[str, float],
        min_length: int,
        max_length: int,
        seasonal_period: int | None,
        n_generations: int,
        population_size: int,
        n_draws_per_candidate: int,
        tolerance: float,
    ) -> None:
        """Validate feature targeting arguments."""
        if min_length < 3:
            raise ValueError("feature targeting requires min_length >= 3")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        if max_length > 1_000_000:
            raise ValueError("feature targeting requires max_length <= 1000000")
        if not target_features:
            raise ValueError("target_features must not be empty")
        unknown = set(target_features) - _SUPPORTED_FEATURES
        if unknown:
            supported = ", ".join(sorted(_SUPPORTED_FEATURES))
            raise ValueError(f"unsupported feature names; supported names: {supported}")
        for name, value in target_features.items():
            lower = -1.0 if name == "acf1" else 0.0
            if not lower <= value <= 1.0:
                raise ValueError(f"target {name} must be in [{lower:g}, 1]")
        if "seasonal_strength" in target_features and seasonal_period is None:
            raise ValueError("targeting seasonal_strength requires seasonal_period")
        if seasonal_period is not None and not 2 <= seasonal_period <= 10_000:
            raise ValueError("seasonal_period must be in [2, 10000] when provided")
        if (
            "seasonal_strength" in target_features
            and seasonal_period is not None
            and min_length < 2 * seasonal_period
        ):
            raise ValueError(
                "targeting seasonal_strength requires min_length >= 2 * seasonal_period"
            )
        if min(n_generations, population_size, n_draws_per_candidate) < 1:
            raise ValueError("search counts must all be >= 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

    @staticmethod
    def _evaluation_lengths(
        min_length: int, max_length: int, n_draws: int
    ) -> np.ndarray:
        """Return deterministic draw lengths spanning the configured range."""
        if n_draws == 1:
            return np.asarray([(min_length + max_length) // 2], dtype=int)
        return np.rint(np.linspace(min_length, max_length, n_draws)).astype(int)

    @classmethod
    def _random_candidate(
        cls, rng: np.random.Generator, seasonal_period: int | None
    ) -> dict[str, Any]:
        """Sample an unconstrained evolutionary candidate."""
        n_components = int(rng.integers(1, 4))
        weights = rng.dirichlet(np.ones(n_components))
        return {
            "weight_logits": np.log(weights),
            "pacf": [
                rng.uniform(-0.9, 0.9, int(rng.integers(1, 6)))
                for _ in range(n_components)
            ],
            "seasonal": [
                float(rng.uniform(-0.9, 0.9))
                if seasonal_period is not None and rng.random() < 0.5
                else None
                for _ in range(n_components)
            ],
            "intercepts": rng.normal(0.0, 1.0, n_components),
            "log_scales": rng.uniform(np.log(0.1), np.log(2.0), n_components),
        }

    @classmethod
    def _mutate_candidate(
        cls, candidate: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Apply Gaussian mutation in logits, PACF, and scale space."""
        pacf = [
            np.clip(
                values + rng.normal(0.0, 0.15, len(values)),
                -0.9,
                0.9,
            )
            for values in candidate["pacf"]
        ]
        seasonal = [
            None
            if value is None
            else float(np.clip(value + rng.normal(0.0, 0.15), -0.9, 0.9))
            for value in candidate["seasonal"]
        ]
        return {
            "weight_logits": candidate["weight_logits"]
            + rng.normal(0.0, 0.2, len(candidate["weight_logits"])),
            "pacf": pacf,
            "seasonal": seasonal,
            "intercepts": candidate["intercepts"]
            + rng.normal(0.0, 0.2, len(candidate["intercepts"])),
            "log_scales": candidate["log_scales"]
            + rng.normal(0.0, 0.15, len(candidate["log_scales"])),
        }

    @classmethod
    def _candidate_to_fixed(
        cls, candidate: dict[str, Any], seasonal_period: int | None
    ) -> tuple[list[float], list[list[float]], list[float], list[float]]:
        """Convert one unconstrained candidate into fixed MAR fields."""
        logits = np.asarray(candidate["weight_logits"], dtype=float)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        coefficients: list[list[float]] = []
        for pacf, seasonal_phi in zip(
            candidate["pacf"], candidate["seasonal"], strict=True
        ):
            component = cls._pacf_to_ar(np.asarray(pacf, dtype=float))
            if seasonal_period is not None and seasonal_phi is not None:
                component = cls._apply_seasonal_factor(
                    component, seasonal_period, seasonal_phi
                )
            coefficients.append(component.tolist())
        scales = np.exp(np.clip(candidate["log_scales"], np.log(0.03), np.log(5.0)))
        return (
            weights.tolist(),
            coefficients,
            np.asarray(candidate["intercepts"], dtype=float).tolist(),
            scales.tolist(),
        )

    @classmethod
    def _candidate_fitness(
        cls,
        candidate,
        target_features,
        eval_lengths,
        freq,
        seasonal_period,
        rng,
        **generator_kwargs,
    ) -> float:
        """Score one candidate with the same native path used by population search."""
        return cls._population_fitness(
            [candidate],
            target_features,
            eval_lengths,
            freq,
            seasonal_period,
            rng,
            generator_kwargs,
            1,
        )[0]

    @classmethod
    def _population_fitness(
        cls,
        candidates,
        target_features,
        eval_lengths,
        freq,
        seasonal_period,
        rng,
        generator_kwargs,
        n_workers,
    ):
        scalars, arrays, seeds, indices = [], [], [], []
        distances = [float("inf")] * len(candidates)
        for i, candidate in enumerate(candidates):
            draw_seeds = rng.integers(
                0, 2**63, size=len(eval_lengths), dtype=np.uint64
            ).tolist()
            fixed = cls._candidate_to_fixed(candidate, seasonal_period)
            try:
                generator = cls(
                    min_length=int(eval_lengths.min()),
                    max_length=int(eval_lengths.max()),
                    freq=freq,
                    seed=0,
                    seasonal_period=seasonal_period,
                    weights=fixed[0],
                    ar_coefficients=fixed[1],
                    intercepts=fixed[2],
                    noise_scales=fixed[3],
                    **generator_kwargs,
                )
            except ValueError:
                continue
            sp, ap = generator._get_batch_params()
            scalars.append(sp.tolist())
            arrays.append([a.tolist() for a in ap])
            seeds.append(draw_seeds)
            indices.append(i)
        if not indices:
            return distances
        results = _rs_augmentation.mar_features_batch(
            scalars, arrays, eval_lengths.tolist(), seeds, seasonal_period, n_workers
        )
        feature_names = [
            "spectral_entropy",
            "trend_strength",
            "seasonal_strength",
            "acf1",
        ]
        selected = [feature_names.index(name) for name in target_features]
        target = np.asarray(list(target_features.values()))
        for i, draws in zip(indices, results, strict=True):
            if draws is not None:
                realized = np.asarray(draws).mean(axis=0)[selected]
                distances[i] = float(np.linalg.norm(realized - target))
        return distances
