"""Temporal Causal Model (TCM) generator: random causal-graph autoregression."""

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, PrivateAttr, model_validator

from synforecast.base import BaseGenerator
from synforecast.exogenous import SeriesMetadata

_EDGE_KINDS = ("linear", "tanh", "relu", "product", "threshold")
_NOISE_TYPES = ("gaussian", "student_t", "laplace")
_EDGE_KIND_CODES = {kind: code for code, kind in enumerate(_EDGE_KINDS)}
# Kinds whose contribution is unbounded-or-linear near 0; they enter the
# stability (companion) matrix with gain = coef. product/threshold outputs
# are bounded by |coef| and cannot destabilize a stable core.
_LINEAR_PART_CODES = (0, 1, 2)  # linear, tanh, relu

# Divergence guard thresholds (mirror tsfm's synthetic-pool guard)
_MAX_ABS = 1e8
_MIN_STD = 1e-8
# SCM redraws allowed before falling back to a guaranteed-stable AR(1)
_MAX_REDRAWS = 5
_MAX_BURN_IN = 200
# Per-series spectral-radius target, as a fraction of stability_margin: with
# probability _PERSISTENT_PROB the target is drawn near the margin (strongly
# persistent, spectrally peaked series); otherwise well below it (noise-like
# to moderately dependent). The mixture yields the entropy spread real pools
# have.
_PERSISTENT_PROB = 0.22
_RADIUS_TARGET_HIGH = (0.95, 1.0)
_RADIUS_TARGET_LOW = (0.1, 0.6)
# Per-series geometric decay of coefficient magnitude with lag; short-lag
# dominance is what lets a high spectral radius concentrate spectral power
# (a lag-24-only system needs coef 0.9**24 for radius 0.9 — a flat spectrum).
_LAG_DECAY_RANGE = (0.4, 1.0)
# Cap on uniformly boosting a weak linear part toward the radius target
_UPSCALE_CAP = 4.0
# Log-uniform range of the softness scale s of saturating edges, which
# contribute c*s*tanh(x/s): slope c near 0 (honest stability gain), output
# bounded by ~|c|*s. Small s = hard regime-like saturation, large s = near
# linear over the typical state range.
_SOFTNESS_RANGE = (0.5, 4.0)
# Heteroscedastic envelope: scale(t) = exp(a * sin(2*pi*t/P + phase))
_ENV_PERIOD_RANGE = (50.0, 500.0)
_ENV_AMP_RANGE = (0.3, 1.0)
_STUDENT_T_DF_RANGE = (3.0, 10.0)


class TCMGenerator(BaseGenerator):
    """Generate series from a random temporal structural causal model (SCM).

    Each series gets a freshly sampled SCM over ``n_vars`` latent variables:
    a sparse dependency graph over the (variable x lag) space is drawn, each
    edge is assigned a random edge function, and the system is rolled out
    autoregressively. Node ``i`` evolves as

        x_i[t] = sum_{e in pa(i)} f_e(x_{j_e}[t - l_e]) + eps_i[t]

    with per-edge functions ``f_e(x)`` in

        c*x, c*tanh(x), c*relu(x), c*tanh(x)*tanh(x'), c*1[x > tau]

    The temporal-SCM framing follows the overview in Runge et al. (2023),
    "Causal inference for time series,"
    https://doi.org/10.1038/s43017-023-00431-y. The particular graph sampler,
    edge-function mixture, stability rescaling, and guards here are original
    SynForecast design choices; this is not a reproduction of a named TCM
    generator from that paper or from Chronos-2.

    where ``x'`` is a second randomly-paired parent (product interaction) and
    ``tau`` a random threshold. Saturating kinds carry a log-uniform softness
    scale ``s`` and contribute ``c*s*tanh(x/s)`` (slope c near 0, bounded
    output). The returned univariate series is node 0 (nodes are exchangeable
    by construction); the remaining nodes act as latent parents, i.e.
    realistic exogenous-looking drivers. This produces genuine causal
    temporal structure — autocorrelation at sampled lags, lead-lag effects,
    nonlinear/regime-like dynamics — that component mixing cannot.

    Diversity is shaped per series: edge kinds follow a random Dirichlet
    mixture over ``edge_kinds`` (some series linear-dominated, others
    nonlinearity-dominated), coefficient magnitudes decay geometrically with
    lag (short-lag dominance), and, when 'linear' is in the pool, every node
    gets a positive linear self lag-1 edge so the observed node carries its
    own persistence.

    Stability: the linear-gain part (linear/tanh/relu edges) is assembled
    into VAR companion form and its coefficients are rescaled toward a
    per-series spectral-radius target below ``stability_margin`` — drawn
    near the margin with probability 0.22 (persistent, spectrally peaked
    series) and well below it otherwise (noise-like series); bounded-output
    edges cannot destabilize the core and keep their coefficients. During
    rollout every state is additionally soft-clamped via
    ``clamp * tanh(x / clamp)`` so nonlinear feedback cannot diverge. If a
    trajectory still fails the finiteness/scale guard, the SCM is redrawn
    (up to 5 times), then a guaranteed-stable linear AR(1) is used. The
    counters ``_redraw_total`` / ``_fallback_total`` and the last accepted
    SCM ``_last_scm`` are exposed for introspection on direct
    ``generate_single_series`` calls.

    Multivariate mode: with ``multivariate=True``, ``generate(n_series)``
    samples a single SCM (with at least ``n_series`` variables — the lower
    bound of ``n_vars_range`` is clamped up as needed) and one shared length,
    rolls the system out once, and returns the first ``n_series`` nodes as
    separate series in the long-format output (one ``unique_id`` per node,
    following ``VARGenerator``). Because the nodes share one causal graph,
    they are genuinely cross-dependent at the sampled lags. The default
    ``multivariate=False`` keeps the univariate behavior: ``n_series``
    independent SCMs, one observed node each.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data. A pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS') or an integer time index step.
        multivariate (bool): When True, generate(n_series) returns n_series
            nodes of one shared SCM as correlated series sharing one length
            (default: False).
        n_vars_range (tuple[int, int]): Inclusive range for the number of
            latent variables per SCM (default: (1, 5)).
        max_lag_range (tuple[int, int]): Inclusive range for the maximum lag
            L of the dependency graph (default: (1, 24)).
        edge_probability_range (tuple[float, float]): Range for the per-slot
            edge probability over the (variable x lag) space
            (default: (0.05, 0.3)).
        edge_kinds (list[str]): Pool of edge function kinds, sampled per
            edge. Subset of ['linear', 'tanh', 'relu', 'product',
            'threshold'] (default: all).
        coef_range (tuple[float, float]): Range for edge coefficient
            magnitudes before stability rescaling; signs are random
            (default: (0.1, 0.8)).
        stability_margin (float): Upper bound (< 1) on the spectral radius
            of the linear-part companion matrix (default: 0.95).
        clamp_threshold (float): Soft-clamp scale for states during rollout;
            generous relative to typical noise scales so it only engages on
            runaway feedback (default: 1e6).
        noise_types (list[str]): Pool of per-node innovation distributions.
            Subset of ['gaussian', 'student_t', 'laplace'] (default: all).
        noise_scale_range (tuple[float, float]): Range for per-node noise
            standard deviation (default: (0.5, 2.0)).
        heteroscedastic_prob (float): Probability that a node's noise scale
            follows a slow random sinusoidal envelope (default: 0.2).
        seed (int | None): Random seed for reproducibility (default: None).
        id_col (str): Name of the ID column (default: 'unique_id').
        time_col (str): Name of the timestamp column (default: 'ds').
        target_col (str): Name of the value column (default: 'y').
        start_datetime (str): First timestamp (default: '2000-01-01').

    Example:
        >>> gen = TCMGenerator(
        ...     min_length=256,
        ...     max_length=512,
        ...     freq="h",
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    multivariate: bool = Field(
        default=False,
        description="Return generate(n_series) as n_series nodes of one shared SCM",
    )
    n_vars_range: tuple[int, int] = Field(
        default=(1, 5), description="Inclusive range for the number of variables"
    )
    max_lag_range: tuple[int, int] = Field(
        default=(1, 24), description="Inclusive range for the maximum lag"
    )
    edge_probability_range: tuple[float, float] = Field(
        default=(0.05, 0.3), description="Range for the per-slot edge probability"
    )
    edge_kinds: list[str] = Field(
        default=list(_EDGE_KINDS), description="Pool of edge function kinds"
    )
    coef_range: tuple[float, float] = Field(
        default=(0.1, 0.8), description="Range for edge coefficient magnitudes"
    )
    stability_margin: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Spectral radius bound for the linear part",
    )
    clamp_threshold: float = Field(
        default=1e6, gt=0.0, description="Soft-clamp scale for rollout states"
    )
    noise_types: list[str] = Field(
        default=list(_NOISE_TYPES), description="Pool of innovation distributions"
    )
    noise_scale_range: tuple[float, float] = Field(
        default=(0.5, 2.0), description="Range for per-node noise scales"
    )
    heteroscedastic_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of a slowly-varying noise envelope per node",
    )

    _redraw_total: int = PrivateAttr(default=0)
    _fallback_total: int = PrivateAttr(default=0)
    _last_scm: dict | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_tcm_params(self) -> "TCMGenerator":
        """Validate range parameters and edge/noise pools."""
        for name, (low, high), min_low in (
            ("n_vars_range", self.n_vars_range, 1),
            ("max_lag_range", self.max_lag_range, 1),
        ):
            if low < min_low or high < low:
                raise ValueError(f"{name} must satisfy {min_low} <= low <= high")

        p_low, p_high = self.edge_probability_range
        if not (0.0 <= p_low <= p_high <= 1.0):
            raise ValueError(
                "edge_probability_range must satisfy 0 <= low <= high <= 1"
            )

        for name, (f_low, f_high) in (
            ("coef_range", self.coef_range),
            ("noise_scale_range", self.noise_scale_range),
        ):
            if f_low <= 0 or f_high < f_low:
                raise ValueError(f"{name} must satisfy 0 < low <= high")

        if not self.edge_kinds:
            raise ValueError("edge_kinds must not be empty")
        for kind in self.edge_kinds:
            if kind not in _EDGE_KINDS:
                raise ValueError(f"edge_kinds must be a subset of {_EDGE_KINDS}")

        if not self.noise_types:
            raise ValueError("noise_types must not be empty")
        for ntype in self.noise_types:
            if ntype not in _NOISE_TYPES:
                raise ValueError(f"noise_types must be a subset of {_NOISE_TYPES}")

        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Encode field-level ranges/pools for the Rust batch kernel.

        The Rust kernel draws the whole per-series SCM itself (graph, edge
        functions, noise config, stability rescale, rollout, guard/redraw/
        fallback) from these ranges, so only range endpoints, probabilities
        and pools are passed. Scalar ordering and pool id encodings must
        match the ``GEN_TCM`` dispatch arm in ``rust/src/batch.rs`` /
        ``rust/src/generators/tcm.rs``:

        scalars: [n_vars_lo, n_vars_hi, max_lag_lo, max_lag_hi,
                  edge_prob_lo, edge_prob_hi, coef_lo, coef_hi,
                  stability_margin, clamp_threshold,
                  noise_scale_lo, noise_scale_hi, heteroscedastic_prob]
        arrays:  [edge kind ids (index into _EDGE_KINDS),
                  noise type ids (index into _NOISE_TYPES)]

        Returns None in multivariate mode: generate() overrides the batch
        machinery there (one shared SCM rolled out jointly), so the
        univariate batch kernel must never be selected for it.
        """
        if self.multivariate:
            return None
        return (
            np.array(
                [
                    float(self.n_vars_range[0]),
                    float(self.n_vars_range[1]),
                    float(self.max_lag_range[0]),
                    float(self.max_lag_range[1]),
                    self.edge_probability_range[0],
                    self.edge_probability_range[1],
                    self.coef_range[0],
                    self.coef_range[1],
                    self.stability_margin,
                    self.clamp_threshold,
                    self.noise_scale_range[0],
                    self.noise_scale_range[1],
                    self.heteroscedastic_prob,
                ],
                dtype=np.float64,
            ),
            [
                np.array(
                    [_EDGE_KIND_CODES[k] for k in self.edge_kinds],
                    dtype=np.float64,
                ),
                np.array(
                    [_NOISE_TYPES.index(t) for t in self.noise_types],
                    dtype=np.float64,
                ),
            ],
        )

    @staticmethod
    def _companion_spectral_radius(lin_mats: list[np.ndarray]) -> float:
        """Spectral radius of the companion matrix of the linear part."""
        p = len(lin_mats)
        k = lin_mats[0].shape[0]
        companion = np.zeros((k * p, k * p))
        for lag, mat in enumerate(lin_mats):
            companion[0:k, lag * k : (lag + 1) * k] = mat
        if p > 1:
            companion[k:, 0:-k] = np.eye(k * (p - 1))
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def _linear_part(self, scm: dict) -> list[np.ndarray]:
        """Assemble per-lag gain matrices from linear/tanh/relu edges."""
        n, max_lag = scm["n_vars"], scm["max_lag"]
        lin_mats = [np.zeros((n, n)) for _ in range(max_lag)]
        for i, node in enumerate(scm["nodes"]):
            mask = node["lin_mask"]
            for pvar, plag, coef in zip(
                node["pvar"][mask],
                node["plag"][mask],
                node["coef"][mask],
                strict=True,
            ):
                lin_mats[plag - 1][i, pvar] += coef
        return lin_mats

    def _sample_scm(self, min_vars: int = 1) -> dict:
        """Draw a random SCM: graph, edge functions, and noise config.

        Args:
            min_vars (int): Lower bound on the number of variables; clamps
                ``n_vars_range`` up so a multivariate call observing
                ``n_series`` nodes always has enough of them (default: 1).
        """
        rng = self.rng
        n_low = max(self.n_vars_range[0], min_vars)
        n_high = max(self.n_vars_range[1], min_vars)
        n = int(rng.integers(n_low, n_high + 1))
        max_lag = int(rng.integers(self.max_lag_range[0], self.max_lag_range[1] + 1))
        p_edge = rng.uniform(*self.edge_probability_range)
        lag_decay = rng.uniform(*_LAG_DECAY_RANGE)
        kind_pool = np.array([_EDGE_KIND_CODES[k] for k in self.edge_kinds])
        # Per-series mixture over edge kinds: some series come out linear-
        # dominated (persistent/smooth), others nonlinearity-dominated
        kind_probs = rng.dirichlet(np.ones(kind_pool.size))

        nodes = []
        for i in range(n):
            present = rng.random((n, max_lag)) < p_edge
            # Every node gets a linear self lag-1 edge so it is well-defined
            # and carries its own persistence; the stability rescale below
            # sets how strong that persistence ends up.
            present[i, 0] = True
            pvar, lag0 = np.nonzero(present)
            plag = lag0 + 1
            m = pvar.size
            kind = rng.choice(kind_pool, size=m, p=kind_probs)
            coef = rng.uniform(*self.coef_range, size=m)
            coef *= rng.choice([-1.0, 1.0], size=m)
            coef *= lag_decay ** (plag - 1)
            if "linear" in self.edge_kinds:
                # Unit positive linear self-persistence on every node, set to
                # the per-series radius target by the stability rescale.
                # Without it the dominant eigenmode rarely loads on the
                # observed node and every series comes out noise-like.
                self_edge = (pvar == i) & (plag == 1)
                kind[self_edge] = _EDGE_KIND_CODES["linear"]
                coef[self_edge] = 1.0
            log_soft = rng.uniform(
                np.log(_SOFTNESS_RANGE[0]), np.log(_SOFTNESS_RANGE[1]), size=m
            )
            nodes.append(
                {
                    "pvar": pvar,
                    "plag": plag,
                    "kind": kind,
                    "coef": coef,
                    "soft": np.exp(log_soft),
                    "tau": rng.normal(0.0, 1.0, size=m),
                    # second parent for product-interaction edges
                    "qvar": rng.integers(0, n, size=m),
                    "qlag": rng.integers(1, max_lag + 1, size=m),
                    "lin_mask": np.isin(kind, _LINEAR_PART_CODES),
                }
            )

        noise = []
        for _ in range(n):
            noise.append(
                {
                    "type": self.noise_types[rng.integers(len(self.noise_types))],
                    "scale": rng.uniform(*self.noise_scale_range),
                    "df": rng.uniform(*_STUDENT_T_DF_RANGE),
                    "hetero": rng.random() < self.heteroscedastic_prob,
                    "env_period": rng.uniform(*_ENV_PERIOD_RANGE),
                    "env_amp": rng.uniform(*_ENV_AMP_RANGE),
                    "env_phase": rng.uniform(0.0, 2.0 * np.pi),
                }
            )

        scm = {"n_vars": n, "max_lag": max_lag, "nodes": nodes, "noise": noise}
        self._rescale_for_stability(scm)
        return scm

    def _rescale_for_stability(self, scm: dict) -> None:
        """Rescale linear-part coefficients toward a per-series radius target.

        Scaling lag-l gains by s**l maps every companion eigenvalue z to s*z,
        so the down-scale sets the spectral radius exactly to the target.
        Weak linear parts are boosted uniformly (capped) instead, then
        re-checked.
        """
        rng = self.rng
        target_range = (
            _RADIUS_TARGET_HIGH
            if rng.random() < _PERSISTENT_PROB
            else _RADIUS_TARGET_LOW
        )
        target = self.stability_margin * rng.uniform(*target_range)
        radius = self._companion_spectral_radius(self._linear_part(scm))
        scm["target_radius"] = target

        def scale_linear_edges(scale_fn) -> None:
            for node in scm["nodes"]:
                mask = node["lin_mask"]
                node["coef"][mask] *= scale_fn(node["plag"][mask])

        if radius > target:
            s = target / radius
            scale_linear_edges(lambda plag: s**plag)
        elif radius > 1e-9:
            u = min(target / radius, _UPSCALE_CAP)
            scale_linear_edges(lambda _plag: u)
            new_radius = self._companion_spectral_radius(self._linear_part(scm))
            if new_radius >= self.stability_margin:
                s = target / new_radius
                scale_linear_edges(lambda plag: s**plag)

    def _draw_noise(self, config: dict, size: int) -> np.ndarray:
        """Draw one node's innovation sequence (unit variance, then scaled)."""
        rng = self.rng
        if config["type"] == "gaussian":
            eps = rng.normal(0.0, 1.0, size)
        elif config["type"] == "student_t":
            df = config["df"]
            eps = rng.standard_t(df, size) * np.sqrt((df - 2.0) / df)
        else:  # laplace, variance 2*b^2 with b = 1/sqrt(2)
            eps = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
        eps *= config["scale"]
        if config["hetero"]:
            t = np.arange(size)
            eps *= np.exp(
                config["env_amp"]
                * np.sin(2.0 * np.pi * t / config["env_period"] + config["env_phase"])
            )
        return eps

    def _rollout(self, scm: dict, length: int) -> np.ndarray:
        """Roll the SCM out and return the target node (node 0)."""
        return self._rollout_states(scm, length)[:, 0]

    def _rollout_states(self, scm: dict, length: int) -> np.ndarray:
        """Roll the SCM out and return all nodes, shape (length, n_vars)."""
        n, max_lag = scm["n_vars"], scm["max_lag"]
        burn_in = min(_MAX_BURN_IN, length)
        total = max_lag + burn_in + length
        clamp = self.clamp_threshold

        noise = np.empty((total, n))
        for i, config in enumerate(scm["noise"]):
            noise[:, i] = self._draw_noise(config, total)

        x = np.zeros((total, n))
        x[:max_lag] = noise[:max_lag]

        nodes = scm["nodes"]
        for t in range(max_lag, total):
            for i, node in enumerate(nodes):
                kind, coef, soft = node["kind"], node["coef"], node["soft"]
                p = x[t - node["plag"], node["pvar"]]
                vals = np.empty_like(p)
                m = kind == 0
                vals[m] = p[m]
                m = kind == 1
                vals[m] = soft[m] * np.tanh(p[m] / soft[m])
                m = kind == 2
                vals[m] = np.maximum(p[m], 0.0)
                m = kind == 3
                if m.any():
                    q = x[t - node["qlag"][m], node["qvar"][m]]
                    vals[m] = soft[m] * np.tanh(p[m] / soft[m]) * np.tanh(q / soft[m])
                m = kind == 4
                vals[m] = (p[m] > node["tau"][m]).astype(np.float64)
                raw = coef @ vals + noise[t, i]
                # soft clamp: identity for |raw| << clamp, saturates at clamp
                x[t, i] = clamp * np.tanh(raw / clamp)

        return x[max_lag + burn_in :]

    @staticmethod
    def _series_ok(values: np.ndarray) -> bool:
        """Finiteness/scale guard: finite, |x| < 1e8, std > 1e-8."""
        return bool(
            np.isfinite(values).all()
            and np.abs(values).max() < _MAX_ABS
            and values.std() > _MIN_STD
        )

    def _fallback_series(self, length: int) -> np.ndarray:
        """Guaranteed-stable linear AR(1) draw used when redraws are exhausted."""
        burn_in = min(_MAX_BURN_IN, length)
        eps = self.rng.normal(0.0, 1.0, burn_in + length)
        x = np.empty(burn_in + length)
        x[0] = eps[0]
        for t in range(1, burn_in + length):
            x[t] = 0.7 * x[t - 1] + eps[t]
        return x[burn_in:]

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single TCM series.

        Samples a fresh random SCM, rolls it out (with burn-in), and returns
        the target node. Redraws the SCM on guard failure, falling back to a
        stable linear AR(1) after ``_MAX_REDRAWS`` redraws.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of time series values.
        """
        for _ in range(_MAX_REDRAWS + 1):
            scm = self._sample_scm()
            values = self._rollout(scm, length)
            if self._series_ok(values):
                self._last_scm = scm
                return values
            self._redraw_total += 1
        self._fallback_total += 1
        return self._fallback_series(length)

    def _generate_multivariate(self, length: int, n_series: int) -> np.ndarray:
        """Roll out one shared SCM and return its first n_series nodes.

        Samples an SCM with at least ``n_series`` variables, rolls it out
        once, and applies the finiteness/scale guard to every observed node.
        Redraws the SCM on guard failure; after ``_MAX_REDRAWS`` redraws a
        correlated stable fallback (one AR(1) common factor plus per-node
        noise) is used so the returned nodes stay cross-dependent.

        Args:
            length (int): The shared length of all series.
            n_series (int): Number of observed nodes.

        Returns:
            Array of shape (length, n_series).
        """
        for _ in range(_MAX_REDRAWS + 1):
            scm = self._sample_scm(min_vars=n_series)
            states = self._rollout_states(scm, length)
            observed = states[:, :n_series]
            if all(self._series_ok(observed[:, i]) for i in range(n_series)):
                self._last_scm = scm
                return observed
            self._redraw_total += 1
        self._fallback_total += 1
        common = self._fallback_series(length)
        noise_std = 0.5 * float(common.std())
        observed = np.empty((length, n_series))
        for i in range(n_series):
            observed[:, i] = common + self.rng.normal(0.0, noise_std, length)
        return observed

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,
    ) -> IntoDataFrameT:
        """Generate n_series time series from temporal causal models.

        With ``multivariate=False`` (default) this is the base behavior:
        n_series independent SCMs, one observed node each. With
        ``multivariate=True`` the n_series series are the first n_series
        nodes of one shared SCM, sharing a single length (following
        VARGenerator); generation is inherently joint, so n_jobs has no
        effect in that mode.

        Args:
            n_series (int): Number of series to generate. In multivariate
                mode, the number of observed nodes of one shared SCM.
            start_id (int): Starting ID for series numbering (default: 0).
            n_jobs (int): Parallel workers for the univariate path; unused
                in multivariate mode.

        Returns:
            DataFrame in long format with columns [id_col, time_col, target_col].
        """
        if not self.multivariate:
            return super().generate(n_series, start_id=start_id, n_jobs=n_jobs)

        length = int(self.rng.integers(self.min_length, self.max_length + 1))
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
