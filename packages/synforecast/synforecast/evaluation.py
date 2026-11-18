"""Native feature extraction and reference-anchored corpus coverage.

Coverage describes occupancy in a two-dimensional feature projection, not
predictive fidelity or privacy. Native feature definitions belong to
SynForecast and do not reproduce an external feature package.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT

from synforecast import _coverage, base
from synforecast._features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    compute_feature_matrix,
)
from synforecast.presets import _seasonal_period_from_freq

logger = logging.getLogger(__name__)

__all__ = [
    "CoverageDiagnostics",
    "CoverageResult",
    "compare_feature_coverage",
    "compute_features",
    "feature_coverage",
]


@dataclass(frozen=True, eq=False)
class CoverageDiagnostics:
    """Series retained or excluded from a corpus, in embedding row order.

    ``drop_reasons`` maps each dropped ID to the names of its non-finite
    features. Instances compare by identity, not by value.
    """

    n_input: int
    retained_ids: tuple[Any, ...]
    dropped_ids: tuple[Any, ...]
    drop_reasons: dict[Any, tuple[str, ...]]


@dataclass(frozen=True, eq=False)
class CoverageResult:
    """Occupancy scores and the parameters needed to interpret them.

    ``miscoverage`` divides uncovered real cells by all grid cells;
    ``uncovered_real_cell_fraction`` divides by occupied real cells instead.
    ``reverse_miscoverage`` counts synthetic-only cells within the grid.
    Out-of-range synthetic points are excluded from occupancy, but included
    in the denominator of ``synthetic_out_of_range_fraction``. These scores
    describe retained series; inspect the diagnostics before comparing corpora.

    ``explained_variance`` is None for t-SNE. ``metadata`` holds the fitted
    ``parameters`` and the ``reference_id`` fingerprint of the retained real
    features; each result owns its copy. Instances compare by identity, since
    several fields are arrays; compare ``summary()`` values instead.
    """

    miscoverage: float
    reverse_miscoverage: float
    uncovered_real_cell_fraction: float
    uncovered_real_series_fraction: float
    n_bins: int
    embedding: str
    explained_variance: tuple[float, float] | None
    real_embedding: np.ndarray
    synthetic_embedding: np.ndarray
    real_occupancy: np.ndarray
    synthetic_occupancy: np.ndarray
    grid_edges: tuple[np.ndarray, np.ndarray]
    uncovered_real_ids: tuple[Any, ...]
    n_synthetic_out_of_range: int
    synthetic_out_of_range_fraction: float
    features_used: tuple[str, ...]
    constant_features_dropped: tuple[str, ...]
    synthetic_constant_feature_deviations: dict[str, int]
    real_diagnostics: CoverageDiagnostics
    synthetic_diagnostics: CoverageDiagnostics
    fit: str
    grid_range: str
    space_id: str
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return the scalar scores and retained counts as a flat dict.

        ``pd.DataFrame({name: r.summary() for name, r in results.items()}).T``
        tabulates the output of :func:`compare_feature_coverage`.
        """
        return {
            "miscoverage": self.miscoverage,
            "reverse_miscoverage": self.reverse_miscoverage,
            "uncovered_real_cell_fraction": self.uncovered_real_cell_fraction,
            "uncovered_real_series_fraction": self.uncovered_real_series_fraction,
            "synthetic_out_of_range_fraction": self.synthetic_out_of_range_fraction,
            "n_real_retained": len(self.real_diagnostics.retained_ids),
            "n_real_input": self.real_diagnostics.n_input,
            "n_synthetic_retained": len(self.synthetic_diagnostics.retained_ids),
            "n_synthetic_input": self.synthetic_diagnostics.n_input,
            "n_features_used": len(self.features_used),
            "explained_variance": (
                None
                if self.explained_variance is None
                else float(sum(self.explained_variance))
            ),
        }

    def plot(self, ax: Any = None) -> Any:
        """Plot occupied cells and uncovered real points; import matplotlib lazily.

        Points outside the grid are omitted from this view and counted in the
        title. Out-of-range means outside the observed projected reference
        box, not necessarily unrealistic. Returns the matplotlib axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if ax is None:
            _, ax = plt.subplots()
        layers = self.real_occupancy.astype(int) + 2 * self.synthetic_occupancy
        colors = ListedColormap(["white", "#d95f02", "#7570b3", "#1b9e77"])
        ax.pcolormesh(*self.grid_edges, layers.T, cmap=colors, vmin=0, vmax=3)
        cells = _coverage.grid_occupancy(self.real_embedding, self.grid_edges)[1]
        uncovered = ~self.synthetic_occupancy[cells[:, 0], cells[:, 1]]
        points = ax.scatter(
            *self.real_embedding[uncovered].T, s=12, c="black", label="Uncovered real"
        )
        ax.set(
            xlabel=f"{self.embedding.upper()} 1",
            ylabel=f"{self.embedding.upper()} 2",
            title=(
                f"fit={self.fit}, grid={self.grid_range}; "
                f"retained real {len(self.real_diagnostics.retained_ids)}/{self.real_diagnostics.n_input}, "
                f"synthetic {len(self.synthetic_diagnostics.retained_ids)}/{self.synthetic_diagnostics.n_input}; "
                f"outside grid: {self.n_synthetic_out_of_range}"
            ),
        )
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="#d95f02", label="Real only"),
                Patch(color="#7570b3", label="Synthetic only"),
                Patch(color="#1b9e77", label="Both"),
                points,
            ]
        )
        return ax


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _workers(n_jobs: int) -> int:
    """Resolve ``n_jobs`` as the generators do: -1 honours RAYON_NUM_THREADS."""
    if not isinstance(n_jobs, bool) and isinstance(n_jobs, Integral) and n_jobs == -1:
        return base._default_n_workers
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError("n_jobs must be -1 (all CPUs) or an integer >= 1")
    return int(n_jobs)


def _seasonal_period(seasonal_period: int | None, freq: str | None) -> int | None:
    """Resolve the declared period; 1 and yearly data mean no seasonality.

    An explicit ``seasonal_period`` wins over ``freq``, matching the presets.
    """
    if seasonal_period is not None:
        period = _integer(seasonal_period, "seasonal_period", 1)
    elif freq is not None:
        if not isinstance(freq, str):
            raise ValueError(
                "freq must be a pandas offset alias such as 'MS' or 'h'; pass "
                "seasonal_period for integer time steps"
            )
        period = _seasonal_period_from_freq(freq)
    else:
        return None
    return None if period == 1 else period


def _feature_names(
    features: Sequence[str] | None, available: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(features, str):
        raise ValueError("features must be a sequence of names, not a string")
    names = tuple(available if features is None else features)
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("features must be a non-empty sequence of column names")
    if len(names) != len(set(names)):
        raise ValueError("features must not contain duplicate names")
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown or missing feature columns: {sorted(unknown)}")
    return names


def _panel(
    df: IntoDataFrameT, id_col: str, time_col: str, target_col: str
) -> tuple[nw.DataFrame[Any], np.ndarray, np.ndarray]:
    """Return one ID row per series, the sorted targets, and series offsets."""
    frame = nw.from_native(df, eager_only=True)
    columns = [id_col, time_col, target_col]
    if len(set(columns)) != 3 or not set(columns) <= set(frame.columns):
        raise ValueError(
            "id_col, time_col, and target_col must name three distinct input columns"
        )
    if not len(frame):
        raise ValueError("Panel must contain at least one series")
    if any(frame[col].is_null().any() for col in columns):
        raise ValueError("Panel IDs, times, and targets must not contain null values")
    for col in (id_col, time_col):
        if frame.schema[col].is_numeric() and not np.all(
            np.isfinite(frame[col].to_numpy())
        ):
            raise ValueError("Numeric panel IDs and times must be finite")
    if not frame.schema[target_col].is_numeric():
        raise ValueError("Panel targets must be numeric and finite")
    if frame.select(id_col, time_col).is_duplicated().any():
        raise ValueError("Panel timestamps must be unique within each series")
    ordered = nw.maybe_reset_index(frame.select(columns).sort([id_col, time_col]))
    values = np.asarray(ordered[target_col].to_numpy(), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Panel targets must be numeric and finite")
    ids = ordered[id_col].to_numpy()
    offsets = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    first_rows = ordered.select(id_col)[offsets[:-1].tolist()]
    return nw.maybe_reset_index(first_rows), values, offsets


def compute_features(
    df: IntoDataFrameT,
    *,
    seasonal_period: int | None = None,
    freq: str | None = None,
    window_size: int | None = None,
    features: Sequence[str] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    n_jobs: int = -1,
) -> IntoDataFrameT:
    """Return one row of native features per series in the caller's engine.

    The ``native_v1`` schema fixes feature definitions and order. Every
    feature is computed on the series normalized to zero mean and unit
    variance, so none depends on the input scale. Features depend on
    observation counts, not on the calendar of ``time_col``.
    ``crossing_points`` is a raw count and therefore grows with series
    length, so compare panels of comparable length. Short series yield NaN
    for undefined features and one summary log warning; invalid raw inputs
    raise.

    Args:
        df: Long-format panel with ID, time, and finite numeric target columns.
        seasonal_period: Seasonal period in observations. 1 means no
            seasonality. Takes precedence over ``freq``.
        freq: Pandas offset alias used to derive the period when
            ``seasonal_period`` is None, with the same convention as the
            presets (hourly 24, daily 7, weekly 52, monthly 12, quarterly 4,
            yearly none). Without either, no seasonality is declared and the
            two seasonal features are zero for every series.
        window_size: Width (>= 2) for lumpiness and the level/variance
            shifts. None uses the seasonal period, or 10 without one. It does
            not change the decomposition or seasonal ACF. Use one fixed width
            per panel; short nonseasonal panels can, for example, use 5.
        features: Feature names to return, in that order. None returns all
            twelve ``native_v1`` features.
        id_col: Series identifier column.
        time_col: Time column, used only for ordering.
        target_col: Target column.
        n_jobs: Native worker threads; -1 uses all CPUs (honouring
            ``RAYON_NUM_THREADS``), as in the generators.

    Returns:
        A frame in the input engine with ``id_col`` followed by the selected
        feature columns, one row per series, sorted by ID.
    """
    names = _feature_names(features, FEATURE_NAMES)
    if id_col in names:
        raise ValueError("id_col must not conflict with a feature name")
    period = _seasonal_period(seasonal_period, freq)
    if window_size is not None:
        window_size = _integer(window_size, "window_size", 2)
    workers = _workers(n_jobs)
    ids, flat, offsets = _panel(df, id_col, time_col, target_col)
    matrix = compute_feature_matrix(flat, offsets, period, window_size, workers)
    values = matrix[:, [FEATURE_NAMES.index(name) for name in names]]
    invalid = ~np.isfinite(values)
    if invalid.any():
        counts = {
            name: int(invalid[:, i].sum())
            for i, name in enumerate(names)
            if invalid[:, i].any()
        }
        logger.warning(
            "Native features undefined for %d series; feature counts: %s",
            invalid.any(axis=1).sum(),
            counts,
        )
    columns = nw.DataFrame.from_dict(
        {name: values[:, i] for i, name in enumerate(names)}, backend=ids.implementation
    )
    return ids.with_columns(*columns.iter_columns()).to_native()


def _feature_matrix(
    df: IntoDataFrameT, names: tuple[str, ...], id_col: str, missing: str, corpus: str
) -> tuple[np.ndarray, CoverageDiagnostics]:
    frame = nw.from_native(df, eager_only=True)
    if id_col not in frame.columns or not len(frame):
        raise ValueError(
            f"{corpus}: feature frame must be non-empty and contain {id_col!r}"
        )
    if frame[id_col].is_null().any() or frame[id_col].n_unique() != len(frame):
        raise ValueError(f"{corpus}: feature IDs must be unique and non-null")
    if frame.schema[id_col].is_numeric() and not np.all(
        np.isfinite(frame[id_col].to_numpy())
    ):
        raise ValueError(f"{corpus}: numeric feature IDs must be finite")
    _feature_names(names, [c for c in frame.columns if c != id_col])
    if any(not frame.schema[name].is_numeric() for name in names):
        raise ValueError(f"{corpus}: selected feature columns must be numeric")
    values = frame.select(names).to_numpy().astype(float)
    ids = frame[id_col].to_list()
    invalid = ~np.isfinite(values)
    retained = ~invalid.any(axis=1)
    reasons = {
        ids[row]: tuple(
            name for name, bad in zip(names, invalid[row], strict=True) if bad
        )
        for row in np.flatnonzero(~retained)
    }
    if reasons:
        if missing == "raise":
            raise ValueError(
                f"{corpus}: {len(reasons)} series have non-finite features; select valid features or explicitly use missing='drop'"
            )
        logger.warning(
            "%s: dropping %d/%d series with non-finite features",
            corpus,
            len(reasons),
            len(ids),
        )
    if not retained.any():
        raise ValueError(f"{corpus}: no retained series remain")
    diagnostics = CoverageDiagnostics(
        len(ids),
        tuple(uid for uid, keep in zip(ids, retained, strict=True) if keep),
        tuple(reasons),
        reasons,
    )
    return values[retained], diagnostics


def compare_feature_coverage(
    real: IntoDataFrameT,
    synthetics: Mapping[str, IntoDataFrameT],
    *,
    seasonal_period: int | None = None,
    freq: str | None = None,
    window_size: int | None = None,
    features: Sequence[str] | None = None,
    precomputed: bool = False,
    embedding: Literal["pca", "tsne"] = "pca",
    fit: Literal["real", "pooled"] = "real",
    grid_range: Literal["real", "pooled"] | None = None,
    n_bins: int = 30,
    seed: int | None = None,
    missing: Literal["raise", "drop"] = "raise",
    n_jobs: int = -1,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> dict[str, CoverageResult]:
    """Compare synthetic corpora using a common feature space and grid.

    Defaults fit the scaler, PCA, constant-column selection, and padded grid
    on retained real data only, so synthetic corpora cannot shift that space
    and adding a corpus leaves existing scores unchanged. Corpus size still
    affects occupancy; compare matched sizes.

    Features are computed per series from observation counts alone, so the
    frequency and calendar of ``time_col`` are never compared across corpora,
    and ``crossing_points`` is a raw count that grows with series length.
    Generate synthetic series at lengths comparable to the real panel; the
    preset pools default to fixed lengths that may not match yours.

    Args:
        real: Reference panel, or its feature frame when ``precomputed``.
        synthetics: Corpora to score, keyed by name, in the same form as
            ``real``.
        seasonal_period: Native extraction period, as in
            :func:`compute_features`. Without it or ``freq``, the seasonal
            features are constant and drop out of the space (logged).
        freq: Pandas offset alias from which to derive the period, as in
            :func:`compute_features`.
        window_size: Native extraction window, as in :func:`compute_features`.
        features: Feature columns to embed. None selects all twelve native
            features or, when ``precomputed``, all non-ID columns, which must
            then match across corpora.
        precomputed: Treat inputs as feature frames (an ID column plus numeric
            feature columns) and skip native extraction. ``seasonal_period``,
            ``freq``, ``window_size``, and non-default ``time_col`` and
            ``target_col`` are then rejected rather than silently ignored;
            record extraction settings alongside the cache.
        embedding: ``"pca"``, or ``"tsne"`` (requires scikit-learn,
            ``fit="pooled"``, an explicit ``seed``, and more than 30 retained
            series).
        fit: ``"real"`` fits the space on the reference alone; ``"pooled"``
            fits on all corpora jointly, a sensitivity analysis whose scores
            change when a corpus is added.
        grid_range: ``"real"`` (padded reference range) or ``"pooled"``.
            None follows ``fit``.
        n_bins: Grid cells per axis. Each result holds two ``n_bins**2``
            boolean grids, about 1.8 KB at the default 30.
        seed: t-SNE random state.
        missing: ``"raise"`` rejects undefined feature values; ``"drop"``
            evaluates complete cases and reports excluded IDs and reasons.
        n_jobs: Threads for native extraction and the t-SNE neighbour search;
            -1 uses all CPUs, as in the generators.
        id_col: Series identifier column.
        time_col: Time column of raw panels.
        target_col: Target column of raw panels.

    Returns:
        A :class:`CoverageResult` per synthetic corpus, keyed as
        ``synthetics``. Each result's metadata stores the fitted parameters
        and the native or caller-defined schema.
    """
    if (
        not isinstance(synthetics, Mapping)
        or not synthetics
        or any(not isinstance(k, str) for k in synthetics)
    ):
        raise ValueError(
            "synthetics must be a non-empty mapping with string corpus names"
        )
    if embedding not in ("pca", "tsne") or fit not in ("real", "pooled"):
        raise ValueError(
            "embedding must be 'pca' or 'tsne'; fit must be 'real' or 'pooled'"
        )
    grid_range = fit if grid_range is None else grid_range
    if grid_range not in ("real", "pooled") or missing not in ("raise", "drop"):
        raise ValueError(
            "grid_range must be 'real' or 'pooled'; missing must be 'raise' or 'drop'"
        )
    n_bins = _integer(n_bins, "n_bins", 1)
    workers = _workers(n_jobs)
    if features is not None and not isinstance(features, str) and id_col in features:
        raise ValueError("id_col must not be a selected feature")
    if embedding == "tsne":
        if fit != "pooled" or seed is None:
            raise ValueError("t-SNE requires fit='pooled' and an explicit seed")
        seed = _integer(seed, "seed", 0)
    corpus_names = list(synthetics)
    frames = [real, *synthetics.values()]
    if precomputed:
        ignored = [
            name
            for name, value in (
                ("seasonal_period", seasonal_period),
                ("freq", freq),
                ("window_size", window_size),
            )
            if value is not None
        ]
        ignored += [
            name
            for name, value, default in (
                ("time_col", time_col, "ds"),
                ("target_col", target_col, "y"),
            )
            if value != default
        ]
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} cannot be combined with precomputed=True; "
                "these describe native extraction, which is skipped for cached "
                "features. Record the extraction settings alongside the cache."
            )
        available = [
            c for c in nw.from_native(real, eager_only=True).columns if c != id_col
        ]
        names = _feature_names(features, available)
        if features is None:
            for frame in frames[1:]:
                columns = set(nw.from_native(frame, eager_only=True).columns) - {id_col}
                if columns != set(names):
                    raise ValueError(
                        "Precomputed feature columns must match across corpora"
                    )
        schema = "precomputed"
        period = None
    else:
        names = _feature_names(features, FEATURE_NAMES)
        period = _seasonal_period(seasonal_period, freq)
        frames = [
            compute_features(
                frame,
                seasonal_period=period,
                window_size=window_size,
                features=names,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                n_jobs=n_jobs,
            )
            for frame in frames
        ]
        schema = FEATURE_SCHEMA
    prepared = [
        _feature_matrix(frame, names, id_col, missing, label)
        for frame, label in zip(frames, ["real", *corpus_names], strict=True)
    ]
    matrices = [item[0] for item in prepared]
    diagnostics = [item[1] for item in prepared]
    embedded = _coverage.embed_features(matrices, embedding, fit, seed, workers)
    edges = _coverage.grid_edges(embedded.coordinates, n_bins, grid_range)
    constant_names = [names[j] for j in embedded.constants]
    seasonal_dropped = [
        name for name in ("seasonal_strength", "seas_acf1") if name in constant_names
    ]
    if not precomputed and period is None and seasonal_dropped:
        logger.warning(
            "No seasonal period declared, so %s are constant and excluded from "
            "the coverage space; pass seasonal_period or freq to compare "
            "seasonality",
            seasonal_dropped,
        )
    parameters = {
        **embedded.parameters,
        "features": names,
        "schema": schema,
        "seasonal_period": period,
        "window_size": window_size if not precomputed else None,
        "effective_window_size": (window_size or period or 10)
        if not precomputed
        else None,
        "embedding": embedding,
        "fit": fit,
        "grid_range": grid_range,
        "grid_edges": [edge.tolist() for edge in edges],
    }
    if embedding == "tsne":
        parameters["fitted_coordinates"] = [x.tolist() for x in embedded.coordinates]
        parameters["fitted_ids"] = [
            [repr(uid) for uid in d.retained_ids] for d in diagnostics
        ]
        parameters["corpus_names"] = corpus_names
    space_id = _coverage.fingerprint(parameters)
    reference_id = _coverage.matrix_fingerprint(
        diagnostics[0].retained_ids, matrices[0]
    )
    real_occupancy, real_cells, _ = _coverage.grid_occupancy(
        embedded.coordinates[0], edges
    )
    results = {}
    for i, corpus in enumerate(corpus_names, start=1):
        occupancy, _, inside = _coverage.grid_occupancy(embedded.coordinates[i], edges)
        miscoverage, reverse, cell_fraction = _coverage.occupancy_scores(
            real_occupancy, occupancy
        )
        uncovered = ~occupancy[real_cells[:, 0], real_cells[:, 1]]
        out_of_range = int((~inside).sum())
        deviations = {
            names[column]: int(
                np.count_nonzero(
                    np.abs(matrices[i][:, column] - value)
                    > _coverage.constant_tolerance(value)
                )
            )
            for column, value in zip(
                embedded.constants, embedded.constant_values, strict=True
            )
        }
        results[corpus] = CoverageResult(
            miscoverage=miscoverage,
            reverse_miscoverage=reverse,
            uncovered_real_cell_fraction=cell_fraction,
            uncovered_real_series_fraction=float(uncovered.mean()),
            n_bins=n_bins,
            embedding=embedding,
            explained_variance=embedded.explained_variance,
            real_embedding=embedded.coordinates[0].copy(),
            synthetic_embedding=embedded.coordinates[i].copy(),
            real_occupancy=real_occupancy.copy(),
            synthetic_occupancy=occupancy,
            grid_edges=(edges[0].copy(), edges[1].copy()),
            uncovered_real_ids=tuple(
                uid
                for uid, bad in zip(diagnostics[0].retained_ids, uncovered, strict=True)
                if bad
            ),
            n_synthetic_out_of_range=out_of_range,
            synthetic_out_of_range_fraction=out_of_range / len(inside),
            features_used=tuple(names[j] for j in embedded.kept),
            constant_features_dropped=tuple(constant_names),
            synthetic_constant_feature_deviations=deviations,
            real_diagnostics=diagnostics[0],
            synthetic_diagnostics=diagnostics[i],
            fit=fit,
            grid_range=grid_range,
            space_id=space_id,
            metadata={
                "parameters": copy.deepcopy(parameters),
                "reference_id": reference_id,
            },
        )
    return results


def feature_coverage(
    real: IntoDataFrameT,
    synthetic: IntoDataFrameT,
    *,
    seasonal_period: int | None = None,
    freq: str | None = None,
    window_size: int | None = None,
    features: Sequence[str] | None = None,
    precomputed: bool = False,
    embedding: Literal["pca", "tsne"] = "pca",
    fit: Literal["real", "pooled"] = "real",
    grid_range: Literal["real", "pooled"] | None = None,
    n_bins: int = 30,
    seed: int | None = None,
    missing: Literal["raise", "drop"] = "raise",
    n_jobs: int = -1,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> CoverageResult:
    """Compare one synthetic corpus; see :func:`compare_feature_coverage` for the protocol."""
    return compare_feature_coverage(
        real,
        {"synthetic": synthetic},
        seasonal_period=seasonal_period,
        freq=freq,
        window_size=window_size,
        features=features,
        precomputed=precomputed,
        embedding=embedding,
        fit=fit,
        grid_range=grid_range,
        n_bins=n_bins,
        seed=seed,
        missing=missing,
        n_jobs=n_jobs,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )["synthetic"]
