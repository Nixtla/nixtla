"""Dataset class for generating synthetic time series from multiple generators."""

import logging
from typing import Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoFrameT

from synforecast._lib import batch as _rs_batch
from synforecast.augment import (
    DBA,
    BaseAugmenter,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
    analyze,
)
from synforecast.base import (
    BaseGenerator,
    _batch_results_to_metadata,
    _categorize_ids,
)

logger = logging.getLogger(__name__)


class SynSet:
    """Generate synthetic time series datasets from multiple generators.

    Combine multiple generators into one long-format panel; each generator
    contributes its own type of time series pattern.

    Args:
        generators (list[BaseGenerator]): List of instantiated generator objects to use for creating time series.

    Examples:
        >>> from synforecast import SynSet
        >>> from synforecast.generators import RandomWalkGenerator, SeasonalGenerator
        >>>
        >>> # Create generators
        >>> rw_gen = RandomWalkGenerator(
        ...     min_length=100,
        ...     max_length=150,
        ...     freq="h",
        ...     seed=42,
        ... )
        >>> seasonal_gen = SeasonalGenerator(
        ...     min_length=100,
        ...     max_length=150,
        ...     freq="h",
        ...     seed=43,
        ... )
        >>>
        >>> # Create dataset
        >>> dataset = SynSet([rw_gen, seasonal_gen])
        >>> df = dataset.generate(n_series_per_generator=5)
    """

    def __init__(self, generators: list[BaseGenerator]) -> None:
        """Initialize the SynSet with a list of generators.

        Args:
            generators (list[BaseGenerator]): List of instantiated generator objects.

        Raises:
            ValueError: If generators list is empty or contains non-BaseGenerator objects.
                If generators have inconsistent column names (id_col, time_col, target_col).
        """
        if not generators:
            raise ValueError("generators list cannot be empty")

        for i, gen in enumerate(generators):
            if not isinstance(gen, BaseGenerator):
                raise ValueError(
                    f"Generator at index {i} is not an instance of BaseGenerator"
                )

        # Check that all generators have the same column names
        if len(generators) > 1:
            first_gen = generators[0]
            reference_id_col = first_gen.id_col
            reference_time_col = first_gen.time_col
            reference_target_col = first_gen.target_col
            reference_freq = first_gen.freq

            for i, gen in enumerate(generators[1:], start=1):
                if gen.id_col != reference_id_col:
                    raise ValueError(
                        f"Generator at index {i} has id_col='{gen.id_col}' "
                        f"but generator at index 0 has id_col='{reference_id_col}'. "
                        "All generators must have the same id_col."
                    )
                if gen.time_col != reference_time_col:
                    raise ValueError(
                        f"Generator at index {i} has time_col='{gen.time_col}' "
                        f"but generator at index 0 has time_col='{reference_time_col}'. "
                        "All generators must have the same time_col."
                    )
                if gen.target_col != reference_target_col:
                    raise ValueError(
                        f"Generator at index {i} has target_col='{gen.target_col}' "
                        f"but generator at index 0 has target_col='{reference_target_col}'. "
                        "All generators must have the same target_col."
                    )
                if gen.freq != reference_freq:
                    logger.warning(
                        f"Generator at index {i} has freq='{gen.freq}' "
                        f"but generator at index 0 has freq='{reference_freq}'. "
                        "Mixing different frequencies may lead to inconsistent time series. "
                        "Consider using the same frequency for all generators."
                    )
                if gen.exogenous != first_gen.exogenous:
                    logger.warning(
                        f"Generator at index {i} has a different exogenous config "
                        f"than generator at index 0. This may cause column mismatches "
                        "when concatenating results."
                    )

        self.id_col = generators[0].id_col
        self.time_col = generators[0].time_col
        self.target_col = generators[0].target_col
        self.generators = generators

    def generate(
        self,
        n_series_per_generator: int,
        n_jobs: int = -1,
    ) -> IntoFrameT:
        """Generate synthetic time series data from all generators.

        Args:
            n_series_per_generator (int): Number of time series to generate from each generator.
            n_jobs (int): Number of parallel workers. -1 (default) uses
                ``RAYON_NUM_THREADS`` if set, otherwise all logical cores.
                Results are seed-deterministic and do not depend on n_jobs.

        Returns:
            DataFrame containing all generated time series from all generators, in long format with columns [id_col, time_col, target_col].

        Notes:
            Series IDs are unique across all generators. With 2 generators and
            3 series per generator, IDs run from 0 to 5.
        """
        all_series = self._generate_parallel(n_series_per_generator, n_jobs)

        # Concatenating categoricals with disjoint categories can degrade or
        # reject the dtype, so concat on plain integer ids and re-categorize.
        as_int = [
            df.with_columns(nw.col(self.id_col).cast(nw.String()).cast(nw.Int64()))
            for df in all_series
        ]
        result = _categorize_ids(nw.concat(as_int), self.id_col)

        return result.to_native()

    def _generate_parallel(
        self,
        n_series_per_generator: int,
        n_jobs: int = -1,
    ) -> list[nw.DataFrame]:
        """Generate from all generators in parallel via Rust rayon.

        Partitions generators into batch-eligible (have _get_batch_params)
        and excluded. Batch-eligible generators are run through a single
        ``generate_multi_batch`` call so rayon can work-steal across all
        series of all generators. Generators without a Rust batch implementation
        use the threaded path (ThreadPoolExecutor).
        """
        n = n_series_per_generator

        # Partition generators
        batch_specs: list[dict] = []
        batch_gen_indices: list[int] = []  # index into self.generators
        excluded_gen_indices: list[int] = []
        # Pre-sampled metadata per batch-eligible generator
        batch_meta: list[dict] = []

        for gi, gen in enumerate(self.generators):
            gen_type = gen._batch_gen_type
            batch_params = gen._get_batch_params() if gen_type is not None else None
            if batch_params is None:
                excluded_gen_indices.append(gi)
                continue

            scalar_params, array_params = batch_params
            lengths = np.array(
                [
                    int(gen.rng.integers(gen.min_length, gen.max_length + 1))
                    for _ in range(n)
                ],
                dtype=np.int32,
            )
            gen_seeds = gen.rng.integers(0, 2**63, size=n, dtype=np.uint64)
            pi_seeds = gen.rng.integers(0, 2**63, size=n * 3, dtype=np.uint64)

            batch_specs.append(
                {
                    "gen_type": gen_type,
                    "scalar_params": scalar_params,
                    "array_params": array_params,
                    "lengths": lengths,
                    "gen_seeds": gen_seeds,
                    "pi_seeds": pi_seeds,
                    "pi_config": gen._build_pi_config_dict(),
                }
            )
            batch_gen_indices.append(gi)
            batch_meta.append({"lengths": lengths, "gen": gen})

        # Run batch-eligible generators via Rust rayon
        batch_results: list[list[dict]] = []
        if batch_specs:
            batch_results = _rs_batch.generate_multi_batch(
                batch_specs,
                n_workers=n_jobs if n_jobs > 0 else 0,
            )

        # Build output list in original generator order
        results_by_gi: dict[int, nw.DataFrame] = {}

        # Convert batch results to DataFrames
        for bi, gen_results in enumerate(batch_results):
            gi = batch_gen_indices[bi]
            gen = batch_meta[bi]["gen"]
            lengths_arr = batch_meta[bi]["lengths"]
            start_id = gi * n

            all_metadata = _batch_results_to_metadata(
                gen, gen_results, lengths_arr, start_id
            )
            df = gen._build_dataframe(all_metadata)
            results_by_gi[gi] = nw.from_native(df)

        # Run excluded generators sequentially
        for gi in excluded_gen_indices:
            gen = self.generators[gi]
            result = gen.generate(n_series=n, start_id=gi * n, n_jobs=n_jobs)
            if isinstance(result, tuple):
                df, _ = result
            else:
                df = result
            results_by_gi[gi] = nw.from_native(df)

        # Assemble in original order
        return [results_by_gi[gi] for gi in range(len(self.generators))]


class SynAugment:
    """Augment time series datasets with synthetic series.

    A thin facade over the augmentation specs in :mod:`synforecast.augment`,
    which are the primary API: :class:`~synforecast.FitAndSimulate`,
    :class:`~synforecast.Mixup`, :class:`~synforecast.MovingBlockBootstrap`
    and :class:`~synforecast.DBA`. Each method builds the matching spec and
    calls its ``apply`` with this instance's columns, engine and a seed. The
    first drawing call uses ``seed`` itself; each later call uses a distinct
    seed derived from it, so repeated calls give fresh draws while a new
    instance with the same ``seed`` replays the same sequence of outputs.
    Call a spec's ``apply`` directly for output that is identical per call.

    - :meth:`augment` fits a parametric generator per series and simulates
      from it (IDs ``"{original_id}_aug_{i}"``).
    - :meth:`mixup` forms convex combinations of scaled series (mean absolute
      scaling by default; z-normalization is optional)
      (IDs ``"mixup_{i}"``).
    - :meth:`mbb` bootstraps decomposition remainders in moving blocks
      (IDs ``"{original_id}_mbb_{i}"``).
    - :meth:`dba` averages nearest neighbours with DTW barycenters
      (IDs ``"{original_id}_dba_{i}"``).

    Series that are entirely missing (or, for :meth:`mbb`, shorter than four
    observations) are skipped with a logged warning by :meth:`mixup`,
    :meth:`mbb`, and :meth:`dba`.

    Args:
        id_col: Name of the ID column (default: 'unique_id')
        time_col: Name of the timestamp column (default: 'ds')
        target_col: Name of the value column (default: 'y')
        seed: Random seed for reproducibility
        engine: Output dataframe library. None (default) matches the input
            DataFrame's library.

    Example:
        >>> from synforecast import SynAugment
        >>> import polars as pl
        >>>
        >>> # Create sample data
        >>> df = pl.DataFrame(
        ...     {
        ...         "unique_id": ["series_0"] * 100,
        ...         "ds": pl.date_range(
        ...             pl.date(2020, 1, 1), pl.date(2020, 4, 9), eager=True
        ...         ),
        ...         "y": [i + np.random.randn() for i in range(100)],
        ...     }
        ... )
        >>>
        >>> # Augment the dataset
        >>> augmenter = SynAugment(seed=42)
        >>> augmented_df = augmenter.augment(df, n_augment=3)
        >>> augmented_df["unique_id"].n_unique()
        4
    """

    def __init__(
        self,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        seed: int | None = None,
        engine: str | None = None,
        on_error: Literal["raise", "ar1"] = "raise",
    ) -> None:
        """Initialize the SynAugment instance.

        Args:
            id_col: Name of the ID column
            time_col: Name of the timestamp column
            target_col: Name of the value column
            seed: Random seed for reproducibility
            engine: Output dataframe library (e.g. 'pandas', 'polars').
                None (default) matches the input DataFrame's library.
            on_error: What to do when a fitted generator fails for a series.
                'raise' (default) propagates the error; 'ar1' substitutes an
                AR(1) series matching the source's mean, std, and lag-1
                autocorrelation, and reports all substitutions in a single
                warning at the end of `augment`.
        """
        if on_error not in ("raise", "ar1"):
            raise ValueError("on_error must be 'raise' or 'ar1'")
        if seed is not None and seed < 0:
            raise ValueError("seed must be >= 0")
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.seed = seed
        self.engine = engine
        self.on_error = on_error
        self._seeds = None if seed is None else np.random.SeedSequence(seed)

    def _next_seed(self) -> int | None:
        """Seed for the next drawing call: ``seed`` first, then derived ones."""
        if self._seeds is None:
            return None
        child = self._seeds.spawn(1)[0]
        if self._seeds.n_children_spawned == 1:
            return self.seed
        return int(child.generate_state(1)[0])

    def _apply(
        self,
        spec: BaseAugmenter,
        df: IntoFrameT,
        include_original: bool = True,
        source_col: str | None = None,
    ) -> IntoFrameT:
        return spec.apply(
            df,
            seed=self._next_seed(),
            include_original=include_original,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            engine=self.engine,
            source_col=source_col,
        )

    def analyze(self, df: IntoFrameT) -> dict[str, dict]:
        """Analyze all series in DataFrame and return properties.

        See :func:`synforecast.analyze`.

        Args:
            df: DataFrame with time series data (must have id_col, time_col, target_col)

        Returns:
            Mapping of each `unique_id` to its analysis results. Every entry holds `recommended_generator` (the chosen generator name), `properties` (all detected statistical properties), and `fitted_params` (the estimated generator parameters).

        Raises:
            ValueError: If DataFrame is missing required columns
        """
        return analyze(
            df, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col
        )

    def augment(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        generator_override: dict[str | int, str] | None = None,
        preserve_timestamps: bool = True,  # noqa: ARG002
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Augment dataset with synthetic series.

        Runs :class:`~synforecast.FitAndSimulate`: for each series in the
        input DataFrame, generates n_augment synthetic series that are
        statistically similar to the original.

        Args:
            df: Input DataFrame with time series (must have id_col, time_col, target_col)
            n_augment: Number of synthetic series to generate per original series
            generator_override: Optional dict mapping unique_id to generator name.
                Overrides automatic generator selection for specified series.
                Example: {"series_0": "SARIMAGenerator"}
            preserve_timestamps: Accepted for compatibility and ignored;
                synthetic series always use the original's timestamps
            source_col: Name of an added column holding each row's source
                series id; None (default) adds no column.

        Returns:
            Combined DataFrame with original and synthetic series. Synthetic series IDs follow the pattern `"{original_id}_aug_{i}"`

        Raises:
            ValueError: If DataFrame is missing required columns, n_augment < 1,
                or generator_override names an unsupported generator
            RuntimeError: If a fitted generator fails for a series and the
                instance was created with on_error='raise' (the default)

        Example:
            >>> augmenter = SynAugment(seed=42)
            >>> # Basic augmentation
            >>> augmented_df = augmenter.augment(df, n_augment=3)  # doctest: +SKIP
            >>>
            >>> # With generator override
            >>> augmented_df = augmenter.augment(  # doctest: +SKIP
            ...     df, n_augment=2, generator_override={"series_0": "SARIMAGenerator"}
            ... )
        """
        spec = FitAndSimulate(
            n_augment=n_augment,
            generator_override=generator_override,
            on_error=self.on_error,
        )
        return self._apply(spec, df, source_col=source_col)

    def mixup(
        self,
        df: IntoFrameT,
        n_series: int | None = None,
        max_mix: int = 3,
        alpha: float = 1.5,
        scaling: Literal["mean", "std", "none"] = "mean",
        include_original: bool = True,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Augment a dataset with TSMixup convex combinations of real series.

        Runs :class:`~synforecast.Mixup`, which documents the method.

        Args:
            df: Input DataFrame with time series (must have id_col, time_col,
                target_col).
            n_series: Total number of synthetic series to create. Defaults to
                the number of input series.
            max_mix: Maximum number of source series combined per synthetic
                series (default: 3).
            alpha: Concentration of the symmetric Dirichlet weight
                distribution (default: 1.5).
            scaling: Per-window scaling before mixing: 'mean', 'std', or
                'none' (default: 'mean').
            include_original: Prepend the original series to the output
                (default: True).
            source_col: Name of an added column holding each row's source
                series id; None (default) adds no column.

        Returns:
            DataFrame with the original series (optional) and synthetic mixup series. Mixup IDs follow the pattern ``"mixup_{i}"``.

        Raises:
            ValueError: If the DataFrame is missing required columns, if
                arguments are out of range, or if it holds no usable series.

        Example:
            >>> augmenter = SynAugment(seed=42)
            >>> mixed_df = augmenter.mixup(df, n_series=50)  # doctest: +SKIP
        """
        spec = Mixup(n_series=n_series, max_mix=max_mix, alpha=alpha, scaling=scaling)
        return self._apply(spec, df, include_original, source_col)

    def mbb(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        seasonal_period: int | None = None,
        block_size: int | None = None,
        include_original: bool = True,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Bootstrap decomposition remainders with moving blocks.

        Runs :class:`~synforecast.MovingBlockBootstrap`, which documents the
        method.

        Args:
            df: Input frame containing the configured ID, time and target columns.
            n_augment: Copies per usable source, from 1 to 1000 (default: 1).
            seasonal_period: Period >= 2, or None to detect it per source.
            block_size: Block length >= 2, or None for the period/length-based
                default, capped per source at ``max(2, len(series) // 3)``.
            include_original: Include unchanged original rows (default: True).
            source_col: Name of an added column holding each row's source
                series id; None (default) adds no column.

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the configured output engine or the input engine.

        Raises:
            ValueError: For invalid arguments, missing required columns, infinite
                target values, or no usable source series.
        """
        spec = MovingBlockBootstrap(
            n_augment=n_augment,
            seasonal_period=seasonal_period,
            block_size=block_size,
        )
        return self._apply(spec, df, include_original, source_col)

    def dba(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        n_neighbors: int = 3,
        n_iterations: int = 5,
        window_fraction: float = 0.1,
        scale: Literal["reference", "none"] = "reference",
        include_original: bool = True,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Augment a panel with randomized nearest-neighbor DTW barycenters.

        Runs :class:`~synforecast.DBA`, which documents the method.

        Args:
            df: Input frame containing the configured ID, time and target columns.
            n_augment: Copies per usable reference, from 1 to 1000 (default: 1).
            n_neighbors: Positive maximum neighbor count (default: 3), capped
                at the number of other usable sources.
            n_iterations: Barycenter updates, from 1 to 1000 (default: 5).
            window_fraction: DTW band fraction in (0, 1] (default: 0.1).
            scale: "reference" to restore reference scale (default), or "none"
                to average raw values.
            include_original: Include unchanged original rows (default: True).
            source_col: Name of an added column holding each row's source
                series id; None (default) adds no column.

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the configured output engine or the input engine.

        Raises:
            ValueError: For invalid arguments, missing required columns, infinite
                target values, excessive alignment/output storage, or fewer
                than two usable source series.
        """
        spec = DBA(
            n_augment=n_augment,
            n_neighbors=n_neighbors,
            n_iterations=n_iterations,
            window_fraction=window_fraction,
            scale=scale,
        )
        return self._apply(spec, df, include_original, source_col)

    def augment_single_series(
        self,
        series_id: str,
        values: np.ndarray,
        timestamps: np.ndarray,
        n_augment: int = 1,
        generator_name: str | None = None,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Augment a single series (lower-level API).

        Runs :meth:`~synforecast.FitAndSimulate.apply_series`.

        Args:
            series_id: ID of the original series
            values: Array of time series values
            timestamps: Array of timestamps
            n_augment: Number of synthetic series to generate
            generator_name: Optional generator name; if None, auto-detect

        Returns:
            List of tuples: (new_id, synthetic_values, timestamps)
        """
        spec = FitAndSimulate(n_augment=n_augment, on_error=self.on_error)
        return spec.apply_series(
            series_id,
            values,
            timestamps,
            generator_name=generator_name,
            seed=self._next_seed(),
        )
