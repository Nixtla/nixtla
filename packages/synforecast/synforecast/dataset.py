"""Dataset class for generating synthetic time series from multiple generators."""

import logging
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoFrameT

from synforecast._analysis import classify_series, detect_seasonality
from synforecast._dtw import nearest_dtw_neighbors
from synforecast._fitting import fit_generator_params
from synforecast._lib import augmentation as _rs_augmentation
from synforecast._lib import batch as _rs_batch
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

    Four strategies share the same column contract and seeding:

    - :meth:`augment` fits a parametric generator per series and simulates
      from it (IDs ``"{original_id}_aug_{i}"``).
    - :meth:`mixup` forms convex combinations of scaled series (mean absolute
      scaling by default; z-normalization is optional)
      (IDs ``"mixup_{i}"``).
    - :meth:`mbb` bootstraps decomposition remainders in moving blocks
      (IDs ``"{original_id}_mbb_{i}"``).
    - :meth:`dba` averages nearest neighbours with DTW barycenters
      (IDs ``"{original_id}_dba_{i}"``).

    The fit-and-simulate process behind :meth:`augment`:
    1. For each unique series in the input DataFrame, analyze its statistical properties
    2. Auto-select the most appropriate generator (or use user override)
    3. Fit generator parameters to match the series' statistical fingerprint
    4. Generate n_augment synthetic series that preserve these properties
    5. Return combined DataFrame with original and synthetic series

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

    # Mapping from generator names to classes
    _GENERATOR_MAP: dict[str, type] = {}

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
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.seed = seed
        self.engine = engine
        self.on_error = on_error
        self.rng = np.random.default_rng(seed)

        # Lazy import of generators to avoid circular imports
        self._load_generators()

    def _load_generators(self) -> None:
        """Lazily load generator classes."""
        if not SynAugment._GENERATOR_MAP:
            from synforecast.generators import (
                FractionalBrownianMotionGenerator,
                GARCHGenerator,
                GeometricBrownianMotionGenerator,
                IntermittentDemandGenerator,
                OrnsteinUhlenbeckGenerator,
                RandomWalkGenerator,
                RegimeSwitchingGenerator,
                SARIMAGenerator,
                SeasonalGenerator,
            )

            SynAugment._GENERATOR_MAP = {
                "RandomWalkGenerator": RandomWalkGenerator,
                "SeasonalGenerator": SeasonalGenerator,
                "SARIMAGenerator": SARIMAGenerator,
                "GARCHGenerator": GARCHGenerator,
                "OrnsteinUhlenbeckGenerator": OrnsteinUhlenbeckGenerator,
                "FractionalBrownianMotionGenerator": FractionalBrownianMotionGenerator,
                "RegimeSwitchingGenerator": RegimeSwitchingGenerator,
                "IntermittentDemandGenerator": IntermittentDemandGenerator,
                "GeometricBrownianMotionGenerator": GeometricBrownianMotionGenerator,
            }

    def _to_output_backend(self, frame: nw.DataFrame, backend: Any) -> nw.DataFrame:
        """Convert an input frame when an explicit output engine was requested."""
        if self.engine is None:
            return frame
        return nw.DataFrame.from_dict(
            frame.to_dict(as_series=False),
            backend=backend,
        )

    def _partition_series(self, frame: nw.DataFrame) -> dict[Any, nw.DataFrame]:
        """Sort once, then take contiguous slices without per-ID panel scans."""
        ordered = frame.sort([self.id_col, self.time_col])
        ids = ordered[self.id_col].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        return {
            ids[start]: ordered[int(start) : int(stop)]
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
            if start < stop
        }

    def analyze(self, df: IntoFrameT) -> dict[str, dict]:
        """Analyze all series in DataFrame and return properties.

        For each unique series, detects statistical properties and recommends
        the most appropriate generator.

        Args:
            df: DataFrame with time series data (must have id_col, time_col, target_col)

        Returns:
            Mapping of each `unique_id` to its analysis results. Every entry holds `recommended_generator` (the chosen generator name), `properties` (all detected statistical properties), and `fitted_params` (the estimated generator parameters).

        Raises:
            ValueError: If DataFrame is missing required columns
        """
        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)

        return self._analyze_partitions(
            self._partition_series(df_nw), df_nw[self.id_col].unique().to_list()
        )

    def _analyze_partitions(
        self, partitions: dict[Any, nw.DataFrame], unique_ids: list[Any]
    ) -> dict:
        results = {}
        for series_id in unique_ids:
            series_data = (
                partitions[series_id].select(self.target_col).to_numpy().flatten()
            )

            # Classify and get properties
            classification = classify_series(series_data)

            # Fit parameters for recommended generator
            fitted_params = fit_generator_params(
                series_data,
                classification["recommended_generator"],
                classification["properties"],
            )

            results[series_id] = {
                "recommended_generator": classification["recommended_generator"],
                "properties": classification["properties"],
                "fitted_params": fitted_params,
            }

        return results

    def augment(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        generator_override: dict[str, str] | None = None,
        preserve_timestamps: bool = True,
    ) -> IntoFrameT:
        """Augment dataset with synthetic series.

        For each series in the input DataFrame, generates n_augment synthetic
        series that are statistically similar to the original.

        Args:
            df: Input DataFrame with time series (must have id_col, time_col, target_col)
            n_augment: Number of synthetic series to generate per original series
            generator_override: Optional dict mapping unique_id to generator name.
                Overrides automatic generator selection for specified series.
                Example: {"series_0": "SARIMAGenerator"}
            preserve_timestamps: If True, synthetic series use the same timestamps
                as the original series

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
        if n_augment < 1:
            raise ValueError("n_augment must be >= 1")
        if n_augment > 1000:
            raise ValueError("n_augment must be <= 1000 to prevent resource exhaustion")

        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)

        # Synthetic frames must match the input's dataframe library, or the
        # final concat would mix backends. An explicit engine wins.
        out_engine = self.engine if self.engine is not None else df_nw.implementation
        output_original = self._to_output_backend(df_nw, out_engine)

        generator_override = generator_override or {}

        partitions = self._partition_series(df_nw)
        unique_ids = df_nw[self.id_col].unique().to_list()
        analysis = self._analyze_partitions(partitions, unique_ids)

        # Collect all series (original + augmented)
        all_dfs = [output_original]

        # (series_id, generator_name, error) for every AR(1) substitution
        # made under on_error='ar1', reported once after the loop.
        fallbacks: list[tuple[str, str, str]] = []

        for series_id in unique_ids:
            series_df = partitions[series_id]
            series_values = series_df.select(self.target_col).to_numpy().flatten()
            timestamps = series_df.select(self.time_col).to_numpy().flatten()

            # Determine which generator to use
            if series_id in generator_override:
                generator_name = generator_override[series_id]
                # Re-fit parameters for the overridden generator
                fitted_params = fit_generator_params(
                    series_values,
                    generator_name,
                    analysis[series_id]["properties"],
                )
            else:
                generator_name = analysis[series_id]["recommended_generator"]
                fitted_params = analysis[series_id]["fitted_params"]

            # Generate augmented series
            augmented = self._generate_augmented_series(
                series_id=series_id,
                series_values=series_values,
                timestamps=timestamps,
                n_augment=n_augment,
                generator_name=generator_name,
                fitted_params=fitted_params,
                _preserve_timestamps=preserve_timestamps,
                engine=out_engine,
                fallbacks=fallbacks,
            )

            all_dfs.extend(augmented)

        self._warn_fallbacks(fallbacks, len(unique_ids) * n_augment)

        # Normalize concat-sensitive dtypes across backends. In particular,
        # pandas datetimes commonly convert to Polars at microsecond precision
        # while NumPy timestamps retain nanosecond precision.
        time_dtype = output_original.schema[self.time_col]
        all_dfs = [
            frame.with_columns(
                nw.col(self.id_col).cast(nw.String()),
                nw.col(self.time_col).cast(time_dtype),
                nw.col(self.target_col).cast(nw.Float64()),
            )
            for frame in all_dfs
        ]
        result = _categorize_ids(nw.concat(all_dfs), self.id_col)

        return result.to_native()

    @staticmethod
    def _warn_skipped(method: str, skipped: list[Any]) -> None:
        """Log one summary for series skipped as unusable by ``method``."""
        if skipped:
            logger.warning(
                "%s skipped %d unusable series (entirely missing or too short): %s",
                method,
                len(skipped),
                ", ".join(repr(series_id) for series_id in skipped[:10])
                + (", ..." if len(skipped) > 10 else ""),
            )

    @staticmethod
    def _warn_fallbacks(fallbacks: list[tuple[str, str, str]], total: int) -> None:
        """Log one summary for AR(1) substitutions made during an API call."""
        if not fallbacks:
            return

        by_generator: dict[str, int] = {}
        for _, gen_name, _ in fallbacks:
            by_generator[gen_name] = by_generator.get(gen_name, 0) + 1
        counts = ", ".join(f"{g}: {c}" for g, c in sorted(by_generator.items()))
        first_id, first_gen, first_err = fallbacks[0]
        logger.warning(
            f"SynAugment substituted AR(1) for {len(fallbacks)} of "
            f"{total} synthetic series ({counts}). "
            f"First failure: {first_gen} on series {first_id!r}: {first_err}"
        )

    def _scale_window(
        self, window: np.ndarray, scaling: Literal["mean", "std", "none"]
    ) -> np.ndarray:
        """Scale a window before mixing so different-magnitude series combine."""
        if scaling == "mean":
            denom = float(np.mean(np.abs(window)))
            return window / denom if denom > 1e-8 else window
        if scaling == "std":
            std = float(np.std(window))
            centered = window - float(np.mean(window))
            return centered / std if std > 1e-8 else centered
        return window

    @staticmethod
    def _interpolate_missing(
        values: np.ndarray, *, series_id: Any = None
    ) -> np.ndarray | None:
        """Linearly interpolate NaNs, or reject a series with no finite values."""
        if np.isinf(values).any():
            source = (
                "source series" if series_id is None else f"source series {series_id!r}"
            )
            raise ValueError(f"{source} must not contain infinite values")
        observed = ~np.isnan(values)
        if not observed.any():
            return None
        if observed.all():
            return values

        positions = np.arange(len(values))
        return np.interp(positions, positions[observed], values[observed])

    @staticmethod
    def _reserve_generated_id(proposed: str, reserved: set[str]) -> str:
        """Reserve a generated ID, adding a numeric suffix on collision."""
        candidate = proposed
        suffix = 1
        while candidate in reserved:
            candidate = f"{proposed}_{suffix}"
            suffix += 1
        reserved.add(candidate)
        return candidate

    def _copy_reference_frames(
        self,
        reference: nw.DataFrame,
        ids: list[str],
        copies: list[np.ndarray],
        backend: Any,
    ) -> nw.DataFrame:
        """Assemble all copies of one source in a single backend operation."""
        return nw.concat([reference] * len(ids)).with_columns(
            nw.new_series(self.id_col, np.repeat(ids, len(reference)), backend=backend),
            nw.new_series(self.target_col, np.concatenate(copies), backend=backend),
        )

    def mixup(
        self,
        df: IntoFrameT,
        n_series: int | None = None,
        max_mix: int = 3,
        alpha: float = 1.5,
        scaling: Literal["mean", "std", "none"] = "mean",
        include_original: bool = True,
    ) -> IntoFrameT:
        """Augment a dataset with TSMixup convex combinations of real series.

        Implements TSMixup, the augmentation used to pretrain the Chronos
        forecasting models (Ansari et al. 2024, "Chronos: Learning the Language
        of Time Series", https://arxiv.org/abs/2403.07815, Apache-2.0). Each
        synthetic series is built by drawing ``1..max_mix`` source series,
        taking a random window of their shared (minimum) length, scaling each
        window, and combining them with convex weights sampled from a
        ``Dirichlet(alpha)`` distribution. Unlike :meth:`augment`, a mixup
        series is not tied to a single source: it blends the dynamics of
        several, which broadens a small panel without fitting any generator.

        Because the mixed series live in scaled space (different sources have
        different magnitudes), the output retains the chosen scaling rather
        than any single source's level. Use ``scaling="none"`` to mix raw
        values when all series already share a scale.

        Args:
            df: Input DataFrame with time series (must have id_col, time_col,
                target_col).
            n_series: Number of synthetic series to create. Defaults to the
                number of input series.
            max_mix: Maximum number of source series combined per synthetic
                series; the count is drawn uniformly from ``1..max_mix`` and
                capped at the number of available series (default: 3).
            alpha: Concentration of the symmetric Dirichlet weight
                distribution. Values < 1 concentrate weight on few sources;
                larger values spread it more evenly (default: 1.5).
            scaling: Per-window scaling before mixing: 'mean' (divide by mean
                absolute value, Chronos default), 'std' (z-normalize), or
                'none' (default: 'mean').
            include_original: Prepend the original series to the output
                (default: True).

        Notes:
            NaNs in source series are linearly interpolated before mixing,
            including nearest-value filling at the ends. Entirely missing
            series are excluded. When ``include_original=True``, the original
            rows and their missing values are retained unchanged.

        Returns:
            DataFrame with the original series (optional) and synthetic mixup series. Mixup IDs follow the pattern ``"mixup_{i}"``.

        Raises:
            ValueError: If the DataFrame is missing required columns, if
                arguments are out of range, or if it holds no usable series.

        Example:
            >>> augmenter = SynAugment(seed=42)
            >>> mixed_df = augmenter.mixup(df, n_series=50)  # doctest: +SKIP
        """
        if max_mix < 1:
            raise ValueError("max_mix must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if scaling not in ("mean", "std", "none"):
            raise ValueError("scaling must be one of 'mean', 'std', 'none'")

        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)

        # Typed as Any to match augment(): from_dict's backend accepts both a
        # library name and an Implementation.
        out_engine: Any = (
            self.engine if self.engine is not None else df_nw.implementation
        )
        output_original = self._to_output_backend(df_nw, out_engine)

        # Sort ids so series selection is reproducible: some backends do not
        # guarantee an order from unique().
        unique_ids = sorted(df_nw[self.id_col].unique().to_list())

        # Cache each series' values and timestamps, sorted by time.
        series: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        skipped: list[Any] = []
        partitions = self._partition_series(df_nw)
        for series_id in unique_ids:
            sdf = partitions[series_id]
            values = sdf.select(self.target_col).to_numpy().flatten().astype(float)
            timestamps = sdf.select(self.time_col).to_numpy().flatten()
            if (
                len(values) >= 1
                and (values := self._interpolate_missing(values)) is not None
            ):
                series[series_id] = (values, timestamps)
            else:
                skipped.append(series_id)

        usable_ids = list(series.keys())
        n_available = len(usable_ids)
        self._warn_skipped("mixup", skipped)
        if n_available == 0:
            raise ValueError("no usable series to mix")

        if n_series is None:
            n_series = n_available
        if n_series < 1:
            raise ValueError("n_series must be >= 1")
        if n_series > 1_000_000:
            raise ValueError("n_series must be <= 1_000_000 to prevent exhaustion")

        reserved_ids = {str(series_id) for series_id in unique_ids}
        mixup_ids: list[str] = []
        candidate = 0
        while len(mixup_ids) < n_series:
            mixup_id = f"mixup_{candidate}"
            candidate += 1
            if mixup_id in reserved_ids:
                continue
            mixup_ids.append(mixup_id)
            reserved_ids.add(mixup_id)

        synthetic_dfs = []
        for mixup_id in mixup_ids:
            k_max = min(max_mix, n_available)
            k = int(self.rng.integers(1, k_max + 1))
            chosen = self.rng.choice(n_available, size=k, replace=False)
            chosen_ids = [usable_ids[c] for c in chosen]

            length = int(min(len(series[c][0]) for c in chosen_ids))
            weights = self.rng.dirichlet(np.full(k, alpha))

            mixed = np.zeros(length)
            anchor_timestamps = None
            for w, series_id in zip(weights, chosen_ids, strict=True):
                values, timestamps = series[series_id]
                start = int(self.rng.integers(0, len(values) - length + 1))
                window = values[start : start + length]
                if anchor_timestamps is None:
                    anchor_timestamps = timestamps[start : start + length]
                mixed += w * self._scale_window(window, scaling)

            mix_df = nw.DataFrame.from_dict(
                {
                    self.id_col: [mixup_id] * length,
                    self.time_col: anchor_timestamps,
                    self.target_col: mixed,
                },
                backend=out_engine,
            )
            synthetic_dfs.append(mix_df)

        frames = [output_original] if include_original else []
        frames.extend(synthetic_dfs)
        # Normalize id (mixup ids are strings) and target (mixed is float) so
        # the frames concatenate across the original and synthetic batches.
        time_dtype = output_original.schema[self.time_col]
        frames = [
            frame.with_columns(
                nw.col(self.id_col).cast(nw.String()),
                nw.col(self.time_col).cast(time_dtype),
                nw.col(self.target_col).cast(nw.Float64()),
            )
            for frame in frames
        ]
        result = _categorize_ids(nw.concat(frames), self.id_col)

        return result.to_native()

    def mbb(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        seasonal_period: int | None = None,
        block_size: int | None = None,
        include_original: bool = True,
    ) -> IntoFrameT:
        """Bootstrap decomposition remainders with moving blocks.

        This non-parametric augmenter follows the moving-block-bootstrap idea
        used for forecasting by Bergmeir, Hyndman, and Benitez (2016,
        https://doi.org/10.1016/j.ijforecast.2015.07.002) and the augmentation
        recipe discussed by Bandara et al. (2021,
        https://arxiv.org/abs/2008.02663). SynForecast uses a classical
        moving-average decomposition rather than Box-Cox plus STL; its
        defaults, block-size choice, and stability guards are its own design
        rather than a reproduction of the reference code.

        This method deliberately bypasses the ``_match_autocorrelation``
        rescaling used by :meth:`augment`. Recombining a source's own trend and
        seasonal components with bootstrapped remainder blocks already anchors
        the result to that source; re-pinning mean and standard deviation would
        distort the remainder variability.

        Classical decomposition and moving-block sampling execute in native
        Rust. Seed determinism is stable within this native path.

        NaNs are interpolated before decomposition. Series without a finite
        observation or with fewer than four observations are skipped with a
        logged warning; at least one series must be usable. Additional
        input columns are copied unchanged from each source series. The
        effective block length, including an explicitly supplied
        ``block_size``, is capped per source series at
        ``max(2, len(series) // 3)``. This keeps multiple remainder blocks in
        play for short inputs; request a smaller value when the exact block
        length matters.

        An explicit period requires at least two cycles per source. Shorter
        sources use nonseasonal decomposition and are listed in a logged warning.
        Infinite values reject the call with the offending series ID; NaNs
        follow the interpolation/skip policy above.

        Args:
            df: Input frame containing the configured ID, time and target columns.
            n_augment: Copies per usable source, from 1 to 1000 (default: 1).
            seasonal_period: Period >= 2, or None to detect it per source.
            block_size: Block length >= 2, or None for the period/length-based
                default. Explicit lengths are also capped as described above.
            include_original: Include unchanged original rows (default: True).

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the configured output engine or the input engine.

        Raises:
            ValueError: For invalid arguments, missing required columns, infinite
                target values, or no usable source series.
        """
        if n_augment < 1:
            raise ValueError("n_augment must be >= 1")
        if n_augment > 1000:
            raise ValueError("n_augment must be <= 1000 to prevent resource exhaustion")
        if seasonal_period is not None and seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2 when provided")
        if block_size is not None and block_size < 2:
            raise ValueError("block_size must be >= 2 when provided")

        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)
        out_engine: Any = (
            self.engine if self.engine is not None else df_nw.implementation
        )
        output_original = self._to_output_backend(df_nw, out_engine)
        unique_ids = sorted(df_nw[self.id_col].unique().to_list())
        if not unique_ids:
            raise ValueError("mbb requires at least 1 series")
        reserved_ids = {str(series_id) for series_id in unique_ids}
        synthetic_dfs: list[nw.DataFrame] = []
        skipped: list[Any] = []
        nonseasonal: list[Any] = []

        partitions = self._partition_series(df_nw)
        for series_id in unique_ids:
            sdf = partitions[series_id]
            raw_values = sdf.select(self.target_col).to_numpy().flatten().astype(float)
            values = self._interpolate_missing(raw_values, series_id=series_id)
            if values is None or len(values) < 4:
                skipped.append(series_id)
                continue
            output_sdf = self._to_output_backend(sdf, out_engine)
            period = (
                seasonal_period
                if seasonal_period is not None
                else detect_seasonality(values)["period"]
            )
            if seasonal_period is not None and seasonal_period > len(values) // 2:
                nonseasonal.append(series_id)
            default_block = (
                2 * period if period else max(2, int(round(len(values) ** 0.5)))
            )
            block = block_size if block_size is not None else default_block
            block = min(block, max(2, len(values) // 3))

            native_seeds = self.rng.integers(0, 2**63, size=n_augment).tolist()
            generated_copies = _rs_augmentation.moving_block_bootstrap_many(
                np.ascontiguousarray(values), block, native_seeds, period
            )
            generated_ids = [
                self._reserve_generated_id(f"{series_id}_mbb_{i}", reserved_ids)
                for i in range(n_augment)
            ]
            synthetic_dfs.append(
                self._copy_reference_frames(
                    output_sdf, generated_ids, generated_copies, out_engine
                )
            )

        self._warn_skipped("mbb", skipped)
        if nonseasonal:
            logger.warning(
                "mbb used nonseasonal decomposition for %s series with fewer than "
                "two cycles of seasonal_period=%s; first IDs: %s",
                len(nonseasonal),
                seasonal_period,
                nonseasonal[:5],
            )
        if len(skipped) == len(unique_ids):
            raise ValueError(
                "mbb requires at least 1 usable series (finite values, >= 4 rows)"
            )

        frames = [output_original] if include_original else []
        frames.extend(synthetic_dfs)
        time_dtype = output_original.schema[self.time_col]
        frames = [
            frame.with_columns(
                nw.col(self.id_col).cast(nw.String()),
                nw.col(self.time_col).cast(time_dtype),
                nw.col(self.target_col).cast(nw.Float64()),
            )
            for frame in frames
        ]
        return _categorize_ids(nw.concat(frames), self.id_col).to_native()

    def dba(
        self,
        df: IntoFrameT,
        n_augment: int = 1,
        n_neighbors: int = 3,
        n_iterations: int = 5,
        window_fraction: float = 0.1,
        scale: Literal["reference", "none"] = "reference",
        include_original: bool = True,
    ) -> IntoFrameT:
        """Augment a panel with randomized nearest-neighbor DTW barycenters.

        DBA comes from Petitjean, Ketterlin, and Gancarski (2011,
        https://doi.org/10.1016/j.patcog.2010.09.013); nearest-neighbor weighted
        augmentation follows Forestier et al. (2017,
        https://doi.org/10.1109/ICDM.2017.106) and Bandara et al. (2021,
        https://arxiv.org/abs/2008.02663). Neighbor weighting, per-copy weight
        randomization, banded DTW, and z-normalized alignment are SynForecast's
        own design rather than a reproduction of the reference code. Pairwise
        panel distances are computed once per call in bounded parallel chunks,
        retaining only the nearest neighbors per source. Neighbor storage is
        O(number of series * n_neighbors); distance computation still examines
        every pair. The band limits the otherwise quadratic alignment cost.

        This panel-aware method caches every usable source series once and
        deliberately bypasses ``_match_autocorrelation``. With the default
        ``scale="reference"``, inputs are z-normalized for alignment and the
        barycenter is mapped to the reference mean and standard deviation. That
        is explicit reference anchoring, not the :meth:`augment` pinning path.
        ``scale="none"`` averages raw values for panels already sharing scale.
        Plain DBA is deterministic, so each requested copy randomizes the
        reference and neighbor weights to create distinct augmentations.
        Additional input columns are copied unchanged from the reference
        series; DBA only barycenters the target column.

        Copies share their initial alignment paths and refine in parallel in
        native Rust. At most eight alignments/copies execute concurrently.
        Parent storage per alignment, cached paths per copy and returned copies
        per reference have separate 64 MiB limits; excessive requests raise
        ValueError. Interrupts are checked
        between work chunks. Seed
        determinism is stable within this native path.

        NaNs are interpolated; entirely missing sources are skipped with a
        warning. Infinite values reject the call with the offending series ID.
        Reference scaling preserves zero scale for constant sources, producing
        constant copies. Sources with standard deviation below 1e-8 are listed
        in a warning because their copies may add little or no variation.

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

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the configured output engine or the input engine.

        Raises:
            ValueError: For invalid arguments, missing required columns, infinite
                target values, excessive alignment/output storage, or fewer
                than two usable source series.
        """
        if n_augment < 1:
            raise ValueError("n_augment must be >= 1")
        if n_augment > 1000:
            raise ValueError("n_augment must be <= 1000 to prevent resource exhaustion")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if not 1 <= n_iterations <= 1000:
            raise ValueError("n_iterations must be in [1, 1000]")
        if not 0 < window_fraction <= 1:
            raise ValueError("window_fraction must satisfy 0 < value <= 1")
        if scale not in ("reference", "none"):
            raise ValueError("scale must be one of 'reference', 'none'")

        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)
        out_engine: Any = (
            self.engine if self.engine is not None else df_nw.implementation
        )
        output_original = self._to_output_backend(df_nw, out_engine)
        unique_ids = sorted(df_nw[self.id_col].unique().to_list())
        reserved_ids = {str(series_id) for series_id in unique_ids}
        series: dict[Any, tuple[np.ndarray, nw.DataFrame]] = {}
        aligned: dict[Any, np.ndarray] = {}
        moments: dict[Any, tuple[float, float]] = {}
        skipped: list[Any] = []
        near_constant: list[Any] = []
        partitions = self._partition_series(df_nw)
        for series_id in unique_ids:
            sdf = partitions[series_id]
            raw_values = sdf.select(self.target_col).to_numpy().flatten().astype(float)
            values = self._interpolate_missing(raw_values, series_id=series_id)
            if values is None:
                skipped.append(series_id)
                continue
            output_sdf = self._to_output_backend(sdf, out_engine)
            mean = float(values.mean())
            std = float(values.std())
            if scale == "reference" and std < 1e-8:
                near_constant.append(series_id)
            divisor = std if std >= 1e-8 else 1.0
            series[series_id] = (values, output_sdf)
            moments[series_id] = (mean, std)
            aligned[series_id] = (
                (values - mean) / divisor if scale == "reference" else values
            )
        usable_ids = list(series)
        self._warn_skipped("dba", skipped)
        if near_constant:
            logger.warning(
                "dba reference scaling preserves near-zero scale for %s series; "
                "copies may add little or no variation; first IDs: %s",
                len(near_constant),
                near_constant[:5],
            )
        if len(usable_ids) < 2:
            raise ValueError("dba requires at least 2 usable series")

        # Native ties use input index; sort by string ID to preserve the
        # established neighbor ordering even when source IDs are numeric.
        neighbor_order = sorted(usable_ids, key=str)
        nearest = nearest_dtw_neighbors(
            [aligned[series_id] for series_id in neighbor_order],
            window_fraction,
            n_neighbors,
        )
        nearest_by_id = dict(zip(neighbor_order, nearest, strict=True))
        synthetic_dfs: list[nw.DataFrame] = []
        for reference_id in usable_ids:
            reference_values = aligned[reference_id]
            neighbor_ids = [
                neighbor_order[index] for index, _ in nearest_by_id[reference_id]
            ]
            neighbor_distances = np.array(
                [distance for _, distance in nearest_by_id[reference_id]]
            )
            positive = neighbor_distances[neighbor_distances > 0]
            if len(positive):
                base_weights = np.exp(
                    np.log(0.5) * neighbor_distances / float(positive.min())
                )
            else:
                base_weights = np.ones(len(neighbor_ids))
            neighbors = [aligned[series_id] for series_id in neighbor_ids]
            max_length = max(
                [len(reference_values), *[len(values) for values in neighbors]]
            )
            group_band = int(np.ceil(window_fraction * max_length))

            generated_ids = []
            copy_weights = []
            for copy_index in range(n_augment):
                generated_ids.append(
                    self._reserve_generated_id(
                        f"{reference_id}_dba_{copy_index}", reserved_ids
                    )
                )
                reference_weight = float(self.rng.uniform(0.4, 0.7))
                randomized = base_weights * self.rng.uniform(
                    0.5, 1.5, len(base_weights)
                )
                neighbor_weights = (
                    (1.0 - reference_weight) * randomized / randomized.sum()
                )
                copy_weights.append(
                    np.concatenate(([reference_weight], neighbor_weights)).tolist()
                )
            copies = _rs_augmentation.dba_barycenters(
                np.ascontiguousarray(reference_values),
                [np.ascontiguousarray(v) for v in neighbors],
                copy_weights,
                n_iterations,
                group_band,
            )
            if scale == "reference":
                mean, std = moments[reference_id]
                copies = [barycenter * std + mean for barycenter in copies]
            synthetic_dfs.append(
                self._copy_reference_frames(
                    series[reference_id][1], generated_ids, copies, out_engine
                )
            )

        frames = [output_original] if include_original else []
        frames.extend(synthetic_dfs)
        time_dtype = output_original.schema[self.time_col]
        frames = [
            frame.with_columns(
                nw.col(self.id_col).cast(nw.String()),
                nw.col(self.time_col).cast(time_dtype),
                nw.col(self.target_col).cast(nw.Float64()),
            )
            for frame in frames
        ]
        return _categorize_ids(nw.concat(frames), self.id_col).to_native()

    def _generate_augmented_series(
        self,
        series_id: str,
        series_values: np.ndarray,
        timestamps: np.ndarray,
        n_augment: int,
        generator_name: str,
        fitted_params: dict,
        _preserve_timestamps: bool,
        engine: Any = "pandas",
        fallbacks: list[tuple[str, str, str]] | None = None,
    ) -> list[nw.DataFrame]:
        """Generate augmented series for a single original series.

        Generates synthetic series from scratch using fitted generators,
        then applies post-processing to match the original's autocorrelation
        structure for realistic output.

        Args:
            series_id: Original series ID
            series_values: Original series values
            timestamps: Original timestamps
            n_augment: Number of augmented series to generate
            generator_name: Name of generator to use
            fitted_params: Fitted parameters for the generator
            _preserve_timestamps: Reserved for future use (timestamp offset support)
            engine: Output dataframe backend for the synthetic frames
            fallbacks: Collector for AR(1) substitutions made under
                on_error='ar1'; the caller reports them once at the end

        Returns:
            List of narwhals DataFrames, one per augmented series

        Raises:
            ValueError: If generator_name is not a supported generator
            RuntimeError: If the generator fails and on_error='raise'
        """
        length = len(series_values)
        augmented_dfs = []

        # Compute target autocorrelation from original series
        valid_values = series_values[~np.isnan(series_values)]
        if len(valid_values) > 1:
            target_ac1 = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
            if np.isnan(target_ac1):
                target_ac1 = 0.9
            target_ac1 = np.clip(target_ac1, 0.0, 0.99)
        else:
            target_ac1 = 0.9

        # Get the generator class
        generator_class = self._GENERATOR_MAP.get(generator_name)
        if generator_class is None:
            raise ValueError(
                f"Unknown generator {generator_name!r} for series {series_id!r}. "
                f"Supported generators: {sorted(self._GENERATOR_MAP)}"
            )

        for i in range(n_augment):
            # Create unique seed for each augmented series
            aug_seed = self.rng.integers(0, 2**31) if self.seed is not None else None

            # Build generator parameters
            gen_params = {
                "min_length": length,
                "max_length": length,
                "freq": "h",  # Placeholder, timestamps are overridden
                "seed": aug_seed,
                "id_col": self.id_col,
                "time_col": self.time_col,
                "target_col": self.target_col,
            }

            # Merge fitted parameters
            gen_params.update(
                self._convert_params_for_generator(
                    generator_name, fitted_params, series_values
                )
            )

            # Create generator and generate values
            try:
                generator = generator_class(**gen_params)
                synthetic_values = generator.generate_single_series(length)
            except Exception as e:
                if self.on_error == "raise":
                    raise RuntimeError(
                        f"Generator {generator_name} failed for series "
                        f"{series_id!r}: {e}. Pass on_error='ar1' to substitute "
                        "an AR(1) series instead."
                    ) from e
                if fallbacks is not None:
                    fallbacks.append((str(series_id), generator_name, str(e)))
                synthetic_values = self._generate_ar1_series(
                    length,
                    target_ac1,
                    np.nanmean(series_values),
                    np.nanstd(series_values),
                    aug_seed,
                )

            # Post-process to match original's autocorrelation and statistics
            synthetic_values = self._match_autocorrelation(
                synthetic_values, target_ac1, series_values, aug_seed
            )

            # Create augmented series DataFrame
            aug_id = f"{series_id}_aug_{i}"
            aug_timestamps = timestamps

            # Create DataFrame using narwhals
            aug_df = nw.DataFrame.from_dict(
                {
                    self.id_col: [aug_id] * length,
                    self.time_col: aug_timestamps,
                    self.target_col: synthetic_values,
                },
                backend=engine,
            )

            augmented_dfs.append(aug_df)

        return augmented_dfs

    def _generate_ar1_series(
        self,
        length: int,
        phi: float,
        mean: float,
        std: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate an AR(1) series from scratch.

        Args:
            length: Length of series to generate
            phi: AR(1) coefficient (autocorrelation)
            mean: Target mean
            std: Target standard deviation
            seed: Random seed

        Returns:
            Generated AR(1) series
        """
        rng = np.random.default_rng(seed)

        # AR(1) process: y_t = phi * y_{t-1} + eps_t
        # Unconditional variance: sigma_y^2 = sigma_eps^2 / (1 - phi^2)
        # So sigma_eps = sigma_y * sqrt(1 - phi^2)
        phi = np.clip(phi, -0.99, 0.99)
        innovation_std = std * np.sqrt(1 - phi**2)

        series = np.zeros(length)
        series[0] = rng.normal(mean, std)

        for t in range(1, length):
            series[t] = (
                mean + phi * (series[t - 1] - mean) + rng.normal(0, innovation_std)
            )

        return series

    def _match_autocorrelation(
        self,
        synthetic: np.ndarray,
        target_ac1: float,
        original: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """Post-process synthetic series to match target autocorrelation.

        Applies exponential smoothing to increase autocorrelation, then
        rescales to match original mean and std.

        Args:
            synthetic: Generated synthetic series
            target_ac1: Target lag-1 autocorrelation
            original: Original series (for mean/std matching)
            seed: Random seed

        Returns:
            Processed series with matched autocorrelation
        """
        rng = np.random.default_rng(seed)
        n = len(synthetic)

        # Compute current autocorrelation
        valid_syn = synthetic[~np.isnan(synthetic)]
        if len(valid_syn) > 1:
            current_ac1 = np.corrcoef(valid_syn[:-1], valid_syn[1:])[0, 1]
            if np.isnan(current_ac1):
                current_ac1 = 0.0
        else:
            current_ac1 = 0.0

        # If autocorrelation is already close enough, just rescale
        if abs(current_ac1 - target_ac1) < 0.1:
            smoothed = synthetic
        else:
            # Apply exponential smoothing to increase autocorrelation
            # Higher alpha = less smoothing, lower alpha = more smoothing
            # We want smoothing that achieves target_ac1
            # For EMA: ac1 ≈ (1 - alpha), so alpha ≈ 1 - target_ac1
            alpha = max(0.01, min(0.5, 1 - target_ac1))

            smoothed = np.zeros(n)
            smoothed[0] = synthetic[0]
            for t in range(1, n):
                smoothed[t] = alpha * synthetic[t] + (1 - alpha) * smoothed[t - 1]

        # Rescale to match original mean and std
        valid_orig = original[~np.isnan(original)]
        orig_mean = np.mean(valid_orig)
        orig_std = np.std(valid_orig)

        smoothed_mean = np.mean(smoothed)
        smoothed_std = np.std(smoothed)

        if smoothed_std > 1e-10:
            result = (smoothed - smoothed_mean) / smoothed_std * orig_std + orig_mean
        else:
            result = np.full(n, orig_mean)

        # Add tiny amount of noise to ensure series are unique
        noise = rng.normal(0, orig_std * 0.001, n)
        result = result + noise

        return result

    def _convert_params_for_generator(
        self, generator_name: str, fitted_params: dict, _series_values: np.ndarray
    ) -> dict:
        """Convert fitted parameters to generator-specific parameter names.

        Args:
            generator_name: Name of the generator
            fitted_params: Fitted parameters from fitting functions
            _series_values: Original series values (reserved for future use)

        Returns:
            dict with parameters in generator-expected format
        """
        # Most generators accept the fitted parameters directly. Only the cases
        # below need a rename or a derived field; everything else falls through.
        params = fitted_params.copy()

        if generator_name == "SeasonalGenerator":
            # SeasonalGenerator uses 'seasonality_period' rather than 'period'.
            if "period" in params:
                params["seasonality_period"] = params.pop("period")
            if "seasonality_period" not in params:
                params["seasonality_period"] = 24
            if "seasonality_amplitude" not in params:
                params["seasonality_amplitude"] = params.get("amplitude", 1.0)

        elif generator_name == "RegimeSwitchingGenerator":
            # Derive the transition matrix from the fitted persistence, placing
            # the persistence on the diagonal and spreading the remainder.
            if "persistence" in params and "n_regimes" in params:
                n = params["n_regimes"]
                p = params["persistence"]
                transition_matrix = np.full((n, n), (1 - p) / (n - 1))
                np.fill_diagonal(transition_matrix, p)
                params["transition_matrix"] = transition_matrix.tolist()

        return params

    def _validate_columns(self, df: nw.DataFrame) -> None:
        """Validate that DataFrame has required columns.

        Args:
            df: narwhals DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {self.id_col, self.time_col, self.target_col}
        actual_cols = set(df.columns)

        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            raise ValueError(
                f"DataFrame is missing required columns: {missing}. "
                f"Expected columns: {required_cols}"
            )

    def augment_single_series(
        self,
        series_id: str,
        values: np.ndarray,
        timestamps: np.ndarray,
        n_augment: int = 1,
        generator_name: str | None = None,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Augment a single series (lower-level API).

        Args:
            series_id: ID of the original series
            values: Array of time series values
            timestamps: Array of timestamps
            n_augment: Number of synthetic series to generate
            generator_name: Optional generator name; if None, auto-detect

        Returns:
            List of tuples: (new_id, synthetic_values, timestamps)
        """
        # Classify if generator not specified
        if generator_name is None:
            classification = classify_series(values)
            generator_name = classification["recommended_generator"]
            properties = classification["properties"]
        else:
            classification = classify_series(values)
            properties = classification["properties"]

        # Fit parameters
        fitted_params = fit_generator_params(values, generator_name, properties)

        # Generate augmented series
        fallbacks: list[tuple[str, str, str]] = []
        augmented_dfs = self._generate_augmented_series(
            series_id=series_id,
            series_values=values,
            timestamps=timestamps,
            n_augment=n_augment,
            generator_name=generator_name,
            fitted_params=fitted_params,
            _preserve_timestamps=True,
            fallbacks=fallbacks,
        )
        self._warn_fallbacks(fallbacks, n_augment)

        # Convert to tuples
        results = []
        for aug_df in augmented_dfs:
            aug_id = aug_df[self.id_col][0]
            aug_values = aug_df[self.target_col].to_numpy()
            aug_timestamps = aug_df[self.time_col].to_numpy()
            results.append((aug_id, aug_values, aug_timestamps))

        return results
