"""Dataset class for generating synthetic time series from multiple generators."""

import logging
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoFrameT

from synforecast._analysis import classify_series
from synforecast._fitting import fit_generator_params
from synforecast.base import (
    BaseGenerator,
    _batch_results_to_metadata,
    _categorize_ids,
)

try:
    from synforecast._lib import batch as _rs_batch
except ImportError:
    _rs_batch = None

logger = logging.getLogger(__name__)


class SynSet:
    """Generate synthetic time series datasets from multiple generators.

    This class allows you to combine multiple generators to create diverse
    synthetic time series datasets. Each generator can produce its own
    type of time series pattern.

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
            DataFrame containing all generated time series from all generators,
            in long format with columns [id_col, time_col, target_col].

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
        series of all generators. Excluded generators fall back to the
        threaded Python path (ThreadPoolExecutor).
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
            if batch_params is None or _rs_batch is None:
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

    Analyzes input time series, auto-selects appropriate generators based on
    statistical properties, fits parameters, and generates statistically
    similar synthetic series.

    The augmentation process:
    1. For each unique series in the input DataFrame, analyze its statistical properties
    2. Auto-select the most appropriate generator (or use user override)
    3. Fit generator parameters to match the series' statistical fingerprint
    4. Generate n_augment synthetic series that preserve these properties
    5. Return combined DataFrame with original and synthetic series

    Synthetic series IDs follow the pattern: "{original_id}_aug_{i}"

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
    ) -> None:
        """Initialize the SynAugment instance.

        Args:
            id_col: Name of the ID column
            time_col: Name of the timestamp column
            target_col: Name of the value column
            seed: Random seed for reproducibility
            engine: Output dataframe library (e.g. 'pandas', 'polars').
                None (default) matches the input DataFrame's library.
        """
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.seed = seed
        self.engine = engine
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

    def analyze(self, df: IntoFrameT) -> dict[str, dict]:
        """Analyze all series in DataFrame and return properties.

        For each unique series, detects statistical properties and recommends
        the most appropriate generator.

        Args:
            df: DataFrame with time series data (must have id_col, time_col, target_col)

        Returns:
            dict mapping unique_id to analysis results:
            {
                'series_0': {
                    'recommended_generator': 'SeasonalGenerator',
                    'properties': {...},  # All detected properties
                    'fitted_params': {...}  # Estimated generator parameters
                },
                ...
            }

        Raises:
            ValueError: If DataFrame is missing required columns
        """
        df_nw = nw.from_native(df)
        self._validate_columns(df_nw)

        results = {}
        unique_ids = df_nw[self.id_col].unique().to_list()

        for series_id in unique_ids:
            # Extract series values
            series_data = (
                df_nw.filter(nw.col(self.id_col) == series_id)
                .sort(self.time_col)
                .select(self.target_col)
                .to_numpy()
                .flatten()
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
            Combined DataFrame with original and synthetic series.
            Synthetic series IDs follow pattern: "{original_id}_aug_{i}"

        Raises:
            ValueError: If DataFrame is missing required columns or n_augment < 1

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

        generator_override = generator_override or {}

        # First, analyze all series
        analysis = self.analyze(df)

        # Collect all series (original + augmented)
        all_dfs = [df_nw]

        unique_ids = df_nw[self.id_col].unique().to_list()

        for series_id in unique_ids:
            # Get series data
            series_df = df_nw.filter(nw.col(self.id_col) == series_id).sort(
                self.time_col
            )
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
            )

            all_dfs.extend(augmented)

        # Augmented ids are strings ("<id>_aug_<i>"), so normalize the id
        # column to String before concatenating, then re-categorize.
        all_dfs = [
            frame.with_columns(nw.col(self.id_col).cast(nw.String()))
            for frame in all_dfs
        ]
        result = _categorize_ids(nw.concat(all_dfs), self.id_col)

        return result.to_native()

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

        Returns:
            DataFrame with the original series (optional) and synthetic mixup
            series. Mixup IDs follow the pattern ``"mixup_{i}"``.

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

        # Sort ids so series selection is reproducible: some backends do not
        # guarantee an order from unique().
        unique_ids = sorted(df_nw[self.id_col].unique().to_list())

        # Cache each series' values and timestamps, sorted by time.
        series: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for series_id in unique_ids:
            sdf = df_nw.filter(nw.col(self.id_col) == series_id).sort(self.time_col)
            values = sdf.select(self.target_col).to_numpy().flatten().astype(float)
            timestamps = sdf.select(self.time_col).to_numpy().flatten()
            if len(values) >= 1:
                series[series_id] = (values, timestamps)

        usable_ids = list(series.keys())
        n_available = len(usable_ids)
        if n_available == 0:
            raise ValueError("no usable series to mix")

        if n_series is None:
            n_series = n_available
        if n_series < 1:
            raise ValueError("n_series must be >= 1")
        if n_series > 1_000_000:
            raise ValueError("n_series must be <= 1_000_000 to prevent exhaustion")

        synthetic_dfs = []
        for i in range(n_series):
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
                    self.id_col: [f"mixup_{i}"] * length,
                    self.time_col: anchor_timestamps,
                    self.target_col: mixed,
                },
                backend=out_engine,
            )
            synthetic_dfs.append(mix_df)

        frames = [df_nw] if include_original else []
        frames.extend(synthetic_dfs)
        # Normalize id (mixup ids are strings) and target (mixed is float) so
        # the frames concatenate across the original and synthetic batches.
        frames = [
            frame.with_columns(
                nw.col(self.id_col).cast(nw.String()),
                nw.col(self.target_col).cast(nw.Float64()),
            )
            for frame in frames
        ]
        result = _categorize_ids(nw.concat(frames), self.id_col)

        return result.to_native()

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

        Returns:
            List of narwhals DataFrames, one per augmented series
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
            # Fallback to RandomWalk
            from synforecast.generators import RandomWalkGenerator

            generator_class = RandomWalkGenerator
            fitted_params = {"drift": 0.0, "volatility": np.std(series_values)}

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
                # Fallback to simple generation if generator fails
                logger.warning(
                    f"Generator {generator_name} failed for {series_id}: {e}. "
                    "Falling back to AR(1) generation."
                )
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
        # Most generators accept parameters directly, but some need renaming
        params = fitted_params.copy()

        if generator_name == "SeasonalGenerator":
            # SeasonalGenerator uses 'seasonality_period' not 'period'
            if "period" in params:
                params["seasonality_period"] = params.pop("period")
            # Ensure required params exist
            if "seasonality_period" not in params:
                params["seasonality_period"] = 24
            if "seasonality_amplitude" not in params:
                params["seasonality_amplitude"] = params.get("amplitude", 1.0)

        elif generator_name == "SARIMAGenerator":
            # SARIMAGenerator has specific parameter names
            # These are already handled by fit_sarima
            pass

        elif generator_name == "GARCHGenerator":
            # GARCHGenerator params are already correct from fit_garch
            pass

        elif generator_name == "OrnsteinUhlenbeckGenerator":
            # OrnsteinUhlenbeckGenerator params are already correct
            pass

        elif generator_name == "FractionalBrownianMotionGenerator":
            # Already correct from fit_fbm
            pass

        elif generator_name == "RegimeSwitchingGenerator":
            # Need to set transition_matrix from persistence
            if "persistence" in params and "n_regimes" in params:
                n = params["n_regimes"]
                p = params["persistence"]
                # Create transition matrix with persistence on diagonal
                transition_matrix = np.full((n, n), (1 - p) / (n - 1))
                np.fill_diagonal(transition_matrix, p)
                params["transition_matrix"] = transition_matrix.tolist()

        elif generator_name == "IntermittentDemandGenerator":
            # Map our params to generator params
            if "demand_probability" in params:
                # The generator might use different param names
                # Check the actual generator interface
                pass

        elif generator_name == "GeometricBrownianMotionGenerator":
            # GBM params are already correct
            if "initial_value" in params:
                params["S0"] = params.pop("initial_value")

        elif generator_name == "RandomWalkGenerator":
            # RandomWalk params are already correct
            pass

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
        augmented_dfs = self._generate_augmented_series(
            series_id=series_id,
            series_values=values,
            timestamps=timestamps,
            n_augment=n_augment,
            generator_name=generator_name,
            fitted_params=fitted_params,
            _preserve_timestamps=True,
        )

        # Convert to tuples
        results = []
        for aug_df in augmented_dfs:
            aug_id = aug_df[self.id_col][0]
            aug_values = aug_df[self.target_col].to_numpy()
            aug_timestamps = aug_df[self.time_col].to_numpy()
            results.append((aug_id, aug_values, aug_timestamps))

        return results
