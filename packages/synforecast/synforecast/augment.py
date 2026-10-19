"""Augmentation strategies as frozen specs executed by ``apply``.

Each strategy is a pydantic model holding only its own settings plus a default
``seed``. ``apply(df, ...)`` builds a fresh random stream from the effective
seed on every call, so the same spec, data and seed always return the same
output.
"""

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoFrameT
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from synforecast._analysis import classify_series, detect_seasonality
from synforecast._dtw import nearest_dtw_neighbors
from synforecast._fitting import fit_generator_params
from synforecast._lib import augmentation as _rs_augmentation
from synforecast.base import _categorize_ids

logger = logging.getLogger(__name__)

FittableGenerator = Literal[
    "RandomWalkGenerator",
    "SeasonalGenerator",
    "SARIMAGenerator",
    "GARCHGenerator",
    "OrnsteinUhlenbeckGenerator",
    "FractionalBrownianMotionGenerator",
    "RegimeSwitchingGenerator",
    "IntermittentDemandGenerator",
    "GeometricBrownianMotionGenerator",
]


@functools.cache
def _generator_map() -> dict[str, type]:
    """Map fittable generator names to classes, imported lazily."""
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

    return {
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


def _check_n_augment(n_augment: int) -> int:
    if n_augment < 1:
        raise ValueError("n_augment must be >= 1")
    if n_augment > 1000:
        raise ValueError("n_augment must be <= 1000 to prevent resource exhaustion")
    return n_augment


def _reject_bool_and_str(value: Any) -> Any:
    """Stop pydantic's lax mode from coercing ``True`` or ``"3"`` to numbers."""
    if isinstance(value, bool | np.bool_ | str | bytes):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    return value


# Python and numpy numbers pass; bools and strings are rejected.
Int = Annotated[int, BeforeValidator(_reject_bool_and_str)]
Float = Annotated[float, BeforeValidator(_reject_bool_and_str)]
NAugment = Annotated[Int, AfterValidator(_check_n_augment)]


class BaseAugmenter(BaseModel, ABC):
    """Base class for augmentation strategies.

    A spec declares a strategy's settings and default ``seed``; ``apply``
    executes it on a frame without mutating the spec.

    Args:
        seed (int | None): Default random seed for ``apply`` (default: None,
            fresh entropy on every call)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: Int | None = Field(
        default=None,
        ge=0,
        description="Default random seed for apply; None draws fresh entropy",
    )

    @abstractmethod
    def apply(
        self,
        df: IntoFrameT,
        *,
        seed: int | None = None,
        include_original: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        engine: str | None = None,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Augment ``df`` with synthetic series.

        Args:
            df: Input frame containing the ID, time and target columns.
            seed: Seed for this call; None uses the spec's ``seed``.
            include_original: Include unchanged original rows (default: True).
            id_col: Name of the ID column (default: 'unique_id').
            time_col: Name of the timestamp column (default: 'ds').
            target_col: Name of the value column (default: 'y').
            engine: Output dataframe library (e.g. 'pandas', 'polars').
                None (default) matches the input DataFrame's library.
            source_col: Name of an added column holding, for every row, the
                input id of the series it derives from; the column has the
                same dtype as the output id column. Original rows name
                themselves; FitAndSimulate and MovingBlockBootstrap copies
                name their single source. Mixup names the anchor (the first
                chosen series, whose timestamps and extra columns are used)
                and DBA the reference series; other blended series also
                contribute values, so grouping CV folds by ``source_col``
                does not prevent leakage for those two strategies. None
                (default) adds no column.

        Returns:
            Frame containing generated series and, optionally, original rows.
        """


class FitAndSimulate(BaseAugmenter):
    """Fit a parametric generator per series and simulate from it.

    The fit-and-simulate process:
    1. For each unique series in the input DataFrame, analyze its statistical properties
    2. Auto-select the most appropriate generator (or use user override)
    3. Fit generator parameters to match the series' statistical fingerprint
    4. Generate n_augment synthetic series that preserve these properties
    5. Return combined DataFrame with original and synthetic series

    Series are processed in order of first appearance in the input, so
    reordering the input rows changes seeded output.

    Synthetic series IDs follow the pattern ``"{original_id}_aug_{i}"``.

    Args:
        n_augment (int): Number of synthetic series to generate per original
            series, from 1 to 1000 (default: 1)
        generator_override (dict | None): Optional dict mapping unique_id to
            generator name. Overrides automatic generator selection for
            specified series; keys match on ``str(unique_id)``. Stored as a
            read-only mapping; ``model_dump`` returns a plain dict.
            Example: {"series_0": "SARIMAGenerator"}
        on_error (str): What to do when a fitted generator fails for a series.
            'raise' (default) propagates the error; 'ar1' substitutes an
            AR(1) series matching the source's mean, std, and lag-1
            autocorrelation, and reports all substitutions in a single
            warning at the end of `apply`.
        seed (int | None): Default random seed for ``apply`` (default: None)

    Example:
        >>> from synforecast import FitAndSimulate
        >>> augmented_df = FitAndSimulate(n_augment=3, seed=42).apply(
        ...     df
        ... )  # doctest: +SKIP
    """

    n_augment: NAugment = Field(
        default=1, description="Synthetic series per source series, 1 to 1000"
    )
    generator_override: dict[str | int, FittableGenerator] | None = Field(
        default=None,
        description="Generator name per series ID, matched on str(id)",
    )
    on_error: Literal["raise", "ar1"] = Field(
        default="raise",
        description="'raise' propagates generator failures; 'ar1' substitutes AR(1)",
    )

    @field_validator("generator_override")
    @classmethod
    def validate_generator_override(
        cls, v: dict[str | int, str] | None
    ) -> MappingProxyType | None:
        """Reject keys that collide as strings and freeze the mapping."""
        if v is None:
            return None
        if len({str(key) for key in v}) != len(v):
            raise ValueError(
                "generator_override has keys that are equal as strings: "
                f"{sorted(map(repr, v))}"
            )
        return MappingProxyType(dict(v))

    @field_serializer("generator_override")
    def _dump_generator_override(
        self, v: MappingProxyType | None
    ) -> dict[str | int, str] | None:
        return None if v is None else dict(v)

    def __hash__(self) -> int:
        """Hash like a frozen model, with the override as a frozenset."""
        return hash(
            (
                type(self),
                *(
                    frozenset(v.items()) if isinstance(v, MappingProxyType) else v
                    for v in self.__dict__.values()
                ),
            )
        )

    def apply(
        self,
        df: IntoFrameT,
        *,
        seed: int | None = None,
        include_original: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        engine: str | None = None,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Augment dataset with synthetic series.

        For each series in the input DataFrame, generates n_augment synthetic
        series that are statistically similar to the original. Arguments are
        described on :meth:`BaseAugmenter.apply`.

        Returns:
            DataFrame with the original series (optional) and synthetic
            series. Synthetic series IDs follow the pattern
            `"{original_id}_aug_{i}"`

        Raises:
            ValueError: If DataFrame is missing required columns
            RuntimeError: If a fitted generator fails for a series and
                on_error='raise' (the default)
        """
        run = _Run.start(
            df,
            self.seed if seed is None else seed,
            engine,
            id_col,
            time_col,
            target_col,
            source_col,
        )
        partitions = run.partitions

        overrides = {
            str(key): name for key, name in (self.generator_override or {}).items()
        }

        unique_ids = run.df[id_col].unique(maintain_order=True).to_list()
        reserved_ids = {str(series_id) for series_id in unique_ids}
        unknown = sorted(set(overrides) - {str(series_id) for series_id in unique_ids})
        if unknown:
            logger.warning(
                "generator_override keys match no series and are ignored: %s",
                unknown,
            )
        analysis = _analyze_partitions(partitions, unique_ids, target_col)

        synthetic_dfs: list[nw.DataFrame] = []

        # (series_id, generator_name, error) for every AR(1) substitution
        # made under on_error='ar1', reported once after the loop.
        fallbacks: list[tuple[str, str, str]] = []

        for series_id in unique_ids:
            series_df = partitions[series_id]
            series_values = series_df.select(target_col).to_numpy().flatten()
            output_sdf = run.out_partitions[series_id]

            # Determine which generator to use
            if str(series_id) in overrides:
                generator_name = overrides[str(series_id)]
                # Re-fit parameters for the overridden generator
                fitted_params = fit_generator_params(
                    series_values,
                    generator_name,
                    analysis[series_id]["properties"],
                )
            else:
                generator_name = analysis[series_id]["recommended_generator"]
                fitted_params = analysis[series_id]["fitted_params"]

            synthetic = self._simulate(
                series_id=series_id,
                series_values=series_values,
                generator_name=generator_name,
                fitted_params=fitted_params,
                rng=run.rng,
                seeded=run.seed is not None,
                fallbacks=fallbacks,
                columns=(id_col, time_col, target_col),
            )
            for i, synthetic_values in enumerate(synthetic):
                generated_id = _reserve_generated_id(
                    f"{series_id}_aug_{i}", reserved_ids
                )
                synthetic_dfs.append(
                    nw.maybe_reset_index(
                        _copy_reference_frames(
                            output_sdf,
                            [generated_id],
                            [synthetic_values],
                            run.out_engine,
                            id_col,
                            target_col,
                        )
                    )
                )

        _warn_fallbacks(fallbacks, len(unique_ids) * self.n_augment)
        return run.finish(synthetic_dfs, include_original)

    def apply_series(
        self,
        series_id: str,
        values: np.ndarray,
        timestamps: np.ndarray,
        *,
        generator_name: FittableGenerator | None = None,
        seed: int | None = None,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Augment a single series (lower-level API).

        Args:
            series_id: ID of the original series
            values: Array of time series values
            timestamps: Array of timestamps
            generator_name: Optional generator name; if None, auto-detect
            seed: Seed for this call; None uses the spec's ``seed``

        Returns:
            List of tuples: (new_id, synthetic_values, timestamps)
        """
        eff = self.seed if seed is None else seed
        rng = np.random.default_rng(eff)

        classification = classify_series(values)
        if generator_name is None:
            generator_name = classification["recommended_generator"]
        fitted_params = fit_generator_params(
            values, generator_name, classification["properties"]
        )

        fallbacks: list[tuple[str, str, str]] = []
        synthetic = self._simulate(
            series_id=series_id,
            series_values=values,
            generator_name=generator_name,
            fitted_params=fitted_params,
            rng=rng,
            seeded=eff is not None,
            fallbacks=fallbacks,
        )
        _warn_fallbacks(fallbacks, self.n_augment)

        return [
            (f"{series_id}_aug_{i}", synthetic_values, np.asarray(timestamps).copy())
            for i, synthetic_values in enumerate(synthetic)
        ]

    def _simulate(
        self,
        series_id: Any,
        series_values: np.ndarray,
        generator_name: str,
        fitted_params: dict,
        rng: np.random.Generator,
        seeded: bool,
        fallbacks: list[tuple[str, str, str]],
        columns: tuple[str, str, str] = ("unique_id", "ds", "y"),
    ) -> list[np.ndarray]:
        """Generate ``n_augment`` synthetic value arrays for one series.

        Generates synthetic series from scratch using fitted generators,
        then applies post-processing to match the original's autocorrelation
        structure for realistic output.

        Args:
            series_id: Original series ID
            series_values: Original series values
            generator_name: Name of generator to use
            fitted_params: Fitted parameters for the generator
            rng: Stream the per-copy generator seeds are drawn from
            seeded: Whether generators get seeds from ``rng`` (else None)
            fallbacks: Collector for AR(1) substitutions made under
                on_error='ar1'; the caller reports them once at the end
            columns: ID, time and target column names for the generators

        Returns:
            One array of synthetic values per augmented series

        Raises:
            ValueError: If generator_name is not a supported generator
            RuntimeError: If the generator fails and on_error='raise'
        """
        length = len(series_values)
        id_col, time_col, target_col = columns
        synthetic = []

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
        generator_class = _generator_map().get(generator_name)
        if generator_class is None:
            raise ValueError(
                f"Unknown generator {generator_name!r} for series {series_id!r}. "
                f"Supported generators: {sorted(_generator_map())}"
            )

        for _ in range(self.n_augment):
            # Create unique seed for each augmented series
            aug_seed = rng.integers(0, 2**31) if seeded else None

            # Build generator parameters
            gen_params = {
                "min_length": length,
                "max_length": length,
                "freq": "h",  # Placeholder, timestamps are overridden
                "seed": aug_seed,
                "id_col": id_col,
                "time_col": time_col,
                "target_col": target_col,
            }

            # Merge fitted parameters
            gen_params.update(
                _convert_params_for_generator(
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
                fallbacks.append((str(series_id), generator_name, str(e)))
                synthetic_values = _generate_ar1_series(
                    length,
                    target_ac1,
                    np.nanmean(series_values),
                    np.nanstd(series_values),
                    aug_seed,
                )

            # Post-process to match original's autocorrelation and statistics
            synthetic.append(
                _match_autocorrelation(
                    synthetic_values, target_ac1, series_values, aug_seed
                )
            )

        return synthetic


class Mixup(BaseAugmenter):
    """Augment a dataset with TSMixup convex combinations of real series.

    Implements TSMixup, the augmentation used to pretrain the Chronos
    forecasting models (Ansari et al. 2024, "Chronos: Learning the Language
    of Time Series", https://arxiv.org/abs/2403.07815, Apache-2.0). Each
    synthetic series is built by drawing ``1..max_mix`` source series,
    taking a random window of their shared (minimum) length, scaling each
    window, and combining them with convex weights sampled from a
    ``Dirichlet(alpha)`` distribution. Unlike :class:`FitAndSimulate`, a mixup
    series is not tied to a single source: it blends the dynamics of
    several, which broadens a small panel without fitting any generator.

    Because the mixed series live in scaled space (different sources have
    different magnitudes), the output retains the chosen scaling rather
    than any single source's level. Use ``scaling="none"`` to mix raw
    values when all series already share a scale.

    NaNs in source series are linearly interpolated before mixing,
    including nearest-value filling at the ends. Entirely missing
    series are excluded. When ``include_original=True``, the original
    rows and their missing values are retained unchanged.

    Sources are indexed by sorted id, so seeded output does not depend on
    the input row order. With ``source_col``, each mixup series names its
    anchor, the first chosen source, whose window supplies the timestamps
    and extra columns. The other blended sources also contribute values, so
    grouping CV folds by ``source_col`` does not prevent leakage.

    Mixup IDs follow the pattern ``"mixup_{i}"``.

    Args:
        n_series (int | None): Total number of synthetic series to create
            (TSMixup's definition), not a count per source, from 1 to
            1_000_000. Defaults to the number of usable input series.
        max_mix (int): Maximum number of source series combined per synthetic
            series; the count is drawn uniformly from ``1..max_mix`` and
            capped at the number of available series (default: 3).
        alpha (float): Concentration of the symmetric Dirichlet weight
            distribution. Values < 1 concentrate weight on few sources;
            larger values spread it more evenly (default: 1.5).
        scaling (str): Per-window scaling before mixing: 'mean' (divide by
            mean absolute value, Chronos default), 'std' (z-normalize), or
            'none' (default: 'mean').
        seed (int | None): Default random seed for ``apply`` (default: None)

    Example:
        >>> from synforecast import Mixup
        >>> mixed_df = Mixup(n_series=50, seed=42).apply(df)  # doctest: +SKIP
    """

    n_series: Int | None = Field(
        default=None,
        description=(
            "Total synthetic series (TSMixup), not per source; "
            "None uses the number of usable input series"
        ),
    )
    max_mix: Int = Field(
        default=3, description="Maximum number of source series mixed per series"
    )
    alpha: Float = Field(
        default=1.5, description="Concentration of the Dirichlet mixing weights"
    )
    scaling: Literal["mean", "std", "none"] = Field(
        default="mean", description="Per-window scaling before mixing"
    )

    @field_validator("n_series")
    @classmethod
    def validate_n_series(cls, v: int | None) -> int | None:
        """Bound n_series when given."""
        if v is not None and v < 1:
            raise ValueError("n_series must be >= 1")
        if v is not None and v > 1_000_000:
            raise ValueError("n_series must be <= 1_000_000 to prevent exhaustion")
        return v

    @field_validator("max_mix")
    @classmethod
    def validate_max_mix(cls, v: int) -> int:
        """Require at least one source per synthetic series."""
        if v < 1:
            raise ValueError("max_mix must be >= 1")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Require a positive Dirichlet concentration."""
        if v <= 0:
            raise ValueError("alpha must be > 0")
        return v

    def apply(
        self,
        df: IntoFrameT,
        *,
        seed: int | None = None,
        include_original: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        engine: str | None = None,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Mix series of ``df``; arguments are described on :meth:`BaseAugmenter.apply`.

        Returns:
            DataFrame with the original series (optional) and synthetic mixup series. Mixup IDs follow the pattern ``"mixup_{i}"``.

        Raises:
            ValueError: If the DataFrame is missing required columns, or if
                it holds no usable series.
        """
        run = _Run.start(
            df,
            self.seed if seed is None else seed,
            engine,
            id_col,
            time_col,
            target_col,
            source_col,
        )
        rng = run.rng

        # Sort ids so series selection is reproducible: some backends do not
        # guarantee an order from unique().
        unique_ids = sorted(run.df[id_col].unique().to_list())

        # Cache each series' values, sorted by time.
        series: dict[Any, np.ndarray] = {}
        skipped: list[Any] = []
        for series_id in unique_ids:
            sdf = run.partitions[series_id]
            values = sdf.select(target_col).to_numpy().flatten().astype(float)
            if (
                len(values) >= 1
                and (values := _interpolate_missing(values)) is not None
            ):
                series[series_id] = values
            else:
                skipped.append(series_id)

        usable_ids = list(series.keys())
        n_available = len(usable_ids)
        _warn_skipped("mixup", skipped)
        if n_available == 0:
            raise ValueError("no usable series to mix")

        n_series = n_available if self.n_series is None else self.n_series

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
            k_max = min(self.max_mix, n_available)
            k = int(rng.integers(1, k_max + 1))
            chosen = rng.choice(n_available, size=k, replace=False)
            chosen_ids = [usable_ids[c] for c in chosen]

            length = int(min(len(series[c]) for c in chosen_ids))
            weights = rng.dirichlet(np.full(k, self.alpha))

            mixed = np.zeros(length)
            anchor = None
            for w, series_id in zip(weights, chosen_ids, strict=True):
                values = series[series_id]
                start = int(rng.integers(0, len(values) - length + 1))
                window = values[start : start + length]
                if anchor is None:
                    # Timestamps and extra columns come from the first
                    # chosen series' window.
                    anchor = run.out_partitions[series_id][start : start + length]
                mixed += w * _scale_window(window, self.scaling)

            synthetic_dfs.append(
                nw.maybe_reset_index(
                    _copy_reference_frames(
                        anchor,
                        [mixup_id],
                        [mixed],
                        run.out_engine,
                        id_col,
                        target_col,
                    )
                )
            )

        return run.finish(synthetic_dfs, include_original)


class MovingBlockBootstrap(BaseAugmenter):
    """Bootstrap decomposition remainders with moving blocks.

    This non-parametric augmenter follows the moving-block-bootstrap idea
    used for forecasting by Bergmeir, Hyndman, and Benitez (2016,
    https://doi.org/10.1016/j.ijforecast.2015.07.002) and the augmentation
    recipe discussed by Bandara et al. (2021,
    https://arxiv.org/abs/2008.02663). SynForecast uses a classical
    moving-average decomposition rather than Box-Cox plus STL; its
    defaults, block-size choice, and stability guards are its own design
    rather than a reproduction of the reference code.

    This strategy deliberately bypasses the ``_match_autocorrelation``
    rescaling used by :class:`FitAndSimulate`. Recombining a source's own
    trend and seasonal components with bootstrapped remainder blocks already
    anchors the result to that source; re-pinning mean and standard deviation
    would distort the remainder variability.

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

    Sources are processed in sorted id order, so seeded output does not
    depend on the input row order.

    IDs follow the pattern ``"{original_id}_mbb_{i}"``.

    Args:
        n_augment (int): Copies per usable source, from 1 to 1000 (default: 1).
        seasonal_period (int | None): Period >= 2, or None to detect it per
            source.
        block_size (int | None): Block length >= 2, or None for the
            period/length-based default. Explicit lengths are also capped as
            described above.
        seed (int | None): Default random seed for ``apply`` (default: None)
    """

    n_augment: NAugment = Field(
        default=1, description="Copies per usable source series, 1 to 1000"
    )
    seasonal_period: Int | None = Field(
        default=None, description="Period >= 2, or None to detect it per source"
    )
    block_size: Int | None = Field(
        default=None,
        description="Block length >= 2, or None for the period/length default",
    )

    @field_validator("seasonal_period")
    @classmethod
    def validate_seasonal_period(cls, v: int | None) -> int | None:
        """Require a period of at least two when given."""
        if v is not None and v < 2:
            raise ValueError("seasonal_period must be >= 2 when provided")
        return v

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: int | None) -> int | None:
        """Require a block of at least two when given."""
        if v is not None and v < 2:
            raise ValueError("block_size must be >= 2 when provided")
        return v

    def apply(
        self,
        df: IntoFrameT,
        *,
        seed: int | None = None,
        include_original: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        engine: str | None = None,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Bootstrap series of ``df``; arguments are described on :meth:`BaseAugmenter.apply`.

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the requested output engine or the input engine.

        Raises:
            ValueError: For missing required columns, infinite target values,
                or no usable source series.
        """
        run = _Run.start(
            df,
            self.seed if seed is None else seed,
            engine,
            id_col,
            time_col,
            target_col,
            source_col,
        )
        rng = run.rng
        n_augment = self.n_augment
        seasonal_period = self.seasonal_period

        unique_ids = sorted(run.df[id_col].unique().to_list())
        if not unique_ids:
            raise ValueError("mbb requires at least 1 series")
        reserved_ids = {str(series_id) for series_id in unique_ids}
        synthetic_dfs: list[nw.DataFrame] = []
        skipped: list[Any] = []
        nonseasonal: list[Any] = []

        for series_id in unique_ids:
            sdf = run.partitions[series_id]
            raw_values = sdf.select(target_col).to_numpy().flatten().astype(float)
            values = _interpolate_missing(raw_values, series_id=series_id)
            if values is None or len(values) < 4:
                skipped.append(series_id)
                continue
            output_sdf = run.out_partitions[series_id]
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
            block = self.block_size if self.block_size is not None else default_block
            block = min(block, max(2, len(values) // 3))

            native_seeds = rng.integers(0, 2**63, size=n_augment).tolist()
            generated_copies = _rs_augmentation.moving_block_bootstrap_many(
                np.ascontiguousarray(values), block, native_seeds, period
            )
            generated_ids = [
                _reserve_generated_id(f"{series_id}_mbb_{i}", reserved_ids)
                for i in range(n_augment)
            ]
            synthetic_dfs.append(
                _copy_reference_frames(
                    output_sdf,
                    generated_ids,
                    generated_copies,
                    run.out_engine,
                    id_col,
                    target_col,
                )
            )

        _warn_skipped("mbb", skipped)
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
        return run.finish(synthetic_dfs, include_original)


class DBA(BaseAugmenter):
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

    This panel-aware strategy caches every usable source series once and
    deliberately bypasses ``_match_autocorrelation``. With the default
    ``scale="reference"``, inputs are z-normalized for alignment and the
    barycenter is mapped to the reference mean and standard deviation. That
    is explicit reference anchoring, not the :class:`FitAndSimulate` pinning
    path. ``scale="none"`` averages raw values for panels already sharing
    scale. Plain DBA is deterministic, so each requested copy randomizes the
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

    References are processed in sorted id order, so seeded output does not
    depend on the input row order. With ``source_col``, each copy names its
    reference series; its neighbors also contribute values, so grouping CV
    folds by ``source_col`` does not prevent leakage.

    IDs follow the pattern ``"{original_id}_dba_{i}"``.

    Args:
        n_augment (int): Copies per usable reference, from 1 to 1000
            (default: 1).
        n_neighbors (int): Positive maximum neighbor count (default: 3),
            capped at the number of other usable sources.
        n_iterations (int): Barycenter updates, from 1 to 1000 (default: 5).
        window_fraction (float): DTW band fraction in (0, 1] (default: 0.1).
        scale (str): "reference" to restore reference scale (default), or
            "none" to average raw values.
        seed (int | None): Default random seed for ``apply`` (default: None)
    """

    n_augment: NAugment = Field(
        default=1, description="Copies per usable reference series, 1 to 1000"
    )
    n_neighbors: Int = Field(
        default=3, description="Maximum neighbor count, capped at usable sources"
    )
    n_iterations: Int = Field(default=5, description="Barycenter updates, 1 to 1000")
    window_fraction: Float = Field(
        default=0.1, description="DTW band fraction in (0, 1]"
    )
    scale: Literal["reference", "none"] = Field(
        default="reference",
        description="'reference' restores reference scale; 'none' averages raw values",
    )

    @field_validator("n_neighbors")
    @classmethod
    def validate_n_neighbors(cls, v: int) -> int:
        """Require at least one neighbor."""
        if v < 1:
            raise ValueError("n_neighbors must be >= 1")
        return v

    @field_validator("n_iterations")
    @classmethod
    def validate_n_iterations(cls, v: int) -> int:
        """Bound the barycenter updates."""
        if not 1 <= v <= 1000:
            raise ValueError("n_iterations must be in [1, 1000]")
        return v

    @field_validator("window_fraction")
    @classmethod
    def validate_window_fraction(cls, v: float) -> float:
        """Require a band fraction in (0, 1]."""
        if not 0 < v <= 1:
            raise ValueError("window_fraction must satisfy 0 < value <= 1")
        return v

    def apply(
        self,
        df: IntoFrameT,
        *,
        seed: int | None = None,
        include_original: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        engine: str | None = None,
        source_col: str | None = None,
    ) -> IntoFrameT:
        """Average neighbors in ``df``; arguments are described on :meth:`BaseAugmenter.apply`.

        Returns:
            Frame containing generated series and, optionally, original rows,
            using the requested output engine or the input engine.

        Raises:
            ValueError: For missing required columns, infinite target values,
                excessive alignment/output storage, or fewer than two usable
                source series.
        """
        run = _Run.start(
            df,
            self.seed if seed is None else seed,
            engine,
            id_col,
            time_col,
            target_col,
            source_col,
        )
        rng = run.rng
        n_augment = self.n_augment
        scale = self.scale
        window_fraction = self.window_fraction

        unique_ids = sorted(run.df[id_col].unique().to_list())
        reserved_ids = {str(series_id) for series_id in unique_ids}
        series: dict[Any, tuple[np.ndarray, nw.DataFrame]] = {}
        aligned: dict[Any, np.ndarray] = {}
        moments: dict[Any, tuple[float, float]] = {}
        skipped: list[Any] = []
        near_constant: list[Any] = []
        for series_id in unique_ids:
            sdf = run.partitions[series_id]
            raw_values = sdf.select(target_col).to_numpy().flatten().astype(float)
            values = _interpolate_missing(raw_values, series_id=series_id)
            if values is None:
                skipped.append(series_id)
                continue
            output_sdf = run.out_partitions[series_id]
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
        _warn_skipped("dba", skipped)
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
            self.n_neighbors,
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
                    _reserve_generated_id(
                        f"{reference_id}_dba_{copy_index}", reserved_ids
                    )
                )
                reference_weight = float(rng.uniform(0.4, 0.7))
                randomized = base_weights * rng.uniform(0.5, 1.5, len(base_weights))
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
                self.n_iterations,
                group_band,
            )
            if scale == "reference":
                mean, std = moments[reference_id]
                copies = [barycenter * std + mean for barycenter in copies]
            synthetic_dfs.append(
                _copy_reference_frames(
                    series[reference_id][1],
                    generated_ids,
                    copies,
                    run.out_engine,
                    id_col,
                    target_col,
                )
            )

        return run.finish(synthetic_dfs, include_original)


def analyze(
    df: IntoFrameT,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> dict[str, dict]:
    """Analyze all series in DataFrame and return properties.

    For each unique series, detects statistical properties and recommends
    the most appropriate generator.

    Args:
        df: DataFrame with time series data (must have id_col, time_col, target_col)
        id_col: Name of the ID column (default: 'unique_id')
        time_col: Name of the timestamp column (default: 'ds')
        target_col: Name of the value column (default: 'y')

    Returns:
        Mapping of each `unique_id` to its analysis results. Every entry holds `recommended_generator` (the chosen generator name), `properties` (all detected statistical properties), and `fitted_params` (the estimated generator parameters).

    Raises:
        ValueError: If DataFrame is missing required columns
    """
    df_nw = nw.from_native(df)
    _validate_columns(df_nw, id_col, time_col, target_col)

    return _analyze_partitions(
        _partition_series(df_nw, id_col, time_col),
        df_nw[id_col].unique(maintain_order=True).to_list(),
        target_col,
    )


def _analyze_partitions(
    partitions: dict[Any, nw.DataFrame], unique_ids: list[Any], target_col: str
) -> dict:
    results = {}
    for series_id in unique_ids:
        series_data = partitions[series_id].select(target_col).to_numpy().flatten()

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


def _validate_columns(
    df: nw.DataFrame, id_col: str, time_col: str, target_col: str
) -> None:
    """Validate that DataFrame has required columns.

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = {id_col, time_col, target_col}
    actual_cols = set(df.columns)

    if not required_cols.issubset(actual_cols):
        missing = required_cols - actual_cols
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Expected columns: {required_cols}"
        )


def _with_source(
    df: nw.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    source_col: str | None,
) -> nw.DataFrame:
    """Validate the columns and append ``source_col`` holding each row's id.

    Synthetic frames are copied from input rows, so they inherit the column
    and every input column with it.
    """
    _validate_columns(df, id_col, time_col, target_col)
    if source_col is None:
        return df
    if source_col in df.columns:
        raise ValueError(
            f"source_col {source_col!r} must not equal id_col, time_col, "
            "target_col or an existing input column"
        )
    return df.with_columns(nw.col(id_col).cast(nw.String()).alias(source_col))


@dataclass(frozen=True)
class _Run:
    """Per-call state shared by every strategy's ``apply``."""

    seed: int | None
    rng: np.random.Generator
    df: nw.DataFrame
    out_engine: Any
    original: nw.DataFrame
    partitions: dict[Any, nw.DataFrame]
    out_partitions: dict[Any, nw.DataFrame]
    columns: tuple[str, str, str, str | None]

    @classmethod
    def start(
        cls,
        df: IntoFrameT,
        seed: int | None,
        engine: str | None,
        id_col: str,
        time_col: str,
        target_col: str,
        source_col: str | None,
    ) -> "_Run":
        """Seed the stream, validate ``df``, and partition it per series."""
        df_nw = _with_source(
            nw.from_native(df), id_col, time_col, target_col, source_col
        )
        # Synthetic frames must match the input's dataframe library, or the
        # final concat would mix backends. An explicit engine wins.
        original = _to_output_backend(df_nw, engine)
        partitions = _partition_series(df_nw, id_col, time_col)
        return cls(
            seed=seed,
            rng=np.random.default_rng(seed),
            df=df_nw,
            out_engine=engine if engine is not None else df_nw.implementation,
            original=original,
            partitions=partitions,
            # Slice the already converted frame so every copy shares one schema.
            out_partitions=(
                partitions
                if engine is None
                else _partition_series(original, id_col, time_col, reset=True)
            ),
            columns=(id_col, time_col, target_col, source_col),
        )

    def finish(
        self, synthetic: list[nw.DataFrame], include_original: bool
    ) -> IntoFrameT:
        """Concatenate the originals (optionally) and synthetic frames."""
        id_col, time_col, target_col, source_col = self.columns
        frames = [self.original] if include_original else []
        frames.extend(synthetic)
        # Normalize concat-sensitive dtypes across backends. In particular,
        # pandas datetimes commonly convert to Polars at microsecond precision
        # while NumPy timestamps retain nanosecond precision.
        return _concat_frames(
            frames,
            self.original.schema[time_col],
            id_col,
            time_col,
            target_col,
            source_col,
        ).to_native()


def _to_output_backend(frame: nw.DataFrame, engine: Any) -> nw.DataFrame:
    """Convert an input frame when an explicit output engine was requested."""
    if engine is None:
        return frame
    return nw.DataFrame.from_dict(
        frame.to_dict(as_series=False),
        backend=engine,
    )


def _partition_series(
    frame: nw.DataFrame, id_col: str, time_col: str, reset: bool = False
) -> dict[Any, nw.DataFrame]:
    """Sort once, then take contiguous slices without per-ID panel scans.

    ``reset`` numbers each slice's pandas index from 0.
    """
    ordered = frame.sort([id_col, time_col])
    ids = ordered[id_col].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    take = nw.maybe_reset_index if reset else (lambda part: part)
    return {
        ids[start]: take(ordered[int(start) : int(stop)])
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if start < stop
    }


def _concat_frames(
    frames: list[nw.DataFrame],
    time_dtype: Any,
    id_col: str,
    time_col: str,
    target_col: str,
    source_col: str | None = None,
) -> nw.DataFrame:
    """Cast the shared columns to one schema, concatenate, and categorize ids.

    With ``source_col``, the stacked id and source values are categorized
    once and both columns get that dtype.
    """
    frames = [
        frame.with_columns(
            nw.col(id_col).cast(nw.String()),
            nw.col(time_col).cast(time_dtype),
            nw.col(target_col).cast(nw.Float64()),
        )
        for frame in frames
    ]
    out = nw.concat(frames)
    if source_col is None:
        return _categorize_ids(out, id_col)
    stacked = _categorize_ids(
        nw.concat([out.select(id_col), out.select(nw.col(source_col).alias(id_col))]),
        id_col,
    )
    if out.implementation.is_pandas_like():
        # narwhals dtypes drop pandas categories, so cast with the native one.
        dtype = stacked[id_col].to_native().dtype
        return nw.from_native(
            out.to_native().astype({id_col: dtype, source_col: dtype}),
            eager_only=True,
        )
    return out.with_columns(nw.col(id_col, source_col).cast(stacked.schema[id_col]))


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
        f"augmentation substituted AR(1) for {len(fallbacks)} of "
        f"{total} synthetic series ({counts}). "
        f"First failure: {first_gen} on series {first_id!r}: {first_err}"
    )


def _scale_window(
    window: np.ndarray, scaling: Literal["mean", "std", "none"]
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
    reference: nw.DataFrame,
    ids: list[str],
    copies: list[np.ndarray],
    backend: Any,
    id_col: str,
    target_col: str,
) -> nw.DataFrame:
    """Assemble all copies of one source in a single backend operation."""
    return nw.concat([reference] * len(ids)).with_columns(
        nw.new_series(id_col, np.repeat(ids, len(reference)), backend=backend),
        nw.new_series(target_col, np.concatenate(copies), backend=backend),
    )


def _generate_ar1_series(
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
        series[t] = mean + phi * (series[t - 1] - mean) + rng.normal(0, innovation_std)

    return series


def _match_autocorrelation(
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
    generator_name: str, fitted_params: dict, _series_values: np.ndarray
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
