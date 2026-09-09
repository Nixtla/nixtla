"""Tests for balanced_pool preset."""

import warnings

import pandas as pd
import polars as pl
import pytest

from synforecast import SynSet, balanced_pool, pretraining_pool
from synforecast.base import BaseGenerator
from synforecast.generators import (
    ETSGenerator,
    GaussianProcessGenerator,
    IntermittentDemandGenerator,
    SARIMAGenerator,
)
from synforecast.presets import _seasonal_period_from_freq


class TestBalancedPool:
    """Tests for the balanced_pool function."""

    def test_returns_list_of_generators(self) -> None:
        """Test that balanced_pool returns a list of BaseGenerator instances."""
        generators = balanced_pool()
        assert isinstance(generators, list)
        assert len(generators) == 42
        for gen in generators:
            assert isinstance(gen, BaseGenerator)

    def test_custom_length_and_freq(self) -> None:
        """Test that custom base parameters are applied to all generators."""
        generators = balanced_pool(
            min_length=50, max_length=75, freq="h", engine="polars"
        )
        for gen in generators:
            assert gen.min_length == 50
            assert gen.max_length == 75
            assert gen.freq == "h"

    def test_custom_seed(self) -> None:
        """Test that seeds are offset per generator."""
        generators = balanced_pool(seed=100)
        seeds = [gen.seed for gen in generators]
        assert seeds[0] == 100
        # Every generator gets a distinct offset from the base seed; the
        # list order interleaves niches, so offsets are unique but not
        # consecutive along the list.
        assert set(seeds) == set(range(100, 142))

    def test_none_seed(self) -> None:
        """Test that seed=None produces None seeds for all generators."""
        generators = balanced_pool(seed=None)
        for gen in generators:
            assert gen.seed is None

    def test_base_kwargs_passed(self) -> None:
        """Test that extra kwargs are forwarded to all generators."""
        generators = balanced_pool(engine="pandas")
        for gen in generators:
            assert gen.engine == "pandas"

    def test_with_synset(self) -> None:
        """Test integration with SynSet."""
        generators = balanced_pool(
            min_length=30, max_length=30, freq="D", engine="polars"
        )
        dataset = SynSet(generators)
        df = dataset.generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        n_series = df["unique_id"].n_unique()
        assert n_series == 42

    def test_all_generators_produce_output(self) -> None:
        """Test that every generator in the pool produces valid time series."""
        generators = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars"
        )
        for i, gen in enumerate(generators):
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame), (
                f"Generator {i} ({gen.alias}) did not return a DataFrame"
            )
            assert len(df) > 0, f"Generator {i} ({gen.alias}) returned empty DataFrame"

    def test_reproducibility(self) -> None:
        """Test that the same seed produces identical results."""
        gen1 = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )
        gen2 = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )

        df1 = gen1[0].generate(n_series=1)
        df2 = gen2[0].generate(n_series=1)

        assert df1["y"].to_list() == df2["y"].to_list()

    def test_diverse_generator_types(self) -> None:
        """Test that the pool contains generators from multiple classes."""
        generators = balanced_pool()
        class_names = {type(gen).__name__ for gen in generators}
        assert len(class_names) == 15
        expected = {
            "SARIMAGenerator",
            "ETSGenerator",
            "FractionalBrownianMotionGenerator",
            "RegimeSwitchingGenerator",
            "GARCHGenerator",
            "CyclicGenerator",
            "IntermittentDemandGenerator",
            "EnergyLoadGenerator",
            "IoTSensorGenerator",
            "VitalSignsGenerator",
            "GaussianProcessGenerator",
            "ChaoticSystemGenerator",
            "INARGenerator",
            "BoundedProcessGenerator",
            "LevyProcessGenerator",
        }
        assert class_names == expected

    def test_prefix_spans_all_niches(self) -> None:
        """The pool is interleaved round-robin across niches: the first 15
        entries cover all 15 generator classes, so any prefix is as
        behaviorally diverse as possible."""
        generators = balanced_pool()
        first_15_classes = [type(gen).__name__ for gen in generators[:15]]
        assert len(set(first_15_classes)) == 15
        # And every prefix of length k <= 15 has k distinct classes.
        for k in range(1, 16):
            assert len({type(g).__name__ for g in generators[:k]}) == k

    def test_interleaving_preserves_pool_contents(self) -> None:
        """Interleaving reorders the pool but keeps the same 42 configured
        variants (same class/seed-offset pairs)."""
        generators = balanced_pool(seed=0)
        assert len(generators) == 42
        seeds_by_class: dict[str, set[int]] = {}
        for gen in generators:
            seeds_by_class.setdefault(type(gen).__name__, set()).add(gen.seed)
        # Variant counts per niche are unchanged.
        expected_counts = {
            "SARIMAGenerator": 5,
            "ETSGenerator": 4,
            "FractionalBrownianMotionGenerator": 3,
            "RegimeSwitchingGenerator": 2,
            "GARCHGenerator": 2,
            "CyclicGenerator": 2,
            "IntermittentDemandGenerator": 3,
            "EnergyLoadGenerator": 2,
            "IoTSensorGenerator": 3,
            "VitalSignsGenerator": 3,
            "GaussianProcessGenerator": 4,
            "ChaoticSystemGenerator": 3,
            "INARGenerator": 2,
            "BoundedProcessGenerator": 2,
            "LevyProcessGenerator": 2,
        }
        assert {c: len(s) for c, s in seeds_by_class.items()} == expected_counts

    def test_balanced_pool_with_patterns(self) -> None:
        """Test balanced_pool with anomalies, changepoints, and missing data."""
        generators = balanced_pool(
            min_length=100,
            max_length=100,
            freq="D",
            engine="polars",
            seed=42,
            anomalies=True,
            changepoints=True,
            missing_data=True,
        )
        assert len(generators) > 0
        for gen in generators:
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0


class TestPretrainingPool:
    """Tests for the pretraining_pool function."""

    META = {"TSIGenerator", "TCMGenerator", "KernelSynthGenerator", "MARGenerator"}

    def test_includes_meta_and_balanced_by_default(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(min_length=64, max_length=64, engine="polars")
        classes = {type(g).__name__ for g in pool}
        # Meta-generators that balanced_pool excludes are present...
        assert classes >= self.META
        # ...alongside the full balanced pool.
        assert len(pool) == 42 + 4 * 3

    def test_meta_only(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(
            include_balanced=False,
            n_meta_variants=2,
            min_length=64,
            max_length=64,
            engine="polars",
        )
        assert len(pool) == 8
        assert {type(g).__name__ for g in pool} == self.META

    def test_all_seeds_unique(self) -> None:
        from synforecast import pretraining_pool

        seeds = [g.seed for g in pretraining_pool(seed=42)]
        assert len(set(seeds)) == len(seeds)

    def test_none_seed(self) -> None:
        from synforecast import pretraining_pool

        assert all(g.seed is None for g in pretraining_pool(seed=None))

    def test_rejects_zero_variants(self) -> None:
        import pytest

        from synforecast import pretraining_pool

        with pytest.raises(ValueError):
            pretraining_pool(n_meta_variants=0)

    def test_with_synset(self) -> None:
        from synforecast import SynSet, pretraining_pool

        pool = pretraining_pool(
            min_length=48, max_length=48, freq="D", engine="polars", seed=1
        )
        df = SynSet(pool).generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        assert df["unique_id"].n_unique() == len(pool)
        assert len(df) > 0


def _skip_if_alias_unsupported(freq: str) -> None:
    """Skip when the installed pandas does not know this offset alias.

    Aliases such as "ME"/"YE" appear in pandas 2.2 and legacy ones such as
    "H"/"T"/"A" are removed later, so each parametrization is only checked
    where the alias parses.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            pd.tseries.frequencies.to_offset(freq)
        except ValueError:
            pytest.skip(f"pandas {pd.__version__} does not support alias {freq!r}")


class TestSeasonalPeriodFromFreq:
    """Tests for deriving the seasonal period from a frequency."""

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [
            ("s", 60),
            ("min", 60),
            ("5min", 12),
            ("15min", 4),
            ("30min", 2),
            ("h", 24),
            ("H", 24),
            ("2h", 12),
            ("D", 7),
            ("1d", 7),
            ("B", 5),
            ("W", 52),
            ("W-MON", 52),
            ("MS", 12),
            ("ME", 12),
            ("3MS", 4),
            ("QS", 4),
            ("QE-DEC", 4),
            ("YS", 1),
            ("YE", 1),
        ],
    )
    def test_calendar_conventions(self, freq: str, expected: int) -> None:
        _skip_if_alias_unsupported(freq)
        assert _seasonal_period_from_freq(freq) == expected

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [
            ("S", 60),
            ("T", 60),
            ("15T", 4),
            ("H", 24),
            ("2H", 12),
            ("M", 12),
            ("Q", 4),
            ("Q-DEC", 4),
            ("A", 1),
            ("Y", 1),
            ("A-DEC", 1),
        ],
    )
    def test_legacy_aliases(self, freq: str, expected: int) -> None:
        # Aliases used by pandas 2.0-2.1 and deprecated since; the lookup must
        # not depend on the version-specific rule code.
        _skip_if_alias_unsupported(freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert _seasonal_period_from_freq(freq) == expected

    def test_integer_freq_uses_default(self) -> None:
        assert _seasonal_period_from_freq(1) == 12
        assert _seasonal_period_from_freq(7, default=4) == 4

    def test_offset_without_convention_uses_default(self) -> None:
        assert _seasonal_period_from_freq("ms") == 12

    def test_never_below_one(self) -> None:
        assert _seasonal_period_from_freq("36h") == 1
        assert _seasonal_period_from_freq("5YS") == 1


def _seasonal_period_of(gen: BaseGenerator) -> int | None:
    """Seasonal period of a pool member, or None for non-seasonal variants."""
    if isinstance(gen, SARIMAGenerator) and (gen.P or gen.D or gen.Q):
        return gen.seasonal_period
    if isinstance(gen, ETSGenerator) and gen.seasonal_type is not None:
        return gen.seasonal_period
    if (
        isinstance(gen, IntermittentDemandGenerator)
        and gen.intermittent_pattern == "seasonal"
    ):
        return gen.seasonal_period
    if isinstance(gen, GaussianProcessGenerator) and gen.kernel == "periodic":
        return int(gen.period)
    return None


class TestBalancedPoolSeasonalPeriod:
    """Tests that balanced_pool's seasonal variants follow freq."""

    @staticmethod
    def _seasonal_periods(generators: list[BaseGenerator]) -> set[int]:
        periods = {_seasonal_period_of(gen) for gen in generators}
        periods.discard(None)
        return periods  # type: ignore[return-value]

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [("h", 24), ("H", 24), ("D", 7), ("W", 52), ("MS", 12), ("QS", 4), (1, 12)],
    )
    def test_derived_from_freq(self, freq: str | int, expected: int) -> None:
        if isinstance(freq, str):
            _skip_if_alias_unsupported(freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generators = balanced_pool(freq=freq)
        assert self._seasonal_periods(generators) == {expected}

    def test_covers_every_seasonal_variant(self) -> None:
        generators = balanced_pool(freq="h")
        seasonal = [g for g in generators if _seasonal_period_of(g) is not None]
        # 2 SARIMA + 2 ETS + 1 intermittent + 1 GP
        assert len(seasonal) == 6

    def test_non_seasonal_variants_unchanged(self) -> None:
        generators = balanced_pool(freq="h")
        plain = [
            gen
            for gen in generators
            if isinstance(gen, SARIMAGenerator) and not (gen.P or gen.D or gen.Q)
        ]
        assert len(plain) == 3
        assert all(gen.seasonal_period == 1 for gen in plain)

    def test_explicit_override(self) -> None:
        generators = balanced_pool(freq="h", seasonal_period=168)
        assert self._seasonal_periods(generators) == {168}

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError, match="seasonal_period"):
            balanced_pool(seasonal_period=0)

    def test_generates_with_derived_period(self) -> None:
        generators = balanced_pool(
            min_length=60, max_length=60, freq="h", engine="polars"
        )
        for gen in generators:
            df = gen.generate(n_series=1)
            assert len(df) == 60, gen.alias

    def test_pretraining_pool_forwards_override(self) -> None:
        pool = pretraining_pool(
            min_length=64, max_length=64, freq="h", seasonal_period=168
        )
        assert self._seasonal_periods(pool) == {168}

    def test_pretraining_pool_derives_from_freq(self) -> None:
        pool = pretraining_pool(min_length=64, max_length=64, freq="MS")
        assert self._seasonal_periods(pool) == {12}
